"""
MIL Attention Pooling + Weighted Multi-Task BCE Clinical Outcome Predictor
===========================================================================

Overview
--------
This script implements a multiple-instance learning (MIL) attention pooling model
(MILAttentionPool) trained with binary cross-entropy (BCE) on clinical admission
data. Each admission's ICD-9 diagnosis codes are converted to 4-character prefix
tokens, embedded, scored with a single-head attention layer, and pooled as an
attention-weighted sum of raw token embeddings. The pooled admission representation
is passed through a small LayerNorm + MLP head to produce one scalar risk score.
Training follows a two-stage weighted multi-task strategy: Stage 1 trains separate
single-task attention models to estimate validation BCE per target, then Stage 2
trains one shared scalar-score model using weights derived from Stage-1 validation
performance. Tasks that are single-class or have no valid training labels are
skipped and assigned zero weight.

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.
2.  Load train/val/test admission CSVs from the CORE dataset and require the
    clinical outcome columns:
        mortality, mortality_30d, long_stay, long_stay_defined, icu_transfer
3.  Load diagnosis ICD-9 CSV.gz files in chunks of 2M rows. Build a TRAIN-only
    vocabulary of 4-character ICD prefix tokens with PAD=0 and UNK=1.
4.  For each admission, collect its ICD prefix tokens from the diagnosis file,
    preserving duplicates and order; truncate to MAX_CODES_PER_ADMISSION=256.
5.  Encode token lists to integer IDs using the TRAIN vocabulary. Admissions with
    no diagnosis tokens are represented as empty sequences and padded during
    batching.
6.  Build label matrices Y (N, T) and validity masks M (N, T):
        - mortality, mortality_30d, and icu_transfer are valid when non-missing.
        - long_stay validity is controlled by long_stay_defined.
7.  Detect TRAIN targets that are single-class or have no valid labels. These
    targets are skipped in Stage 1 and assigned weight 0 in Stage 2.
8.  Create PyTorch datasets and dataloaders using collate_batch:
        - pad ICD code sequences with PAD=0
        - enforce minimum batch sequence length of 1
        - return padded codes, sequence lengths, labels, and masks.
9.  Stage 1 — Per-target single-task BCE training:
        - Train a fresh MILAttentionPool model for each usable target.
        - Train for EPOCHS_STAGE1=10 epochs using AdamW.
        - Optimize masked single-task BCE over valid labels only.
        - Select the best checkpoint by lowest per-task validation BCE.
        - Record best validation BCE for each target.
10. Derive Stage-2 task weights from Stage-1 validation BCEs:
        score_t  = max(log(2) - val_bce_t, 0)
        weight_t = clip((max_score / max(score_t, WEIGHT_EPS))^WEIGHT_ALPHA,
                        1/WEIGHT_CAP, WEIGHT_CAP)
    Skipped or unusable tasks receive weight 0. Active weights are normalized to
    mean 1. If all scores are zero, the script falls back to uniform weights over
    non-skipped tasks.
11. Stage 2 — Weighted multi-task BCE training:
        - Train one MILAttentionPool model that outputs a single scalar score.
        - Broadcast the scalar score across all tasks.
        - Compute normalized weighted masked BCE:
              numerator   = sum_t w_t * sum_i BCE(score_i, y_it)
              denominator = sum_t w_t * number_of_valid_labels_t
              loss        = numerator / denominator
        - Train for EPOCHS_STAGE2=10 epochs using AdamW.
        - Select the best checkpoint by lowest weighted validation BCE.
12. Score the TEST split using the best Stage-2 model.
13. Report TEST correlations and dependence metrics between the scalar score and
    each clinical outcome:
        Distance Correlation, and Mutual Information.

Architecture: MILAttentionPool
-------------------------------
- Embedding layer:
      vocab_size -> EMB_DIM=128, padding_idx=0
- Single-head attention layer:
      Linear(EMB_DIM -> 1)
- Attention masking:
      PAD positions are filled with -inf before softmax.
- Attention pooling:
      softmax attention weights over valid diagnosis tokens
      weighted sum of raw token embeddings
- Head network:
      LayerNorm(EMB_DIM)
      Linear(EMB_DIM -> MLP_HIDDEN=128)
      ReLU
      Linear(MLP_HIDDEN -> 1)
- Output:
      one scalar logit per admission, shape (B, 1)

Loss Functions
--------------
Single-task masked BCE:
    mean BCEWithLogitsLoss over admissions with valid labels for one target.

Multi-task weighted masked BCE:
    The scalar admission score is broadcast to all targets, invalid labels are
    masked out, and task weights derived from Stage 1 control each target's
    contribution. The loss is normalized by the weighted count of valid labels.

Evaluation Metrics
------------------
For each target on the TEST split, the script computes:
    - Distance correlation
    - Mutual information in nats

Mutual information uses sklearn.feature_selection.mutual_info_classif with a
fixed EVAL_SEED=12345 and up to MI_BASE_NEIGHBORS=5 neighbors.

Key Configuration
-----------------
  SEED: controlled via SEED environment variable, default 1001
  EVAL_SEED=12345
  BATCH_SIZE=256
  EPOCHS_STAGE1=10
  EPOCHS_STAGE2=10
  LR=1e-3
  EMB_DIM=128
  MLP_HIDDEN=128
  WEIGHT_DECAY=0.0
  NUM_WORKERS=4
  PIN_MEMORY=True
  PREFIX_LEN=4
  MAX_CODES_PER_ADMISSION=256
  DX_CHUNKSIZE=2_000_000
  WEIGHT_EPS=0.02
  WEIGHT_ALPHA=0.25
  WEIGHT_CAP=3.0
  MI_BASE_NEIGHBORS=5
  DEVICE: cuda if available, otherwise cpu

Expected Directory Layout
--------------------------
    <data_root>/
    └── core/
        ├── admissions_core_train.csv
        ├── admissions_core_val.csv
        ├── admissions_core_test.csv
        ├── diagnoses_icd9_core_train.csv.gz
        ├── diagnoses_icd9_core_val.csv.gz
        └── diagnoses_icd9_core_test.csv.gz

Required admission columns:
    hadm_id, mortality, mortality_30d, long_stay, long_stay_defined, icu_transfer

Required diagnosis columns:
    hadm_id, icd_code

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor torch

Running
-------
    python MILAttentionPooling_mimiciii.py --data_root /path/to/final_datasets

Override training seed:
    SEED=42 python MILAttentionPooling_mimiciii.py --data_root /path/to/final_datasets

Example:
    python MILAttentionPooling_mimiciii.py --data_root ./data/final_datasets
"""

################################################################
# IMPORTS
################################################################
import os
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor


################################################################
# CONFIG
################################################################
SEED = int(os.environ.get("SEED", 1001))   # training seed varies per run
EVAL_SEED = 12345                        # fixed for MI eval determinism
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)
BATCH_SIZE = 256
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10
LR = 1e-3
EMB_DIM = 128
MLP_HIDDEN = 128
WEIGHT_DECAY = 0.0
NUM_WORKERS = 4
PIN_MEMORY = True
PREFIX_LEN = 4
MAX_CODES_PER_ADMISSION = 256
DX_CHUNKSIZE = 2_000_000
WEIGHT_EPS   = 0.02   # floor for score when computing weights (try 0.01–0.05)
WEIGHT_ALPHA = 0.25   # gentler than sqrt (sqrt would be 0.5)
WEIGHT_CAP   = 3.0    # max ratio cap (prevents domination)
MI_BASE_NEIGHBORS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to final_datasets directory containing core/admissions_core_train.csv, etc.",
    )
    return parser.parse_args()


ARGS = parse_args()
FINAL_ROOT = Path(ARGS.data_root)

DATASETS = {
    "CORE": {
        "adm_train": FINAL_ROOT / "core" / "admissions_core_train.csv",
        "adm_val":   FINAL_ROOT / "core" / "admissions_core_val.csv",
        "adm_test":  FINAL_ROOT / "core" / "admissions_core_test.csv",
        "dx_train":  FINAL_ROOT / "core" / "diagnoses_icd9_core_train.csv.gz",
        "dx_val":    FINAL_ROOT / "core" / "diagnoses_icd9_core_val.csv.gz",
        "dx_test":   FINAL_ROOT / "core" / "diagnoses_icd9_core_test.csv.gz",
    },
}


################################################################
# REPRO
################################################################
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(SEED)

################################################################
# UTILS
################################################################
def require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

def norm_icd(code) -> str:
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()

def to_prefix(code: str, k: int) -> str:
    c = norm_icd(code)
    return c[:k] if c else ""

def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def task_is_single_class(Y: np.ndarray, M: np.ndarray, j: int) -> bool:
    valid = M[:, j].astype(bool)
    if valid.sum() < 1:
        return True
    yy = Y[valid, j].astype(int)
    return np.unique(yy).size < 2

################################################################
# DATA LOADING AND PREPROCESSING
################################################################

def build_vocab_from_dx_file(dx_path: Path, chunksize: int = DX_CHUNKSIZE) -> dict[str, int]:
    toks = set()
    for chunk in pd.read_csv(dx_path, usecols=["icd_code"], chunksize=chunksize):
        for raw in chunk["icd_code"].to_numpy():
            tok = to_prefix(raw, PREFIX_LEN)
            if tok:
                toks.add(tok)

    toks = sorted(toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, t in enumerate(toks, start=2):
        vocab[t] = i
    return vocab


def load_dx_prefix_map(dx_path: Path, chunksize: int = DX_CHUNKSIZE) -> dict[int, list[str]]:
    if not dx_path.exists():
        raise FileNotFoundError(f"Missing diagnoses file: {dx_path}")

    hadm_to_list: dict[int, list[str]] = {}
    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=chunksize):
        hadm_arr = chunk["hadm_id"].astype(int).to_numpy()
        code_arr = chunk["icd_code"].to_numpy()

        for hadm, raw in zip(hadm_arr, code_arr):
            tok = to_prefix(raw, PREFIX_LEN)
            if tok == "":
                continue
            hadm_to_list.setdefault(int(hadm), []).append(tok)  # keep duplicates + order

    # truncate like NHSIC scripts
    for h, lst in hadm_to_list.items():
        if len(lst) > MAX_CODES_PER_ADMISSION:
            hadm_to_list[h] = lst[:MAX_CODES_PER_ADMISSION]

    return hadm_to_list

def encode_admissions(adm: pd.DataFrame, dx_map: dict[int, list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    unk = vocab["<UNK>"]
    out = []
    for h in adm["hadm_id"].astype(int).tolist():
        toks = dx_map.get(int(h), [])
        if not toks:
            out.append(np.asarray([], dtype=np.int64))  # empty list (collate pads)
            continue
        ids = [vocab.get(t, unk) for t in toks]  # keep duplicates + order
        if len(ids) > MAX_CODES_PER_ADMISSION:
            ids = ids[:MAX_CODES_PER_ADMISSION]
        out.append(np.asarray(ids, dtype=np.int64))
    return out

def build_Y_and_mask(df: pd.DataFrame, targets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Y: float32 (N,T) with 0-filled where invalid
    M: float32 (N,T) valid mask (1=valid)
    long_stay validity uses long_stay_defined.
    """
    N = len(df)
    T = len(targets)
    Y = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)

    for j, t in enumerate(targets):
        if t == "long_stay":
            require_cols(df, ["long_stay", "long_stay_defined"], "admissions")
            valid = df["long_stay_defined"].fillna(0).astype(int).to_numpy().astype(bool)
            y = df["long_stay"].fillna(0).astype(int).to_numpy()
            Y[:, j] = y.astype(np.float32)
            M[:, j] = valid.astype(np.float32)
        else:
            require_cols(df, [t], "admissions")
            s = pd.to_numeric(df[t], errors="coerce")
            valid = (~s.isna()).to_numpy(dtype=bool)
            y = s.fillna(0).astype(int).to_numpy()
            Y[:, j] = y.astype(np.float32)
            M[:, j] = valid.astype(np.float32)

    return Y, M


################################################################
# EVALUATION METRICS
################################################################



def kendall_from_score(score: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = valid_mask.astype(bool)
    if valid.sum() < 3:
        return 0.0

    ss = score[valid].astype(float)
    yy = y[valid].astype(float)

    if np.std(ss) < 1e-8 or np.std(yy) < 1e-8:
        return 0.0

    try:
        tau, _ = kendalltau(ss, yy, variant="b")  # best (tie-aware)
    except TypeError:
        tau, _ = kendalltau(ss, yy)               # older SciPy fallback

    return 0.0 if not np.isfinite(tau) else float(tau)

def pearson_from_score(score: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = valid_mask.astype(bool)
    if valid.sum() < 3:
        return 0.0
    yy = y[valid].astype(float)
    ss = score[valid].astype(float)
    if np.std(ss) < 1e-8 or np.std(yy) < 1e-8:
        return 0.0
    corr, _ = pearsonr(ss, yy)
    return float(corr)

def dcor_from_score(score: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
    valid = valid_mask.astype(bool)
    if valid.sum() < 3:
        return 0.0
    yy = y[valid].astype(float)
    ss = score[valid].astype(float)
    return float(dcor.distance_correlation(ss, yy))

def mi_nats_masked(
    score: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    seed: int = EVAL_SEED,
    base_neighbors: int = MI_BASE_NEIGHBORS,
) -> float:
    valid = valid.astype(bool)
    if valid.sum() < 3:
        return 0.0
    yy = y[valid].astype(int)
    if np.unique(yy).size < 2:
        return 0.0
    nn = int(min(base_neighbors, valid.sum() - 1))
    if nn < 1:
        return 0.0
    s_valid = zscore_1d(score[valid]).reshape(-1, 1)
    return float(mutual_info_classif(s_valid, yy, random_state=seed, n_neighbors=nn)[0])

################################################################
# DATASET + COLLATE
################################################################

class ICDSetDataset(Dataset):
    def __init__(self, codes_list, Y, M):
        self.codes_list = codes_list
        self.Y = Y.astype(np.float32)
        self.M = M.astype(np.float32)

    def __len__(self):
        return len(self.codes_list)

    def __getitem__(self, idx):
        return self.codes_list[idx], self.Y[idx], self.M[idx]


def collate_batch(batch):
    codes, Y, M = zip(*batch)
    lengths = torch.tensor([len(x) for x in codes], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(codes) > 0 else 0
    max_len = max(max_len, 1)

    x = torch.zeros((len(codes), max_len), dtype=torch.long)  # PAD=0
    for i, seq in enumerate(codes):
        if len(seq) == 0:
            continue
        x[i, :len(seq)] = torch.from_numpy(seq)

    Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)  # (B,T)
    M = torch.tensor(np.stack(M, axis=0), dtype=torch.float32)  # (B,T)
    return x, lengths, Y, M


# -------------------------
# MODEL (ONE scalar score)
# -------------------------
class MILAttentionPool(nn.Module):
    """
    Pure MIL attention pooling (single-head):
      Embedding -> attention weights -> weighted sum -> small head -> scalar
    No token-wise DeepSets "phi" MLP.
    """
    def __init__(self, vocab_size: int, emb_dim: int, head_hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # attention on raw embeddings
        self.attn = nn.Linear(emb_dim, 1)

        # small post-pool head (can be 0-layer = Linear(emb_dim,1) for ultra-minimal)
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        E = self.emb(x_codes)                 # (B,L,D)
        mask = (x_codes != 0)                 # (B,L)

        logits = self.attn(E).squeeze(-1)     # (B,L)
        logits = logits.masked_fill(~mask, float("-inf"))

        w = torch.softmax(logits, dim=1)      # (B,L)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        pooled = torch.sum(E * w.unsqueeze(-1), dim=1)  # (B,D)
        return self.head(pooled)



# -------------------------
# LOSS HELPERS
# -------------------------
def masked_single_task_bce(
    s: torch.Tensor,     # (B,) scalar logit
    y: torch.Tensor,     # (B,) labels in {0,1}
    m: torch.Tensor,     # (B,) mask in {0,1}
) -> torch.Tensor:
    """
    mean_{i: m_i=1} BCEWithLogits(s_i, y_i)
    Returns 0*s.sum() if no valid labels in batch, so backward() is safe.
    """
    valid = (m > 0.5)
    if valid.sum().item() < 1:
        return s.sum() * 0.0
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    return loss_fn(s[valid], y[valid])


def masked_multitask_bce(
    s: torch.Tensor,          # (B,) scalar logit
    Y: torch.Tensor,          # (B,T)
    M: torch.Tensor,          # (B,T) mask in {0,1}
    weights: torch.Tensor,    # (T,)
) -> torch.Tensor:
    """
    NORMALIZED weighted masked BCE (matches set_transformer_weighted_bce_2stage_weighted.py):
      num = sum_t w_t * sum_{i:valid} BCE(i,t)
      den = sum_t w_t * (#valid_t)
      loss = num / den
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    B, T = Y.shape

    # broadcast scalar score to all tasks
    s2 = s.view(-1, 1).expand(-1, T)  # (B,T)

    per_elem = loss_fn(s2, Y)         # (B,T)
    per_elem = per_elem * M           # mask invalid

    per_task_sum = per_elem.sum(dim=0)        # (T,)
    valid_counts = M.sum(dim=0)               # (T,)

    num = (weights * per_task_sum).sum()
    den = (weights * valid_counts).sum()

    if den.item() <= 0:
        return s.sum() * 0.0
    return num / den


@torch.no_grad()
def predict_scores(model, loader) -> np.ndarray:
    model.eval()
    scores = []
    for x, lengths, _Y, _M in loader:
        x = x.to(DEVICE)
        lengths = lengths.to(DEVICE)
        s = model(x, lengths).squeeze(1).detach().cpu().numpy()
        scores.append(s)
    return np.concatenate(scores, axis=0)


@torch.no_grad()
def eval_single_task_val_loss(model, loader, task_j: int) -> float:
    """
    MATCH SetTransformer stage-1: mean over batches of (mean BCE over valid in that batch).
    Returns +inf if there are no valid labels across all batches.
    """
    model.eval()
    losses = []
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for x, lengths, Yb, Mb in loader:
        x = x.to(DEVICE)
        lengths = lengths.to(DEVICE)
        Yb = Yb.to(DEVICE)
        Mb = Mb.to(DEVICE)

        s = model(x, lengths).squeeze(1)  # (B,)
        y = Yb[:, task_j]
        m = Mb[:, task_j]
        valid = (m > 0.5)

        if valid.sum().item() < 1:
            continue

        per = loss_fn(s, y)                 # (B,)
        batch_mean = per[valid].mean()      # mean over valid in THIS batch
        losses.append(float(batch_mean.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("inf")



@torch.no_grad()
def eval_multitask_val_loss(model, loader, weights: torch.Tensor) -> float:
    """
    Computes the weighted masked multitask BCE over the ENTIRE val loader:
      average of per-batch losses (like training loop)
    This matches the existing baseline behavior (mean over batches).
    Returns +inf if loader is empty.
    """
    model.eval()
    losses = []
    for x, lengths, Yb, Mb in loader:
        x = x.to(DEVICE)
        lengths = lengths.to(DEVICE)
        Yb = Yb.to(DEVICE)
        Mb = Mb.to(DEVICE)

        s = model(x, lengths).squeeze(1)
        loss = masked_multitask_bce(s, Yb, Mb, weights=weights)
        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        return float("inf")
    return float(np.mean(losses))


# -------------------------
# TRAINING: STAGE 1
# -------------------------
def train_stage1_single_task(
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_j: int,
    task_name: str,
) -> tuple[dict, float]:
    """
    Stage-1: Train a single-task model for task_j for EPOCHS_STAGE1 epochs.
    Select best checkpoint by masked val loss for this task (lower is better).
    Always runs full EPOCHS_STAGE1 epochs.
    Returns: best_state_dict, best_val_loss
    """
    model = MILAttentionPool(vocab_size=vocab_size, emb_dim=EMB_DIM, head_hidden=MLP_HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        tr_losses = []

        for x, lengths, Yb, Mb in train_loader:
            x = x.to(DEVICE)
            lengths = lengths.to(DEVICE)
            Yb = Yb.to(DEVICE)
            Mb = Mb.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x, lengths).squeeze(1)       # (B,)
            y = Yb[:, task_j]                      # (B,)
            m = Mb[:, task_j]                      # (B,)
            loss = masked_single_task_bce(s, y, m)  # scalar
            loss.backward()
            opt.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

        val_loss = eval_single_task_val_loss(model, val_loader, task_j=task_j)

        print(f"    [Stage1 task={task_name}] Epoch {ep:02d}/{EPOCHS_STAGE1} | train_bce={tr_loss:.6f} | val_bce={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, float(best_val)


# -------------------------
# TRAIN/EVAL
# -------------------------
def get_eval_y_valid(te: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "long_stay":
        valid = te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
        y = te["long_stay"].fillna(0).astype(int).to_numpy()
        return y, valid
    else:
        s = pd.to_numeric(te[target], errors="coerce")
        valid = (~s.isna()).to_numpy(dtype=bool)
        y = s.fillna(0).astype(int).to_numpy()
        return y, valid


def train_one_dataset(ds_name: str):
    cfg = DATASETS[ds_name]

    print("\n" + "=" * 100)
    print(f"DATASET: {ds_name}")
    print("=" * 100)

    tr = pd.read_csv(cfg["adm_train"])
    va = pd.read_csv(cfg["adm_val"])
    te = pd.read_csv(cfg["adm_test"])

    require_cols(tr, ["hadm_id"], ds_name + "/TRAIN")
    require_cols(va, ["hadm_id"], ds_name + "/VAL")
    require_cols(te, ["hadm_id"], ds_name + "/TEST")

    core_targets = ["mortality", "mortality_30d", "long_stay", "icu_transfer"]

    if ds_name == "CORE":
        train_targets = core_targets
        eval_targets  = core_targets
        require_cols(tr, core_targets + ["long_stay_defined"], ds_name + "/TRAIN")
        require_cols(va, core_targets + ["long_stay_defined"], ds_name + "/VAL")
        require_cols(te, core_targets + ["long_stay_defined"], ds_name + "/TEST")
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    print(f"TRAIN n={len(tr):,} | VAL n={len(va):,} | TEST n={len(te):,}")

    # Build (Y,M) for train/val
    Ytr, Mtr = build_Y_and_mask(tr, train_targets)
    Yva, Mva = build_Y_and_mask(va, train_targets)

    # Determine which tasks are usable (skip if single-class among valid rows, NHSIC-style)
    skipped = np.zeros(len(train_targets), dtype=bool)
    for j, t in enumerate(train_targets):
        if task_is_single_class(Ytr, Mtr, j):
            skipped[j] = True
            print(f"  [TRAIN] task '{t}' is single-class (or no valid) -> skipped in Stage-1 and weight=0 in Stage-2")

    if skipped.all():
        print("All train tasks are skipped (no valid variability). Skipping training.")
        return

    # Load diagnoses + vocab (TRAIN only)
    print("\nLoading diagnoses + building vocab from TRAIN dx...")
    vocab = build_vocab_from_dx_file(cfg["dx_train"])

    dx_tr_map = load_dx_prefix_map(cfg["dx_train"])
    dx_va_map = load_dx_prefix_map(cfg["dx_val"])
    dx_te_map = load_dx_prefix_map(cfg["dx_test"])

    tr_codes = encode_admissions(tr, dx_tr_map, vocab)
    va_codes = encode_admissions(va, dx_va_map, vocab)
    te_codes = encode_admissions(te, dx_te_map, vocab)

    # Datasets
    train_ds = ICDSetDataset(tr_codes, Ytr, Mtr)
    val_ds   = ICDSetDataset(va_codes, Yva, Mva)

    # test: dummy Y/M (not used)
    Yte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    Mte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    test_ds  = ICDSetDataset(te_codes, Yte0, Mte0)

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=collate_batch
    )

    # -------------------------
    # STAGE 1: per-task single-task models -> derive weights
    # -------------------------
    print("\n=== Stage 1: Single-task BCE models (select by VAL masked BCE) ===")
    stage1_val_losses = np.full(len(train_targets), np.inf, dtype=np.float64)

    for j, t in enumerate(train_targets):
        if skipped[j]:
            print(f"\n  Skipping Stage-1 model for {t}: single-class (or no valid labels) in TRAIN")
            stage1_val_losses[j] = np.inf
            continue

        print(f"\n  Training Stage-1 single-task model for: {t}")
        _best_state, best_val = train_stage1_single_task(
            vocab_size=len(vocab),
            train_loader=train_loader,
            val_loader=val_loader,
            task_j=j,
            task_name=t,
        )
        stage1_val_losses[j] = float(best_val)
        print(f"  Best VAL masked BCE for {t}: {float(best_val):.6f}")

        # Compute weights from Stage-1 VAL losses (harder => higher loss => higher weight)
    finite_mask = np.isfinite(stage1_val_losses) & (~skipped)
    if not finite_mask.any():
        print("No usable Stage-1 validation losses found. Skipping Stage-2.")
        return

    LOG2 = float(math.log(2.0))

    # score_t = max(log(2) - val_bce_t, 0)
    score = np.maximum(LOG2 - stage1_val_losses, 0.0)
    score[~np.isfinite(score)] = 0.0
    score[skipped] = 0.0

    max_score = float(score.max()) if score.size > 0 else 0.0

    eps   = WEIGHT_EPS
    alpha = WEIGHT_ALPHA
    cap   = WEIGHT_CAP

    # If everything is impossible (all scores ~0), just use uniform weights on active tasks
    if max_score <= 0.0:
        weights = np.ones_like(score, dtype=np.float32)
    else:
        denom = np.maximum(score, eps)
        weights = (np.maximum(max_score, eps) / denom) ** alpha
        weights = np.clip(weights, 1.0 / cap, cap).astype(np.float32)

    # skipped tasks => 0
    weights[skipped] = 0.0

    # normalize active weights to mean 1 (keeps objective scale stable across runs/datasets)
    active = weights > 0
    if active.any():
        weights[active] = weights[active] / weights[active].mean()


    print("\nStage 1 VAL BCE, score_t=(log2 - BCE)+, and derived weights:")
    for j, t in enumerate(train_targets):
        tag = " (SKIPPED)" if skipped[j] else ""
        vl = stage1_val_losses[j]
        vl_str = f"{vl:.6f}" if np.isfinite(vl) else "inf"
        print(f"  {t:20s}: val_bce={vl_str:>12s} | score={float(score[j]):.6f} | w={float(weights[j]):.6f}{tag}")

    bad = ~np.isfinite(stage1_val_losses)
    score[bad] = 0.0
    weights[bad] = 0.0   # treat as unusable, like skipped

    W_t = torch.tensor(weights, device=DEVICE, dtype=torch.float32)

    if float(W_t.sum().detach().cpu().item()) == 0.0:
        print("All Stage-2 weights are zero. Skipping Stage-2.")
        return

    # -------------------------
    # STAGE 2: one multi-task model with derived weights
    # -------------------------
    print("\n=== Stage 2: Multi-task weighted masked BCE (ONE scalar) ===")
    model = MILAttentionPool(vocab_size=len(vocab), emb_dim=EMB_DIM, head_hidden=MLP_HIDDEN).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        tr_losses = []

        for x, lengths, Yb, Mb in train_loader:
            x = x.to(DEVICE)
            lengths = lengths.to(DEVICE)
            Yb = Yb.to(DEVICE)
            Mb = Mb.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x, lengths).squeeze(1)  # (B,)
            loss = masked_multitask_bce(s, Yb, Mb, weights=W_t)
            loss.backward()
            opt.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_loss = eval_multitask_val_loss(model, val_loader, weights=W_t)

        print(f"[Stage2] Epoch {ep:02d}/{EPOCHS_STAGE2} | train_wmt_bce={tr_loss:.6f} | val_wmt_bce={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Score TEST (raw logits = s(x))
    scores_test = predict_scores(model, test_loader)

    # Eval on TEST
    print("\nMetrics on TEST using AttentionPool->MLP scalar score s(x):")
 

    for t in eval_targets:
        if t not in te.columns:
            raise ValueError(f"[{ds_name}/TEST] Missing eval target column: {t}")

        y, valid = get_eval_y_valid(te, t)

        mi = mi_nats_masked(scores_test, y, valid, seed=EVAL_SEED)
        pearson = pearson_from_score(scores_test, y, valid)
        dcorr = dcor_from_score(scores_test, y, valid)


        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )

    print("=" * 100)


# -------------------------
# RUN ALL
# -------------------------
if __name__ == "__main__":
    for name in ["CORE"]:
        train_one_dataset(name)