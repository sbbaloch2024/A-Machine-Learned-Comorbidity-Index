"""
Deep Sets (Mean+Max Pooling) + Weighted Multi-Task BCE Clinical Outcome Predictor
==================================================================================

Overview
--------
This script implements a Deep Sets model with concatenated mean and max pooling
(DeepSetsSingleIndex) trained with binary cross-entropy (BCE) on clinical admission
data. Each admission's ICD-9 codes are embedded and transformed per-element via a
phi network. Mean pooling (masked) and max pooling (PAD positions filled with -inf)
are computed independently and concatenated into a 2*EMB_DIM representation, which
is passed through a rho MLP to produce a single scalar risk score. Training follows
a two-stage weighted multi-task strategy with an additional guard: tasks whose
Stage-1 VAL BCE is no better than random (≥ log(2)) are dropped from Stage-2
weighting entirely. Checkpoint selection in both stages uses validation BCE.

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.
2.  Load train/val/test admission CSVs and diagnosis ICD-9 CSV.gz files in chunks
    (2M rows). Build a TRAIN-only vocabulary of 4-char ICD prefix tokens (PAD=0,
    UNK=1) from per-admission token lists (duplicates kept, order preserved).
3.  Encode token lists to integer IDs; truncate to 256 codes per admission;
    pad into batches via collate_pad (minimum batch length 1).
4.  Build label matrices Y (N, T) and validity masks M (N, T); long_stay validity
    driven by long_stay_defined. Zero out masks for single-class TRAIN targets
    before Stage 2 (both TRAIN and VAL masks).
5.  Stage 1 — Per-target single-task BCE training (10 epochs, AdamW, LR=1e-3):
        - Fresh DeepSetsSingleIndex model per target.
        - Skip targets that are single-class or have no valid labels on TRAIN.
        - Select best checkpoint by lowest per-task VAL BCE.
        - Record best VAL BCE per task for weight derivation.
6.  Derive per-task loss weights from Stage-1 VAL BCEs:
        score_t  = max(log(2) − val_bce_t, 0)
        weight_t = clip((max_score / max(score_t, ε))^0.25, 1/3, 3)
    Tasks with val_bce ≥ log(2) (no better than random) receive weight 0 and are
    excluded from Stage-2. Remaining active weights are normalized to mean 1.
    Falls back to uniform weights on non-skipped tasks if all scores are zero.
7.  Stage 2 — Weighted multi-task BCE training (10 epochs, AdamW, LR=1e-3):
        - Single DeepSetsSingleIndex model trained on normalized weighted masked BCE
          (weighted sum of per-task BCE / weighted sum of valid counts).
        - Select best checkpoint by lowest weighted VAL BCE.
8.  Score the TEST split with the best Stage-2 model scalar output.
9.  Report Distance Correlation, and Mutual Information between model
    scores and each binary test label.

Architecture: DeepSetsSingleIndex (Mean+Max Pooling)
------------------------------------------------------
- Embedding layer (vocab_size → EMB_DIM=128), padding_idx=0
- phi network (per-element transform): Linear(128→128) → ReLU → Linear(128→128) → ReLU
- Mean pooling: masked sum of phi(E) / token count (clamped to 1.0)
- Max pooling: element-wise max of phi(E), PAD positions filled with -1e9
- Concatenation of mean and max pools → (B, 2*EMB_DIM=256)
- LayerNorm(256) applied to the concatenated representation
- rho network (output MLP): Linear(256→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1)
- Output: single scalar logit per admission (B,)

Key Configuration
-----------------
  SEED: controlled via SEED env var (default 1111); EVAL_SEED=12345 (fixed)
  EMB_DIM=128, BATCH_SIZE=256, LR=1e-3, WEIGHT_DECAY=0.0, EPOCHS=10
  PREFIX_LEN=4, MAX_CODES_PER_ADMISSION=256, CHUNK_DX=2_000_000
  WEIGHT_EPS=0.02, WEIGHT_ALPHA=0.25, WEIGHT_CAP=3.0

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
    python deepsets_minmax_pooling_mimiciii.py --data_root /path/to/final_datasets

Override training seed:
    SEED=42 python deepsets_minmax_pooling_mimiciii.py --data_root /path/to/final_datasets

Example:
    python deepsets_minmax_pooling_mimiciii.py --data_root ./data/final_datasets
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
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor
import math

################################################################
# CONFIG
################################################################

SEED = int(os.environ.get("SEED", 1111))   # training varies per run
EVAL_SEED = 12345                        # fixed for MI (and any subsampling, if added)
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)
PREFIX_LEN = 4              # ICD-9 prefix length used as token (e.g., "I509" from "I50.9")
EMB_DIM = 128
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 4
PIN_MEMORY = True
MI_BASE_NEIGHBORS = 5       # for mutual_info_classif
MAX_CODES_PER_ADMISSION = 256
WEIGHT_EPS   = 0.02   # floor for score when computing weights (try 0.01–0.05)
WEIGHT_ALPHA = 0.25   # gentler than sqrt (sqrt would be 0.5)
WEIGHT_CAP   = 3.0    # max ratio cap (prevents domination)


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
        "train_targets": ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
        "eval_targets":  ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
    },
}

################################################################
# UTILITY HELPERS
################################################################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_icd(code) -> str:
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()

def to_prefix(code: str, k: int) -> str:
    c = norm_icd(code)
    if not c:
        return ""
    return c[:k]

def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def is_single_class_train(Y: np.ndarray, M: np.ndarray, j: int) -> bool:
    valid = M[:, j].astype(bool)
    if valid.sum() == 0:
        return True
    yy = Y[valid, j].astype(int)
    return np.unique(yy).size < 2

def require_cols(df: pd.DataFrame, cols, name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

################################################################
# Data loading & preprocessing 
################################################################

CHUNK_DX = 2_000_000  # add near config

def load_dx_prefix_map_chunked(dx_path: Path, prefix_len: int, chunksize: int = CHUNK_DX) -> dict[int, list[str]]:
    hadm_to_list: dict[int, list[str]] = {}
    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=chunksize):
        hadm_arr = chunk["hadm_id"].astype(int).to_numpy()
        code_arr = chunk["icd_code"].to_numpy()

        for hadm, raw in zip(hadm_arr, code_arr):
            tok = to_prefix(raw, prefix_len)
            if tok == "":
                continue
            lst = hadm_to_list.get(hadm)
            if lst is None:
                hadm_to_list[hadm] = [tok]
            else:
                lst.append(tok)
    return hadm_to_list

# 1) vocab: add PAD + UNK
def build_vocab_from_train(dx_map):
    toks = sorted({t for lst in dx_map.values() for t in lst})
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, t in enumerate(toks, start=2):
        vocab[t] = i
    return vocab

def encode_hadm_tokens(adm_df: pd.DataFrame, dx_map: dict[int, list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    unk = vocab["<UNK>"]
    out = []
    for h in adm_df["hadm_id"].astype(int).tolist():
        toks = dx_map.get(int(h), [])

        # apples-to-apples: missing dx -> empty list
        if not toks:
            out.append(np.asarray([], dtype=np.int64))
            continue

        # keep duplicates + order (NO set(), NO sorting)
        ids = [vocab.get(t, unk) for t in toks]

        # truncate first 256 in encounter order
        if MAX_CODES_PER_ADMISSION is not None and len(ids) > MAX_CODES_PER_ADMISSION:
            ids = ids[:MAX_CODES_PER_ADMISSION]

        out.append(np.asarray(ids, dtype=np.int64))
    return out

def build_Y_and_mask(df: pd.DataFrame, targets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Y:    float32 (N,T) with 0/1 targets (NaN filled with 0)
      Mask: float32 (N,T) where 1 indicates valid label for loss
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
# DATASET + COLLATE
################################################################

class TokenDataset(Dataset):
    def __init__(self, seqs: list[np.ndarray], Y: np.ndarray, M: np.ndarray):
        self.seqs = seqs
        self.Y = Y
        self.M = M

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.Y[idx], self.M[idx]

def collate_pad(batch, pad_id: int = 0):
    seqs, Y, M = zip(*batch)
    B = len(seqs)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    max_len = int(lengths.max().item()) if B > 0 else 0
    max_len = max(max_len, 1)  # <-- ADD THIS (apples-to-apples)

    X = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        X[i, :len(s)] = torch.from_numpy(s).long()

    Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
    M = torch.tensor(np.stack(M, axis=0), dtype=torch.float32)
    return X, lengths, Y, M


################################################################
# MODEL: DeepSetsSingleIndex
################################################################
class DeepSetsSingleIndex(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.phi = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        # pooling output is now 2*emb_dim
        self.pool_ln = nn.LayerNorm(2 * emb_dim)

        # rho input dim becomes 2*emb_dim
        self.rho = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x_ids: (B, L) with PAD=0
        E = self.emb(x_ids)                 # (B, L, d)
        H = self.phi(E)                     # (B, L, d)

        pad_mask = (x_ids == 0)             # (B, L) True where PAD
        keep = (~pad_mask).float()          # (B, L) 1 where valid

        # ---- mean pooling ----
        H_mean = H * keep.unsqueeze(-1)     # mask PAD tokens
        den = keep.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1)
        pooled_mean = H_mean.sum(dim=1) / den               # (B,d)

        # ---- max pooling ----
        H_max_in = H.masked_fill(pad_mask.unsqueeze(-1), -1e9)  # PAD -> -inf
        pooled_max = H_max_in.max(dim=1).values                 # (B,d)

        # ---- concat mean + max ----
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)   # (B,2d)
        pooled = self.pool_ln(pooled)

        s = self.rho(pooled).squeeze(-1)    # (B,)
        return s

################################################################
# Loss functions (BCE variants)
################################################################

def masked_bce_loss_single_task(logits: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, task_idx: int) -> torch.Tensor:
    """
    logits: (B,) scalar s(x)
    Y:      (B,T)
    M:      (B,T) 0/1 mask
    task_idx: which task to compute loss for
    Loss: average BCE for this task only
    """
    y_t = Y[:, task_idx]
    m_t = M[:, task_idx]
    
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction="none")  # (B,)
    loss = loss * m_t
    
    denom = m_t.sum()
    if denom.item() <= 0:
        return logits.sum() * 0.0

    return loss.sum() / denom

def masked_multi_bce_loss_weighted(logits: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    NORMALIZED weighted masked BCE:
      per_elem = BCE(logits, Y) * M
      per_task_sum = sum_i per_elem(i,t)
      valid_counts = sum_i M(i,t)
      loss = sum_t w_t * per_task_sum[t] / sum_t w_t * valid_counts[t]
    """
    B, T = Y.shape
    logit_mat = logits.unsqueeze(1).expand(B, T)  # (B,T)

    per_elem = nn.functional.binary_cross_entropy_with_logits(logit_mat, Y, reduction="none")
    per_elem = per_elem * M

    per_task_sum = per_elem.sum(dim=0)        # (T,)
    valid_counts = M.sum(dim=0)               # (T,)

    num = (weights * per_task_sum).sum()
    den = (weights * valid_counts).sum()

    if den.item() <= 0:
        return logits.sum() * 0.0
    return num / den

################################################################
# TRAINING HELPERS
################################################################
@torch.no_grad()
def compute_task_loss(model, loader, task_idx: int) -> float:
    """Compute BCE loss for a specific task on a loader."""
    model.eval()
    losses = []
    for x_ids, lengths, Y, M in loader:
        x_ids = x_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        Y = Y.to(DEVICE)
        M = M.to(DEVICE)

        s = model(x_ids, lengths)
        loss = masked_bce_loss_single_task(s, Y, M, task_idx)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")

@torch.no_grad()
def compute_epoch_loss_weighted(model, loader, weights: torch.Tensor) -> float:
    model.eval()
    losses = []
    for x_ids, lengths, Y, M in loader:
        x_ids = x_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        Y = Y.to(DEVICE)
        M = M.to(DEVICE)

        s = model(x_ids, lengths)
        loss = masked_multi_bce_loss_weighted(s, Y, M, weights=weights)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("inf")

@torch.no_grad()
def predict_scores(model, loader) -> np.ndarray:
    model.eval()
    out = []
    for x_ids, lengths, _Y, _M in loader:
        x_ids = x_ids.to(DEVICE)
        lengths = lengths.to(DEVICE)
        s = model(x_ids, lengths).detach().cpu().numpy()
        out.append(s)
    return np.concatenate(out, axis=0)

def train_single_task_bce(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_idx: int,
) -> tuple[dict, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x_ids, lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids, lengths)
            loss = masked_bce_loss_single_task(s, Y, M, task_idx)
            loss.backward()
            opt.step()

        val_loss = compute_task_loss(model, val_loader, task_idx)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, float(best_val)


################################################################
# Evaluation metrics 
################################################################



def pearson_from_score(score: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
   """
   Compute Pearson correlation between scalar score and binary outcome.
   """
   valid = valid_mask.astype(bool)
   if valid.sum() < 3:
       return 0.0
  
   yy = y[valid].astype(float)
   ss = score[valid].astype(float)
  
   # Check for constant values (no variance)
   if np.std(ss) < 1e-8 or np.std(yy) < 1e-8:
       return 0.0
  
   corr, _ = pearsonr(ss, yy)  # returns (correlation, p-value)
   return float(corr)


def dcor_from_score(score: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
   """
   Compute distance correlation between scalar score and binary outcome.
   """
   valid = valid_mask.astype(bool)
   if valid.sum() < 3:
       return 0.0
  
   yy = y[valid].astype(float)
   ss = score[valid].astype(float)
  
   dc = dcor.distance_correlation(ss, yy)
   return float(dc)


def mi_nats_masked(
    score: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    seed: int = EVAL_SEED,
    base_neighbors: int = MI_BASE_NEIGHBORS
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

    # z-score ONLY within valid rows (apples-to-apples with NHSIC scripts)
    s_valid = zscore_1d(score[valid]).reshape(-1, 1)
    return float(mutual_info_classif(s_valid, yy, random_state=seed, n_neighbors=nn)[0])



################################################################
# MAIN PIPELINE
################################################################

def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]

    print("\n" + "=" * 100)
    print(f"DATASET: {ds_name}")
    print("=" * 100)

    tr = pd.read_csv(cfg["adm_train"])
    va = pd.read_csv(cfg["adm_val"])
    te = pd.read_csv(cfg["adm_test"])

    train_targets = cfg["train_targets"]
    eval_targets  = cfg["eval_targets"]

    require_cols(tr, ["hadm_id"], f"{ds_name}/TRAIN")
    require_cols(va, ["hadm_id"], f"{ds_name}/VAL")
    require_cols(te, ["hadm_id"], f"{ds_name}/TEST")
    require_cols(tr, train_targets, f"{ds_name}/TRAIN")
    require_cols(va, train_targets, f"{ds_name}/VAL")
    require_cols(te, eval_targets,  f"{ds_name}/TEST")

    print(f"TRAIN n={len(tr):,} | VAL n={len(va):,} | TEST n={len(te):,}")

    # Build labels + masks
    Ytr, Mtr = build_Y_and_mask(tr, train_targets)
    Yva, Mva = build_Y_and_mask(va, train_targets)

    # -------------------------
    # SKIP TASKS THAT ARE SINGLE-CLASS IN TRAIN (apples-to-apples with HSIC)
    # -------------------------
    skipped = np.zeros(len(train_targets), dtype=bool)

    for j, t in enumerate(train_targets):
        if is_single_class_train(Ytr, Mtr, j):
            skipped[j] = True
            Mtr[:, j] = 0.0   # remove from Stage-2 TRAIN loss
            Mva[:, j] = 0.0   # remove from Stage-2 VAL loss (early stopping)


    # Diagnostics: positive rates on valid rows
    print("\nTRAIN label positive rates (valid-only):")
    for j, t in enumerate(train_targets):
        valid = Mtr[:, j].astype(bool)
        if valid.sum() == 0:
            print(f"  {t:16s}: n_valid=0")
        else:
            print(f"  {t:16s}: {float(Ytr[valid, j].mean()):.6f} (n_valid={int(valid.sum())})")

    # Load dx maps
    print("\nLoading diagnoses + building TRAIN vocab...")
    dx_tr = load_dx_prefix_map_chunked(cfg["dx_train"], PREFIX_LEN)
    dx_va = load_dx_prefix_map_chunked(cfg["dx_val"], PREFIX_LEN)
    dx_te = load_dx_prefix_map_chunked(cfg["dx_test"], PREFIX_LEN)

    vocab = build_vocab_from_train(dx_tr)
    vocab_size = len(vocab)   # (already includes PAD/UNK)
    print(f"Vocab size (TRAIN ICD9 prefixes): {len(vocab):,} (+PAD)")

    # Encode sequences
    tr_seqs = encode_hadm_tokens(tr, dx_tr, vocab)
    va_seqs = encode_hadm_tokens(va, dx_va, vocab)
    te_seqs = encode_hadm_tokens(te, dx_te, vocab)

    # Dataloaders
    train_ds = TokenDataset(tr_seqs, Ytr, Mtr)
    val_ds   = TokenDataset(va_seqs, Yva, Mva)

    # For test we don't need Y/M in loader, but keep placeholders
    Yte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    Mte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    test_ds  = TokenDataset(te_seqs, Yte0, Mte0)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_pad
    )

    # =========================
    # STAGE 1: Single-task models
    # =========================
    print("\n=== Stage 1: Single-task BCE models ===")
    stage1_val = {}

    for j, t in enumerate(train_targets):
        if skipped[j]:
            print(f"  Skipping single-task model for {t}: single-class in TRAIN")
            stage1_val[t] = float("inf")   # recommended for skipped
            continue

        print(f"  Training single-task model for: {t}")
        model = DeepSetsSingleIndex(vocab_size=vocab_size, emb_dim=EMB_DIM).to(DEVICE)

        _, best_val = train_single_task_bce(model, train_loader, val_loader, j)
        print(f"    best VAL BCE: {best_val:.6f}")

        stage1_val[t] = best_val


    # =========================
    # Compute weights from Stage-1 VAL BCE (with "below-random => drop" guard)
    # =========================
    vals = np.array([float(stage1_val[t]) for t in train_targets], dtype=np.float64)

    LOG2 = float(np.log(2.0))

    # tasks that are not better than random on VAL (or invalid) -> drop from weighting
    below_random = (~np.isfinite(vals)) | (vals >= LOG2)

    # quality score (higher is better); 0 for below-random anyway
    score = np.maximum(LOG2 - vals, 0.0)

    # skipped tasks => score=0
    score[skipped] = 0.0
    score[~np.isfinite(score)] = 0.0

    eps   = WEIGHT_EPS
    alpha = WEIGHT_ALPHA
    cap   = WEIGHT_CAP

    # Active tasks for weighting: not skipped AND better than random
    active_mask = (~skipped) & (~below_random) & (score > 0)

    weights = np.zeros_like(score, dtype=np.float32)

    if active_mask.any():
        max_score = float(score[active_mask].max())

        denom = np.maximum(score, eps)
        w = (np.maximum(max_score, eps) / denom) ** alpha
        w = np.clip(w, 1.0 / cap, cap).astype(np.float32)

        # only keep weights for active tasks; others remain 0
        weights[active_mask] = w[active_mask]

        # mean-normalize active weights to mean 1
        weights[active_mask] /= weights[active_mask].mean()
    else:
        print("No tasks are better than random on VAL; using uniform weights on non-skipped tasks.")
        weights[~skipped] = 1.0
        weights[~skipped] /= weights[~skipped].mean()

    W_t = torch.tensor(weights, device=DEVICE, dtype=torch.float32)


    if float(W_t.sum().detach().cpu().item()) == 0.0:
        print("All Stage-2 weights are zero. Skipping Stage-2.")
        return

    print("\nDerived Stage-2 weights (after cap + mean-normalize):")
    for j, t in enumerate(train_targets):
        tag = " (SKIPPED)" if skipped[j] else ""
        print(f"  {t:16s}: w={float(weights[j]):.6f}{tag}")

    # =========================
    # STAGE 2: Weighted multi-task BCE
    # =========================
    print("\n=== Stage 2: Weighted multi-task BCE model (ONE scalar) ===")
    model = DeepSetsSingleIndex(vocab_size=vocab_size, emb_dim=EMB_DIM).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x_ids, lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            lengths = lengths.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids, lengths)
            loss = masked_multi_bce_loss_weighted(s, Y, M, weights=W_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        tr_loss = float(np.mean(losses)) if losses else float("inf")
        va_loss = compute_epoch_loss_weighted(model, val_loader, weights=W_t)


        print(f"Epoch {ep:02d}/{EPOCHS} | train_BCE={tr_loss:.6f} | val_BCE={va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict scalar score on TEST
    score_test = predict_scores(model, test_loader)

    # Metrics on TEST vs eval_targets
    print("\nTEST Metrics (DistCorr, MI_nats) using DeepSets 2-stage WEIGHTED scalar score s(x):")

    for t in eval_targets:
        if t == "long_stay":
            require_cols(te, ["long_stay", "long_stay_defined"], f"{ds_name}/TEST")
            valid = te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            y = te["long_stay"].fillna(0).astype(int).to_numpy()
        else:
            s = pd.to_numeric(te[t], errors="coerce")
            valid = (~s.isna()).to_numpy(dtype=bool)
            y = s.fillna(0).astype(int).to_numpy()

        pearson = pearson_from_score(score_test, y, valid)
        dcorr = dcor_from_score(score_test, y, valid)
        mi = mi_nats_masked(score_test, y, valid)

   
        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    for ds in ["CORE"]:
        run_dataset(ds)