"""
Set Transformer + Weighted Multi-Task BCE Clinical Outcome Scorer
=================================================================

Overview
--------
This script implements a Set Transformer single-scalar model trained with binary
cross-entropy (BCE). Each hospital admission is represented as a variable-length
set of ICD-10 prefix tokens. Tokens are embedded, processed with Induced Set
Attention Blocks (ISAB), pooled with Pooling by Multihead Attention (PMA), and
mapped to one scalar logit per admission. The same scalar logit is trained to
predict multiple binary clinical outcomes using a weighted, masked multi-task BCE
objective.

The training procedure has two stages. Stage 1 trains separate single-task BCE
models for each outcome and records the best validation BCE. These validation BCE
values are converted into task weights by measuring how much each task beats the
random-guessing BCE baseline log(2). Stage 2 trains one weighted multi-task BCE
model that learns a single shared score for all active targets. The test split is 
then evaluated by measuring association between the learned scalar score and each 
clinical outcome using distance correlation, and mutual information.

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.

2.  Define the CORE dataset paths:
        - admissions_core_train.csv
        - admissions_core_val.csv
        - admissions_core_test.csv
        - diagnoses_icd10_core_train.csv.gz
        - diagnoses_icd10_core_val.csv.gz
        - diagnoses_icd10_core_test.csv.gz

3.  Load train, validation, and test admission CSV files.

4.  Validate required admission columns:
        - hadm_id must be present in train, validation, and test splits.
        - Training targets must be present in train and validation.
        - Evaluation targets must be present in test.
        - long_stay uses long_stay_defined as its validity indicator.

5.  Build TRAIN label and mask matrices:
        - Y has shape (N, T) and stores binary target values.
        - M has shape (N, T) and stores validity indicators.
        - Invalid labels are filled with zero in Y and masked out by M.

6.  Identify tasks to skip:
        - A task is skipped if it has no valid TRAIN labels.
        - A task is skipped if valid TRAIN labels contain only one class.
        - Skipped tasks receive zero weight during Stage 2.

7.  Build a TRAIN-only ICD-10 prefix vocabulary:
        - Normalize ICD codes by uppercasing and removing periods/spaces.
        - Truncate each ICD code to PREFIX_LEN=4 characters.
        - Reserve <PAD>=0 and <UNK>=1.
        - Read diagnosis files in chunks of CHUNK_DX=2,000,000 rows.
        - Build the vocabulary from the full TRAIN diagnoses file only.

8.  Build hadm_id-to-token-list mappings for TRAIN, VAL, and TEST:
        - Read diagnosis files in chunks.
        - Convert each ICD code to a normalized prefix token.
        - Preserve diagnosis token order within each admission.
        - Truncate each admission to MAX_CODES_PER_ADMISSION=256 tokens.

9.  Encode admission token lists:
        - Map TRAIN, VAL, and TEST prefixes to integer IDs.
        - Map unseen validation/test prefixes to <UNK>.
        - Represent admissions without diagnosis tokens as empty arrays before
          collation.

10. Construct TokenDataset objects:
        - Store token ID sequences per admission.
        - Store label matrix Y.
        - Store validity mask matrix M.
        - Use dummy zero labels and masks for the test loader because test labels
          are evaluated separately from the original dataframe.

11. Pad variable-length ICD token sets in collate_pad:
        - Create padded token tensor X of shape (B, L).
        - Use <PAD>=0 for padding positions.
        - Replace empty sequences with a single <UNK> token to avoid all-masked
          rows.
        - Return X, sequence lengths, label matrix Y, and validity mask M.

12. Train Stage 1 single-task BCE models:
        - For each active target, initialize a fresh SetTransformerSingleLogit
          model.
        - Train for EPOCHS_STAGE1=10 epochs.
        - Optimize masked BCE for that one target only.
        - Compute validation BCE for that target after each epoch.
        - Select the checkpoint with the lowest validation BCE.

13. Convert Stage-1 validation BCE values into task weights:
        - Use log(2) as the BCE baseline for random guessing at p=0.5.
        - Compute task skill:
              score_t = max(log(2) - val_bce_t, 0)
        - Replace non-finite scores with zero.
        - Set skipped-task scores to zero.
        - If no task beats chance, use uniform active weights.
        - Otherwise compute stabilized inverse-skill weights:
              weight_t = (max(max_score, eps) / max(score_t, eps))^alpha
        - Clip weights to [1 / WEIGHT_CAP, WEIGHT_CAP].
        - Set skipped-task weights to zero.
        - Normalize active weights so their mean is one.

14. Train Stage 2 weighted multi-task BCE model:
        - Initialize one SetTransformerSingleLogit model.
        - For each batch, produce one scalar logit per admission.
        - Broadcast the scalar logit across all targets.
        - Compute masked BCE for every valid target label.
        - Weight each task by the Stage-1-derived task weight.
        - Normalize by the weighted number of valid labels.
        - Train for EPOCHS_STAGE2=10 epochs.
        - Select the checkpoint with the lowest weighted validation BCE.

15. Score the TEST split:
        - Load the best Stage-2 checkpoint.
        - Generate one scalar score/logit per test admission.
        - Use this scalar score for all evaluation targets.

16. Evaluate TEST association:
        - For each evaluation target, construct the target-specific validity mask.
        - For long_stay, use long_stay_defined as the validity mask.
        - For other targets, treat non-missing values as valid.
        - Compute distance correlation.
        - Compute mutual information in nats.

Architecture: SetTransformerSingleLogit
---------------------------------------
- Embedding layer:
      vocab_size -> EMB_DIM=128, padding_idx=0

- Set encoder:
      N_LAYERS=2 Induced Set Attention Blocks

- Each ISAB contains:
      NUM_INDUCING_POINTS=16 learned inducing points
      MAB(inducing points, input tokens)
      MAB(input tokens, induced representation)

- Each MAB contains:
      MultiheadAttention(dim=128, num_heads=4, batch_first=True)
      Residual connection
      LayerNorm
      Feed-forward network:
          Linear(128 -> 128) -> ReLU -> Linear(128 -> 128)
      Residual connection
      LayerNorm

- Pooling:
      PMA with one learned seed vector
      Multihead attention pools the encoded set into one vector.

- Output head:
      Linear(128 -> 1)

- Output:
      one scalar logit s(x) per admission, shape (B,).

Loss Objective
--------------
Stage 1 uses masked single-task BCE. For task t, BCE is computed only over rows
where the task label is valid:

    loss_t = mean BCEWithLogits(s_i, y_i,t) over valid rows

If a batch contains no valid rows for the task, the loss is returned as
0 * logits.sum() so that the computation graph remains valid.

Stage 2 uses weighted masked multi-task BCE. The scalar logit is broadcast across
all tasks:

    logits_expanded = s(x).unsqueeze(1).expand(B, T)

The unreduced BCE matrix has shape (B, T), is multiplied by the validity mask M,
and is aggregated with task weights:

    loss = sum_t w_t * sum_i M_i,t * BCE(s_i, y_i,t)
           ------------------------------------------------
           sum_t w_t * sum_i M_i,t

This trains one scalar score to be jointly predictive of all active clinical
targets.

Validity Handling
-----------------
Each target has its own validity mask.

For long_stay:
    valid = long_stay_defined == 1

For all other targets:
    valid = target value is not missing

Invalid labels are filled with zero in the label matrix, but they do not
contribute to the loss or evaluation metrics because their mask value is zero.

Single-Class Task Handling
--------------------------
Before training, each task is checked on the TRAIN split. A task is skipped if it
has no valid labels or if all valid labels belong to one class. Skipped tasks:

    - are not trained in Stage 1
    - receive validation BCE = infinity
    - receive score = 0
    - receive Stage-2 weight = 0

Task Weighting Rule
-------------------
Stage-1 validation BCE is converted into a nonnegative skill score:

    score_t = max(log(2) - val_bce_t, 0)

Tasks with smaller skill receive larger weights through a stabilized inverse-skill
rule:

    weight_t = (max(max_score, WEIGHT_EPS) / max(score_t, WEIGHT_EPS))^WEIGHT_ALPHA

Weights are clipped to:

    [1 / WEIGHT_CAP, WEIGHT_CAP]

Skipped-task weights are forced to zero. Active weights are mean-normalized so the
average active weight is one.

Evaluation Metrics
------------------
The Stage-2 scalar score is evaluated against each target using:

    - Distance correlation
    - Mutual information in nats

Metric safeguards:
    - Return 0.0 when fewer than three valid examples are available.
    - Return 0.0 when the score or label vector is constant or near-constant.
    - Return 0.0 for mutual information when the target has fewer than two classes.
    - Z-score scores within valid rows before mutual information estimation.

Key Configuration
-----------------
  SEED: controlled via SEED environment variable, default 1001
  EVAL_SEED=12345
  DEVICE: cuda if available, otherwise cpu
  PREFIX_LEN=4
  EMB_DIM=128
  NUM_HEADS=4
  N_LAYERS=2
  NUM_INDUCING_POINTS=16
  DROPOUT=0.0
  MAX_CODES_PER_ADMISSION=256
  CHUNK_DX=2_000_000
  BATCH_SIZE=256
  EPOCHS_STAGE1=10
  EPOCHS_STAGE2=10
  LR=1e-3
  WEIGHT_DECAY=0.0
  NUM_WORKERS=4
  PIN_MEMORY=True
  MI_BASE_NEIGHBORS=5
  WEIGHT_EPS=0.02
  WEIGHT_ALPHA=0.25
  WEIGHT_CAP=3.0

Expected Directory Layout
--------------------------
    <data_root>/
    └── core/
        ├── admissions_core_train.csv
        ├── admissions_core_val.csv
        ├── admissions_core_test.csv
        ├── diagnoses_icd10_core_train.csv.gz
        ├── diagnoses_icd10_core_val.csv.gz
        └── diagnoses_icd10_core_test.csv.gz

Required admission columns:
    hadm_id, mortality, mortality_30d, long_stay, long_stay_defined, icu_transfer

Required diagnosis columns:
    hadm_id, icd_code

Outputs
-------
The script prints:

    - random seed and evaluation seed
    - dataset sizes
    - training-label positive rates
    - skipped tasks
    - ICD-10 prefix vocabulary size
    - Stage-1 training and validation BCE by epoch
    - best Stage-1 validation BCE by task
    - derived task scores and weights
    - Stage-2 weighted training and validation BCE by epoch
    - test-set distance correlation, and mutual information for each
      evaluation target

This script does not write model checkpoints or diagnostic CSV files to disk.

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor torch

Running
-------
Run the script with:

    python set_transformer_mimiciv.py --data_root /path/to/final_datasets

Override the training seed:

    SEED=42 python set_transformer_mimiciv.py --data_root /path/to/final_datasets

Example:
    python set_transformer_mimiciv.py --data_root ./data/final_datasets
"""

################################################################
# IMPORTS
################################################################
import os
import argparse
import math
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

################################################################
# CONFIG
################################################################

SEED = int(os.environ.get("SEED", 1001))   # training seed varies per run
EVAL_SEED = 12345                        # fixed for eval subsampling + MI
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)
PREFIX_LEN = 4
EMB_DIM = 128
NUM_HEADS = 4
N_LAYERS = 2
NUM_INDUCING_POINTS = 16
DROPOUT = 0.0
MAX_CODES_PER_ADMISSION = 256
CHUNK_DX = 2_000_000
BATCH_SIZE = 256
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10
LR = 1e-3
WEIGHT_DECAY = 0.0
NUM_WORKERS = 4
PIN_MEMORY = True
MI_BASE_NEIGHBORS = 5
WEIGHT_EPS   = 0.02
WEIGHT_ALPHA = 0.25
WEIGHT_CAP   = 3.0
LOG2 = float(math.log(2.0))  # BCE for random guessing at p=0.5
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
        "dx_train":  FINAL_ROOT / "core" / "diagnoses_icd10_core_train.csv.gz",
        "dx_val":    FINAL_ROOT / "core" / "diagnoses_icd10_core_val.csv.gz",
        "dx_test":   FINAL_ROOT / "core" / "diagnoses_icd10_core_test.csv.gz",
        "train_targets": ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
        "eval_targets":  ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
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
# HELPERS 
################################################################

def build_vocab_from_dx_file(dx_path: Path, prefix_len: int, chunksize: int = CHUNK_DX) -> dict[str, int]:
    """
    Build vocab by scanning ENTIRE TRAIN dx file for unique prefixes (no truncation effect).
    """
    toks = set()
    for chunk in pd.read_csv(dx_path, usecols=["icd_code"], chunksize=chunksize):
        for raw in chunk["icd_code"].to_numpy():
            tok = to_prefix(raw, prefix_len)
            if tok:
                toks.add(tok)

    toks = sorted(toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, t in enumerate(toks, start=2):
        vocab[t] = i
    return vocab

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
    if not c:
        return ""
    return c[:k]

def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def build_Y_and_mask(df: pd.DataFrame, targets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      Y: float32 (N,T) with 0-filled where invalid
      M: float32 (N,T) valid mask (1=valid)
    Special handling:
      long_stay uses long_stay_defined as validity.
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


def is_single_class_train(Y: np.ndarray, M: np.ndarray, j: int) -> bool:
    """
    True if task j is single-class on TRAIN among valid rows (or has no valid rows).
    Mirrors HSIC skip logic.
    """
    valid = M[:, j].astype(bool)
    if valid.sum() == 0:
        return True
    yy = Y[valid, j].astype(int)
    return np.unique(yy).size < 2

################################################################
# EVALUATION METRICS 
################################################################



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

    # Match nhsic_* scripts: z-score ONLY within valid rows for this task
    s_valid = zscore_1d(score[valid]).reshape(-1, 1)
    return float(mutual_info_classif(s_valid, yy, random_state=seed, n_neighbors=nn)[0])

################################################################
# DATA-LOADING
################################################################

def load_dx_prefix_map_chunked(dx_path: Path, prefix_len: int, chunksize: int = CHUNK_DX) -> dict[int, list[str]]:
    if not dx_path.exists():
        raise FileNotFoundError(f"Missing diagnoses file: {dx_path}")

    hadm_to_list: dict[int, list[str]] = {}
    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=chunksize):
        if "hadm_id" not in chunk.columns or "icd_code" not in chunk.columns:
            raise ValueError(f"diagnoses file must contain hadm_id and icd_code: {dx_path}")

        hadm_arr = chunk["hadm_id"].astype(int).to_numpy()
        code_arr = chunk["icd_code"].to_numpy()

        for hadm, raw in zip(hadm_arr, code_arr):
            tok = to_prefix(raw, prefix_len)
            if tok == "":
                continue
            lst = hadm_to_list.get(int(hadm))
            if lst is None:
                hadm_to_list[int(hadm)] = [tok]
            else:
                lst.append(tok)

    if MAX_CODES_PER_ADMISSION is not None:
        for hadm, lst in hadm_to_list.items():
            if len(lst) > MAX_CODES_PER_ADMISSION:
                hadm_to_list[hadm] = lst[:MAX_CODES_PER_ADMISSION]

    return hadm_to_list

def encode_hadm_tokens(adm_df: pd.DataFrame, dx_map: dict[int, list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    unk = vocab["<UNK>"]
    out = []

    for h in adm_df["hadm_id"].astype(int).tolist():
        toks = dx_map.get(int(h), [])

        if not toks:
            out.append(np.asarray([], dtype=np.int64))
            continue

        ids = [vocab.get(t, unk) for t in toks]

        if MAX_CODES_PER_ADMISSION is not None and len(ids) > MAX_CODES_PER_ADMISSION:
            ids = ids[:MAX_CODES_PER_ADMISSION]

        out.append(np.asarray(ids, dtype=np.int64))

    return out

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


def collate_pad(batch, pad_id: int = 0, unk_id: int = 1):
    seqs, Y, M = zip(*batch)
    B = len(seqs)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    max_len = max(int(lengths.max().item()) if B > 0 else 0, 1)
    X = torch.full((B, max_len), pad_id, dtype=torch.long)

    for i, s in enumerate(seqs):
        if len(s) == 0:
            X[i, 0] = unk_id          # <-- prevents all-masked row
            continue
        X[i, :len(s)] = torch.from_numpy(s).long()

    Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
    M = torch.tensor(np.stack(M, axis=0), dtype=torch.float32)
    return X, lengths, Y, M


################################################################
# SET TRANSFORMER COMPONENTS ARCHITECTURE
################################################################

class MAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, key_padding_mask: torch.Tensor | None):
        attn, _ = self.mha(Q, K, K, key_padding_mask=key_padding_mask, need_weights=False)
        H = self.ln1(Q + attn)
        H2 = self.ff(H)
        return self.ln2(H + H2)

class ISAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab1 = MAB(dim, num_heads, dropout)
        self.mab2 = MAB(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor | None):
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, key_padding_mask=key_padding_mask)
        return self.mab2(X, H, key_padding_mask=None)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads, dropout)

    def forward(self, X: torch.Tensor, key_padding_mask: torch.Tensor | None):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, key_padding_mask=key_padding_mask)


class SetTransformerSingleLogit(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_heads: int, n_layers: int, num_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.blocks = nn.ModuleList([ISAB(dim, num_heads, num_inducing, dropout) for _ in range(n_layers)])
        self.pma = PMA(dim, num_heads, num_seeds=1, dropout=dropout)
        self.to_logit = nn.Linear(dim, 1)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        X = self.emb(x_ids)
        key_padding_mask = (x_ids == 0)
        for blk in self.blocks:
            X = blk(X, key_padding_mask=key_padding_mask)
        pooled = self.pma(X, key_padding_mask=key_padding_mask)  # (B,1,D)
        logit = self.to_logit(pooled.squeeze(1)).squeeze(1)      # (B,)
        return logit



################################################################
# LOSS FUNCTIONS
################################################################
def masked_bce_single_task(logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,)
    y,m:   (B,)
    returns mean BCE over valid, or 0*logits.sum() if no valid.
    """
    valid = (m > 0.5)
    if valid.sum().item() <= 0:
        return logits.sum() * 0.0
    loss = nn.functional.binary_cross_entropy_with_logits(logits[valid], y[valid], reduction="mean")
    return loss


def masked_weighted_multitask_bce(logits: torch.Tensor, Y: torch.Tensor, M: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,) one scalar per example
    Y,M:    (B,T)
    W:      (T,) nonnegative weights
    Weighted masked BCE with normalization:
        sum_t w_t * sum_{valid} bce / sum_t w_t * (#valid)
    """
    B, T = Y.shape
    loss_mat = nn.functional.binary_cross_entropy_with_logits(
        logits.unsqueeze(1).expand(B, T),
        Y,
        reduction="none",
    )  # (B,T)

    # only valid entries
    loss_mat = loss_mat * M

    # per-task valid counts
    valid_counts = M.sum(dim=0)  # (T,)

    # numerator: sum_t w_t * sum_i loss_{i,t}
    per_task_sum = loss_mat.sum(dim=0)  # (T,)
    num = (W * per_task_sum).sum()

    # denom: sum_t w_t * (#valid_t)
    den = (W * valid_counts).sum()

    if den.item() <= 0:
        return logits.sum() * 0.0
    return num / den


@torch.no_grad()
def predict_scores(model, loader) -> np.ndarray:
    model.eval()
    out = []
    for x_ids, _lengths, _Y, _M in loader:
        x_ids = x_ids.to(DEVICE)
        s = model(x_ids).detach().cpu().numpy()
        out.append(s)
    return np.concatenate(out, axis=0)


# -------------------------
# STAGE 1: train single-task BCE models, select by VAL BCE
# -------------------------
def train_stage1_single_task(
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_j: int,
) -> tuple[dict, float]:
    """
    Returns (best_state_dict, best_val_bce_for_task).
    """
    model = SetTransformerSingleLogit(
        vocab_size=vocab_size,
        dim=EMB_DIM,
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        num_inducing=NUM_INDUCING_POINTS,
        dropout=DROPOUT,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS_STAGE1 + 1):
        # TRAIN
        model.train()
        tr_losses = []
        for x_ids, _lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids)
            loss = masked_bce_single_task(s, Y[:, task_j], M[:, task_j])
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.detach().cpu().item()))
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("inf")

        # VAL (task-specific)
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x_ids, _lengths, Y, M in val_loader:
                x_ids = x_ids.to(DEVICE)
                Y = Y.to(DEVICE)
                M = M.to(DEVICE)
                s = model(x_ids)
                vloss = masked_bce_single_task(s, Y[:, task_j], M[:, task_j])
                va_losses.append(float(vloss.detach().cpu().item()))
        va_loss = float(np.mean(va_losses)) if va_losses else float("inf")

        print(f"    [Stage1 task={task_j}] Epoch {ep:02d}/{EPOCHS_STAGE1} | train_BCE={tr_loss:.6f} | val_BCE={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, float(best_val)


# -------------------------
# STAGE 2: weighted multi-task BCE, select by weighted VAL loss
# -------------------------
def train_stage2_weighted_multitask(
    vocab_size: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weights: np.ndarray,
) -> dict:
    model = SetTransformerSingleLogit(
        vocab_size=vocab_size,
        dim=EMB_DIM,
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        num_inducing=NUM_INDUCING_POINTS,
        dropout=DROPOUT,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    W = torch.tensor(weights.astype(np.float32), device=DEVICE)

    best_val = float("inf")
    best_state = None

    for ep in range(1, EPOCHS_STAGE2 + 1):
        # TRAIN
        model.train()
        tr_losses = []
        for x_ids, _lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids)
            loss = masked_weighted_multitask_bce(s, Y, M, W)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.detach().cpu().item()))
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("inf")

        # VAL (weighted)
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x_ids, _lengths, Y, M in val_loader:
                x_ids = x_ids.to(DEVICE)
                Y = Y.to(DEVICE)
                M = M.to(DEVICE)
                s = model(x_ids)
                vloss = masked_weighted_multitask_bce(s, Y, M, W)
                va_losses.append(float(vloss.detach().cpu().item()))
        va_loss = float(np.mean(va_losses)) if va_losses else float("inf")

        print(f"  [Stage2] Epoch {ep:02d}/{EPOCHS_STAGE2} | train_wBCE={tr_loss:.6f} | val_wBCE={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state


# -------------------------
# RUN ONE DATASET
# -------------------------
def run_dataset(ds_name: str):
    cfg = DATASETS[ds_name]

    print("\n" + "=" * 100)
    print(f"DATASET: {ds_name}")
    print("=" * 100)

    tr = pd.read_csv(cfg["adm_train"])
    va = pd.read_csv(cfg["adm_val"])
    te = pd.read_csv(cfg["adm_test"])

    train_targets = cfg["train_targets"]
    eval_targets = cfg["eval_targets"]

    require_cols(tr, ["hadm_id"], f"{ds_name}/TRAIN")
    require_cols(va, ["hadm_id"], f"{ds_name}/VAL")
    require_cols(te, ["hadm_id"], f"{ds_name}/TEST")
    require_cols(tr, train_targets, f"{ds_name}/TRAIN")
    require_cols(va, train_targets, f"{ds_name}/VAL")
    require_cols(te, eval_targets, f"{ds_name}/TEST")

    print(f"TRAIN n={len(tr):,} | VAL n={len(va):,} | TEST n={len(te):,}")

    Ytr, Mtr = build_Y_and_mask(tr, train_targets)
    Yva, Mva = build_Y_and_mask(va, train_targets)

    print("\nTRAIN label positive rates (valid-only):")
    for j, t in enumerate(train_targets):
        valid = Mtr[:, j].astype(bool)
        if valid.sum() == 0:
            print(f"  {t:16s}: n_valid=0")
        else:
            print(f"  {t:16s}: {float(Ytr[valid, j].mean()):.6f} (n_valid={int(valid.sum())})")

    # ---- apples-to-apples skip logic (single-class TRAIN) ----
    skipped = np.zeros(len(train_targets), dtype=bool)
    for j, t in enumerate(train_targets):
        if is_single_class_train(Ytr, Mtr, j):
            skipped[j] = True
            print(f"  -> SKIP task entirely (single-class or no-valid in TRAIN): {t}")

    print("\nLoading diagnoses (chunked) + building TRAIN vocab...")
    # Build vocab from FULL TRAIN dx (not affected by per-admission truncation)
    vocab = build_vocab_from_dx_file(cfg["dx_train"], PREFIX_LEN, chunksize=CHUNK_DX)
    vocab_size = len(vocab)
    print(f"Vocab size (TRAIN ICD10 prefixes): {vocab_size:,} (incl PAD/UNK)")

    # Then build hadm->prefix lists (still OK to truncate to 256 for the input sequences)
    dx_tr = load_dx_prefix_map_chunked(cfg["dx_train"], PREFIX_LEN, chunksize=CHUNK_DX)
    dx_va = load_dx_prefix_map_chunked(cfg["dx_val"], PREFIX_LEN, chunksize=CHUNK_DX)
    dx_te = load_dx_prefix_map_chunked(cfg["dx_test"], PREFIX_LEN, chunksize=CHUNK_DX)

    tr_seqs = encode_hadm_tokens(tr, dx_tr, vocab)
    va_seqs = encode_hadm_tokens(va, dx_va, vocab)
    te_seqs = encode_hadm_tokens(te, dx_te, vocab)

    train_ds = TokenDataset(tr_seqs, Ytr, Mtr)
    val_ds   = TokenDataset(va_seqs, Yva, Mva)

    # test placeholders for loader shape consistency
    Yte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    Mte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    test_ds = TokenDataset(te_seqs, Yte0, Mte0)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_pad,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_pad,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_pad,
        drop_last=False,
    )

    # -------------------------
    # STAGE 1: single-task BCE -> VAL BCE -> weights
    # -------------------------
    print("\n=== Stage 1: Single-task BCE models (select by lowest VAL BCE) ===")
    val_bce = []
    for j, t in enumerate(train_targets):
        if skipped[j]:
            val_bce.append(float("inf"))
            continue
        print(f"\n  Training single-task model for: {t}")
        _best_state, best_val = train_stage1_single_task(
            vocab_size=vocab_size,
            train_loader=train_loader,
            val_loader=val_loader,
            task_j=j,
        )
        val_bce.append(best_val)
        print(f"  Best VAL BCE for {t}: {best_val:.6f}")

    val_bce = np.asarray(val_bce, dtype=np.float64)

    # Convert BCE -> "skill score" where higher is better, and chance ~= 0.
    # score_t = max(log(2) - val_bce_t, 0)
    score = np.maximum(LOG2 - val_bce, 0.0)
    score[~np.isfinite(score)] = 0.0  # inf/NaN -> 0
    score[skipped] = 0.0

    max_score = float(score.max()) if score.size > 0 else 0.0

    eps   = WEIGHT_EPS
    alpha = WEIGHT_ALPHA
    cap   = WEIGHT_CAP

    if max_score <= 0.0:
        # uniform weights if no task beats chance
        weights = np.ones_like(score, dtype=np.float32)
    else:
        denom = np.maximum(score, eps)
        weights = (np.maximum(max_score, eps) / denom) ** alpha
        weights = np.clip(weights, 1.0 / cap, cap).astype(np.float32)

    # skipped tasks must contribute nothing
    weights[skipped] = 0.0

    # mean-normalize active weights so mean(w_active)=1
    active = weights > 0
    if active.any():
        weights[active] = weights[active] / weights[active].mean()


    print("\nStage 1 VAL BCE, score_t=(log2 - BCE)+, and derived weights:")
    for j, t in enumerate(train_targets):
        tag = " (SKIPPED)" if skipped[j] else ""
        vb = float(val_bce[j]) if np.isfinite(val_bce[j]) else float("inf")
        sc = float(score[j])
        w  = float(weights[j])
        print(f"  {t:16s}: val_BCE={vb:.6f} | score={sc:.6f} | w={w:.6f}{tag}")

    # -------------------------
    # STAGE 2: weighted multi-task BCE
    # -------------------------
    print("\n=== Stage 2: Weighted multi-task BCE (ONE scalar) ===")
    best_state2 = train_stage2_weighted_multitask(
        vocab_size=vocab_size,
        train_loader=train_loader,
        val_loader=val_loader,
        weights=weights,
    )

    # Load best Stage-2 model
    model = SetTransformerSingleLogit(
        vocab_size=vocab_size,
        dim=EMB_DIM,
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        num_inducing=NUM_INDUCING_POINTS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(best_state2)
    model.eval()

    # TEST scores
    score_test = predict_scores(model, test_loader)

    # -------------------------
    # EVAL
    # -------------------------
    print("\n=== TEST Metrics (DistCorr, MI_nats) using Stage-2 score s(x) ===")
    for t in eval_targets:
        if t == "long_stay":
            require_cols(te, ["long_stay", "long_stay_defined"], f"{ds_name}/TEST")
            valid = te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            y = te["long_stay"].fillna(0).astype(int).to_numpy()
        else:
            s_col = pd.to_numeric(te[t], errors="coerce")
            valid = (~s_col.isna()).to_numpy(dtype=bool)
            y = s_col.fillna(0).astype(int).to_numpy()

        pearson = pearson_from_score(score_test, y, valid)
        dcorr = dcor_from_score(score_test, y, valid)
        mi = mi_nats_masked(score_test, y, valid)


        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )

    print("=" * 100)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    for ds in ["CORE"]:
        run_dataset(ds)