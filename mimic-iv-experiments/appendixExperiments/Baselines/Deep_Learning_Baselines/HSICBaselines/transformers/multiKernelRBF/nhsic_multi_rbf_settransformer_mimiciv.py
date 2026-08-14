"""
SetTransformer (ISAB + PMA) + Multi-RBF NHSIC Multi-Task Clinical Outcome Predictor
=====================================================================================

Overview
--------
This script replaces the DeepSets encoder with a full SetTransformer architecture
(Induced Set Attention Blocks + Pooling by Multihead Attention) while retaining
the two-stage multi-RBF NHSIC training strategy. The SetTransformer uses
self-attention over inducing points to achieve O(n·m) complexity (n=tokens,
m=inducing points), making it more expressive than simple pooling while
remaining tractable for large code sets.

Pipeline Flow
-------------
1.  Parse --data_root (path to final_datasets) and --validity_mode
    ("taskwise" or "intersection") controlling which admissions are valid
    per NHSIC objective.
2.  Load train/val/test admission CSVs. Build a TRAIN vocabulary by scanning
    the FULL TRAIN diagnosis file (no per-admission truncation at vocab-build
    time), then load hadm_id → token-list mappings for each split (truncated
    to 256 codes in file-encounter order).
3.  Encode token lists to integer IDs; pad into batches via collate_pad.
4.  Estimate sigma_base for the multi-RBF kernel by computing the median
    pairwise absolute score difference over up to 5000 TRAIN predictions
    from the randomly initialized model (uses global NumPy RNG).
5.  Stage 1 — Per-target single-task NHSIC training (10 epochs, AdamW, LR=1e-3):
        - Fresh SetTransformer model and fresh sigma estimate per target.
        - Skip targets that are single-class or have no valid labels on TRAIN.
        - Select best checkpoint by highest per-task VAL NHSIC (subsampled).
        - Record best VAL NHSIC per task for weight derivation.
6.  Derive per-task weights from Stage-1 VAL NHSIC scores:
        weight_t = clip((max_score / max(score_t, ε))^0.25, 1/3, 3)
    then normalize active weights to mean 1; skipped tasks receive weight 0.
7.  Stage 2 — Weighted multi-task NHSIC training (10 epochs, AdamW, LR=1e-3):
        - Fresh SetTransformer model and fresh sigma estimate.
        - Objective is torch.zeros-accumulated weighted sum of per-task NHSIC.
        - Select best checkpoint by highest weighted VAL NHSIC sum
          (per-task subsampled, summed with task weights).
8.  Orient the final score's sign using VAL Pearson correlation against mortality
    (flip so that higher score implies higher risk).
9.  Evaluate on TEST: per-task multi-RBF NHSIC (subsampled, target-hash seeds),
     Distance Correlation, and Mutual Information.

Architecture: SetTransformerSingleLogit
----------------------------------------
- Embedding layer (vocab_size → EMB_DIM=128), padding_idx=0
- N_LAYERS=2 ISAB blocks (Induced Set Attention Block):
    Each ISAB has NUM_INDUCING_POINTS=16 learnable inducing points and two
    MAB (Multihead Attention Block) sub-layers with LayerNorm and feed-forward.
    Key padding mask from PAD tokens is passed through.
- PMA (Pooling by Multihead Attention): 1 seed vector attends over the encoded
    set to produce a single (B, 1, d) summary.
- Linear(d → 1) → scalar logit score per admission.

NHSIC Kernel
------------
Multi-RBF kernel over scores at scales [0.25, 0.50, 1.00, 2.00, 4.00] × sigma_base;
delta (equality) kernel over binary labels. NHSIC = Frobenius inner product of
centered kernels, divided by sqrt(HSIC(K,K) × HSIC(L,L) + eps).

Key Configuration
-----------------
  SEED: controlled via SEED env var (default 1001); EVAL_SEED=12345 (fixed)
  EMB_DIM=128, NUM_HEADS=4, N_LAYERS=2, NUM_INDUCING_POINTS=16, DROPOUT=0.0
  BATCH_SIZE=256, LR=1e-3, WEIGHT_DECAY=0.0
  EPOCHS_STAGE1=10, EPOCHS_STAGE2=10
  PREFIX_LEN=4, MAX_CODES_PER_ADMISSION=256, CHUNK_DX=2_000_000
  SUBSAMPLE_N=5000, RBF_SCALES=[0.25, 0.50, 1.00, 2.00, 4.00]
  WEIGHT_EPS=0.02, WEIGHT_ALPHA=0.25, WEIGHT_CAP=3.0

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

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor torch

Running
-------
    python nhsic_multi_rbf_settransformer_mimiciv.py --data_root /path/to/final_datasets

With optional arguments:
    python nhsic_multi_rbf_settransformer_mimiciv.py \\
        --data_root /path/to/final_datasets \\
        --validity_mode intersection

Override training seed:
    SEED=42 python nhsic_multi_rbf_settransformer_mimiciv.py --data_root /path/to/final_datasets

Example:
    python nhsic_multi_rbf_settransformer_mimiciv.py --data_root ./data/final_datasets
"""

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
import hashlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to final_datasets directory containing core/admissions_core_train.csv, etc.",
    )
    parser.add_argument(
        "--validity_mode",
        type=str,
        default="taskwise",
        choices=["taskwise", "intersection"],
        help=(
            "Validity convention for HSIC/NHSIC objectives. "
            "'taskwise' uses each outcome's valid-label mask; "
            "'intersection' uses admissions valid for all outcomes."
        ),
    )
    return parser.parse_args()


ARGS = parse_args()
FINAL_ROOT = Path(ARGS.data_root)
VALIDITY_MODE = ARGS.validity_mode
print("DATA_ROOT:", FINAL_ROOT)
print("VALIDITY_MODE for HSIC/NHSIC objectives:", VALIDITY_MODE)


def global_flip_from_val_anchor(
    score_val: np.ndarray,
    Yva: np.ndarray,
    Mva: np.ndarray,
    targets: list[str],
    anchor: str = "mortality",
) -> tuple[int, float]:
    """
    Decide a single global flip (+1 or -1) so that higher score => higher risk for anchor on VAL.
    Uses Pearson on VAL (mask-aware).
    Returns: (flip, pearson_before_flip)
    """
    if anchor not in targets:
        return +1, 0.0

    j = targets.index(anchor)
    corr = pearson_from_score(score_val, Yva[:, j], Mva[:, j])
    flip = -1 if corr < 0 else +1
    return flip, float(corr)






def stable_hash_mod(s: str, mod: int = 1000) -> int:
    # stable across runs/machines (unlike Python's built-in hash())
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


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


# -------------------------
# CONFIG
# -------------------------
SEED = int(os.environ.get("SEED", 1001))   # training seed (varies across runs)
EVAL_SEED = 12345                       # fixed for val/test subsampling + MI
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)

PREFIX_LEN = 4

EMB_DIM = 128
NUM_HEADS = 4
N_LAYERS = 2
NUM_INDUCING_POINTS = 16
DROPOUT = 0.0
MAX_CODES_PER_ADMISSION = 256  # match single-RBF

BATCH_SIZE = 256
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10
LR = 1e-3
WEIGHT_DECAY = 0.0

NUM_WORKERS = 4
PIN_MEMORY = True

SUBSAMPLE_N = 5000
DIFFICULTY_FLOOR = 1e-4

MI_BASE_NEIGHBORS = 5

RBF_SCALES = [0.25, 0.50, 1.00, 2.00, 4.00]
NHSIC_EPS = 1e-8

CHUNK_DX = 2_000_000
WEIGHT_EPS = 0.02     # floor for nhsic in weight computation (try 0.01–0.05)
WEIGHT_ALPHA = 0.25   # 0.25 is gentler than sqrt(·) which is 0.5
WEIGHT_CAP = 3.0      # max weight ratio (prevents domination)



# -------------------------
# PATHS (FINAL DATASETS)
# -------------------------
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


# -------------------------
# REPRO / DEVICE
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def estimate_sigma_median_multi(model: nn.Module, loader: DataLoader, n_samples: int = SUBSAMPLE_N) -> float:
    """
    Median heuristic on |s_i - s_j| for 1D scores.
    MATCHES single-RBF behavior:
      - sample k scores (k<=n_samples)
      - compute full |s_i - s_j| matrix
      - take median over all nonzero entries (includes both upper+lower triangle)
    NOTE: still uses global NumPy RNG, exactly like the single-RBF script.
    """
    model.eval()
    scores = []

    for x_ids, _lengths, _Y, _M in loader:
        x_ids = x_ids.to(DEVICE)
        s = model(x_ids).detach().float().cpu().numpy()
        scores.append(s)
        if sum(len(a) for a in scores) >= n_samples:
            break

    s_all = np.concatenate(scores, axis=0).astype(np.float64)
    if s_all.size < 10:
        return 1.0

    k = int(min(s_all.size, n_samples))
    idx = np.random.choice(s_all.size, size=k, replace=False)  # global RNG, like single-RBF
    ss = s_all[idx]

    # FULL distance matrix (k,k)
    d = np.abs(ss.reshape(-1, 1) - ss.reshape(1, -1))

    # Median over all nonzero entries (includes duplicated distances)
    nz = d[d > 0]
    if nz.size == 0:
        return 1.0

    med = float(np.median(nz))
    if (not np.isfinite(med)) or med <= 0:
        return 1.0
    return med



# -------------------------
# UTILS
# -------------------------
def build_vocab_from_dx_file(dx_path: Path, prefix_len: int, chunksize: int = CHUNK_DX) -> dict[str, int]:
    """
    Script-2 style vocab: scan the ENTIRE TRAIN dx file and collect all unique ICD prefixes.
    No per-admission truncation affects the vocab.
    """
    if not dx_path.exists():
        raise FileNotFoundError(f"Missing diagnoses file: {dx_path}")

    toks = set()
    for chunk in pd.read_csv(dx_path, usecols=["icd_code"], chunksize=chunksize):
        code_arr = chunk["icd_code"].to_numpy()
        for raw in code_arr:
            tok = to_prefix(raw, prefix_len)
            if tok:
                toks.add(tok)

    toks = sorted(toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, t in enumerate(toks, start=2):
        vocab[t] = i
    return vocab


def is_single_class_train(Y: np.ndarray, M: np.ndarray, j: int) -> bool:
    """
    True if task j is single-class on TRAIN among valid rows (or has no valid rows).
    Matches Single-RBF skip logic.
    """
    valid = M[:, j].astype(bool)
    if valid.sum() == 0:
        return True
    yy = Y[valid, j].astype(int)
    return np.unique(yy).size < 2


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


def mi_nats_masked(score: np.ndarray, y: np.ndarray, valid: np.ndarray, seed: int = EVAL_SEED, base_neighbors: int = MI_BASE_NEIGHBORS) -> float:
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


def subsample_indices(valid_mask: np.ndarray, n: int, seed: int = EVAL_SEED) -> np.ndarray:
    idx = np.where(valid_mask.astype(bool))[0]
    if idx.size == 0:
        return idx
    if idx.size <= n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=n, replace=False)


def get_valid_mask_np(
    M_all: np.ndarray,
    task_idx: int,
    mode: str = VALIDITY_MODE,
) -> np.ndarray:
    if mode == "taskwise":
        return M_all[:, task_idx] > 0.5
    if mode == "intersection":
        return (M_all > 0.5).all(axis=1)
    raise ValueError(f"Unknown validity_mode: {mode}")


def get_valid_mask_torch(
    M_all: torch.Tensor,
    task_idx: int,
    mode: str = VALIDITY_MODE,
) -> torch.Tensor:
    if mode == "taskwise":
        return M_all[:, task_idx] > 0.5
    if mode == "intersection":
        return (M_all > 0.5).all(dim=1)
    raise ValueError(f"Unknown validity_mode: {mode}")


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


# -------------------------
# DX LOADING (CHUNKED) -> hadm_id -> set(prefixes)
# -------------------------
def load_dx_prefix_map_chunked(dx_path: Path, prefix_len: int, chunksize: int = CHUNK_DX) -> dict[int, list[str]]:
    if not dx_path.exists():
        raise FileNotFoundError(f"Missing diagnoses file: {dx_path}")

    hadm_to_list: dict[int, list[str]] = {}
    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=chunksize):
        if "hadm_id" not in chunk.columns or "icd_code" not in chunk.columns:
            raise ValueError(f"diagnoses file must contain hadm_id and icd_code: {dx_path}")

        # NOTE: DO NOT drop_duplicates -> keep duplicates like single-RBF
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

    # truncate like single-RBF
    if MAX_CODES_PER_ADMISSION is not None:
        for hadm, lst in hadm_to_list.items():
            if len(lst) > MAX_CODES_PER_ADMISSION:
                hadm_to_list[hadm] = lst[:MAX_CODES_PER_ADMISSION]

    return hadm_to_list


def build_vocab_from_train(dx_map: dict[int, list[str]]) -> dict[str, int]:
    toks = set()
    for lst in dx_map.values():
        for t in lst:
            toks.add(t)
    toks = sorted(toks)

    # 0=PAD, 1=UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, t in enumerate(toks, start=2):
        vocab[t] = i
    return vocab


def encode_hadm_tokens(adm_df: pd.DataFrame, dx_map: dict[int, list[str]], vocab: dict[str, int]) -> list[np.ndarray]:
    unk = vocab["<UNK>"]
    out = []

    for h in adm_df["hadm_id"].astype(int).tolist():
        toks = dx_map.get(int(h), [])

        # EXACTLY like Single-RBF: missing dx -> empty list (collate will pad with PAD=0)
        if not toks:
            out.append(np.asarray([], dtype=np.int64))
            continue

        # keep duplicates + order
        ids = [vocab.get(t, unk) for t in toks]

        # truncate first-256 in encounter order
        if MAX_CODES_PER_ADMISSION is not None and len(ids) > MAX_CODES_PER_ADMISSION:
            ids = ids[:MAX_CODES_PER_ADMISSION]

        out.append(np.asarray(ids, dtype=np.int64))

    return out



# -------------------------
# DATASET + COLLATE
# -------------------------
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
    max_len = max(max_len, 1)   # EXACTLY like Single-RBF

    X = torch.full((B, max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) == 0:
            continue
        X[i, :len(s)] = torch.from_numpy(s).long()

    Y = torch.tensor(np.stack(Y, axis=0), dtype=torch.float32)
    M = torch.tensor(np.stack(M, axis=0), dtype=torch.float32)
    return X, lengths, Y, M


# -------------------------
# SET TRANSFORMER BLOCKS
# -------------------------
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


# -------------------------
# NHSIC (multi-RBF on score, delta kernel on binary label)
# -------------------------
def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    # Kc = K - row_mean - col_mean + grand_mean
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    grand = K.mean()
    return K - row_mean - col_mean + grand


def _rbf_kernel_multi(s: torch.Tensor, sigma_base: float) -> torch.Tensor:
    # s: (n,)
    s = s.view(-1, 1)
    d2 = (s - s.t()).pow(2)  # (n,n)

    K_sum = 0.0
    for a in RBF_SCALES:
        sig = float(sigma_base) * float(a)
        sig = max(sig, 1e-6)
        K_sum = K_sum + torch.exp(-d2 / (2.0 * (sig ** 2)))
    return K_sum / float(len(RBF_SCALES))


def _delta_kernel(y: torch.Tensor) -> torch.Tensor:
    # y: (n,) int
    y = y.view(-1, 1)
    return (y == y.t()).to(dtype=torch.float32)



def nhsic_multi_rbf_batch(s: torch.Tensor, y: torch.Tensor, sigma_base: float) -> torch.Tensor:
    """
    Normalized HSIC:
      NHSIC = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L) + eps)
    """
    K = _rbf_kernel_multi(s, sigma_base)
    L = _delta_kernel(y)

    Kc = _center_kernel(K)
    Lc = _center_kernel(L)

    # Frobenius inner products
    hsic = (Kc * Lc).sum()
    kk = (Kc * Kc).sum()
    ll = (Lc * Lc).sum()

    denom = torch.sqrt(kk * ll + NHSIC_EPS)
    return hsic / denom


def nhsic_task_from_batch(
    s: torch.Tensor,
    y: torch.Tensor,
    M_all: torch.Tensor,
    sigma_base: float,
    task_j: int,
    validity_mode: str = VALIDITY_MODE,
) -> torch.Tensor:
    """
    s: (B,)
    y: (B,) float or int (0/1)
    M_all: (B,T) mask float (0/1)
    returns NHSIC for selected valid rows only.
    IMPORTANT: if the batch is single-class or too small, return 0 * s.sum()
               so the result keeps a grad_fn and backward() won't crash.
    """
    valid = get_valid_mask_torch(M_all, task_j, mode=validity_mode)
    if valid.sum().item() < 3:
        return s.sum() * 0.0  # keeps graph

    sv = s[valid]
    yv = y[valid].to(torch.int64)

    # single-class batch -> no usable delta-kernel signal
    if torch.unique(yv).numel() < 2:
        return s.sum() * 0.0  # keeps graph

    return nhsic_multi_rbf_batch(sv, yv, sigma_base)



# -------------------------
# PREDICTION HELPERS
# -------------------------
@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    out = []
    for x_ids, _lengths, _Y, _M in loader:
        x_ids = x_ids.to(DEVICE)
        s = model(x_ids).detach().cpu().numpy()
        out.append(s)
    return np.concatenate(out, axis=0)


def build_loader(seqs, Y, M, shuffle: bool) -> DataLoader:
    ds = TokenDataset(seqs, Y, M)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_pad,
        drop_last=False,
    )


# -------------------------
# STAGE 1: single-task training
# -------------------------
def train_single_task(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_j: int,
    validation_anchor_idx: int,
    validation_anchor_name: str,
    sigma_train_base: float,
) -> tuple[dict, float]:
    """
    Trains the PROVIDED model instance (same one used for sigma estimation).
    Returns: best_state_dict, best_val_nhsic
    """
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1e9
    best_state = None

    for ep in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        vals = []

        for x_ids, _lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids)

            y = Y[:, task_j]

            obj = nhsic_task_from_batch(s, y, M, sigma_train_base, task_j)
            loss = -obj
            loss.backward()
            opt.step()

            vals.append(float(obj.detach().cpu().item()))

        train_obj = float(np.mean(vals)) if vals else 0.0

        # Stage-1 checkpoint selection uses the shared mortality anchor.
        s_val = predict_scores(model, val_loader)
        Yv = val_loader.dataset.Y[:, validation_anchor_idx].astype(np.float32)
        Mv_full = val_loader.dataset.M.astype(np.float32)
        valid_v = get_valid_mask_np(Mv_full, validation_anchor_idx, mode=VALIDITY_MODE)
        idx = subsample_indices(valid_v, SUBSAMPLE_N, seed=EVAL_SEED + 1000 + task_j)

        if idx.size < 3:
            val_obj = 0.0
        else:
            yy = Yv[idx].astype(int)
            if np.unique(yy).size < 2:
                val_obj = 0.0
            else:
                sv = torch.tensor(s_val[idx], dtype=torch.float32, device=DEVICE)
                yv = torch.tensor(yy, dtype=torch.int64, device=DEVICE)
                val_obj = float(nhsic_multi_rbf_batch(sv, yv, sigma_train_base).detach().cpu().item())

        print(
            f"    [Stage1 task={task_j}] Epoch {ep:02d}/{EPOCHS_STAGE1} "
            f"| train_NHSIC={train_obj:.6f} "
            f"| val_NHSIC_vs_{validation_anchor_name}={val_obj:.6f}"
        )

        if val_obj > best_val:
            best_val = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, float(best_val)


# -------------------------
# STAGE 2: weighted multi-task training
# -------------------------
def train_multi_task(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weights: np.ndarray,
    sigma_train_base: float,
) -> dict:
    """
    Trains the PROVIDED model instance (same one used for sigma estimation).
    Returns: best_state_dict
    """
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1e9
    best_state = None

    W = torch.tensor(weights.astype(np.float32), device=DEVICE)

    for ep in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        vals = []

        for x_ids, _lengths, Y, M in train_loader:
            x_ids = x_ids.to(DEVICE)
            Y = Y.to(DEVICE)
            M = M.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            s = model(x_ids)

            obj = torch.zeros((), device=DEVICE)
            T = Y.shape[1]
            for j in range(T):
                obj = obj + W[j] * nhsic_task_from_batch(s, Y[:, j], M, sigma_train_base, j)

            loss = -obj
            loss.backward()
            opt.step()

            vals.append(float(obj.detach().cpu().item()))

        train_obj = float(np.mean(vals)) if vals else 0.0

        # VAL weighted NHSIC (subsampled per-task valid rows)
        s_val = predict_scores(model, val_loader)
        Yv_full = val_loader.dataset.Y.astype(np.float32)
        Mv_full = val_loader.dataset.M.astype(np.float32)

        val_obj_t = 0.0
        for j in range(Yv_full.shape[1]):
            valid_j = get_valid_mask_np(Mv_full, j, mode=VALIDITY_MODE)
            idx_j = subsample_indices(valid_j, SUBSAMPLE_N, seed=EVAL_SEED + 9999 + j)

            if idx_j.size < 3:
                continue

            yy = Yv_full[idx_j, j].astype(int)
            if np.unique(yy).size < 2:
                continue

            sj = torch.tensor(s_val[idx_j], dtype=torch.float32, device=DEVICE)
            yj = torch.tensor(yy, dtype=torch.int64, device=DEVICE)

            nh = float(nhsic_multi_rbf_batch(sj, yj, sigma_train_base).detach().cpu().item())
            val_obj_t += float(W[j].detach().cpu().item()) * nh

        val_obj = float(val_obj_t)

        print(f"  [Stage2] Epoch {ep:02d}/{EPOCHS_STAGE2} | train_wNHSIC={train_obj:.6f} | val_wNHSIC={val_obj:.6f}")

        if val_obj > best_val:
            best_val = val_obj
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
    eval_targets  = cfg["eval_targets"]

    require_cols(tr, ["hadm_id"], f"{ds_name}/TRAIN")
    require_cols(va, ["hadm_id"], f"{ds_name}/VAL")
    require_cols(te, ["hadm_id"], f"{ds_name}/TEST")

    require_cols(tr, train_targets, f"{ds_name}/TRAIN")
    require_cols(va, train_targets, f"{ds_name}/VAL")
    require_cols(te, eval_targets,  f"{ds_name}/TEST")

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

    print("\nTEST label positive rates (valid-only) [EVAL TARGETS]:")
    for t in eval_targets:
        if t == "long_stay":
            valid = te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            y = te["long_stay"].fillna(0).astype(int).to_numpy()
        else:
            s = pd.to_numeric(te[t], errors="coerce")
            valid = (~s.isna()).to_numpy(dtype=bool)
            y = s.fillna(0).astype(int).to_numpy()
        if valid.sum() == 0:
            print(f"  {t:16s}: n_valid=0")
        else:
            print(f"  {t:16s}: {float(y[valid].mean()):.6f} (n_valid={int(valid.sum())})")

    print("\nLoading diagnoses (chunked) + building TRAIN vocab (FULL TRAIN scan)...")
    vocab = build_vocab_from_dx_file(cfg["dx_train"], PREFIX_LEN, chunksize=CHUNK_DX)
    vocab_size = len(vocab)
    print(f"Vocab size (TRAIN ICD10 prefixes): {vocab_size:,} (incl PAD/UNK)")

    # Now build hadm->prefix lists for each split (can remain truncated to 256)
    dx_tr = load_dx_prefix_map_chunked(cfg["dx_train"], PREFIX_LEN, chunksize=CHUNK_DX)
    dx_va = load_dx_prefix_map_chunked(cfg["dx_val"], PREFIX_LEN, chunksize=CHUNK_DX)
    dx_te = load_dx_prefix_map_chunked(cfg["dx_test"], PREFIX_LEN, chunksize=CHUNK_DX)


    tr_seqs = encode_hadm_tokens(tr, dx_tr, vocab)
    va_seqs = encode_hadm_tokens(va, dx_va, vocab)
    te_seqs = encode_hadm_tokens(te, dx_te, vocab)

    train_loader = build_loader(tr_seqs, Ytr, Mtr, shuffle=True)
    val_loader   = build_loader(va_seqs, Yva, Mva, shuffle=False)

    # For test, loader just needs sequences; labels handled separately per eval target
    Yte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    Mte0 = np.zeros((len(te), len(train_targets)), dtype=np.float32)
    test_loader = build_loader(te_seqs, Yte0, Mte0, shuffle=False)



    # -------------------------
    # STAGE 1
    # -------------------------
    validation_anchor_name = "mortality"
    mortality_idx = train_targets.index(validation_anchor_name)
    print(
        "\n=== Stage 1: Single-task NHSIC models "
        f"(checkpoint selection anchored to VAL {validation_anchor_name}) ==="
    )
    val_nhsic = []
    skipped = np.zeros(len(train_targets), dtype=bool)

    for j, t in enumerate(train_targets):

        # ---- SKIP LOGIC (apples-to-apples with Single-RBF) ----
        if is_single_class_train(Ytr, Mtr, j):
            print(f"\n  Skipping single-task model for {t}: single-class (or no valid labels) in TRAIN")
            val_nhsic.append(0.0)
            skipped[j] = True
            continue

        print(f"\n  Training single-task model for: {t}")

        # per-task sigma (fresh untrained probe, single-RBF-analog)
        model_task = SetTransformerSingleLogit(
            vocab_size=vocab_size,
            dim=EMB_DIM,
            num_heads=NUM_HEADS,
            n_layers=N_LAYERS,
            num_inducing=NUM_INDUCING_POINTS,
            dropout=DROPOUT,
        ).to(DEVICE)

        sigma_t = estimate_sigma_median_multi(model_task, train_loader, n_samples=SUBSAMPLE_N)
        sigma_t = float(max(sigma_t, 1e-6))
        print(f"  Stage1 sigma_t for {t}: {sigma_t:.6f}")

        _best_state, best_val = train_single_task(
            model=model_task,
            train_loader=train_loader,
            val_loader=val_loader,
            task_j=j,
            validation_anchor_idx=mortality_idx,
            validation_anchor_name=validation_anchor_name,
            sigma_train_base=sigma_t,
        )


        val_nhsic.append(best_val)
        print(f"  Best VAL NHSIC vs {validation_anchor_name} for model {t}: {best_val:.6f}")
    val_nhsic = np.asarray(val_nhsic, dtype=np.float32)

    # clip negatives to 0 (keep this)
    vals = np.maximum(val_nhsic, 0.0)
    max_n = float(vals.max()) if vals.size > 0 else 0.0

    # ---- NEW bounded weighting rule ----
    eps   = float(WEIGHT_EPS)
    alpha = float(WEIGHT_ALPHA)
    cap   = float(WEIGHT_CAP)

    den = np.maximum(vals, eps)
    num = max(max_n, eps)

    weights = (num / den) ** alpha
    weights = np.clip(weights, 1.0 / cap, cap).astype(np.float32)

    # zero-out skipped tasks (CRITICAL)
    weights[skipped] = 0.0

    # normalize active weights to mean 1 (keeps objective scale stable)
    active = (weights > 0)
    if active.any():
        weights[active] = weights[active] / weights[active].mean()



    print("\nStage 1 VAL NHSIC and derived weights:")
    for j, t in enumerate(train_targets):
        tag = " (SKIPPED)" if skipped[j] else ""
        print(f"  {t:16s}: val_NHSIC={float(val_nhsic[j]):.6f} | w={float(weights[j]):.6f}{tag}")


    # -------------------------
    # STAGE 2
    # -------------------------
    model = SetTransformerSingleLogit(
        vocab_size=vocab_size,
        dim=EMB_DIM,
        num_heads=NUM_HEADS,
        n_layers=N_LAYERS,
        num_inducing=NUM_INDUCING_POINTS,
        dropout=DROPOUT,
    ).to(DEVICE)

    sigma_stage2 = estimate_sigma_median_multi(model, train_loader, n_samples=SUBSAMPLE_N)
    sigma_stage2 = float(max(sigma_stage2, 1e-6))
    print(f"Stage-2 sigma_stage2 (median heuristic on TRAIN scores): {sigma_stage2:.6f}")

    print("\n=== Stage 2: Multi-task weighted NHSIC (ONE scalar) ===")
    best_multi_state = train_multi_task(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        weights=weights,
        sigma_train_base=sigma_stage2,
    )

    model.load_state_dict(best_multi_state)
    model.eval()

    # -------------------------
    # ORIENT SCORE USING VAL ANCHOR (global sign convention)
    # -------------------------
    s_val = predict_scores(model, val_loader)

    flip, corr_before = global_flip_from_val_anchor(
        score_val=s_val,
        Yva=Yva,
        Mva=Mva,
        targets=train_targets,
        anchor="mortality",
    )

    print(f"\nScore orientation (VAL anchor='mortality'): pearson_before={corr_before:.6f} | flip={flip}")

    # TEST scores (apply same flip)
    s_test = predict_scores(model, test_loader)
    s_test = s_test * flip


    # -------------------------
    # EVAL: NHSIC (subsampled) + MI_nats (mask-aware)
    # -------------------------
    print("\n=== TEST Evaluation ===")
    print(f"Bandwidth sigma_base (Stage-2 sigma): {sigma_stage2:.6f}")

    print("\nTEST NHSIC (subsampled) using multi-RBF score kernel + delta label kernel:")
    for j, t in enumerate(eval_targets):
        if t == "long_stay":
            base_valid = te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            y = te["long_stay"].fillna(0).astype(int).to_numpy()
        else:
            s = pd.to_numeric(te[t], errors="coerce")
            base_valid = (~s.isna()).to_numpy(dtype=bool)
            y = s.fillna(0).astype(int).to_numpy()

        if VALIDITY_MODE == "taskwise":
            valid = base_valid
        elif VALIDITY_MODE == "intersection":
            _, Mte_eval = build_Y_and_mask(te, eval_targets)
            valid = (Mte_eval > 0.5).all(axis=1)
        else:
            raise ValueError(f"Unknown validity_mode: {VALIDITY_MODE}")

        idx = subsample_indices(valid, SUBSAMPLE_N, seed=EVAL_SEED + 777 + stable_hash_mod(t, 1000))


        if idx.size < 3 or np.unique(y[idx]).size < 2:
            nh = 0.0
        else:
            sv = torch.tensor(s_test[idx], dtype=torch.float32, device=DEVICE)
            yv = torch.tensor(y[idx].astype(int), dtype=torch.int64, device=DEVICE)
            nh = float(nhsic_multi_rbf_batch(sv, yv, sigma_stage2).detach().cpu().item())


        print(f"  {t:16s} NHSIC={nh:.6f}")

    print("\nTEST Metrics (DistCorr, MI_nats) between score and each eval target (mask-aware):")
    for t in eval_targets:
        if t == "long_stay":
            valid = te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            y = te["long_stay"].fillna(0).astype(int).to_numpy()
        else:
            s = pd.to_numeric(te[t], errors="coerce")
            valid = (~s.isna()).to_numpy(dtype=bool)
            y = s.fillna(0).astype(int).to_numpy()

        pearson = pearson_from_score(s_test, y, valid)
        dcorr = dcor_from_score(s_test, y, valid)
        mi = mi_nats_masked(s_test, y, valid, seed=EVAL_SEED)

   
        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )



# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    for ds in ["CORE"]:
        run_dataset(ds)
