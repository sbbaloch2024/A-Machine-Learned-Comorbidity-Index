"""
SetTransformer (ISAB + PMA) + Single-RBF HSIC Multi-Task Clinical Outcome Predictor
=====================================================================================

Overview
--------
This script is a variant of the SetTransformer + NHSIC pipeline that uses a
single-scale RBF kernel with the Frobenius kernel alignment normalization
(hsic_normalized via centering with the explicit H = I - 11^T/n matrix). The
SetTransformer architecture (ISAB blocks + PMA pooling) is identical to the
multi-RBF variant, but the NHSIC objective uses only one RBF scale (sigma
estimated per stage/task via the median heuristic), and the kernel alignment
denominator is clamped with clamp_min rather than adding eps inside sqrt.

Pipeline Flow
-------------
1.  Parse --data_root (path to final_datasets) and --validity_mode
    ("taskwise" or "intersection") controlling which admissions are valid
    per HSIC objective.
2.  Load train/val/test admission CSVs and diagnosis ICD-10 CSV.gz files in
    chunks (2M rows). Build a TRAIN-only vocabulary of 4-char ICD prefix tokens
    (plus PAD and UNK tokens).
3.  Build hadm_id → token-list mappings for each split, capped at 256 codes
    per admission in file-encounter order.
4.  Construct AdmissionsSetDataset instances with per-target label matrices Y
    (N, T) and validity masks M (N, T); long_stay validity is driven by
    long_stay_defined.
5.  Estimate sigma for the single RBF kernel by computing the median pairwise
    absolute score difference over up to 5000 TRAIN predictions from the
    randomly initialized model (uses global NumPy RNG).
6.  Stage 1 — Per-target single-task HSIC training (10 epochs, AdamW, LR=1e-3):
        - Fresh SetTransformer model and fresh sigma estimate per target.
        - Skip targets that are single-class or have no valid labels on TRAIN.
        - Select best checkpoint by highest per-task VAL HSIC (subsampled).
        - Record best VAL HSIC per task for weight derivation.
7.  Derive per-task weights from Stage-1 VAL HSIC scores:
        weight_t = clip((max_score / max(score_t, ε))^0.25, 1/3, 3)
    then normalize active weights to mean 1; skipped tasks receive weight 0.
8.  Stage 2 — Weighted multi-task HSIC training (10 epochs, AdamW, LR=1e-3):
        - Fresh SetTransformer model and fresh sigma estimate.
        - Objective is torch.zeros-accumulated weighted sum of per-task HSIC.
        - Select best checkpoint by highest weighted VAL HSIC sum.
9.  Orient the final score's sign using VAL Pearson correlation against mortality
    (flip so that higher score implies higher risk).
10. Evaluate on TEST: per-task HSIC (subsampled, target-hash seeds), mean HSIC,
     Distance Correlation, and Mutual Information.

Architecture: SetTransformerSingleScalar
-----------------------------------------
- Embedding layer (vocab_size → EMB_DIM=128), padding_idx=0
- N_LAYERS=2 ISAB blocks (Induced Set Attention Block):
    MAB1: inducing points (16) attend to input X (key_padding_mask applied)
    MAB2: input X attends to inducing point outputs (no masking)
    Each MAB has MultiheadAttention + LayerNorm + feed-forward + LayerNorm.
- PMA (Pooling by Multihead Attention): 1 seed vector attends over encoded set.
- Linear(d → 1) → scalar score per admission.

Key Difference from Multi-RBF SetTransformer Variant
------------------------------------------------------
This variant uses a single RBF kernel (rbf_kernel_1d) rather than the average
of five scales. The center_kernel function uses the explicit hat matrix
H = I - 11^T/n rather than row/col/grand mean subtraction. The denominator
normalization uses clamp_min(eps=1e-12) rather than adding eps inside sqrt.
Also reports mean HSIC across eval targets at TEST time.

Key Configuration
-----------------
  SEED: controlled via SEED env var (default 1001); EVAL_SEED=12345 (fixed)
  EMB_DIM=128, NUM_HEADS=4, N_LAYERS=2, NUM_INDUCING_POINTS=16, DROPOUT=0.0
  BATCH_SIZE=256, LR=1e-3, WEIGHT_DECAY=0.0
  EPOCHS_STAGE1=10, EPOCHS_STAGE2=10
  PREFIX_LEN=4, MAX_CODES_PER_ADMISSION=256, DX_CHUNKSIZE=2_000_000
  HSIC_SUBSAMPLE=5000
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
    python nhsic_single_rbf_settransformer_mimiciv.py --data_root /path/to/final_datasets

With optional arguments:
    python nhsic_single_rbf_settransformer_mimiciv.py \\
        --data_root /path/to/final_datasets \\
        --validity_mode intersection

Override training seed:
    SEED=42 python nhsic_single_rbf_settransformer_mimiciv.py --data_root /path/to/final_datasets

Example:
    python nhsic_single_rbf_settransformer_mimiciv.py --data_root ./data/final_datasets
"""

import os
import argparse
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor


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
ROOT = Path(ARGS.data_root)
VALIDITY_MODE = ARGS.validity_mode
print("DATA_ROOT:", ROOT)
print("VALIDITY_MODE for HSIC/NHSIC objectives:", VALIDITY_MODE)


def orient_score_by_anchor(
    score: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    targets: List[str],
    anchor: str = "mortality",
) -> Tuple[int, float]:
    """
    Decide a GLOBAL sign so that higher score => higher risk for anchor (y=1).
    Uses Pearson on the provided split (use VAL).
    Returns (flip, pearson_before), where flip is +1 or -1.
    """
    if anchor not in targets:
        return +1, 0.0

    j = targets.index(anchor)
    corr = pearson_from_score(score, Y[:, j], M[:, j])
    flip = -1 if corr < 0 else +1
    return flip, float(corr)





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


# =========================
# PATHS (FINAL DATASETS)
# =========================
DATASETS = {
    "CORE": {
        "train_adm": ROOT / "core" / "admissions_core_train.csv",
        "val_adm":   ROOT / "core" / "admissions_core_val.csv",
        "test_adm":  ROOT / "core" / "admissions_core_test.csv",
        "train_dx":  ROOT / "core" / "diagnoses_icd10_core_train.csv.gz",
        "val_dx":    ROOT / "core" / "diagnoses_icd10_core_val.csv.gz",
        "test_dx":   ROOT / "core" / "diagnoses_icd10_core_test.csv.gz",
        "train_targets": ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
        "eval_targets":  ["mortality", "mortality_30d", "long_stay", "icu_transfer"],
    },
}


# =========================
# CONFIG
# =========================
SEED = int(os.environ.get("SEED", 1001))
EVAL_SEED = 12345
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREFIX_LEN = 4
EMB_DIM = 128
N_LAYERS = 2
NUM_HEADS = 4
NUM_INDUCING_POINTS = 16
DROPOUT = 0.0

BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.0
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10

MAX_CODES_PER_ADMISSION = 256

HSIC_SUBSAMPLE = 5000
HSIC_FLOOR = 1e-4

MI_BASE_NEIGHBORS = 5

DX_CHUNKSIZE = 2_000_000
WEIGHT_EPS = 0.02
WEIGHT_ALPHA = 0.25
WEIGHT_CAP = 3.0


def stable_hash_mod(s: str, mod: int = 1000) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def subsample_indices(valid_mask: np.ndarray, n: int, seed: int = EVAL_SEED) -> np.ndarray:
    idx = np.where(valid_mask.astype(bool))[0]
    if idx.size == 0:
        return idx
    if idx.size <= n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=n, replace=False)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


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


# =========================
# SEEDING
# =========================
def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# ICD PREFIX TOKENIZATION
# =========================
def norm_icd(code: str) -> str:
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()


def icd_prefix(code_norm: str, k: int = PREFIX_LEN) -> str:
    if not code_norm:
        return ""
    return code_norm[:k] if len(code_norm) >= k else code_norm


# =========================
# DIAGNOSES LOADING
# =========================
def build_vocab_from_dx(dx_path: Path) -> Dict[str, int]:
    vocab = set()
    for chunk in pd.read_csv(
        dx_path,
        usecols=["hadm_id", "icd_code"],
        dtype={"icd_code": str},
        chunksize=DX_CHUNKSIZE,
    ):
        c = chunk["icd_code"].map(norm_icd).map(lambda s: icd_prefix(s, PREFIX_LEN))
        c = c[c != ""]
        vocab.update(c.unique().tolist())

    vocab = sorted(vocab)
    stoi = {"<PAD>": 0, "<UNK>": 1}
    for i, tok in enumerate(vocab, start=2):
        stoi[tok] = i
    return stoi


def hsic_normalized(K: torch.Tensor, L: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalized HSIC (kernel alignment):
        <Kc, Lc>_F / (||Kc||_F ||Lc||_F)
    """
    n = K.size(0)
    if n < 3:
        return K.sum() * 0.0

    Kc = center_kernel(K)
    Lc = center_kernel(L)

    num = (Kc * Lc).sum()
    den = torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()).clamp_min(eps)

    return num / den


def build_hadm_to_tokens(dx_path: Path, stoi: Dict[str, int]) -> Dict[int, List[int]]:
    hadm2toks: Dict[int, List[int]] = {}
    for chunk in pd.read_csv(
        dx_path,
        usecols=["hadm_id", "icd_code"],
        dtype={"icd_code": str},
        chunksize=DX_CHUNKSIZE,
    ):
        hadm = chunk["hadm_id"].to_numpy()
        codes = chunk["icd_code"].to_numpy()
        for h, raw in zip(hadm, codes):
            try:
                hid = int(h)
            except Exception:
                continue
            cn = icd_prefix(norm_icd(raw), PREFIX_LEN)
            if not cn:
                continue
            tid = stoi.get(cn, 1)
            lst = hadm2toks.get(hid)
            if lst is None:
                hadm2toks[hid] = [tid]
            else:
                lst.append(tid)

    if MAX_CODES_PER_ADMISSION is not None:
        for hid, lst in hadm2toks.items():
            if len(lst) > MAX_CODES_PER_ADMISSION:
                hadm2toks[hid] = lst[:MAX_CODES_PER_ADMISSION]
    return hadm2toks


# =========================
# DATASET
# =========================
class AdmissionsSetDataset(Dataset):
    def __init__(
        self,
        admissions_df: pd.DataFrame,
        hadm2toks: Dict[int, List[int]],
        targets: List[str],
    ):
        self.hadm_ids = admissions_df["hadm_id"].astype(int).to_numpy()
        self.targets = targets

        self.tokens = [hadm2toks.get(int(h), []) for h in self.hadm_ids]

        Y = admissions_df[targets].copy()
        for t in targets:
            Y[t] = pd.to_numeric(Y[t], errors="coerce")

        mask = (~Y.isna()).to_numpy(dtype=np.bool_)

        if "long_stay" in targets:
            if "long_stay_defined" not in admissions_df.columns:
                raise ValueError("Expected long_stay_defined column for long_stay masking.")
            j = targets.index("long_stay")
            valid_ls = admissions_df["long_stay_defined"].fillna(0).astype(int).to_numpy().astype(bool)
            mask[:, j] = valid_ls

        self.mask = mask
        self.y = Y.fillna(0).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        return self.tokens[idx], self.y[idx], self.mask[idx]


def collate_batch(batch):
    toks, y, m = zip(*batch)
    lens = [len(x) for x in toks]
    max_len = max(lens) if max(lens) > 0 else 1

    B = len(toks)
    x = torch.zeros((B, max_len), dtype=torch.long)
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)

    for i, seq in enumerate(toks):
        if len(seq) == 0:
            continue
        s = torch.tensor(seq, dtype=torch.long)
        x[i, : len(seq)] = s
        pad_mask[i, : len(seq)] = False

    y = torch.tensor(np.asarray(y), dtype=torch.float32)
    m = torch.tensor(np.asarray(m), dtype=torch.float32)
    return x, pad_mask, y, m


# =========================
# SET TRANSFORMER MODULES
# =========================
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

    def forward(self, Q, K, key_padding_mask: Optional[torch.Tensor] = None):
        H, _ = self.mha(Q, K, K, key_padding_mask=key_padding_mask, need_weights=False)
        Q = self.ln1(Q + H)
        H2 = self.ff(Q)
        Q = self.ln2(Q + H2)
        return Q


class ISAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab1 = MAB(dim, num_heads, dropout)
        self.mab2 = MAB(dim, num_heads, dropout)

    def forward(self, X, key_padding_mask: torch.Tensor):
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, key_padding_mask=key_padding_mask)
        out = self.mab2(X, H, key_padding_mask=None)
        return out


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.mab = MAB(dim, num_heads, dropout)

    def forward(self, X, key_padding_mask: torch.Tensor):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        out = self.mab(S, X, key_padding_mask=key_padding_mask)
        return out


class SetTransformerSingleScalar(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = EMB_DIM,
        n_layers: int = N_LAYERS,
        num_heads: int = NUM_HEADS,
        num_inducing: int = NUM_INDUCING_POINTS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.blocks = nn.ModuleList([ISAB(dim, num_heads, num_inducing, dropout) for _ in range(n_layers)])
        self.pma = PMA(dim, num_heads, num_seeds=1, dropout=dropout)
        self.out = nn.Linear(dim, 1)

    def forward(self, x_tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        X = self.emb(x_tokens)
        for blk in self.blocks:
            X = blk(X, key_padding_mask=pad_mask)
        pooled = self.pma(X, key_padding_mask=pad_mask)
        s = self.out(pooled[:, 0, :]).squeeze(-1)
        return s


# =========================
# HSIC
# =========================
def rbf_kernel_1d(s: torch.Tensor, sigma: float) -> torch.Tensor:
    s = s.view(-1, 1)
    d2 = (s - s.t()).pow(2)
    K = torch.exp(-d2 / (2.0 * (sigma ** 2)))
    return K


def delta_kernel(y: torch.Tensor) -> torch.Tensor:
    return (y.view(-1, 1) == y.view(1, -1)).float()


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    return H @ K @ H


def hsic_task_from_batch(
    s: torch.Tensor,
    y: torch.Tensor,
    m_all: torch.Tensor,
    sigma: float,
    task_idx: int,
    validity_mode: str = VALIDITY_MODE,
) -> torch.Tensor:
    valid = get_valid_mask_torch(m_all, task_idx, mode=validity_mode)
    if valid.sum().item() < 3:
        return s.sum() * 0.0

    sv = s[valid]
    yv = y[valid].to(torch.int64)
    if torch.unique(yv).numel() < 2:
        return s.sum() * 0.0

    K = rbf_kernel_1d(sv, sigma)
    L = delta_kernel(yv)
    return hsic_normalized(K, L)


@torch.no_grad()
def estimate_sigma_median(model: nn.Module, loader: DataLoader, n_samples: int = HSIC_SUBSAMPLE) -> float:
    model.eval()
    scores = []
    for x, pad_mask, _y, _m in loader:
        x = x.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        s = model(x, pad_mask).detach().float().cpu().numpy()
        scores.append(s)
        if sum(len(a) for a in scores) >= n_samples:
            break
    s_all = np.concatenate(scores, axis=0)
    if len(s_all) < 10:
        return 1.0

    k = min(len(s_all), n_samples)
    idx = np.random.choice(len(s_all), size=k, replace=False)
    s = s_all[idx]
    d = np.abs(s.reshape(-1, 1) - s.reshape(1, -1))
    med = np.median(d[d > 0])
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return float(med)


@torch.no_grad()
def hsic_on_split(
    model: nn.Module,
    loader: DataLoader,
    targets: List[str],
    sigma: float,
    subsample: int = HSIC_SUBSAMPLE,
    seed: int = EVAL_SEED,
    use_target_hash: bool = False,
    target_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    model.eval()
    S, Y, M = [], [], []
    for x, pad_mask, y, m in loader:
        x = x.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        s = model(x, pad_mask).detach().float().cpu().numpy()
        S.append(s)
        Y.append(y.numpy())
        M.append(m.numpy())
    S = np.concatenate(S, axis=0).astype(np.float32)
    Y = np.concatenate(Y, axis=0).astype(np.float32)
    M = np.concatenate(M, axis=0).astype(np.float32)

    out = {}
    if target_indices is None:
        target_indices = list(range(len(targets)))
    if len(target_indices) != len(targets):
        raise ValueError("target_indices must have the same length as targets")

    for local_j, (t, j) in enumerate(zip(targets, target_indices)):
        valid = get_valid_mask_np(M, j, mode=VALIDITY_MODE)
        if valid.sum() < 3:
            out[t] = 0.0
            continue

        yv = Y[valid, j].astype(np.int64)
        if np.unique(yv).size < 2:
            out[t] = 0.0
            continue

        seed_j = seed + (stable_hash_mod(t, 1000) if use_target_hash else local_j)
        idx = subsample_indices(valid, subsample, seed=seed_j)

        if idx.size < 3:
            out[t] = 0.0
            continue

        yy = Y[idx, j].astype(np.int64)
        if np.unique(yy).size < 2:
            out[t] = 0.0
            continue

        sv = torch.tensor(S[idx], device=DEVICE, dtype=torch.float32)
        yv_t = torch.tensor(yy, device=DEVICE, dtype=torch.int64)

        K = rbf_kernel_1d(sv, sigma)
        L = delta_kernel(yv_t)
        hs = hsic_normalized(K, L).detach().float().cpu().item()
        out[t] = float(hs)

    return out


# =========================
# MI (masked)
# =========================
def mi_nats_masked(
    score: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    targets: List[str],
    seed: int = EVAL_SEED,
    base_neighbors: int = MI_BASE_NEIGHBORS,
) -> Dict[str, float]:
    out = {}
    for j, t in enumerate(targets):
        valid = M[:, j].astype(bool)
        if valid.sum() < 3:
            out[t] = 0.0
            continue

        yy = Y[valid, j].astype(int)
        if np.unique(yy).size < 2:
            out[t] = 0.0
            continue

        nn = int(min(base_neighbors, valid.sum() - 1))
        if nn < 1:
            out[t] = 0.0
            continue

        s_valid = zscore_1d(score[valid]).reshape(-1, 1)

        out[t] = float(mutual_info_classif(s_valid, yy, random_state=seed, n_neighbors=nn)[0])
    return out


# =========================
# TRAINING
# =========================
@torch.no_grad()
def label_rates(df: pd.DataFrame, targets: List[str]) -> Dict[str, float]:
    out = {}
    for t in targets:
        if t == "long_stay":
            valid = df["long_stay_defined"].fillna(0).astype(int).to_numpy().astype(bool)
            y = df["long_stay"].fillna(0).astype(float).to_numpy()
            out[t] = float(y[valid].mean()) if valid.sum() > 0 else 0.0
        else:
            y = pd.to_numeric(df[t], errors="coerce")
            valid = (~y.isna()).to_numpy()
            out[t] = float(y[valid].fillna(0).mean()) if valid.sum() > 0 else 0.0
    return out


def is_single_class_train_from_df(df_tr: pd.DataFrame, target: str) -> bool:
    if target == "long_stay":
        if "long_stay_defined" not in df_tr.columns or "long_stay" not in df_tr.columns:
            return True
        valid = df_tr["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
        yy = pd.to_numeric(df_tr["long_stay"], errors="coerce").fillna(0).astype(int).to_numpy()
        yy = yy[valid]
        if yy.size == 0:
            return True
        return np.unique(yy).size < 2

    ycol = pd.to_numeric(df_tr[target], errors="coerce")
    valid = ~ycol.isna()
    if valid.sum() == 0:
        return True
    yy = ycol[valid].astype(int).to_numpy()
    return np.unique(yy).size < 2


def train_single_task(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_idx: int,
    task_name: str,
    validation_anchor_idx: int,
    validation_anchor_name: str,
    sigma: float,
) -> Tuple[Dict[str, torch.Tensor], float]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = -1e18

    for ep in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        for x, pad_mask, y, m in train_loader:
            x = x.to(DEVICE)
            pad_mask = pad_mask.to(DEVICE)
            y = y.to(DEVICE)
            m = m.to(DEVICE)

            s = model(x, pad_mask)
            hs = hsic_task_from_batch(s, y[:, task_idx], m, sigma, task_idx)
            loss = -hs

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        hsic_val = hsic_on_split(
            model,
            val_loader,
            [validation_anchor_name],
            sigma,
            subsample=HSIC_SUBSAMPLE,
            seed=EVAL_SEED + 1000 + task_idx,
            target_indices=[validation_anchor_idx],
        )[validation_anchor_name]

        if hsic_val > best_val:
            best_val = hsic_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, float(best_val)


def train_weighted_multitask(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_targets: List[str],
    weights: np.ndarray,
    sigma: float,
) -> Dict[str, torch.Tensor]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = -1e18

    w = torch.tensor(weights, device=DEVICE, dtype=torch.float32)

    for ep in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        for x, pad_mask, y, m in train_loader:
            x = x.to(DEVICE)
            pad_mask = pad_mask.to(DEVICE)
            y = y.to(DEVICE)
            m = m.to(DEVICE)

            s = model(x, pad_mask)

            obj = torch.zeros((), device=DEVICE)
            for j in range(len(train_targets)):
                if w[j].item() == 0.0:
                    continue
                hs = hsic_task_from_batch(s, y[:, j], m, sigma, j)
                obj = obj + w[j] * hs

            loss = -obj
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        hs_val_map = hsic_on_split(
            model,
            val_loader,
            train_targets,
            sigma,
            subsample=HSIC_SUBSAMPLE,
            seed=EVAL_SEED + 9999,
        )

        hs_val = 0.0
        for j, t in enumerate(train_targets):
            hs_val += float(weights[j]) * hs_val_map[t]

        if hs_val > best_val:
            best_val = hs_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state


@torch.no_grad()
def scores_and_labels(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    S, Y, M = [], [], []
    for x, pad_mask, y, m in loader:
        x = x.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        s = model(x, pad_mask).detach().float().cpu().numpy()
        S.append(s)
        Y.append(y.numpy())
        M.append(m.numpy())
    return (
        np.concatenate(S, axis=0).astype(np.float32),
        np.concatenate(Y, axis=0).astype(np.float32),
        np.concatenate(M, axis=0).astype(np.float32),
    )


# =========================
# RUN ONE DATASET
# =========================
def run_dataset(name: str):
    cfg = DATASETS[name]

    print("\n" + "=" * 100)
    print(f"DATASET: {name}")
    print("=" * 100)

    df_tr = pd.read_csv(cfg["train_adm"])
    df_va = pd.read_csv(cfg["val_adm"])
    df_te = pd.read_csv(cfg["test_adm"])

    train_targets = cfg["train_targets"]
    eval_targets = cfg["eval_targets"]

    print(f"\nTRAIN n={len(df_tr):,} | VAL n={len(df_va):,} | TEST n={len(df_te):,}")
    tr_rates = label_rates(df_tr, train_targets)
    te_rates = label_rates(df_te, eval_targets)

    print("\nTRAIN label positive rates (valid-only):")
    for t in train_targets:
        print(f"  {t:16s}: {tr_rates[t]:.6f}")
    print("\nTEST label positive rates (valid-only) [EVAL TARGETS]:")
    for t in eval_targets:
        print(f"  {t:16s}: {te_rates[t]:.6f}")

    print("\nLoading diagnoses (chunked) + building TRAIN vocab...")
    stoi = build_vocab_from_dx(cfg["train_dx"])
    vocab_size = len(stoi)
    print(f"Vocab size (TRAIN ICD10 prefixes): {vocab_size:,} (incl PAD/UNK)")

    print("Building hadm->tokens for TRAIN/VAL/TEST...")
    hadm2tr = build_hadm_to_tokens(cfg["train_dx"], stoi)
    hadm2va = build_hadm_to_tokens(cfg["val_dx"], stoi)
    hadm2te = build_hadm_to_tokens(cfg["test_dx"], stoi)

    ds_tr = AdmissionsSetDataset(df_tr, hadm2tr, train_targets)
    ds_va = AdmissionsSetDataset(df_va, hadm2va, train_targets)
    ds_te = AdmissionsSetDataset(df_te, hadm2te, eval_targets)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_batch)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_batch)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_batch)

    validation_anchor_name = "mortality"
    mortality_idx = train_targets.index(validation_anchor_name)
    print(
        "\n=== Stage 1: Single-task HSIC models "
        f"(checkpoint selection anchored to VAL {validation_anchor_name}) ==="
    )
    stage1_best = {}
    stage1_val_hsic = {}
    skipped = np.zeros(len(train_targets), dtype=bool)

    for j, t in enumerate(train_targets):
        if is_single_class_train_from_df(df_tr, t):
            print(f"\n  Skipping single-task model for {t}: single-class (or no valid labels) in TRAIN")
            stage1_best[t] = None
            stage1_val_hsic[t] = 0.0
            skipped[j] = True
            continue

        print(f"\n  Training single-task model for: {t}")
        model = SetTransformerSingleScalar(vocab_size=vocab_size).to(DEVICE)

        sigma = estimate_sigma_median(model, dl_tr, n_samples=HSIC_SUBSAMPLE)
        sigma = float(max(sigma, 1e-6))
        best_state, best_val = train_single_task(
            model, dl_tr, dl_va, j, t,
            mortality_idx, validation_anchor_name, sigma,
        )

        stage1_best[t] = best_state
        stage1_val_hsic[t] = best_val
        print(f"    best VAL HSIC vs {validation_anchor_name}: {best_val:.6f} (sigma={sigma:.6f})")

    vals = np.array([max(stage1_val_hsic[t], 0.0) for t in train_targets], dtype=np.float64)
    max_h = float(vals.max()) if vals.size > 0 else 0.0

    eps = WEIGHT_EPS
    alpha = WEIGHT_ALPHA
    w_cap = WEIGHT_CAP

    denom = np.maximum(vals, eps)
    weights = (np.maximum(max_h, eps) / denom) ** alpha
    weights = np.clip(weights, 1.0 / w_cap, w_cap).astype(np.float32)

    for j, t in enumerate(train_targets):
        if stage1_best[t] is None:
            weights[j] = 0.0

    active = weights > 0
    if active.any():
        weights[active] = weights[active] / weights[active].mean()

    print("\nStage-1 VAL HSIC and derived weights:")
    for j, t in enumerate(train_targets):
        tag = " (SKIPPED)" if skipped[j] else ""
        print(f"  {t:16s}: val_HSIC={float(stage1_val_hsic[t]):.6f} | w={float(weights[j]):.6f}{tag}")

    print("\n=== Stage 2: Weighted-sum HSIC multi-task model ===")
    model = SetTransformerSingleScalar(vocab_size=vocab_size).to(DEVICE)
    sigma2 = estimate_sigma_median(model, dl_tr, n_samples=HSIC_SUBSAMPLE)
    sigma2 = float(max(sigma2, 1e-6))

    best_state2 = train_weighted_multitask(model, dl_tr, dl_va, train_targets, weights, sigma2)
    model.load_state_dict(best_state2)
    print(f"Stage-2 sigma={sigma2:.6f}")

    score_va, Y_va, M_va = scores_and_labels(model, dl_va)
    flip, corr_before = orient_score_by_anchor(
        score_va, Y_va, M_va, train_targets, anchor="mortality"
    )
    print(f"\nScore orientation (VAL anchor='mortality'): pearson_before={corr_before:.6f} | flip={flip}")

    print("\n=== TEST EVALUATION (ONE score s(x)) ===")
    score_te, Y_te, M_te = scores_and_labels(model, dl_te)
    score_te = score_te * flip

    hsic_te = hsic_on_split(
        model,
        dl_te,
        eval_targets,
        sigma2,
        subsample=HSIC_SUBSAMPLE,
        seed=EVAL_SEED + 777,
        use_target_hash=True,
    )

    mean_hsic = float(np.mean([hsic_te[t] for t in eval_targets])) if len(eval_targets) > 0 else 0.0

    print("\nTEST HSIC (subsampled, masked) using Stage-2 score:")
    for t in eval_targets:
        print(f"  {t:16s} HSIC={hsic_te[t]:.6f}")
    print(f"  {'MEAN':16s} HSIC={mean_hsic:.6f}")

    print("\nTEST Metrics (DistCorr, MI_nats) using Stage-2 score:")
    mi_te = mi_nats_masked(score_te, Y_te, M_te, eval_targets)

    for j, t in enumerate(eval_targets):
        valid = M_te[:, j].astype(bool)
        y = Y_te[:, j]

        pearson = pearson_from_score(score_te, y, valid)
        dcorr = dcor_from_score(score_te, y, valid)
        mi = mi_te[t]

  
        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )
    print("=" * 100)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    seed_all(SEED)
    for ds_name in ["CORE"]:
        run_dataset(ds_name)
