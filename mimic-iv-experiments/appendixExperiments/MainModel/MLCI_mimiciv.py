"""
MLCI 
================================================================================

Overview
--------
This script implements MLCI trained with normalized Hilbert-Schmidt Independence 
Criterion (NHSIC). Each hospital admission is represented as a variable-length set 
of ICD-10 prefix tokens. Tokens are embedded, processed independently by a phi network, 
pooled using both masked mean pooling and masked max pooling, concatenated, layer-normalized, 
and passed through a rho network to produce one scalar score per admission.

The scalar score is optimized to maximize statistical dependence with multiple
binary clinical outcomes. The training procedure has two stages. Stage 1 trains
separate single-task NHSIC models for each outcome and records mortality-anchored
validation NHSIC. These values are converted into task weights. Stage 2 trains one
weighted multi-task NHSIC model that learns a single shared score for all active
targets.

After Stage 2, the learned scalar score is globally oriented using validation
mortality so that higher scores correspond to higher mortality risk when possible.
The test split is then evaluated using NHSIC, distance correlation, 
and mutual information in nats. A diagnostic CSV containing the learned test score and 
outcomes is also written to disk.

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

4.  Define training and evaluation targets:
        - mortality
        - mortality_30d
        - long_stay
        - icu_transfer

5.  Print valid-only label positive rates for TRAIN and TEST:
        - For long_stay, validity is determined by long_stay_defined.
        - For all other targets, validity is determined by non-missing labels.

6.  Build a TRAIN-only ICD-10 prefix vocabulary:
        - Normalize ICD codes by uppercasing and removing periods/spaces.
        - Truncate each normalized ICD code to PREFIX_LEN=4 characters.
        - Reserve <PAD>=0 and <UNK>=1.
        - Read the training diagnoses file in chunks of DX_CHUNKSIZE=2,000,000
          rows.
        - Use only training diagnosis prefixes to build the vocabulary.

7.  Build hadm_id-to-token-ID mappings for TRAIN, VAL, and TEST:
        - Read diagnosis files in chunks.
        - Convert each ICD code to a normalized prefix.
        - Map prefixes through the TRAIN vocabulary.
        - Map unseen validation/test prefixes to <UNK>.
        - Preserve diagnosis token order within each admission.
        - Truncate each admission to MAX_CODES_PER_ADMISSION=256 tokens.

8.  Construct AdmissionsSetDataset objects:
        - Store hadm_id values.
        - Store token ID lists per admission.
        - Build label matrix Y for all targets.
        - Build mask matrix M for valid labels.
        - Use long_stay_defined as the validity mask for long_stay.

9.  Pad variable-length ICD token sets in collate_batch:
        - Create padded token tensor x of shape (B, L).
        - Create pad_mask of shape (B, L), where True indicates PAD.
        - Leave admissions with no diagnosis tokens as all-PAD rows.
        - Return x, pad_mask, label matrix y, and validity mask m.

10. Train Stage 1 single-task NHSIC models:
        - For each target, skip training if the target is single-class or has no
          valid TRAIN labels.
        - Initialize a fresh DeepSetsSingleScalar model.
        - Estimate an RBF kernel bandwidth sigma from model scores using the
          median pairwise distance of standardized sampled scores.
        - Train for EPOCHS_STAGE1=10 epochs by maximizing NHSIC between the scalar
          score and the target label.
        - Select the best checkpoint by highest validation NHSIC with mortality,
          which is the explicit Stage-1 severity anchor.

11. Compute task weights from Stage-1 mortality-anchored validation NHSIC:
        - Clamp mortality-anchored validation NHSIC values at zero.
        - Compute stabilized inverse-NHSIC weights:
              weight_t = (max(max_h, eps) / max(nhsic_t, eps))^alpha
        - Use eps=0.02, alpha=0.25, and w_cap=3.0.
        - Clip weights to [1 / w_cap, w_cap].
        - Set skipped-task weights to zero.
        - Normalize active weights so their mean is approximately one.

12. Train Stage 2 weighted multi-task NHSIC model:
        - Initialize one DeepSetsSingleScalar model.
        - Estimate a new sigma for the Stage-2 model.
        - For each batch, compute a weighted sum of per-task NHSIC objectives.
        - Skip any task whose weight is zero.
        - Maximize the weighted NHSIC objective for EPOCHS_STAGE2=10 epochs.
        - Select the best checkpoint by highest weighted validation NHSIC.

13. Orient the learned scalar score:
        - Score the validation set.
        - Compute Pearson correlation between validation score and validation
          mortality.
        - If USE_INTERSECTION_VALIDITY=True, use rows valid for all targets.
        - Otherwise, use the mortality-specific validity mask.
        - Flip the global sign if mortality Pearson correlation is negative.
        - This makes higher score correspond to higher mortality risk when the
          mortality anchor is available.

14. Score the TEST split:
        - Generate one scalar score per test admission.
        - Apply the validation-derived sign flip.
        - Print diagnostic score spread information:
              unique fraction
              standard deviation

15. Save a diagnostic CSV:
        - Include hadm_id.
        - Include learned_score.
        - Include evaluation target columns.
        - Include long_stay_defined when available.
        - Save to the configured DiagnosticCSV directory.

16. Evaluate TEST dependence and association:
        - Compute NHSIC using the Stage-2 score.
        - Compute distance correlation.
        - Compute mutual information in nats.

Architecture: DeepSetsSingleScalar
----------------------------------
- Embedding layer:
      vocab_size -> EMB_DIM=128, padding_idx=0

- Phi network applied independently to each token embedding:
      Linear(128 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU

- Masked mean pooling:
      valid token embeddings are summed and divided by valid token count.

- Masked max pooling:
      PAD positions are filled with -1e9 before max pooling.

- Concatenation:
      pooled = concat(mean_pool, max_pool), producing a 2*EMB_DIM=256 vector.

- Layer normalization:
      LayerNorm(256)

- Rho network:
      Linear(256 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> Linear(128 -> 1)

- Output:
      one scalar score s(x) per admission, shape (B,).

NHSIC Objective
---------------
For each task, the model computes normalized HSIC between the scalar score and the
binary target label. The score kernel is an RBF kernel and the label kernel is a
delta kernel.

The score kernel is:

    K_ij = exp(-(s_i - s_j)^2 / (2 sigma^2))

The label kernel is:

    L_ij = 1[y_i = y_j]

Both kernels are centered:

    K_c = K - row_mean(K) - col_mean(K) + mean(K)
    L_c = L - row_mean(L) - col_mean(L) + mean(L)

NHSIC is computed as:

    NHSIC(K, L) = <K_c, L_c>_F / (max(||K_c||_F, epsilon_0) * ||L_c||_F)

The model maximizes NHSIC by minimizing the negative NHSIC objective.

RBF Bandwidth Estimation
------------------------
The RBF bandwidth sigma is estimated from model scores before training each model:

    - Collect up to NHSIC_SUBSAMPLE scores from the training loader.
    - Standardize the sampled scores.
    - Deterministically sample up to NHSIC_SUBSAMPLE values using EVAL_SEED.
    - Compute pairwise absolute score distances.
    - Use the median nonzero pairwise distance as sigma.
    - Return sigma=1.0 if too few scores or invalid distances are available.
    - Floor sigma at 0.05 to avoid extremely narrow kernels.

Validity Handling
-----------------
USE_INTERSECTION_VALIDITY=True by default.

When USE_INTERSECTION_VALIDITY=True:
    A row is considered valid for NHSIC if all task labels are valid for that
    admission.

When USE_INTERSECTION_VALIDITY=False:
    Each task uses its own per-target validity mask.

For long_stay:
    valid = long_stay_defined == 1

For all other targets:
    valid = target value is not missing

Invalid labels are filled with zero in the label matrix, but they do not
contribute to NHSIC, orientation, or evaluation metrics because their mask value is
false.

Single-Class Task Handling
--------------------------
Before Stage 1, each target is checked on the TRAIN split. A target is skipped if
it has no valid labels or if all valid labels belong to one class. Skipped tasks:

    - are not trained in Stage 1
    - receive Stage-1 mortality-anchored validation NHSIC = 0
    - receive Stage-2 weight = 0
    - do not contribute to the weighted multi-task NHSIC objective

Score Orientation
-----------------
The final scalar score is oriented using the validation split and the mortality
anchor. The script computes Pearson(score, mortality) on validation rows. If this
correlation is negative, the test score is multiplied by -1.

This produces an interpretable direction where higher score generally means higher
mortality risk, when the mortality anchor is present and valid.

Task Weighting Rule
-------------------
Stage-1 mortality-anchored validation NHSIC values are converted into weights
using a stabilized inverse-NHSIC rule:

    weight_t = (max(max_h, eps) / max(nhsic_t, eps))^alpha

where:

    max_h = max Stage-1 mortality-anchored validation NHSIC across targets
    eps = 0.02
    alpha = 0.25

Weights are clipped to:

    [1 / w_cap, w_cap]

where:

    w_cap = 3.0

Skipped tasks are assigned weight zero. Active weights are mean-normalized so the
average active weight is one.

Evaluation Metrics
------------------
The Stage-2 scalar score is evaluated against each target using:

    - NHSIC
    - Distance correlation
    - Mutual information in nats

Metric safeguards:
    - Return 0.0 when fewer than three valid examples are available.
    - Return 0.0 when the score or label vector is constant or near-constant.
    - Return 0.0 for mutual information when the target has fewer than two classes.
    - Z-score scores within valid rows before mutual information estimation.
    - Cap mutual-information neighbors at valid sample size minus one.

Key Configuration
-----------------
  SEED: controlled via SEED environment variable, default 1001
  EVAL_SEED=12345
  DEVICE: cuda if available, otherwise cpu
  USE_INTERSECTION_VALIDITY=True
  PREFIX_LEN=4
  EMB_DIM=128
  BATCH_SIZE=256
  LR=1e-3
  WEIGHT_DECAY=0.0
  EPOCHS_STAGE1=10
  EPOCHS_STAGE2=10
  MAX_CODES_PER_ADMISSION=256
  DX_CHUNKSIZE=2_000_000
  NHSIC_SUBSAMPLE=20000
  NHSIC_FLOOR=1e-4
  EPS=1e-8
  MI_BASE_NEIGHBORS=5

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
    - TRAIN label positive rates
    - TEST label positive rates
    - ICD-10 prefix vocabulary size
    - Stage-1 skipped tasks
    - Stage-1 best mortality-anchored validation NHSIC by target
    - derived task weights
    - Stage-2 sigma
    - validation mortality orientation correlation and flip value
    - diagnostic CSV path
    - test score unique fraction and standard deviation
    - test NHSIC by target
    - test distance correlation, and mutual
      information by target

The script writes a diagnostic CSV to:

    /OUPUTPATH-HERE/DiagnosticCSV/

Output filename:

    CORE_seed<SEED>_test_scores_and_outcomes.csv

The CSV contains:

    hadm_id, learned_score, mortality, mortality_30d, long_stay, icu_transfer,
    and long_stay_defined when available.

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor torch

Running
-------
Run the script with:

    python MLCI_mimiciv.py --data_root /path/to/final_datasets

Override the training seed:

    SEED=42 python MLCI_mimiciv.py --data_root /path/to/final_datasets

Example:
    python MLCI_mimiciv.py --data_root ./data/final_datasets
"""

import os
import argparse
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor

USE_INTERSECTION_VALIDITY = True

def intersection_mask_np(M: np.ndarray) -> np.ndarray:
    # M: (N, T) float/bool mask
    return (M > 0.5).all(axis=1)

def intersection_mask_torch(m: torch.Tensor) -> torch.Tensor:
    # m: (B, T) float/bool mask
    return (m > 0.5).all(dim=1)



def orient_score_by_anchor(
    score: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    targets: List[str],
    anchor: str = "mortality",
) -> Tuple[int, float]:
    """
    Decide global sign flip so that higher score => higher risk for anchor (y=1).
    Uses Pearson on the provided split (use VAL).
    Returns (flip, pearson_before), where flip is +1 or -1.
    """
    if anchor not in targets:
        return +1, 0.0

    j = targets.index(anchor)
    if USE_INTERSECTION_VALIDITY:
        valid = intersection_mask_np(M)
    else:
        valid = (M[:, j] > 0.5)

    corr = pearson_from_score(score, Y[:, j], valid)
    flip = -1 if corr < 0 else +1
    return flip, float(corr)






def stable_hash_mod(s: str, mod: int = 1000) -> int:
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

    if np.std(ss) < 1e-8 or np.std(yy) < 1e-8:
        return 0.0

    corr, _ = pearsonr(ss, yy)
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


# =========================
# PATHS (FINAL DATASETS)
# =========================
# PLEASE ADJUST THE BASE PATH TO CORRECTLY IMPORT THE DATASETS

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
ROOT = Path(ARGS.data_root)

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
    }
}

# =========================
# CONFIG
# =========================
SEED = int(os.environ.get("SEED", 1001))   # training seed varies per run
EVAL_SEED = 12345                        # fixed for val/test subsampling + MI
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREFIX_LEN = 4
EMB_DIM = 128

BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 0.0

EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 10

MAX_CODES_PER_ADMISSION = 256
DX_CHUNKSIZE = 2_000_000

# NHSIC
NHSIC_SUBSAMPLE = 20000
NHSIC_FLOOR = 1e-4
EPS = 1e-8

# MI
MI_BASE_NEIGHBORS = 5


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def subsample_indices(valid_mask: np.ndarray, n: int, seed: int = EVAL_SEED) -> np.ndarray:
    idx = np.where(valid_mask.astype(bool))[0]
    if idx.size == 0:
        return idx
    if idx.size <= n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=n, replace=False)


def is_single_class_train_from_df(df_tr: pd.DataFrame, target: str) -> bool:
    """
    Skip if no valid rows or <2 unique labels among valid rows on TRAIN.
    long_stay validity uses long_stay_defined (NOT NaN).
    """
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
    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=DX_CHUNKSIZE):
        c = chunk["icd_code"].map(norm_icd).map(lambda s: icd_prefix(s, PREFIX_LEN))
        c = c[c != ""]
        vocab.update(c.unique().tolist())

    vocab = sorted(vocab)
    stoi = {"<PAD>": 0, "<UNK>": 1}
    for i, tok in enumerate(vocab, start=2):
        stoi[tok] = i
    return stoi


def build_hadm_to_tokens(dx_path: Path, stoi: Dict[str, int]) -> Dict[int, List[int]]:
    hadm2toks: Dict[int, List[int]] = {}
    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=DX_CHUNKSIZE):
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
            tid = stoi.get(cn, 1)  # UNK
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
    def __init__(self, admissions_df: pd.DataFrame, hadm2toks: Dict[int, List[int]], targets: List[str]):
        self.hadm_ids = admissions_df["hadm_id"].astype(int).to_numpy()
        self.targets = targets
        self.tokens = [hadm2toks.get(int(h), []) for h in self.hadm_ids]

        Y = admissions_df[targets].copy()
        for t in targets:
            Y[t] = pd.to_numeric(Y[t], errors="coerce")  # allow NaN

        mask = (~Y.isna()).to_numpy(dtype=np.bool_)

        # ---- SPECIAL CASE: long_stay validity comes from long_stay_defined ----
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
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)  # True = PAD

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
# DEEPSETS MODEL (ONE SCALAR)
# =========================
class DeepSetsSingleScalar(nn.Module):
    def __init__(self, vocab_size: int, dim: int = EMB_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)

        # pooling output is now 2*dim (mean + max)
        self.pool_ln = nn.LayerNorm(2 * dim)

        self.phi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # Rho now takes 2*dim
        self.rho = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x_tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        E = self.emb(x_tokens)          # (B, L, d)
        H = self.phi(E)                 # (B, L, d)

        # ---- mean pooling (masked) ----
        keep = (~pad_mask).float().unsqueeze(-1)   # (B, L, 1) 1=valid, 0=pad
        H_mean = H * keep
        valid_count = keep.sum(dim=1).clamp(min=1.0)      # (B, 1)
        pooled_mean = H_mean.sum(dim=1) / valid_count     # (B, d)

        # ---- max pooling (masked) ----
        # pad positions become very negative so they don't win max
        H_max_in = H.masked_fill(pad_mask.unsqueeze(-1), -1e9)  # (B, L, d)
        pooled_max = H_max_in.max(dim=1).values                 # (B, d)

        # ---- concat mean+max ----
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)   # (B, 2d)
        pooled = self.pool_ln(pooled)

        s = self.rho(pooled).squeeze(-1)  # (B,)
        return s



# =========================
# NHSIC
# =========================
def center_kernel(K: torch.Tensor) -> torch.Tensor:
    mean_row = K.mean(dim=1, keepdim=True)
    mean_col = K.mean(dim=0, keepdim=True)
    mean_all = K.mean()
    return K - mean_row - mean_col + mean_all

def hsic_normalized(
    K: torch.Tensor,
    L: torch.Tensor,
    eps_floor: float = NHSIC_FLOOR,  # <-- this is ε0 from the paper
    eps_num: float = EPS,            # <-- tiny numeric stabilizer
) -> torch.Tensor:
    Kc = center_kernel(K)
    Lc = center_kernel(L)

    num = (Kc * Lc).sum()

    k_norm = torch.sqrt((Kc * Kc).sum() + eps_num)  # ||Kc||_F
    l_norm = torch.sqrt((Lc * Lc).sum() + eps_num)  # ||Lc||_F

    # Floor ONLY the Kc norm (matches paper’s max(||Kc||, ε0) spirit)
    k_norm_f = torch.clamp(k_norm, min=eps_floor)

    den = k_norm_f * l_norm
    return num / den

def delta_kernel(y: torch.Tensor) -> torch.Tensor:
    return (y.view(-1, 1) == y.view(1, -1)).float()


def rbf_kernel_1d(s: torch.Tensor, sigma: float) -> torch.Tensor:
    s = s.view(-1, 1)
    d2 = (s - s.t()).pow(2)
    K = torch.exp(-d2 / (2.0 * (sigma ** 2)))
    return K


def nhsic_rbf(s: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    K = rbf_kernel_1d(s, sigma)
    L = delta_kernel(y)
    return hsic_normalized(K, L)



def nhsic_task_from_batch(s: torch.Tensor, y_task: torch.Tensor, m_all: torch.Tensor,
                         sigma: float, task_idx: int) -> torch.Tensor:
    # m_all: (B, T)
    if USE_INTERSECTION_VALIDITY:
        valid = intersection_mask_torch(m_all)
    else:
        valid = (m_all[:, task_idx] > 0.5)

    if valid.sum().item() < 3:
        return s.sum() * 0.0

    sv = s[valid]
    yv = y_task[valid].to(torch.int64)

    if torch.unique(yv).numel() < 2:
        return s.sum() * 0.0

    return nhsic_rbf(sv, yv, sigma)



@torch.no_grad()
def estimate_sigma_median(model: nn.Module, loader: DataLoader, n_samples: int = NHSIC_SUBSAMPLE) -> float:
    model.eval()
    scores = []
    for x, pad_mask, y, m in loader:
        x = x.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)
        s = model(x, pad_mask).detach().float().cpu().numpy()
        scores.append(s)
        if sum(len(a) for a in scores) >= n_samples:
            break

    s_all = np.concatenate(scores, axis=0).astype(np.float64)
    if len(s_all) < 10:
        return 1.0

    # --- NEW: standardize so sigma is on a stable scale ---
    s_all = (s_all - s_all.mean()) / (s_all.std() + 1e-12)

    # --- NEW: deterministic sampling ---
    k = min(len(s_all), n_samples)
    rng = np.random.default_rng(EVAL_SEED)
    idx = rng.choice(len(s_all), size=k, replace=False)

    s = s_all[idx]
    d = np.abs(s.reshape(-1, 1) - s.reshape(1, -1))
    med = np.median(d[d > 0])

    if not np.isfinite(med) or med <= 0:
        return 1.0

    # --- NEW: floor sigma to avoid micro-kernel collapse ---
    med = max(float(med), 0.05)   # try 0.02–0.1 if needed
    return float(med)


@torch.no_grad()
def nhsic_on_split(
    model: nn.Module,
    loader: DataLoader,
    targets: List[str],
    sigma: float,
    subsample: int = NHSIC_SUBSAMPLE,
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

    # ✅ ADD IT HERE (right after M exists)
    if USE_INTERSECTION_VALIDITY:
        valid_all = intersection_mask_np(M)   # shape (N,)
    else:
        valid_all = None

    out = {}
    if target_indices is None:
        target_indices = list(range(len(targets)))
    if len(target_indices) != len(targets):
        raise ValueError("target_indices must have the same length as targets")

    for local_j, (t, j) in enumerate(zip(targets, target_indices)):
        # ✅ then use valid_all instead of per-task masks when enabled
        valid = valid_all if USE_INTERSECTION_VALIDITY else (M[:, j] > 0.5)

        if valid.sum() < 3:
            out[t] = 0.0
            continue
        yv = Y[valid, j].astype(np.int64)
        if np.unique(yv).size < 2:
            out[t] = 0.0
            continue

        seed_j = seed + (stable_hash_mod(t, 1000) if use_target_hash else local_j)
        idx = subsample_indices(valid, subsample, seed=seed_j)

        sv = torch.tensor(S[idx], device=DEVICE, dtype=torch.float32)
        yv_t = torch.tensor(Y[idx, j].astype(np.int64), device=DEVICE, dtype=torch.int64)

        out[t] = float(nhsic_rbf(sv, yv_t, sigma).detach().cpu().item())

    return out



# =========================
# MI (masked, MI_nats)
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
    bad = 0

    for _ep in range(1, EPOCHS_STAGE1 + 1):
        model.train()
        for x, pad_mask, y, m in train_loader:
            x = x.to(DEVICE)
            pad_mask = pad_mask.to(DEVICE)
            y = y.to(DEVICE)
            m = m.to(DEVICE)

            s = model(x, pad_mask)
            obj = nhsic_task_from_batch(s, y[:, task_idx], m, sigma, task_idx)

            loss = -obj

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_map = nhsic_on_split(
            model,
            val_loader,
            [validation_anchor_name],
            sigma,
            subsample=NHSIC_SUBSAMPLE,
            seed=EVAL_SEED + 1000 + task_idx,
            use_target_hash=False,
            target_indices=[validation_anchor_idx],
        )
        val_score = val_map[validation_anchor_name]

        if val_score > best_val + 1e-12:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

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
    bad = 0

    w = torch.tensor(weights, device=DEVICE, dtype=torch.float32)

    for _ep in range(1, EPOCHS_STAGE2 + 1):
        model.train()
        for x, pad_mask, y, m in train_loader:
            x = x.to(DEVICE)
            pad_mask = pad_mask.to(DEVICE)
            y = y.to(DEVICE)
            m = m.to(DEVICE)

            s = model(x, pad_mask)

            obj = 0.0
            for j in range(len(train_targets)):
                if w[j].item() == 0.0:
                    continue
                obj = obj + w[j] * nhsic_task_from_batch(s, y[:, j], m, sigma, j)


            loss = -obj
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        val_map = nhsic_on_split(
            model,
            val_loader,
            train_targets,
            sigma,
            subsample=NHSIC_SUBSAMPLE,
            seed=EVAL_SEED + 9999,
            use_target_hash=False,
        )
        val_score = 0.0
        for j, t in enumerate(train_targets):
            val_score += float(weights[j]) * float(val_map[t])

        if val_score > best_val + 1e-12:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

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

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate_batch)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_batch)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_batch)

    # =========================
    # Stage 1: single-task NHSIC models
    # =========================
    validation_anchor_name = "mortality"
    mortality_idx = train_targets.index(validation_anchor_name)
    print(
        "\n=== Stage 1: Single-task NHSIC models "
        f"(checkpoint selection anchored to VAL {validation_anchor_name}) ==="
    )
    stage1_best = {}
    stage1_val = {}

    for j, t in enumerate(train_targets):
        if is_single_class_train_from_df(df_tr, t):
            print(f"\n  Skipping single-task model for {t}: single-class (or no valid labels) in TRAIN")
            stage1_best[t] = None
            stage1_val[t] = 0.0
            continue

        print(f"\n  Training single-task model for: {t}")
        model = DeepSetsSingleScalar(vocab_size=vocab_size).to(DEVICE)

        sigma = estimate_sigma_median(model, dl_tr, n_samples=NHSIC_SUBSAMPLE)
        best_state, best_val = train_single_task(
            model, dl_tr, dl_va, j, t,
            mortality_idx, validation_anchor_name, sigma,
        )

        stage1_best[t] = best_state
        stage1_val[t] = best_val
        print(f"    best VAL NHSIC vs {validation_anchor_name}: {best_val:.6f} (sigma={sigma:.6f})")

    vals = np.array([max(stage1_val[t], 0.0) for t in train_targets], dtype=np.float64)
    max_h = float(vals.max()) if vals.size > 0 else 0.0

    # --- safer weighting ---
    eps = 0.02          # floor in "nhsic units" (try 0.01–0.05)
    alpha = 0.25        # weaker than sqrt (sqrt=0.5)
    w_cap = 3.0         # prevent domination

    denom = np.maximum(vals, eps)
    weights = (np.maximum(max_h, eps) / denom) ** alpha
    weights = np.clip(weights, 1.0 / w_cap, w_cap).astype(np.float32)

    # zero-out skipped tasks
    for j, t in enumerate(train_targets):
        if stage1_best[t] is None:
            weights[j] = 0.0

    # normalize so average weight ~ 1 over active tasks
    active = weights > 0
    if active.any():
        weights[active] = weights[active] / weights[active].mean()


    for j, t in enumerate(train_targets):
        if stage1_best[t] is None:
            weights[j] = 0.0

    print("\nStage-1 VAL NHSIC (per train target):")
    for t in train_targets:
        print(f"  {t:16s}: {stage1_val[t]:.6f}")

    print("\nComputed stabilized inverse-strength weights:")
    for j, t in enumerate(train_targets):
        print(f"  {t:16s}: w={weights[j]:.6f}")

    # =========================
    # Stage 2: weighted multi-task NHSIC
    # =========================
    print("\n=== Stage 2: Weighted-sum NHSIC multi-task model ===")
    model = DeepSetsSingleScalar(vocab_size=vocab_size).to(DEVICE)
    sigma2 = estimate_sigma_median(model, dl_tr, n_samples=NHSIC_SUBSAMPLE)
    best_state2 = train_weighted_multitask(model, dl_tr, dl_va, train_targets, weights, sigma2)
    model.load_state_dict(best_state2)
    print(f"Stage-2 sigma={sigma2:.6f}")

    # =========================
    # ORIENT SCORE USING VAL ANCHOR (GLOBAL SIGN)
    # =========================
    score_va, Y_va, M_va = scores_and_labels(model, dl_va)
    flip, pearson_before = orient_score_by_anchor(
        score_va, Y_va, M_va, train_targets, anchor="mortality"
    )
    print(f"\nScore orientation (VAL anchor='mortality'): pearson_before={pearson_before:.6f} | flip={flip}")


    # =========================
    # TEST: NHSIC + MI_nats
    # =========================
    print("\n=== TEST EVALUATION (ONE score s(x)) ===")
    score_te, Y_te, M_te = scores_and_labels(model, dl_te)
    score_te = score_te * flip

    # =========================
    # SAVE CSV FOR DIAGNOSTIC SCRIPT
    # =========================

    out_dir = Path(os.environ.get("MLCI_OUTPUT_DIR", "DiagnosticCSV"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # columns to include (keep hadm_id for sanity/debug)
    cols = ["hadm_id"] + eval_targets
    if "long_stay" in eval_targets and "long_stay_defined" in df_te.columns:
        cols.append("long_stay_defined")

    # build output df (same row order as df_te / dl_te / score_te)
    assert len(score_te) == len(df_te), "score_te and df_te length mismatch!"
    out_df = df_te[cols].copy()
    out_df.insert(1, "learned_score", score_te.astype(np.float32))

    out_path = out_dir / f"{name}_seed{SEED}_test_scores_and_outcomes.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved diagnostic CSV: {out_path}")


    # ---- DIAGNOSTIC: score spread + ties (global, unmasked) ----
    u = np.unique(score_te).size / score_te.size
    print("TEST score unique fraction:", u, "| std:", float(score_te.std()))


    nh_te = nhsic_on_split(
        model,
        dl_te,
        eval_targets,
        sigma2,
        subsample=NHSIC_SUBSAMPLE,
        seed=EVAL_SEED + 777,
        use_target_hash=True,
    )
    print("\nTEST NHSIC (subsampled, masked) using Stage-2 score:")
    for t in eval_targets:
        print(f"  {t:16s} NHSIC={nh_te[t]:.6f}")

    mi_te = mi_nats_masked(score_te, Y_te, M_te, eval_targets, seed=EVAL_SEED)
    print("\nTEST Metrics (DistCorr, MI_nats) using Stage-2 score:")
    for j, t in enumerate(eval_targets):
        y_t = Y_te[:, j]
        m_t = M_te[:, j]
        pearson = pearson_from_score(score_te, y_t, m_t)
        dcorr = dcor_from_score(score_te, y_t, m_t)
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
