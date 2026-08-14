"""
FM-Based Multi-Task Clinical Outcome Predictor
===============================================

Overview
--------
This script trains a Factorization Machine (FM) model on structured clinical admission data
to predict multiple binary outcomes (mortality, 30-day mortality, long stay, ICU transfer).
It follows a two-stage training pipeline:

  Stage 1 — Per-task single-FM training to derive validation BCE scores, which are
             converted into per-task loss weights.
  Stage 2 — A single pooled FM trained across all tasks using the Stage-1-derived
             weights, selected by best multi-task validation BCE.

Final evaluation reports Distance Correlation, and Mutual Information
between the FM logit scores and each binary test label.

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.
2.  Load train/val/test admission CSVs and diagnosis ICD-9 CSVs (gzipped).
3.  Normalize ICD codes to 4-char uppercase prefixes; build a vocabulary from TRAIN only.
4.  Construct binary Bag-of-Words sparse matrices (CSR) per split, capped at 256
    codes per admission, then apply TF-IDF transformation (fit on TRAIN only).
5.  Build pooled multi-task training rows: for each kept target (non-single-class
    on TRAIN), collect valid admission indices, labels, and task IDs.
6.  Stage 1 — For each kept target, independently train a single-task FM with
    AdaGrad SGD (K=16 factors, 10 epochs, LR=0.05) and record best VAL BCE.
7.  Convert Stage-1 VAL BCEs to task weights:
        score_t  = max(log(2) − val_bce_t, 0)
        weight_t = clip((max_score / max(score_t, ε))^0.25, 1/3, 3)
    then normalize active weights to mean 1.
8.  Stage 2 — Train one pooled FM on all kept tasks simultaneously, using the
    computed weights in the gradient step, selecting the best epoch by weighted
    multi-task VAL BCE (mean-over-batches, DL-style).
9.  Score the TEST split with the best Stage-2 FM logits.
10. Report per-target evaluation metrics on TEST.

Key Hyperparameters (fixed)
---------------------------
  SEED=42, EVAL_SEED=12345
  K=16 (FM factors), EPOCHS=10, LR=0.05
  REG_W=1e-6, REG_V=1e-6, ADAGRAD_EPS=1e-8
  PREFIX_LEN=4, MAX_CODES_PER_ADMISSION=256
  VAL_BATCH_SIZE=256
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
    pip install numpy pandas scipy scikit-learn dcor

Running
-------
    python fm_mimiciii.py --data_root /path/to/final_datasets

Example:
    python fm_mimiciii.py --data_root ./data/final_datasets
"""

# ----------------
# IMPORTS
# ----------------
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor
import math
from sklearn.feature_extraction.text import TfidfTransformer


WEIGHT_EPS   = 0.02   # floor on score_t
WEIGHT_ALPHA = 0.25   # gentler than sqrt
WEIGHT_CAP   = 3.0    # clip weights

import zlib  # NEW

def stable_seed(base_seed: int, key: str) -> int:
    """
    Deterministic 32-bit seed from (base_seed, key), independent of Python's hash randomization.
    """
    return (base_seed ^ zlib.crc32(key.encode("utf-8"))) & 0xFFFFFFFF



def build_pooled_rows(
    df: pd.DataFrame,
    targets: list[str],
    kept_targets: list[str] | None = None,
):
    """
    Returns:
      rows_pool: np.int64 indices into X rows (admissions), repeated across tasks
      y_pool:    np.int32 labels aligned with rows_pool
      task_pool: np.int32 task index aligned with rows_pool (index into `targets`)
      kept:      targets kept (TRAIN-derived) if kept_targets is None, else []
    """
    rows_list, y_list, task_list = [], [], []
    kept = []

    # IMPORTANT: task IDs must be indices into `targets` (train_targets order)
    use_targets = kept_targets if kept_targets is not None else targets
    target_to_j = {t: j for j, t in enumerate(targets)}

    for t in use_targets:
        y, m = get_y_mask(df, t)
        idx = np.where(m)[0]
        if idx.size < 1:
            continue

        yy = y[idx].astype(int)

        if kept_targets is None:
            # TRAIN decides kept tasks: must be non-single-class on TRAIN valid rows
            if np.unique(yy).size < 2:
                continue
            kept.append(t)

        j = target_to_j[t]
        rows_list.append(idx.astype(np.int64))
        y_list.append(yy.astype(np.int32))
        task_list.append(np.full(idx.size, j, dtype=np.int32))  # <-- NEW

    if not rows_list:
        return None, None, None, kept

    return (
        np.concatenate(rows_list).astype(np.int64),
        np.concatenate(y_list).astype(np.int32),
        np.concatenate(task_list).astype(np.int32),
        kept,
    )


# -------------------------
# CONFIG
# -------------------------
SEED = 42
EVAL_SEED = 12345

PREFIX_LEN = 4
MAX_CODES_PER_ADMISSION = 256

# FM hyperparams (fixed; best epoch selected by VAL)
K = 16
EPOCHS = 10
LR = 0.05
REG_W = 1e-6
REG_V = 1e-6
ADAGRAD_EPS = 1e-8

# DL-style VAL loss definition uses mean-over-batches
VAL_BATCH_SIZE = 256

# MI
MI_BASE_NEIGHBORS = 5


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
BASE = Path(ARGS.data_root)


CORE_TARGETS = ["mortality", "mortality_30d", "long_stay", "icu_transfer"]

def single_task_bce_mean_over_batches_from_logits(
    logits_all: np.ndarray,
    y_all: np.ndarray,
    m_all: np.ndarray,
    batch_size: int = VAL_BATCH_SIZE,
) -> float:
    n = int(len(y_all))
    if n == 0:
        return float("inf")

    losses = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        mb = m_all[start:end].astype(bool)
        if mb.sum() == 0:
            losses.append(0.0)  # DL-parity
            continue

        x = logits_all[start:end][mb].astype(np.float64)
        y = y_all[start:end][mb].astype(np.float64)
        bce = (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) - y * x).mean()
        losses.append(float(bce))

    return float(np.mean(losses)) if losses else float("inf")

def train_fm_single_task_with_val(
    Xtr: sparse.csr_matrix,
    ytr: np.ndarray,
    mtr: np.ndarray,
    Xva: sparse.csr_matrix,
    yva: np.ndarray,
    mva: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """
    Train a single-task FM on TRAIN valid rows for one target.
    Select best epoch by lowest VAL BCE (DL-style mean-over-batches).
    Returns: best_val_bce
    """
    tr_idx = np.where(mtr.astype(bool))[0]
    if tr_idx.size < 10:
        return float("inf")
    yy = ytr[tr_idx].astype(int)
    if np.unique(yy).size < 2:
        return float("inf")

    n, d = Xtr.shape
    w0 = np.float32(0.0)
    w = np.zeros(d, dtype=np.float32)
    V = (0.01 * rng.standard_normal((d, K))).astype(np.float32)

    gw0 = np.float32(0.0)
    gw = np.zeros(d, dtype=np.float32)
    gV = np.zeros((d, K), dtype=np.float32)

    best_val = float("inf")

    order = tr_idx.astype(np.int64).copy()

    for ep in range(1, EPOCHS + 1):
        rng.shuffle(order)

        for r in order:
            y = np.float32(ytr[r])

            start, end = Xtr.indptr[r], Xtr.indptr[r + 1]
            idx = Xtr.indices[start:end].astype(np.int64)

            y_hat = fm_predict_row(float(w0), w, V, idx)
            p = sigmoid(float(y_hat))
            grad = np.float32(p - y)

            gw0 += grad * grad
            w0 -= (LR / np.sqrt(float(gw0) + ADAGRAD_EPS)) * grad

            if idx.size == 0:
                continue

            V_i = V[idx, :]
            sum_v = V_i.sum(axis=0)

            g_w = grad + REG_W * w[idx]
            gw[idx] += g_w * g_w
            w[idx] -= (LR * g_w) / (np.sqrt(gw[idx]) + ADAGRAD_EPS)

            g_V = grad * (sum_v[None, :] - V_i) + REG_V * V_i
            gV[idx, :] += g_V * g_V
            V[idx, :] = V_i - (LR * g_V) / (np.sqrt(gV[idx, :]) + ADAGRAD_EPS)

        # compute val BCE for this task
        va_logits = fm_score_all(Xva, float(w0), w, V)
        val_bce = single_task_bce_mean_over_batches_from_logits(va_logits, yva, mva, batch_size=VAL_BATCH_SIZE)

        if val_bce < best_val:
            best_val = val_bce

    return float(best_val)


# -------------------------
# UTILITY HELPERS
# -------------------------
def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

def norm_icd(code) -> str:
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()

# -------------------------
# DATA LOADING
# -------------------------

def load_admissions(ds_dir: Path, stem: str, split: str) -> pd.DataFrame:
    p = ds_dir / f"{stem}_{split}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def load_diagnoses(ds_dir: Path, diag_stem: str, split: str) -> pd.DataFrame:
    p = ds_dir / f"{diag_stem}_{split}.csv.gz"
    if not p.exists():
        raise FileNotFoundError(p)

    dx = pd.read_csv(p, usecols=["hadm_id", "icd_code"], dtype={"icd_code": str})

    # ICD-9 prefix tokenization (match DL scripts)
    dx["code"] = dx["icd_code"].map(norm_icd).str[:PREFIX_LEN]
    dx = dx.loc[dx["code"] != "", ["hadm_id", "code"]]

    return dx.reset_index(drop=True)

def build_vocab(dx_train):
    codes = dx_train["code"].unique()
    codes.sort()
    vocab = {c: i for i, c in enumerate(codes)}
    vocab["<UNK>"] = len(vocab)
    return vocab

def build_bow_matrix_binary(adm: pd.DataFrame, dx: pd.DataFrame, vocab: dict) -> sparse.csr_matrix:
    hadm_ids = adm["hadm_id"].astype(int).to_numpy()
    hadm_to_row = {hid: i for i, hid in enumerate(hadm_ids)}
    unk_col = vocab.get("<UNK>", None)

    # Track how many codes we've kept per admission row (truncate at 256)
    kept_counts = np.zeros(len(adm), dtype=np.int32)

    # Track which columns already set for each row (binary presence)
    # Use Python sets per row to avoid duplicate entries
    seen_cols = [set() for _ in range(len(adm))]
    unk_hit = np.zeros(len(adm), dtype=bool)

    hadm_arr = dx["hadm_id"].to_numpy()
    code_arr = dx["code"].to_numpy()

    for hid, code in zip(hadm_arr, code_arr):
        try:
            hid = int(hid)
        except Exception:
            continue

        r = hadm_to_row.get(hid)
        if r is None:
            continue

        if kept_counts[r] >= MAX_CODES_PER_ADMISSION:
            continue  # exact DeepSets-style truncation: keep first 256 encountered

        kept_counts[r] += 1

        j = vocab.get(code, None)
        if j is None:
            unk_hit[r] = True
        else:
            seen_cols[r].add(int(j))

    rows, cols, data = [], [], []
    for r, colset in enumerate(seen_cols):
        for j in colset:
            rows.append(r); cols.append(j); data.append(1.0)
        if unk_hit[r] and (unk_col is not None):
            rows.append(r); cols.append(int(unk_col)); data.append(1.0)

    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(len(adm), len(vocab)),
        dtype=np.float32,
    )
    X.sum_duplicates()
    return X


# -------------------------
# LABEL MASKING AND POOLING
# -------------------------

def get_y_mask(df: pd.DataFrame, t: str) -> tuple[np.ndarray, np.ndarray]:
    if t == "long_stay":
        require_cols(df, ["long_stay", "long_stay_defined"], "admissions")
        m = df["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
        y = pd.to_numeric(df["long_stay"], errors="coerce").to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0).astype(int)
        return y, m
    else:
        require_cols(df, [t], "admissions")
        y = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(y)
        y = np.nan_to_num(y, nan=0.0).astype(int)
        return y, m



def build_YM_for_targets(df: pd.DataFrame, targets: list[str], kept: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    DL-style: build Y (N,T) and M (N,T) over admissions rows.
    Then zero-out masks for tasks not kept (TRAIN decided).
    """
    N, T = len(df), len(targets)
    Y = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)

    kept_set = set(kept)
    for j, t in enumerate(targets):
        y, m = get_y_mask(df, t)          # y int, m bool over admissions rows
        Y[:, j] = y.astype(np.float32)
        M[:, j] = m.astype(np.float32)

        if t not in kept_set:
            M[:, j] = 0.0                # TRAIN decided task doesn’t exist for the objective

    return Y, M

# -------------------------
# LOSS FUNCTIONS
# -------------------------

def bce_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Mean BCEWithLogits (stable):
      softplus(x) - y*x, averaged
    """
    x = logits.astype(np.float64)
    y = y.astype(np.float64)
    return float((np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) - y * x).mean())

def masked_multitask_bce_mean_over_batches_from_logits(
    logits_all: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    weights_vec: np.ndarray | None = None,
    batch_size: int = VAL_BATCH_SIZE,
) -> float:
    """
    Weighted masked multitask BCE over admissions batches:
      num = sum_t w_t * sum_{i in batch} m_{i,t} * BCE(logit_i, y_{i,t})
      den = sum_t w_t * sum_{i in batch} m_{i,t}
      loss_batch = num/den, with DL-parity: if den==0 -> 0.0
    then mean over batches.
    """
    n = int(Y.shape[0])
    T = int(Y.shape[1])

    if weights_vec is None:
        w = np.ones(T, dtype=np.float64)
    else:
        w = weights_vec.astype(np.float64)

    losses = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        s = logits_all[start:end].astype(np.float64)  # (B,)
        yb = Y[start:end].astype(np.float64)          # (B,T)
        mb = M[start:end].astype(np.float64)          # (B,T)

        x = s[:, None]  # (B,1) -> broadcast to (B,T)

        per_entry = (np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) - yb * x)  # (B,T)
        per_entry = per_entry * mb

        num = (per_entry * w[None, :]).sum()
        den = (mb * w[None, :]).sum()

        losses.append(0.0 if den <= 0 else float(num / den))

    return float(np.mean(losses)) if losses else float("inf")


# -------------------------
# EVALUATION METRICS
# -------------------------



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

def mi_masked_fixed_rng(
    score: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    random_state: int = EVAL_SEED,
    base_neighbors: int = MI_BASE_NEIGHBORS,
) -> float:
    valid = valid_mask.astype(bool)
    if valid.sum() < 3:
        return 0.0
    yy = y[valid].astype(int)
    if np.unique(yy).size < 2:
        return 0.0
    nn = int(min(base_neighbors, valid.sum() - 1))
    if nn < 1:
        return 0.0
    s_valid = zscore_1d(score[valid]).reshape(-1, 1)
    return float(
        mutual_info_classif(
            s_valid,
            yy,
            random_state=random_state,
            n_neighbors=nn,
        )[0]
    )

# -------------------------
# FM MODEL IMPLEMENATION
# -------------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)

def fm_predict_row(w0: float, w: np.ndarray, V: np.ndarray, idx: np.ndarray) -> float:
    # binary x_i=1
    lin = w0 + float(w[idx].sum()) if idx.size else float(w0)
    if idx.size == 0:
        return lin

    V_i = V[idx, :]                  # (m, k)
    sum_v = V_i.sum(axis=0)          # (k,)
    sum_v2 = (V_i * V_i).sum(axis=0) # (k,)
    inter = 0.5 * float((sum_v * sum_v - sum_v2).sum())
    return lin + inter

def fm_score_all(X: sparse.csr_matrix, w0: float, w: np.ndarray, V: np.ndarray) -> np.ndarray:
    scores = np.zeros(X.shape[0], dtype=np.float32)
    for r in range(X.shape[0]):
        start, end = X.indptr[r], X.indptr[r + 1]
        idx = X.indices[start:end].astype(np.int64)
        scores[r] = np.float32(fm_predict_row(w0, w, V, idx))
    return scores

def train_fm_logistic_with_val(
    Xtr: sparse.csr_matrix,
    tr_rows: np.ndarray,
    y_tr: np.ndarray,
    tr_task_ids: np.ndarray,
    Xva: sparse.csr_matrix | None,
    Yva: np.ndarray | None,
    Mva: np.ndarray | None,
    rng: np.random.Generator,
    weights_vec: np.ndarray | None = None,
):


    """
    Train FM on pooled multi-task examples (NO composite label).
    Select best checkpoint by pooled VAL BCE (mean-over-batches).
    """
    n, d = Xtr.shape

    w0 = np.float32(0.0)
    w = np.zeros(d, dtype=np.float32)
    V = (0.01 * rng.standard_normal((d, K))).astype(np.float32)

    gw0 = np.float32(0.0)
    gw = np.zeros(d, dtype=np.float32)
    gV = np.zeros((d, K), dtype=np.float32)

    if tr_rows is None or tr_rows.size < 10 or np.unique(y_tr).size < 2:
        return w0, w, V

    use_val = True
    if (Xva is None) or (Yva is None) or (Mva is None) or (Yva.shape[0] == 0):
        raise RuntimeError("[Stage2 FM] VAL is required but missing/empty.")




    best_val = float("inf")
    best_w0, best_w, best_V = w0, w.copy(), V.copy()

    order = np.arange(tr_rows.size, dtype=np.int64)

    for ep in range(1, EPOCHS + 1):
        rng.shuffle(order)

        for k in order:
            r = int(tr_rows[k])
            y = np.float32(y_tr[k])

            start, end = Xtr.indptr[r], Xtr.indptr[r + 1]
            idx = Xtr.indices[start:end].astype(np.int64)

            y_hat = fm_predict_row(float(w0), w, V, idx)
            p = sigmoid(float(y_hat))
            t = int(tr_task_ids[k])
            wt = 1.0 if (weights_vec is None) else float(weights_vec[t])
            if wt <= 0.0:
                continue

            grad = np.float32((p - y) * wt)


            gw0 += grad * grad
            w0 -= (LR / np.sqrt(float(gw0) + ADAGRAD_EPS)) * grad

            if idx.size == 0:
                continue

            V_i = V[idx, :]
            sum_v = V_i.sum(axis=0)

            g_w = grad + REG_W * w[idx]
            gw[idx] += g_w * g_w
            w[idx] -= (LR * g_w) / (np.sqrt(gw[idx]) + ADAGRAD_EPS)

            g_V = grad * (sum_v[None, :] - V_i) + REG_V * V_i
            gV[idx, :] += g_V * g_V
            V[idx, :] = V_i - (LR * g_V) / (np.sqrt(gV[idx, :]) + ADAGRAD_EPS)

        if use_val:
            va_logits_all = fm_score_all(Xva, float(w0), w, V)  # logits per admission row
            val_bce = masked_multitask_bce_mean_over_batches_from_logits(
                va_logits_all,
                Yva,
                Mva,
                weights_vec=weights_vec,
                batch_size=VAL_BATCH_SIZE,
            )

            print(f"  Epoch {ep:02d}/{EPOCHS} | pooled_VAL_BCE(mean-over-batches)={val_bce:.6f}")

            if val_bce < best_val:
                best_val = val_bce
                best_w0 = np.float32(w0)
                best_w = w.copy()
                best_V = V.copy()

    return (best_w0, best_w, best_V) if use_val else (w0, w, V)


# -------------------------
# RUN ONE DATASET
# -------------------------
def run_dataset(
    name: str,
    ds_dir: Path,
    adm_stem: str,
    diag_stem: str,
    train_targets: list[str],
    eval_targets: list[str],
):
    print("\n" + "=" * 100)
    print(f"DATASET: {name}")
    print("=" * 100)

    # load TRAIN / VAL / TEST
    adm_tr = load_admissions(ds_dir, adm_stem, "train")
    adm_va = load_admissions(ds_dir, adm_stem, "val")
    adm_te = load_admissions(ds_dir, adm_stem, "test")

    # require long_stay_defined if long_stay involved (DL-parity)
    if "long_stay" in set(train_targets) | set(eval_targets):
        require_cols(adm_tr, ["long_stay", "long_stay_defined"], f"{name}/TRAIN")
        require_cols(adm_va, ["long_stay", "long_stay_defined"], f"{name}/VAL")
        require_cols(adm_te, ["long_stay", "long_stay_defined"], f"{name}/TEST")

    dx_tr = load_diagnoses(ds_dir, diag_stem, "train")
    dx_va = load_diagnoses(ds_dir, diag_stem, "val")
    dx_te = load_diagnoses(ds_dir, diag_stem, "test")

    # vocab from TRAIN only
    vocab = build_vocab(dx_tr)

    # binary BoW matrices (FM parity)
    Xtr = build_bow_matrix_binary(adm_tr, dx_tr, vocab)
    Xva = build_bow_matrix_binary(adm_va, dx_va, vocab)
    Xte = build_bow_matrix_binary(adm_te, dx_te, vocab)

    # Xtr, Xva, Xte are CSR count matrices already
    tfidf = TfidfTransformer(
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,  # set True for log(1+tf)
        norm="l2",           # common default; try None for raw scaling
    )

    Xtr = tfidf.fit_transform(Xtr)   # FIT ONLY on train
    Xva = tfidf.transform(Xva)
    Xte = tfidf.transform(Xte)


    # ---- pooled multi-task TRAIN examples (NO composite label) ----
    tr_rows, y_tr_pool, tr_task_ids, kept = build_pooled_rows(adm_tr, train_targets)
    print("Kept targets (TRAIN-derived):", kept)

    # ---- Stage 1: train FM per task, get VAL BCE per task (for weights) ----
    # no shared rng here
    val_bce_by_task = {t: float("inf") for t in train_targets}

    print("\n=== Stage 1 (FM): single-task models -> VAL_BCE per task (for weights) ===")
    for t in train_targets:
        if t not in kept:
            print(f"  [Stage1] {t:16s}: not kept (single-class on TRAIN) -> val_bce=inf")
            continue

        ytr, mtr = get_y_mask(adm_tr, t)
        yva, mva = get_y_mask(adm_va, t)

        rng_t = np.random.default_rng(stable_seed(SEED, f"{name}|stage1|{t}"))  # NEW

        vb = train_fm_single_task_with_val(
            Xtr=Xtr, ytr=ytr, mtr=mtr,
            Xva=Xva, yva=yva, mva=mva,
            rng=rng_t,  # NEW
        )

        val_bce_by_task[t] = float(vb)
        vb_str = f"{vb:.6f}" if np.isfinite(vb) else "inf"
        print(f"  [Stage1] {t:16s}: best_val_bce={vb_str}")

    # ---- compute weights from Stage-1 VAL BCE (same rule as weighted BCE baseline) ----
    LOG2 = float(math.log(2.0))
    scores = np.zeros(len(train_targets), dtype=np.float64)

    for j, t in enumerate(train_targets):
        vb = val_bce_by_task[t]
        if (t not in kept) or (not np.isfinite(vb)):
            scores[j] = 0.0
        else:
            scores[j] = max(LOG2 - float(vb), 0.0)

    max_s = float(scores.max()) if scores.size > 0 else 0.0
    denom = np.maximum(scores, WEIGHT_EPS)
    w = (np.maximum(max_s, WEIGHT_EPS) / denom) ** WEIGHT_ALPHA
    w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP).astype(np.float32)

    # zero weights for non-kept tasks
    for j, t in enumerate(train_targets):
        if t not in kept:
            w[j] = 0.0

    # normalize active weights to mean 1
    active = w > 0
    if active.any():
        w[active] = w[active] / w[active].mean()

    print("\nStage-1 VAL_BCE -> score=(log2-BCE)+ -> weights:")
    for j, t in enumerate(train_targets):
        vb = val_bce_by_task[t]
        vb_str = f"{vb:.6f}" if np.isfinite(vb) else "inf"
        print(f"  {t:16s}: val_bce={vb_str:>10s} | score={scores[j]:.6f} | w={float(w[j]):.6f}")

    weights_vec = w  # aligned to train_targets indexing

    # ---- build VAL objective matrices (masks zeroed for non-kept tasks) ----
    Yva, Mva = build_YM_for_targets(adm_va, train_targets, kept)

    if Mva.sum() <= 0:
        raise RuntimeError("[Stage2 FM] VAL has zero valid labels after masking (kept tasks).")



    # ---- Stage 2: pooled weighted FM ----
    if tr_rows is None or tr_rows.size < 10 or np.unique(y_tr_pool).size < 2:
        print("TRAIN pooled labels unusable (too small or single-class). Using zero scores.")
        score_te = np.zeros(len(adm_te), dtype=np.float32)
    else:
        rng_stage2 = np.random.default_rng(stable_seed(SEED, f"{name}|stage2"))  # NEW
        w0, w_lin, V = train_fm_logistic_with_val(
            Xtr=Xtr,
            tr_rows=tr_rows,
            y_tr=y_tr_pool,
            tr_task_ids=tr_task_ids,
            Xva=Xva,
            Yva=Yva,
            Mva=Mva,
            rng=rng_stage2,        # NEW
            weights_vec=weights_vec,
        )
        score_te = fm_score_all(Xte, float(w0), w_lin, V)  # <-- RESTORE THIS




    print("\nTEST Metrics (DistCorr, MI_nats) using FM logit score:")
    for t in eval_targets:
        if t == "long_stay":
            valid_mask = adm_te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            yte_raw = pd.to_numeric(adm_te["long_stay"], errors="coerce").to_numpy(dtype=float)
        else:
            if t not in adm_te.columns:
                raise ValueError(f"[{name}/TEST] Missing eval target column: {t}")
            yte_raw = pd.to_numeric(adm_te[t], errors="coerce").to_numpy(dtype=float)
            valid_mask = ~np.isnan(yte_raw)

        yte = np.nan_to_num(yte_raw, nan=0.0).astype(int)

        pearson = pearson_from_score(score_te, yte, valid_mask)
        dcorr = dcor_from_score(score_te, yte, valid_mask)
        mi = mi_masked_fixed_rng(
            score_te,
            yte,
            valid_mask,
            random_state=EVAL_SEED,
            base_neighbors=MI_BASE_NEIGHBORS,
        )


        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )
    print("=" * 100)


def main():
    run_dataset(
        name="CORE",
        ds_dir=BASE / "core",
        adm_stem="admissions_core",
        diag_stem="diagnoses_icd9_core",
        train_targets=CORE_TARGETS,
        eval_targets=CORE_TARGETS,
    )


if __name__ == "__main__":
    main()