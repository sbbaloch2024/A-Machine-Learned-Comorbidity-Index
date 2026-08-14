
"""
GBT-Based Multi-Task Clinical Outcome Predictor
================================================

Overview
--------
This script trains a Gradient Boosted Tree (GBT) model — defaulting to XGBoost —
on structured clinical admission data to predict multiple binary outcomes
(mortality, 30-day mortality, long stay, ICU transfer). It mirrors the two-stage
weighted multi-task training strategy of the FM-based counterpart, but replaces
the FM with a GBT trained on TF-IDF-weighted ICD-9 code bag-of-words features.

The backend can be switched via the GBT_BACKEND environment variable (default: "xgb").

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.
2.  Load train/val/test admission CSVs and diagnosis ICD-9 CSVs (gzipped).
3.  Normalize ICD codes to 4-char uppercase prefixes; build a vocabulary (+ UNK
    token) from TRAIN diagnoses only.
4.  Construct count-based Bag-of-Words sparse matrices (CSR), capped at 256 codes
    per admission in file-encounter order, then apply TF-IDF transformation
    (fit on TRAIN only; transform val/test).
5.  Determine kept targets: tasks with at least 2 classes and ≥1 valid label on TRAIN.
6.  Build pooled multi-task feature matrices and label arrays for TRAIN and VAL,
    each row tagged with its task ID.
7.  Stage 1 — For each kept target independently:
        - Train a GBT on valid TRAIN rows only (2000 rounds, depth 4, LR 0.05).
        - Score all VAL admissions to compute a masked mean-over-batches VAL BCE.
        - Store best VAL BCE per task for weight derivation.
8.  Derive per-task loss weights from Stage-1 VAL BCEs:
        score_t  = max(log(2) − val_bce_t, 0)
        weight_t = clip((max_score / max(score_t, ε))^0.25, 1/3, 3)
    then normalize active weights to mean 1.
9.  Stage 2 — Train one pooled GBT on all kept tasks simultaneously, using
    per-sample weights derived from Stage-1 task weights. Best iteration is
    selected by minimum pooled VAL logloss tracked internally by XGBoost
    (no early stopping; full 2000 rounds trained, best checkpoint applied).
10. Score the TEST split with Stage-2 logits.
11. Report Distance Correlation, and Mutual Information between logit
    scores and each binary test label.

Key Configuration
-----------------
  SEED=42, EVAL_SEED=12345
  N_ROUNDS_STAGE1=2000, N_ROUNDS_STAGE2=2000
  max_depth=4, learning_rate=0.05
  PREFIX_LEN=4, MAX_CODES_PER_ADMISSION=256
  VAL_BATCH_SIZE=256
  WEIGHT_EPS=0.02, WEIGHT_ALPHA=0.25, WEIGHT_CAP=3.0
  GBT_BACKEND: controlled via environment variable (default "xgb")

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
    pip install numpy pandas scipy scikit-learn dcor xgboost

Running
-------
    python gbt_mimiciii.py --data_root /path/to/final_datasets

Switching backend (XGBoost is the only fully implemented option):
    GBT_BACKEND=xgb python gbt_mimiciii.py --data_root /path/to/final_datasets

Example:
    python gbt_mimiciii.py --data_root ./data/final_datasets
"""

# ----------------
# IMPORTS
# ----------------
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor
import math
import zlib
from sklearn.feature_extraction.text import TfidfTransformer


WEIGHT_EPS   = 0.02
WEIGHT_ALPHA = 0.25
WEIGHT_CAP   = 3.0
# add near top / config for:
N_ROUNDS_STAGE1 = 2000
N_ROUNDS_STAGE2 = 2000

def stable_seed(base_seed: int, key: str) -> int:
    return (base_seed ^ zlib.crc32(key.encode("utf-8"))) & 0xFFFFFFFF



# ----------------
# CONFIG
# ----------------
MAX_CODES_PER_ADMISSION = 256

VAL_BATCH_SIZE = 256
EPS = 1e-7
GBT_BACKEND = os.environ.get("GBT_BACKEND", "xgb").lower()  # fixed default for apples-to-apples
SEED = 42
EVAL_SEED = 12345
PREFIX_LEN = 4


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


# ----------------
# UTILITY HELPERS
# ----------------
def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

def norm_icd(code) -> str:
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()

# ----------------
# DATA LOADING
# ----------------

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
    return dx

def build_vocab(dx_train: pd.DataFrame) -> tuple[dict, int]:
    codes = dx_train["code"].unique()
    codes.sort()
    vocab = {c: i for i, c in enumerate(codes)}
    unk_idx = len(vocab)
    return vocab, unk_idx


def build_bow_matrix(adm: pd.DataFrame, dx: pd.DataFrame, vocab: dict, unk_idx: int) -> sparse.csr_matrix:
    hadm_ids = adm["hadm_id"].astype(int).to_numpy()
    hadm_to_row = {hid: i for i, hid in enumerate(hadm_ids)}

    # keep first 256 token-ids per admission in FILE encounter order
    kept_cols_by_row = [[] for _ in range(len(adm))]
    kept_count_by_row = np.zeros(len(adm), dtype=np.int32)

    hadm_arr = dx["hadm_id"].to_numpy()
    code_arr = dx["code"].to_numpy()

    for hid, code in zip(hadm_arr, code_arr):
        try:
            hid = int(hid)
        except Exception:
            continue

        r = hadm_to_row.get(hid, None)
        if r is None:
            continue

        if kept_count_by_row[r] >= MAX_CODES_PER_ADMISSION:
            continue  # truncate exactly like DeepSets: keep first 256 encountered

        c = vocab.get(code, unk_idx)
        kept_cols_by_row[r].append(c)
        kept_count_by_row[r] += 1

    # convert kept token-ids into CSR count matrix
    rows, cols, data = [], [], []
    for r, col_list in enumerate(kept_cols_by_row):
        if not col_list:
            continue
        u, cnt = np.unique(np.asarray(col_list, dtype=np.int32), return_counts=True)
        rows.extend([r] * len(u))
        cols.extend(u.tolist())
        data.extend(cnt.astype(np.float32).tolist())

    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(len(adm), len(vocab) + 1),  # +1 for UNK
        dtype=np.float32,
    )
    X.sum_duplicates()
    return X



# ----------------
# LABEL MASKING AND POOLING
# ----------------
def get_y_mask(df: pd.DataFrame, t: str) -> tuple[np.ndarray, np.ndarray]:
    if t == "long_stay":
        if "long_stay_defined" not in df.columns:
            raise ValueError("Missing long_stay_defined for long_stay")
        m = df["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
        y = pd.to_numeric(df["long_stay"], errors="coerce").to_numpy(dtype=float)
        y = np.nan_to_num(y, nan=0.0).astype(int)
        return y, m
    else:
        if t not in df.columns:
            raise ValueError(f"Missing target column: {t}")
        y_raw = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(y_raw)
        y = np.nan_to_num(y_raw, nan=0.0).astype(int)
        return y, m

def kept_targets_from_train(adm_tr: pd.DataFrame, targets: list[str], min_valid: int = 1) -> list[str]:
    kept = []
    for t in targets:
        y, m = get_y_mask(adm_tr, t)
        idx = np.where(m)[0]
        if idx.size < min_valid:
            continue
        yy = y[idx].astype(int)
        if np.unique(yy).size < 2:
            continue
        kept.append(t)
    return kept

def build_pooled_xy_keep_with_task_ids(X, adm_df: pd.DataFrame, kept: list[str], all_targets: list[str]):
    idx_list, y_list, task_list = [], [], []
    target_to_j = {t: j for j, t in enumerate(all_targets)}  # stable indexing

    for t in kept:
        y, m = get_y_mask(adm_df, t)
        idx = np.where(m)[0]
        if idx.size < 1:
            continue

        idx_list.append(idx.astype(np.int64))
        y_list.append(y[idx].astype(np.int32))
        task_list.append(np.full(idx.size, target_to_j[t], dtype=np.int32))

    if not idx_list:
        return None, None, None

    pool_idx = np.concatenate(idx_list, axis=0)
    X_pool = X[pool_idx]
    y_pool = np.concatenate(y_list, axis=0).astype(np.int32)
    task_ids = np.concatenate(task_list, axis=0).astype(np.int32)
    return X_pool, y_pool, task_ids


# ----------------
# LOSS FUNCTIONS
# ----------------
def bce_from_proba(p1: np.ndarray, y: np.ndarray) -> float:
    p1 = np.clip(p1.astype(np.float64), EPS, 1.0 - EPS)
    y = y.astype(np.float64)
    return float(-(y * np.log(p1) + (1.0 - y) * np.log(1.0 - p1)).mean())

def bce_mean_over_batches_masked(p1_all: np.ndarray, y_all: np.ndarray, m_all: np.ndarray,
                                 batch_size: int = VAL_BATCH_SIZE) -> float:
    n = int(len(y_all))
    if n == 0:
        return float("inf")

    losses = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        mb = m_all[start:end].astype(bool)
        if mb.sum() == 0:
            losses.append(0.0)  # DL-parity: masked batch contributes 0
            continue
        losses.append(bce_from_proba(p1_all[start:end][mb], y_all[start:end][mb]))

    return float(np.mean(losses)) if losses else float("inf")


# ----------------
# EVALUATION METRICS
# ----------------
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

def mi_masked_fixed_rng(
    score: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    random_state: int = EVAL_SEED,   # default matches rule
    base_neighbors: int = 5,
) -> float:
    """
    MI with fixed RNG; z-score score[valid] like DL scripts.
    y is binary (0/1). valid_mask selects valid rows.
    """
    valid = valid_mask.astype(bool)
    if valid.sum() < 3:
        return 0.0

    yy = y[valid].astype(int)
    if np.unique(yy).size < 2:
        return 0.0

    nn = int(min(base_neighbors, valid.sum() - 1))
    if nn < 1:
        return 0.0

    # DL-parity: z-score ONLY within valid rows
    s_valid = zscore_1d(score[valid]).reshape(-1, 1)

    return float(
        mutual_info_classif(
            s_valid,
            yy,
            random_state=random_state,
            n_neighbors=nn,
        )[0]
    )


# ----------------
# GBT MODEL & TRAINING
# ----------------

def train_and_score_logit_backend(
    Xtr, ytr, Xva, yva, Xte,
    backend: str, params: dict,
    early_stop_rounds: int | None = None,   # kept for API symmetry, NOT used
    seed: int = SEED,
    wtr: np.ndarray | None = None,
    wva: np.ndarray | None = None,
):
    # returns (val_logits, test_logits)  [logits = raw margin]
    if Xtr is None or Xtr.shape[1] == 0:
        s_va = np.zeros(Xva.shape[0], dtype=np.float32) if Xva is not None else np.zeros(0, dtype=np.float32)
        s_te = np.zeros(Xte.shape[0], dtype=np.float32)
        return s_va, s_te

    has_val = (Xva is not None) and (yva is not None) and (len(yva) >= 10) and (np.unique(yva).size >= 2)

    if backend == "xgb":
        import xgboost as xgb

        ytr_ = (np.asarray(ytr).astype(int) > 0).astype(int)
        dtr = xgb.DMatrix(Xtr, label=ytr_, weight=wtr)

        evals = []
        evals_result = {}

        if has_val:
            yva_ = (np.asarray(yva).astype(int) > 0).astype(int)
            dva = xgb.DMatrix(Xva, label=yva_, weight=wva)
            evals = [(dva, "val")]
        else:
            dva = None

        dte = xgb.DMatrix(Xte)

        eta = float(params.get("learning_rate", params.get("eta", 0.05)))
        max_depth = int(params.get("max_depth", 4))
        num_round = int(params.get("n_estimators", 2000))

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": eta,
            "max_depth": max_depth,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "lambda": 1.0,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "seed": int(seed),
            "tree_method": "hist",
            "nthread": 1,
        }

        # NO early stopping; we still log val metric each round
        bst = xgb.train(
            params=xgb_params,
            dtrain=dtr,
            num_boost_round=num_round,
            evals=evals,
            evals_result=evals_result if has_val else None,
            early_stopping_rounds=None,
            verbose_eval=False,
        )

        # Pick best iteration by min val logloss (but still trained full num_round)
        best_round = num_round
        if has_val:
            best_round = int(np.argmin(evals_result["val"]["logloss"])) + 1  # 1-based

        def _predict_margin(bst_, dmat_, best_round_):
            # use best checkpoint without early stopping
            try:
                return bst_.predict(dmat_, output_margin=True, iteration_range=(0, best_round_))
            except TypeError:
                return bst_.predict(dmat_, output_margin=True, ntree_limit=best_round_)

        s_va = _predict_margin(bst, dva, best_round) if has_val else np.zeros(0, dtype=np.float32)
        s_te = _predict_margin(bst, dte, best_round)

        return np.asarray(s_va, np.float32), np.asarray(s_te, np.float32)

    # ---- non-xgb fallback only happens here ----
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingClassifier

    n_comp = int(min(256, max(2, Xtr.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Xtr_d = svd.fit_transform(Xtr)
    Xte_d = svd.transform(Xte)

    model = HistGradientBoostingClassifier(random_state=seed, **params)
    model.fit(Xtr_d, (np.asarray(ytr).astype(int) > 0).astype(int), sample_weight=wtr)

    s_va = np.zeros(0, dtype=np.float32)
    s_te = model.decision_function(Xte_d).astype(np.float32)
    return s_va, s_te

# ----------------
# MAIN PIPELINE
# ----------------
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

    adm_tr = load_admissions(ds_dir, adm_stem, "train")
    adm_va = load_admissions(ds_dir, adm_stem, "val")
    adm_te = load_admissions(ds_dir, adm_stem, "test")

    dx_tr = load_diagnoses(ds_dir, diag_stem, "train")
    dx_va = load_diagnoses(ds_dir, diag_stem, "val")
    dx_te = load_diagnoses(ds_dir, diag_stem, "test")


    vocab, unk_idx = build_vocab(dx_tr)
    Xtr = build_bow_matrix(adm_tr, dx_tr, vocab, unk_idx)
    Xva = build_bow_matrix(adm_va, dx_va, vocab, unk_idx)
    Xte = build_bow_matrix(adm_te, dx_te, vocab, unk_idx)

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




    kept = kept_targets_from_train(adm_tr, train_targets)
    print("Kept targets (TRAIN-derived):", kept)


    Xtr_pool, ytr_pool, task_tr = build_pooled_xy_keep_with_task_ids(Xtr, adm_tr, kept, train_targets)
    Xva_pool, yva_pool, task_va = build_pooled_xy_keep_with_task_ids(Xva, adm_va, kept, train_targets)

    val_ok = (Xva_pool is not None) and (yva_pool is not None) and (yva_pool.size >= 10) and (np.unique(yva_pool).size >= 2)
    if not val_ok:
        raise RuntimeError("[Stage2] pooled VAL unusable but validation is REQUIRED. Aborting.")


    if Xtr_pool is None or ytr_pool.size < 10 or np.unique(ytr_pool).size < 2:
        print("TRAIN pooled target unusable -> zero scores")
        score_te = np.zeros(len(adm_te), dtype=np.float32)

    else:
        backend = GBT_BACKEND
        if backend == "xgb":
            import xgboost  # noqa
        elif backend == "lgb":
                raise ValueError("GBT_BACKEND='lgb' not implemented in train_and_score_logit_backend() yet.")
        else:
            raise ValueError(f"GBT_BACKEND must be 'xgb' or 'lgb' for apples-to-apples (got {backend}).")
        print("GBT backend:", backend)



        # Fixed recipe params (NO GRID SEARCH)

        # inside run_dataset(), replace fixed_params with two:
        fixed_params_stage1 = {"n_estimators": N_ROUNDS_STAGE1, "max_depth": 4, "learning_rate": 0.05}
        fixed_params_stage2 = {"n_estimators": N_ROUNDS_STAGE2, "max_depth": 4, "learning_rate": 0.05}


        LOG2 = float(math.log(2.0))
        val_bce_by_task = {t: float("inf") for t in train_targets}

        print("\n=== Stage 1 (GBT): single-task models -> VAL_BCE per task (for weights) ===")
        for t in train_targets:
            if t not in kept:
                print(f"  [Stage1] {t:16s}: not kept -> val_bce=inf")
                continue

            ytr_all, mtr = get_y_mask(adm_tr, t)
            yva_all, mva = get_y_mask(adm_va, t)

            if mva.sum() == 0:
                print(f"  [Stage1] {t:16s}: VAL has no valid labels -> val_bce=inf")
                val_bce_by_task[t] = float("inf")
                continue


            tr_idx = np.where(mtr)[0]
            # TRAIN must be usable (enough + both classes)
            if tr_idx.size < 10 or np.unique(ytr_all[tr_idx]).size < 2:
                print(f"  [Stage1] {t:16s}: TRAIN unusable -> val_bce=inf")
                continue

            # train on valid TRAIN rows only; early-stop on valid VAL rows only
            Xtr_t = Xtr[tr_idx]
            ytr_t = ytr_all[tr_idx].astype(np.int32)

            # IMPORTANT TRICK:
            # Call train_and_score... with Xte = Xva (full) so we can compute masked VAL_BCE over all admissions.
            # Returned "test_logits" will actually be logits over full VAL admissions rows.
            seed_t = stable_seed(SEED, f"{name}|stage1|{t}")

            # For xgb, xgb_params already uses "seed": SEED; make it per-task by temporarily overriding global SEED or passing seed in params.
            # Easiest: temporarily set SEED-like value via params (see note below) OR add a 'seed' field into xgb_params from outside.
            va_idx = np.where(mva)[0]          # valid VAL rows for that task
            Xva_eval = Xva[va_idx]
            yva_eval = yva_all[va_idx].astype(np.int32)

            _, s_va_full_logits = train_and_score_logit_backend(
                Xtr_t, ytr_t,
                Xva_eval, yva_eval,   # <-- give xgb a real labeled val set
                Xva,                  # <-- still score ALL val admissions here
                backend=backend,
                seed=seed_t,
                params=fixed_params_stage1,
                early_stop_rounds=None,
                wtr=None, wva=None,
            )



            p1_va_full = sigmoid(s_va_full_logits)
            vb = bce_mean_over_batches_masked(p1_va_full, yva_all.astype(np.int32), mva, batch_size=VAL_BATCH_SIZE)
            val_bce_by_task[t] = float(vb)

            print(f"  [Stage1] {t:16s}: val_bce={vb:.6f}")

        scores = np.zeros(len(train_targets), dtype=np.float64)
        for j, t in enumerate(train_targets):
            vb = val_bce_by_task[t]
            if (t not in kept) or (not np.isfinite(vb)):
                scores[j] = 0.0
            else:
                scores[j] = max(LOG2 - float(vb), 0.0)

        max_s = float(scores.max()) if scores.size else 0.0
        denom = np.maximum(scores, WEIGHT_EPS)
        w = (np.maximum(max_s, WEIGHT_EPS) / denom) ** WEIGHT_ALPHA
        w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP).astype(np.float32)

        # zero non-kept tasks
        for j, t in enumerate(train_targets):
            if (t not in kept) or (not np.isfinite(val_bce_by_task[t])):
                w[j] = 0.0

        # normalize active weights to mean 1
        active = w > 0
        if active.any():
            w[active] = w[active] / w[active].mean()

        weights_vec = w  # aligned with train_targets order



        print("\nStage-1 VAL_BCE -> score=(log2-BCE)+ -> weights:")
        for j, t in enumerate(train_targets):
            vb = val_bce_by_task[t]
            vb_str = f"{vb:.6f}" if np.isfinite(vb) else "inf"
            print(f"  {t:16s}: val_bce={vb_str:>10s} | score={scores[j]:.6f} | w={float(weights_vec[j]):.6f}")
        
        # ---- pooled sample weights for Stage 2 (NOW weights_vec exists) ----
        wtr_pool = weights_vec[task_tr].astype(np.float32)
        wva_pool = weights_vec[task_va].astype(np.float32)

        seed_stage2 = stable_seed(SEED, f"{name}|stage2")

        s_va_pool_logits, s_te_logits = train_and_score_logit_backend(
            Xtr_pool, ytr_pool,
            Xva_pool, yva_pool,      # <-- pooled VAL is used
            Xte,
            backend=backend,
            params=fixed_params_stage2,
            early_stop_rounds=None,  # <-- NO EARLY STOPPING
            seed=seed_stage2,
            wtr=wtr_pool,
            wva=wva_pool,
        )

        score_te = s_te_logits.astype(np.float32)

        # --- Stage-2 VAL is report-only (NO checkpoint selection / NO early stop) ---
        p1_va_pool = sigmoid(s_va_pool_logits)
        val_bce_stage2 = bce_from_proba(p1_va_pool, yva_pool.astype(np.int32))
        print(f"  [Stage2] pooled_VAL_BCE(report-only)={val_bce_stage2:.6f}")



  


    print("\nTEST Metrics (DistCorr, MI_nats) using GBT logit score:")
    for t in eval_targets:
        if t == "long_stay":
            # DL-parity: validity comes from long_stay_defined, not NaN in long_stay
            valid_mask = adm_te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            yte_raw = pd.to_numeric(adm_te["long_stay"], errors="coerce").to_numpy(dtype=float)
        else:
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
            base_neighbors=5,
        )

        print(
            f"  {t:16s} DistCorr={dcorr:.6f} | MI_nats={mi:.6f}"
        )


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