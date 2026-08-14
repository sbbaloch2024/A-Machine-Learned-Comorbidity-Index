
"""
Logistic Regression Multi-Task Clinical Outcome Predictor with Logit Fusion
============================================================================

Overview
--------
This script trains per-target Logistic Regression (LR) models on TF-IDF-weighted
ICD-9 code bag-of-words features, then fuses the resulting per-task logit vectors
into a single scalar score using a second-stage pooled LR with task-derived weights.
The final fused score is evaluated against each binary outcome on the TEST split.

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.
2.  Load train/val/test admission CSVs; read diagnosis ICD-9 CSV.gz files in
    chunks (2M rows at a time) to build vocabulary and BoW matrices.
3.  Normalize ICD codes to 4-char uppercase prefixes; build a vocabulary (+ UNK
    token) from TRAIN diagnoses only.
4.  Construct count-based Bag-of-Words sparse matrices (CSR), capped at 256 codes
    per admission in file-encounter order, then apply TF-IDF transformation
    (fit on TRAIN only; transform val/test).
5.  Stage 1 — Per-target LR (saga, L2, C=0.2, max_iter=1000):
        - Fit on valid TRAIN rows for each target.
        - Compute masked mean-over-batches VAL BCE from decision_function logits.
        - Store logits over ALL admissions (train/val/test) for each target.
        - Skip and zero-fill targets that are single-class or have too few labels.
6.  Stack per-target logit arrays into matrices Ltr (N_tr, T), Lva (N_va, T),
    Lte (N_te, T), where each column is one target's raw logit signal.
7.  Derive per-task loss weights from Stage-1 VAL BCEs:
        score_t  = max(log(2) − val_bce_t, 0)
        weight_t = clip((max_score / max(score_t, ε))^0.25, 1/3, 3)
    then normalize active weights to mean 1.
8.  Stage 2 — Fusion LR on pooled logit features:
        - Pool (Ltr rows, labels, task IDs) across all kept fusion targets.
        - Assign per-sample weights from Stage-1 task weights.
        - Train fusion LR at fixed iteration checkpoints [50, 100, 200, 300, 500]
          (lbfgs, L2, C=1.0); select the checkpoint with lowest pooled VAL BCE.
        - Produce final scalar fused_te scores via decision_function(Lte).
9.  Report Distance Correlation, and Mutual Information between fused
    scores and each binary test label.

Key Configuration
-----------------
  SEED=42, EVAL_SEED=12345
  LR Stage 1: solver=saga, penalty=l2, C=0.2, max_iter=1000
  LR Stage 2: solver=lbfgs, penalty=l2, C=1.0, checkpoints=[50,100,200,300,500]
  PREFIX_LEN=4, MAX_CODES_PER_ADMISSION=256, DX_CHUNKSIZE=2_000_000
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
    python lr_mimiciii.py --data_root /path/to/final_datasets

Example:
    python lr_mimiciii.py --data_root ./data/final_datasets
"""

# ----------------
# IMPORTS
# ----------------
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfTransformer


WEIGHT_EPS   = 0.02
WEIGHT_ALPHA = 0.25
WEIGHT_CAP   = 3.0


# ----------------
# CONFIG
# ----------------
DX_CHUNKSIZE = 2_000_000   # match HSIC/DL scripts
MAX_CODES_PER_ADMISSION = 256
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

MAX_ITER = 1000
C = 1.0
SOLVER = "saga"          # good for large sparse
PENALTY = "l2"
N_JOBS = 1
CLASS_WEIGHT = None      # or "balanced" for
CORE_TARGETS = ["mortality", "mortality_30d", "long_stay", "icu_transfer"]


# ----------------
# UTILITY HELPERS
# ----------------
def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

def softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def norm_icd(code) -> str:
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()

# ----------------
# DATA LOADING AND VOCAB
# ----------------
def load_admissions(ds_dir: Path, stem: str, split: str) -> pd.DataFrame:
    p = ds_dir / f"{stem}_{split}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def build_vocab_from_dx_path(dx_path: Path, chunksize: int = DX_CHUNKSIZE) -> dict:
    toks = set()

    for chunk in pd.read_csv(dx_path, usecols=["icd_code"], chunksize=chunksize):
        codes = chunk["icd_code"].map(norm_icd).astype(str).str[:PREFIX_LEN]
        for c in codes.to_numpy():
            if c and c != "":   # skip empty
                toks.add(c)

    toks = sorted(toks)
    vocab = {c: i for i, c in enumerate(toks)}
    vocab["<UNK>"] = len(vocab)
    return vocab

def build_bow_matrix_chunked(
    adm_df: pd.DataFrame,
    dx_path: Path,
    vocab: dict,
    chunksize: int = DX_CHUNKSIZE,
    max_codes_per_adm: int = MAX_CODES_PER_ADMISSION,
):
    hadm_ids = adm_df["hadm_id"].astype(int).to_numpy()
    hadm_to_row = {hid: i for i, hid in enumerate(hadm_ids)}
    unk = vocab["<UNK>"]

    # track how many dx rows we've kept so far for each hadm (after empty filtering)
    seen = defaultdict(int)

    # sparse accumulator: row -> {col -> count}
    row_counts = defaultdict(lambda: defaultdict(float))

    for chunk in pd.read_csv(dx_path, usecols=["hadm_id", "icd_code"], chunksize=chunksize):
        hadm_arr = chunk["hadm_id"].astype(int).to_numpy()
        code_arr = chunk["icd_code"].to_numpy()

        for hid, raw in zip(hadm_arr, code_arr):
            # tokenization must match DL scripts
            code = norm_icd(raw)[:PREFIX_LEN]
            if code == "":
                continue

            r = hadm_to_row.get(int(hid))
            if r is None:
                continue

            if seen[int(hid)] >= max_codes_per_adm:
                continue
            seen[int(hid)] += 1

            j = vocab.get(code, unk)
            row_counts[r][j] += 1.0

    # materialize CSR
    rows, cols, data = [], [], []
    for r, cdict in row_counts.items():
        for j, v in cdict.items():
            rows.append(r)
            cols.append(int(j))
            data.append(float(v))

    X = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(adm_df), len(vocab)),
        dtype=np.float32,
    )
    X.sum_duplicates()
    return X

# ----------------
# LABEL MASKING AND POOLING
# ----------------
def get_y_and_mask(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "long_stay":
        m = df["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
        y = pd.to_numeric(df["long_stay"], errors="coerce").fillna(0).to_numpy(dtype=int)
        return y, m
    else:
        y_raw = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(y_raw)
        y = np.nan_to_num(y_raw, nan=0.0).astype(int)
        return y, m

def kept_targets_from_train_for_fusion(adm_tr: pd.DataFrame, targets: list[str], min_valid: int = 3) -> list[str]:
    kept = []
    for t in targets:
        y, m = get_y_and_mask(adm_tr, t)
        idx = np.where(m.astype(bool))[0]
        if idx.size < min_valid:
            continue
        yy = y[idx].astype(int)
        if np.unique(yy).size < 2:
            continue
        kept.append(t)
    return kept


def build_pooled_from_logits_keep_with_task_ids(
    adm_df: pd.DataFrame,
    L: np.ndarray,
    kept_targets: list[str],
    all_targets: list[str],   # for stable indexing
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    X_list, y_list, task_list = [], [], []
    target_to_j = {t: j for j, t in enumerate(all_targets)}  # stable indexing

    for t in kept_targets:
        y, m = get_y_and_mask(adm_df, t)
        idx = np.where(m.astype(bool))[0]
        if idx.size < 1:
            continue
        X_list.append(L[idx])                                # full logit vector as features
        y_list.append(y[idx].astype(int))                    # label for this task
        task_list.append(np.full(idx.size, target_to_j[t], dtype=np.int32))

    if not X_list:
        return None, None, None

    return (
        np.vstack(X_list).astype(np.float32),
        np.concatenate(y_list).astype(int),
        np.concatenate(task_list).astype(np.int32),
    )

def compute_task_weights_from_val_bce(tasks: list[str], val_bce_by_task: dict[str, float]) -> np.ndarray:
    LOG2 = float(math.log(2.0))

    scores = np.zeros(len(tasks), dtype=np.float64)
    active = np.zeros(len(tasks), dtype=bool)

    for j, t in enumerate(tasks):
        vb = float(val_bce_by_task.get(t, float("inf")))
        if np.isfinite(vb):
            scores[j] = max(LOG2 - vb, 0.0)
            active[j] = True
        else:
            scores[j] = 0.0
            active[j] = False

    if not active.any():
        # no usable VAL for any fusion task => hard fail (VAL is required)
        raise RuntimeError("[FUSION] No fusion tasks have usable VAL_BCE; cannot derive weights.")

    max_s = float(scores[active].max())

    denom = np.maximum(scores, WEIGHT_EPS)
    w = (np.maximum(max_s, WEIGHT_EPS) / denom) ** WEIGHT_ALPHA
    w = np.clip(w, 1.0 / WEIGHT_CAP, WEIGHT_CAP).astype(np.float32)

    # tasks with no usable VAL get weight 0 (masked out)
    w[~active] = 0.0

    # normalize active weights to mean 1
    w_active = w[w > 0]
    w[w > 0] = w_active / w_active.mean()

    return w


# ----------------
# LOSS FUNCTIONS
# ----------------
def bce_with_logits_from_logits(logits: np.ndarray, y: np.ndarray) -> float:
    # assumes y is 0/1 and logits are real numbers, returns mean BCEWithLogits
    logits = logits.astype(np.float32)
    y = y.astype(np.float32)
    return float(np.mean(softplus(logits) - y * logits))

def bce_mean_over_batches_masked_logits(
    logits_all: np.ndarray,
    y_all: np.ndarray,
    m_all: np.ndarray,
    batch_size: int = 256,
) -> float:
    n = int(len(y_all))
    if n == 0:
        return float("inf")

    losses = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        mb = m_all[start:end].astype(bool)
        if mb.sum() == 0:
            losses.append(0.0)  # DL-parity: include zero-loss batches
            continue

        lb = bce_with_logits_from_logits(logits_all[start:end][mb], y_all[start:end][mb])
        losses.append(lb)

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
    print("\n" + "=" * 90)
    print(f"DATASET: {name}")
    print(f"DIR    : {ds_dir}")
    print(f"TARGETS: {train_targets}")
    print("=" * 90)

    adm_tr = load_admissions(ds_dir, adm_stem, "train")
    adm_va = load_admissions(ds_dir, adm_stem, "val")
    adm_te = load_admissions(ds_dir, adm_stem, "test")

    dx_tr_path = ds_dir / f"{diag_stem}_train.csv.gz"
    dx_va_path = ds_dir / f"{diag_stem}_val.csv.gz"
    dx_te_path = ds_dir / f"{diag_stem}_test.csv.gz"

    vocab = build_vocab_from_dx_path(dx_tr_path)
    print(f"TRAIN vocab size (#unique ICD codes): {len(vocab):,}")

    Xtr = build_bow_matrix_chunked(adm_tr, dx_tr_path, vocab)
    Xva = build_bow_matrix_chunked(adm_va, dx_va_path, vocab)
    Xte = build_bow_matrix_chunked(adm_te, dx_te_path, vocab)

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



    print(f"Xtr shape: {Xtr.shape}, nnz={Xtr.nnz:,}")
    print(f"Xte shape: {Xte.shape}, nnz={Xte.nnz:,}")

    # -------- 1) Per-target LR -> logits --------
    logits_tr, logits_va, logits_te = [], [], []
    skipped_targets = []
    kept_targets = []

    val_bce_by_target = {t: float("inf") for t in train_targets}   # <-- ADD THIS LINE HERE



    for t in train_targets:
        ytr, mtr = get_y_and_mask(adm_tr, t)
        yva, mva = get_y_and_mask(adm_va, t)

        ytr_bin = ytr[mtr].astype(int) if mtr.any() else np.array([], dtype=int)

        if mtr.sum() < 3 or np.unique(ytr_bin).size < 2:
            print(f"[WARN] Skipping target '{t}' (too few / single-class in TRAIN). Using zeros.")
            skipped_targets.append(t)

            lt_tr = np.zeros(len(adm_tr), dtype=np.float32)
            lt_va = np.zeros(len(adm_va), dtype=np.float32)
            lt_te = np.zeros(len(adm_te), dtype=np.float32)
        else:
            kept_targets.append(t)

            model = LogisticRegression(
                solver=SOLVER, penalty=PENALTY, C=0.2, max_iter=MAX_ITER,
                random_state=SEED, n_jobs=N_JOBS, class_weight=CLASS_WEIGHT,
            )
            model.fit(Xtr[mtr], ytr_bin)

            va_logits = model.decision_function(Xva).astype(np.float32)
            if int(mva.sum()) == 0:
                va_bce = float("inf")
            else:
                va_bce = bce_mean_over_batches_masked_logits(va_logits, yva, mva, batch_size=256)

            val_bce_by_target[t] = float(va_bce)
            print(
                f"  target={t:16s} C={C} val_BCE={va_bce:.6f} "
                f"n_tr_valid={int(mtr.sum()):,} n_va_valid={int(mva.sum()):,}"
            )

            lt_tr = model.decision_function(Xtr).astype(np.float32)
            lt_va = va_logits
            lt_te = model.decision_function(Xte).astype(np.float32)

        logits_tr.append(lt_tr)
        logits_va.append(lt_va)
        logits_te.append(lt_te)

    Ltr = np.stack(logits_tr, axis=1)
    Lva = np.stack(logits_va, axis=1)
    Lte = np.stack(logits_te, axis=1)

# -------- 2) Fusion LR on TRAIN logits -> single scalar (POOLED multitask; NO composite label) --------
    if len(kept_targets) == 0:
        print("[WARN] All tasks skipped; fused score = mean(per-task logits).")
        fused_te = Lte.mean(axis=1).astype(np.float32)
    else:
        kept_fuse = kept_targets_from_train_for_fusion(adm_tr, kept_targets, min_valid=3)
        if len(kept_fuse) == 0:
            print("[WARN] No fusion-kept targets; fused score = mean(per-task logits).")
            fused_te = Lte.mean(axis=1).astype(np.float32)
        else:
            print("Fusion kept targets (TRAIN-derived):", kept_fuse)

            # pooled TRAIN/VAL + task ids
            Xtr_pool, ytr_pool, task_tr = build_pooled_from_logits_keep_with_task_ids(
                adm_tr, Ltr, kept_fuse, all_targets=kept_fuse
            )
            Xva_pool, yva_pool, task_va = build_pooled_from_logits_keep_with_task_ids(
                adm_va, Lva, kept_fuse, all_targets=kept_fuse
            )

            # --- STRICT: VAL REQUIRED ---
            train_ok = (Xtr_pool is not None) and (ytr_pool is not None) and (ytr_pool.size >= 3) and (np.unique(ytr_pool).size >= 2)
            val_ok   = (Xva_pool is not None) and (yva_pool is not None) and (yva_pool.size >= 3) and (np.unique(yva_pool).size >= 2)

            if not train_ok:
                raise RuntimeError("[FUSION] Pooled TRAIN degenerate but validation is required. Aborting.")
            if not val_ok:
                raise RuntimeError("[FUSION] Pooled VAL unusable but validation is required. Aborting.")

            # weights from Stage-1 VAL_BCE (strict: tasks without VAL => weight 0 / fail if none)
            weights_vec = compute_task_weights_from_val_bce(kept_fuse, val_bce_by_target)

            # sample weights per pooled row (based on which task that row came from)
            wtr_pool = weights_vec[task_tr].astype(np.float32)

            print("\nStage-1 VAL_BCE -> weights (fusion tasks):")
            for j, t in enumerate(kept_fuse):
                vb = val_bce_by_target.get(t, float("inf"))
                vb_str = f"{vb:.6f}" if np.isfinite(vb) else "inf"
                print(f"  {t:16s}: val_bce={vb_str:>10s} | w={float(weights_vec[j]):.6f}")

            # --- DL-style checkpointing: fixed list of iterations, select best by pooled VAL_BCE ---
            ITER_CKPTS = [50, 100, 200, 300, 500]   # fixed “epochs” for LR
            best_val = float("inf")
            best_model = None

            for it in ITER_CKPTS:
                fusion = LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    C=C,
                    max_iter=int(it),
                    random_state=SEED,
                )
                fusion.fit(Xtr_pool, ytr_pool, sample_weight=wtr_pool)

                va_logits = fusion.decision_function(Xva_pool).astype(np.float32)
                m_all = np.ones_like(yva_pool, dtype=bool)  # pooled rows are all “valid”
                va_bce = bce_mean_over_batches_masked_logits(va_logits, yva_pool, m_all, batch_size=256)

                print(f"  [FUSION] it={it:4d} pooled_VAL_BCE={va_bce:.6f}")

                if va_bce < best_val:
                    best_val = va_bce
                    best_model = fusion

            print(f"[FUSION] selected checkpoint pooled_VAL_BCE={best_val:.6f}")

            fused_te = best_model.decision_function(Lte).astype(np.float32)


    print("\nTEST Metrics (DistCorr, MI_nats) using fused score:")
    for t in eval_targets:
        if t == "long_stay":
            # DL-parity: validity comes from long_stay_defined, not NaN in long_stay
            valid_mask = adm_te["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)
            yte_raw = pd.to_numeric(adm_te["long_stay"], errors="coerce").to_numpy(dtype=float)
        else:
            yte_raw = pd.to_numeric(adm_te[t], errors="coerce").to_numpy(dtype=float)
            valid_mask = ~np.isnan(yte_raw)

        yte = np.nan_to_num(yte_raw, nan=0.0).astype(int)

        pearson = pearson_from_score(fused_te.astype(np.float32), yte, valid_mask)
        dcorr = dcor_from_score(fused_te.astype(np.float32), yte, valid_mask)
        mi = mi_masked_fixed_rng(
            fused_te.astype(np.float32),
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




    print("\n✅ Done.")


if __name__ == "__main__":
    main()