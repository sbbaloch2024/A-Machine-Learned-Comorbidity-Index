"""
Naive Bayes Multi-Task Clinical Outcome Predictor with Weighted Logit Fusion
=============================================================================

Overview
--------
This script trains per-target Naive Bayes (NB) classifiers — either Multinomial
or Complement NB — on TF-IDF-weighted ICD-10 code bag-of-words features, converts
the resulting log-probability outputs to logit scores, then fuses them into a
single scalar via a VAL-BCE-derived weighted mean. There is no second-stage
learner; fusion is a direct normalized weighted average of Stage-1 log-odds scores.

NB variant is controlled by NB_KIND ("multinomial" or "complement", default: "complement").

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.
2.  Load train/val/test admission CSVs; read diagnosis ICD-10 CSV.gz files in
    chunks (2M rows at a time) to build vocabulary and BoW matrices.
3.  Normalize ICD codes to 4-char uppercase prefixes; build a vocabulary (+ UNK
    token) from TRAIN diagnoses only.
4.  Construct count-based Bag-of-Words sparse matrices (CSR), capped at 256 codes
    per admission in file-encounter order, then apply TF-IDF transformation
    (fit on TRAIN only; transform val/test).
5.  Stage 1 — Per-target NB (alpha=1.0 Laplace smoothing):
        - Fit on valid TRAIN rows for each target.
        - Compute log-odds logit scores via log P(y=1|x) − log P(y=0|x) for all splits.
        - Compute masked mean-over-batches VAL BCE from the logit scores.
        - Skip and zero-fill targets that are single-class or have too few labels.
6.  Stack per-target logit arrays into matrices Ltr (N_tr, T), Lva (N_va, T),
    Lte (N_te, T).
7.  Derive per-task fusion weights from Stage-1 VAL BCEs:
        score_t  = max(log(2) − val_bce_t, 0)
        weight_t = clip((max_score / max(score_t, ε))^0.25, 1/3, 3)
    then normalize active weights to mean 1.
8.  Fusion — compute fused_te as the normalized weighted mean of Lte columns
    corresponding to kept fusion targets (no second-stage model trained).
9.  Report Distance Correlation, and Mutual Information between fused
    scores and each binary test label.

Key Configuration
-----------------
  SEED=42, EVAL_SEED=12345
  NB_KIND="complement" (set to "multinomial" to switch)
  NB_ALPHA=1.0 (Laplace smoothing)
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
        ├── diagnoses_icd10_core_train.csv.gz
        ├── diagnoses_icd10_core_val.csv.gz
        └── diagnoses_icd10_core_test.csv.gz

Required admission columns:
    hadm_id, mortality, mortality_30d, long_stay, long_stay_defined, icu_transfer

Required diagnosis columns:
    hadm_id, icd_code

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor

Running
-------
    python nb_mimiciv.py --data_root /path/to/final_datasets

Switch NB variant at runtime:
    NB_KIND=multinomial python nb_mimiciv.py --data_root /path/to/final_datasets

Example:
    python nb_mimiciv.py --data_root ./data/final_datasets
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
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB

NB_KIND  = "complement"   # "multinomial" or "complement"
print(NB_KIND)
NB_ALPHA = 1.0            # Laplace smoothing


def nb_logit(model, X) -> np.ndarray:
    """
    Return a logit-like score for binary NB: log P(y=1|x) - log P(y=0|x).
    Works for MultinomialNB and ComplementNB.
    """
    logp = model.predict_log_proba(X)  # shape (n, 2) for binary
    classes = model.classes_
    # find column indices for class 0 and class 1
    i0 = int(np.where(classes == 0)[0][0])
    i1 = int(np.where(classes == 1)[0][0])
    return (logp[:, i1] - logp[:, i0]).astype(np.float32)


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

N_JOBS = 1
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
        sublinear_tf=False,  # set True if for log(1+tf)
        norm="l2",           # common default; 
    )

    Xtr = tfidf.fit_transform(Xtr)   # FIT ONLY on train
    Xva = tfidf.transform(Xva)
    Xte = tfidf.transform(Xte)



    print(f"Xtr shape: {Xtr.shape}, nnz={Xtr.nnz:,}")
    print(f"Xte shape: {Xte.shape}, nnz={Xte.nnz:,}")

# -------- 1) Per-target Naive Bayes -> logits --------
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

            # ---- Stage 1: Naive Bayes instead of LogisticRegression ----
            if NB_KIND == "multinomial":
                model = MultinomialNB(alpha=NB_ALPHA)
            elif NB_KIND == "complement":
                model = ComplementNB(alpha=NB_ALPHA)
            else:
                raise ValueError(f"Unknown NB_KIND={NB_KIND}")

            model.fit(Xtr[mtr], ytr_bin)

            va_logits = nb_logit(model, Xva)  # log-odds score (acts like logits)
            if int(mva.sum()) == 0:
                va_bce = float("inf")
            else:
                va_bce = bce_mean_over_batches_masked_logits(va_logits, yva, mva, batch_size=256)

            val_bce_by_target[t] = float(va_bce)
            print(
                f"  target={t:16s} NB={NB_KIND} alpha={NB_ALPHA} val_BCE={va_bce:.6f} "
                f"n_tr_valid={int(mtr.sum()):,} n_va_valid={int(mva.sum()):,}"
            )

            lt_tr = nb_logit(model, Xtr)
            lt_va = va_logits
            lt_te = nb_logit(model, Xte)

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

            # weights from Stage-1 VAL_BCE
            weights_vec = compute_task_weights_from_val_bce(kept_fuse, val_bce_by_target)

            # map target -> column index in L* (since L columns follow train_targets order)
            target_to_col = {t: j for j, t in enumerate(train_targets)}
            cols = [target_to_col[t] for t in kept_fuse]

            Lte_fuse = Lte[:, cols].astype(np.float32)

            # simple weighted mean of logits across tasks
            w = weights_vec.astype(np.float32)
            w = w / (w.sum() + 1e-8)
            fused_te = (Lte_fuse * w.reshape(1, -1)).sum(axis=1).astype(np.float32)



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
        diag_stem="diagnoses_icd10_core",
        train_targets=CORE_TARGETS,
        eval_targets=CORE_TARGETS,
    )




    print("\n✅ Done.")


if __name__ == "__main__":
    main()