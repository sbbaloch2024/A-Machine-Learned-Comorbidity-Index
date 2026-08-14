"""
CCI + Van Walraven ECI Baseline Clinical Outcome Scorer
=======================================================

Overview
--------
This script implements a simple non-neural baseline clinical outcome scorer using
two admission-level comorbidity features: Charlson Comorbidity Index (CCI) and
Elixhauser Comorbidity Index van Walraven total score (ECI-VW). The script fits a
mean imputer and standard scaler on the TRAIN split only, applies the fitted
preprocessing pipeline to the TEST split, and computes one scalar score per test
admission as:

    score = z(cci) + z(eci_vw_total)

The resulting scalar score is evaluated against multiple binary clinical outcomes
on the CORE test split. Evaluation is mask-aware: each target is evaluated only on
rows where that target is valid. For long_stay, validity is determined by
long_stay_defined.

Unlike the neural Set Transformer or Deep Sets experiments, this script does not
train a model. It provides a transparent feature-based baseline that summarizes
comorbidity burden using standardized CCI and ECI-VW scores.

Pipeline Flow
-------------
1.  Parse --data_root argument pointing to the final_datasets directory.

2.  Define the CORE dataset:
        - Dataset directory:
              <data_root>/core
        - Admission file stem:
              admissions_core
        - Train file:
              admissions_core_train.csv
        - Test file:
              admissions_core_test.csv

3.  Define clinical outcome targets:
        - mortality
        - mortality_30d
        - long_stay
        - icu_transfer

4.  Define baseline feature columns:
        - cci
        - eci_vw_total

5.  Load TRAIN and TEST admission CSV files.

6.  Print label positive rates for TRAIN and TEST:
        - Rates are computed on valid-only rows.
        - For long_stay, validity is determined by long_stay_defined.
        - For all other targets, validity is determined by non-missing target
          values.

7.  Build the scalar TEST score using TRAIN-fitted preprocessing:
        - Read cci and eci_vw_total from TRAIN and TEST.
        - Convert feature columns to numeric values.
        - Treat non-numeric entries as missing.
        - Fit SimpleImputer(strategy="mean") on TRAIN features only.
        - Apply the fitted imputer to TEST features.
        - Fit StandardScaler on imputed TRAIN features only.
        - Apply the fitted scaler to TEST features.
        - Compute:
              score_test = z_test(cci) + z_test(eci_vw_total)

8.  Evaluate the TEST score for each target:
        - Build a target-specific validity mask.
        - Fill invalid or missing labels with zero only as a placeholder.
        - Use the validity mask to exclude invalid rows from all metrics.
        - Compute mutual information in nats.
        - Compute Kendall tau-b correlation.
        - Compute distance correlation.

9.  Print a formatted TEST metrics table.

10. Print completion message.

Score Definition
----------------
The scalar score is computed from two comorbidity features:

    score = z(cci) + z(eci_vw_total)

where z(.) denotes standardization using the TRAIN-fitted StandardScaler. The
scaler is fit only on TRAIN data to avoid test-set leakage. The same TRAIN-fitted
scaler is then applied to TEST data.

Feature Preprocessing
---------------------
Feature preprocessing is fit on TRAIN and applied to TEST:

- Numeric conversion:
      pd.to_numeric(..., errors="coerce")

- Missing-value imputation:
      SimpleImputer(strategy="mean")

- Standardization:
      StandardScaler()

The imputer and scaler are both fit only on TRAIN features. TEST features are
transformed using those fitted objects.

Validity Handling
-----------------
Each target has its own validity mask.

For long_stay:
    valid = long_stay_defined == 1

For all other targets:
    valid = target value is not missing

Invalid labels are filled with zero only as placeholders. They do not contribute
to any evaluation metric because the corresponding validity mask value is false.

Evaluation Metrics
------------------
The scalar comorbidity score is evaluated against each clinical outcome using:

    - Mutual information in nats
    - Distance correlation

Metric safeguards:
    - Return 0.0 when fewer than three valid examples are available.
    - Return 0.0 when the score or label vector is constant or near-constant.
    - Return 0.0 for mutual information when the target has fewer than two classes.
    - Z-score scores within valid rows before mutual information estimation.
    - Use a fixed EVAL_SEED for mutual information estimation.

Architecture
------------
This script has no learned neural architecture.

The baseline scorer is a deterministic linear combination of two standardized
comorbidity features:

    input features:
        cci
        eci_vw_total

    preprocessing:
        TRAIN-fitted mean imputation
        TRAIN-fitted standardization

    scalar output:
        z(cci) + z(eci_vw_total)

No parameters are optimized with respect to the clinical outcomes.

Key Configuration
-----------------
  SEED: controlled via SEED environment variable, default 42
  EVAL_SEED=12345
  CORE_TARGETS=["mortality", "mortality_30d", "long_stay", "icu_transfer"]
  FEATURE_COLS=["cci", "eci_vw_total"]
  Mutual information base neighbors=5

Expected Directory Layout
--------------------------
    <data_root>/
    └── core/
        ├── admissions_core_train.csv
        └── admissions_core_test.csv

Required admission columns:
    hadm_id, mortality, mortality_30d, long_stay, long_stay_defined,
    icu_transfer, cci, eci_vw_total

Outputs
-------
The script prints:

    - random seed and evaluation seed
    - dataset name and directory
    - TRAIN label positive rates
    - TEST label positive rates
    - TEST metrics table containing:
          MI
          dcor
    - completion message

This script does not write scores, metrics, checkpoints, or diagnostic CSV files
to disk.

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor

Running
-------
Run the script with:

    python CCIECI_zsum_mimiciii.py --data_root /path/to/final_datasets

Override the seed:

    SEED=42 python CCIECI_zsum_mimiciii.py --data_root /path/to/final_datasets

Example:
    python CCIECI_zsum_mimiciii.py --data_root ./data/final_datasets
"""


# ----------------
# IMPORTS
# ----------------
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
import dcor  # pip install dcor
import os
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr

# ----------------
# CONFIG
# ----------------
SEED = int(os.environ.get("SEED", 42))   # varies across runs (even if unused here)
EVAL_SEED = 12345                        # fixed eval seed
print("SEED:", SEED, "| EVAL_SEED:", EVAL_SEED)

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
FEATURE_COLS = ["cci", "eci_vw_total"]  # columns in admissions_* files

# ----------------
# UTILITY HELPERS
# ----------------
def zscore_1d(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-8)

# ----------------
# DATA LOADING AND PREPROCESSING
# ----------------
def load_split_csvs(ds_dir: Path, stem: str):
    tr = pd.read_csv(ds_dir / f"{stem}_train.csv")
    te = pd.read_csv(ds_dir / f"{stem}_test.csv")
    return tr, te

def build_score_train_test(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> np.ndarray:
    """
    Fit imputer+scaler on TRAIN features, apply to TEST, then compute scalar score.
    """
    missing = [c for c in FEATURE_COLS if c not in df_tr.columns or c not in df_te.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    # define feature matrices
    Xtr = df_tr[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xte = df_te[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # fit imputer on train only
    imp = SimpleImputer(strategy="mean")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)

    # fit scaler on train only
    scaler = StandardScaler()
    Xtr_z = scaler.fit_transform(Xtr)
    Xte_z = scaler.transform(Xte)

    # score on test
    score_te = (Xte_z[:, 0] + Xte_z[:, 1]).astype(np.float32)  # z(cci) + z(eci)
    return score_te

# ----------------
# LABEL INSPECTION
# ----------------
def label_stats(name: str, df: pd.DataFrame, cols):
    print(f"\nLabel positive rates ({name}) [computed on valid-only rows]:")
    for c in cols:
        if c not in df.columns and c != "long_stay":
            print(f"  {c:20s}: MISSING COLUMN")
            continue

        if c == "long_stay":
            if "long_stay_defined" not in df.columns or "long_stay" not in df.columns:
                print(f"  {c:20s}: MISSING COLUMN(S)")
                continue
            valid = df["long_stay_defined"].fillna(0).astype(int).to_numpy().astype(bool)
            n_valid = int(valid.sum())
            if n_valid == 0:
                print(f"  {c:20s}: n_valid=0")
                continue
            y = df["long_stay"].fillna(0).astype(float).to_numpy()
            rate = float(y[valid].mean())
            print(f"  {c:20s}: {rate:.4f} (n_valid={n_valid:,})")
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        valid = (~s.isna()).to_numpy(dtype=bool)
        n_valid = int(valid.sum())
        if n_valid == 0:
            print(f"  {c:20s}: n_valid=0")
            continue
        rate = float(s[valid].astype(float).mean())
        print(f"  {c:20s}: {rate:.4f} (n_valid={n_valid:,})")

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

def mi_from_score(
    score: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    seed: int = EVAL_SEED,
    base_neighbors: int = 5
) -> float:
    valid = valid_mask.astype(bool)
    if valid.sum() < 3:
        return 0.0

    yy = y[valid].astype(int)
    if np.unique(yy).size < 2:
        return 0.0

    n_neighbors = int(min(base_neighbors, valid.sum() - 1))
    if n_neighbors < 1:
        return 0.0

    # z-score ONLY within valid rows (match DL)
    s_valid = zscore_1d(score[valid]).reshape(-1, 1).astype(np.float32)
    return float(mutual_info_classif(X=s_valid, y=yy, random_state=seed, n_neighbors=n_neighbors)[0])


# ----------------
# MAIN PIPELINE
# ----------------
def eval_dataset(name: str, ds_dir: Path, stem: str, targets):
    """
    Fit on TRAIN, evaluate MI, and dcor on TEST for targets.
    """
    print("\n" + "=" * 80)
    print(f"DATASET: {name}")
    print(f"DIR    : {ds_dir}")
    print("=" * 80)

    df_tr, df_te = load_split_csvs(ds_dir, stem)

    # Print label stats
    label_stats("TRAIN", df_tr, targets)
    label_stats("TEST ", df_te, targets)

    # Build scalar score on TEST using TRAIN-fitted imputer/scaler
    score_te = build_score_train_test(df_tr, df_te)

    # Compute all metrics per target on TEST
    print("\nMetrics on TEST (train-fit scaler, mask-aware):")
    print(f"{'Target':20s} {'MI':>10s} {'dcor':>10s}")
    print("-" * 86)


    for t in targets:
        if t not in df_te.columns and t != "long_stay":
            print(f"{t:20s} MISSING COLUMN")
            continue

        if t == "long_stay":
            valid = df_te["long_stay_defined"].fillna(0).astype(int).to_numpy().astype(bool)
            y01 = df_te["long_stay"].fillna(0).astype(int).to_numpy()
        else:
            y = pd.to_numeric(df_te[t], errors="coerce").to_numpy()
            valid = ~np.isnan(y)
            y01 = np.where(np.isnan(y), 0, y).astype(int)

        mi = mi_from_score(score_te, y01, valid_mask=valid, seed=EVAL_SEED)
        pearson = pearson_from_score(score_te, y01, valid_mask=valid)
        dc = dcor_from_score(score_te, y01, valid_mask=valid)

        print(f"{t:20s} {mi:10.6f}{dc:10.6f}")


def main():
    # CORE
    core_dir = BASE / "core"
    eval_dataset(
        name="CORE",
        ds_dir=core_dir,
        stem="admissions_core",
        targets=CORE_TARGETS,
    )

   
    print("\n✅ Done.")

if __name__ == "__main__":
    main()