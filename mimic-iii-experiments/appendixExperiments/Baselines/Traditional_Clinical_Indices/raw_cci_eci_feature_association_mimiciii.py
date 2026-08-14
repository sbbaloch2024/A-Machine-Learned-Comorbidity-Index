"""
Raw CCI/ECI Feature Association Evaluator
=========================================

Overview
--------
This script evaluates the raw association between individual comorbidity summary
features and binary clinical outcomes. It does not train a model, fit a scaler, or
combine features into a learned score. Instead, it treats each selected feature as
a standalone scalar score and computes dependence/association metrics against each
clinical target.

The script currently evaluates two raw admission-level features:

    - cci_raw: raw Charlson Comorbidity Index value
    - eci_raw: raw Elixhauser Comorbidity Index van Walraven total score

For each requested dataset split, the script computes metrics between each raw
feature and each available binary outcome. The supported dataset is CORE, and the
default split is TEST. Results are printed as a table and can optionally be saved
to CSV.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the final_datasets directory
        - --dataset: dataset to evaluate, either ALL or CORE
        - --splits: comma-separated split list, default test
        - --out_csv: optional path for saving results

2.  Resolve datasets:
        - If --dataset ALL, evaluate CORE.
        - Otherwise, evaluate the requested dataset.

3.  Resolve splits:
        - Split the --splits argument by commas.
        - Examples:
              test
              train,val,test

4.  Load each requested admissions split:
        - For CORE:
              <data_root>/core/admissions_core_<split>.csv

5.  Identify available clinical targets:
        - mortality
        - mortality_30d
        - long_stay
        - icu_transfer

6.  Validate required feature columns:
        - cci
        - eci_vw_total

7.  Build standalone raw feature scores:
        - cci_raw = cci with missing values filled as 0
        - eci_raw = eci_vw_total with missing values filled as 0

8.  For each target:
        - Build a target-specific validity mask.
        - For long_stay, use long_stay_defined when available.
        - For other targets, use non-missing target values as valid.
        - Build an integer label vector with missing labels filled as 0.

9.  For each raw feature:
        - Treat the feature as a scalar score.
        - Compute association metrics between the score and the target.
        - Append one result row containing dataset, split, feature, target, and
          metrics.

10. Sort the result table by:
        - dataset
        - split
        - feature
        - target

11. Print the full result table.

12. If --out_csv is provided:
        - Create parent directories if needed.
        - Save the result table to CSV.
        - Print the output path.

Score Definition
----------------
This script evaluates raw features directly as scalar scores.

For CCI:

    score = cci

For ECI-VW:

    score = eci_vw_total

Missing feature values are filled with zero before evaluation:

    cci_raw = fillna(cci, 0)
    eci_raw = fillna(eci_vw_total, 0)

No imputation model, standardization, calibration, weighting, or training is
performed.

Validity Handling
-----------------
Each target has its own validity mask.

For long_stay:
    valid = long_stay_defined == 1

For all other targets:
    valid = target value is not missing

Missing or invalid target labels are filled with zero only as placeholders. They
do not contribute to any metric because the validity mask excludes them.

Evaluation Metrics
------------------
Each raw feature score is evaluated against each binary clinical outcome using:

    - Distance correlation
    - Mutual information in nats

Metric safeguards:
    - Return 0.0 when fewer than three valid examples are available.
    - Return 0.0 when the score or label vector is constant or near-constant.
    - Return 0.0 for mutual information when the target has fewer than two classes.
    - Z-score scores within valid rows before mutual information estimation.
    - Use a fixed EVAL_SEED for mutual information estimation.
    - Use MI_BASE_NEIGHBORS=5, capped at valid sample size minus one.

Supported Dataset
-----------------
CORE is the only supported dataset.

For CORE, split files are loaded from:

    <data_root>/core/admissions_core_<split>.csv

For example:

    <data_root>/core/admissions_core_test.csv
    <data_root>/core/admissions_core_train.csv
    <data_root>/core/admissions_core_val.csv

Key Configuration
-----------------
  EVAL_SEED=12345
  MI_BASE_NEIGHBORS=5

Default command-line arguments:
  --dataset ALL
  --splits test
  --out_csv ""

Expected Directory Layout
--------------------------
    <data_root>/
    └── core/
        ├── admissions_core_train.csv
        ├── admissions_core_val.csv
        └── admissions_core_test.csv

Required admission columns:
    cci, eci_vw_total

Expected clinical target columns:
    mortality, mortality_30d, long_stay, long_stay_defined, icu_transfer

Outputs
-------
The script prints a table titled:

    CCI/ECI vs Binary Labels (per split)

Each output row contains:

    - dataset
    - split
    - feature
    - target
    - dcor
    - mi_nats

If --out_csv is provided, the same table is written to the specified CSV path.

Dependencies
------------
    pip install numpy pandas scipy scikit-learn dcor

Running
-------
Evaluate the CORE test split:

    python raw_cci_eci_feature_association_mimiciii.py --data_root /path/to/final_datasets

Evaluate train, validation, and test splits:

    python raw_cci_eci_feature_association_mimiciii.py \
        --data_root /path/to/final_datasets \
        --splits train,val,test

Save results to CSV:

    python raw_cci_eci_feature_association_mimiciii.py \
        --data_root /path/to/final_datasets \
        --out_csv ./results/raw_cci_eci_metrics.csv

Example:
    python raw_cci_eci_feature_association_mimiciii.py --data_root ./data/final_datasets
"""

# ----------------
# IMPORTS
# ----------------
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import dcor  # pip install dcor
# ----------------
# CONSTANTS
# ----------------
EVAL_SEED = 12345
MI_BASE_NEIGHBORS = 5

# ----------------
# UTILITY HELPERS
# ----------------
def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)

# ----------------
# DATA LOADING
# ---------------
def valid_mask_for_target(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "long_stay" and "long_stay_defined" in df.columns:
        return df["long_stay_defined"].fillna(0).astype(int).to_numpy(dtype=bool)

    s = pd.to_numeric(df[target], errors="coerce")
    return (~s.isna()).to_numpy(dtype=bool)


def y_for_target(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "long_stay" and "long_stay_defined" in df.columns:
        return df["long_stay"].fillna(0).astype(int).to_numpy()

    s = pd.to_numeric(df[target], errors="coerce")
    return s.fillna(0).astype(int).to_numpy()

def load_split(data_root: Path, dataset: str, split: str) -> pd.DataFrame:
    if dataset == "CORE":
        p = data_root / "core" / f"admissions_core_{split}.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not p.exists():
        raise FileNotFoundError(f"Missing admissions file: {p}")
    return pd.read_csv(p)

# ----------------
# CORRELATIONS
# ---------------
def mi_nats(score: np.ndarray, y: np.ndarray, valid: np.ndarray, seed: int = EVAL_SEED) -> float:
    valid = valid.astype(bool)
    if valid.sum() < 3:
        return 0.0
    yy = y[valid].astype(int)
    if np.unique(yy).size < 2:
        return 0.0

    nn = int(min(MI_BASE_NEIGHBORS, valid.sum() - 1))
    if nn < 1:
        return 0.0

    s_valid = zscore(score[valid]).reshape(-1, 1)
    return float(mutual_info_classif(s_valid, yy, random_state=seed, n_neighbors=nn)[0])



def dist_corr(score: np.ndarray, y: np.ndarray, valid: np.ndarray) -> float:
    valid = valid.astype(bool)
    if valid.sum() < 3:
        return 0.0
    ss = score[valid].astype(float)
    yy = y[valid].astype(float)
    return float(dcor.distance_correlation(ss, yy))


# ----------------
# MAIN COMPUTE
# ---------------
def compute_metrics(score: np.ndarray, y: np.ndarray, valid: np.ndarray) -> dict:
    valid = valid.astype(bool)
    if valid.sum() < 3:
        return {
            "dcor": 0.0,
            "mi_nats": 0.0,
        }

    return {
        "dcor": dist_corr(score, y, valid),
        "mi_nats": mi_nats(score, y, valid),
    }

# ----------------
# MAIN PIPELINE
# ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to final_datasets directory containing core/admissions_core_train.csv, etc.",
    )
    ap.add_argument("--dataset", type=str, default="ALL", choices=["ALL", "CORE"])
    ap.add_argument("--splits", type=str, default="test",
                    help="comma-separated, e.g. test or train,val,test")

    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    datasets = ["CORE"] if args.dataset == "ALL" else [args.dataset]

    rows = []

    for ds in datasets:
        for sp in splits:
            df = load_split(data_root, ds, sp)

            base_targets = ["mortality", "mortality_30d", "long_stay", "icu_transfer"]
            targets = [t for t in (base_targets) if t in df.columns]

            if "cci" not in df.columns or "eci_vw_total" not in df.columns:
                raise ValueError(f"{ds}/{sp} missing cci or eci_vw_total columns.")

            features = {
                "cci_raw": df["cci"].fillna(0).to_numpy(dtype=float),
                "eci_raw": df["eci_vw_total"].fillna(0).to_numpy(dtype=float),
            }

            for tgt in targets:
                valid = valid_mask_for_target(df, tgt)
                y = y_for_target(df, tgt)

                for feat_name, score in features.items():
                    met = compute_metrics(score, y, valid)
                    rows.append({
                        "dataset": ds,
                        "split": sp,
                        "feature": feat_name,
                        "target": tgt,
                        **met
                    })

    out = pd.DataFrame(rows)

    out = out.sort_values(["dataset", "split", "feature", "target"]).reset_index(drop=True)

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 160)

    print("\n=== CCI/ECI vs Binary Labels (per split) ===")
    print(out.to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\n✅ wrote results to: {out_path}")


if __name__ == "__main__":
    main()