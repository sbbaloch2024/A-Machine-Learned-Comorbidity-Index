"""
MIMIC-III ICD-9 Final Dataset Creator
=====================================

Overview
--------
This script creates the final MIMIC-III ICD-9 CORE dataset used for downstream
clinical outcome modeling. It merges admission-level binary labels, comorbidity
baseline scores, and deterministic patient-disjoint train/validation/test splits.
It also writes split-specific ICD-9 diagnosis-code files for each split.

The final CORE admission table is anchored on the in-hospital mortality label
file. Additional labels and baseline scalar features are merged by subject_id and
hadm_id. Diagnosis features are created by splitting the raw MIMIC-III
DIAGNOSES_ICD table according to the admission split assignment.

The train/validation/test split is deterministic and patient-disjoint. Each
subject_id is assigned to exactly one split using a stable MD5 hash, ensuring that
all admissions for the same patient remain in the same split.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-III root directory containing
          DIAGNOSES_ICD.csv or DIAGNOSES_ICD.csv.gz
        - --dataset_creation_root: path to the DatasetCreation directory
          containing binaryLabels, charlson, elixhauser, and final_datasets

2.  Define DatasetCreation paths:
        - binaryLabels/
        - final_datasets/
        - charlson/
        - elixhauser/

3.  Define required CORE label files:
        - label_mortality_inhouse_icd9_only.csv
        - label_mortality_30d_icd9_only.csv
        - label_long_stay_icd9_only.csv
        - label_icu_transfer_icd9_only.csv

4.  Define required comorbidity baseline files:
        - baseline_charlson_mimiciii_icd9.csv
        - baseline_elixhauser_vw_mimiciii_icd9.csv

5.  Define raw diagnosis source:
        - DIAGNOSES_ICD.csv or DIAGNOSES_ICD.csv.gz

6.  Load all label files with flexible CSV loading:
        - Try .csv.gz first.
        - If the compressed file is not found, try .csv.
        - Raise FileNotFoundError if neither exists.

7.  Anchor the CORE cohort on the in-hospital mortality file:
        - Keep subject_id, hadm_id, and age.
        - Drop duplicate admission rows.
        - Assign each row to a deterministic patient-level split.

8.  Assign patient-disjoint splits:
        - Hash subject_id using MD5.
        - Convert the hash to a value in [0, 1).
        - Assign train if value < 0.70.
        - Assign validation if value is in [0.70, 0.80).
        - Assign test otherwise.
        - This produces approximately 70% train, 10% validation, and 20% test.

9.  Load CCI and ECI scalar files:
        - cci from the Charlson baseline file.
        - eci_vw_total from the Elixhauser Van Walraven baseline file.

10. Merge comorbidity scalar features:
        - Merge cci by subject_id and hadm_id.
        - Merge eci_vw_total by subject_id and hadm_id.
        - Fill missing cci with 0 and cast to int.
        - Fill missing eci_vw_total with 0 and cast to int.

11. Merge clinical outcome labels:
        - mortality
        - mortality_30d
        - in_hosp_death
        - post_discharge_30d_death
        - los_days
        - long_stay
        - icu_any
        - time_to_icu_hours
        - icu_transfer

12. Fill labels where missing means no event:
        - mortality
        - mortality_30d
        - icu_any
        - icu_transfer

13. Preserve missingness where missing is meaningful:
        - los_days
        - long_stay
        - time_to_icu_hours

14. Create long_stay_defined:
        - long_stay_defined = 1 if long_stay is not missing
        - long_stay_defined = 0 otherwise

15. Write split-specific CORE admission files:
        - admissions_core_train.csv
        - admissions_core_val.csv
        - admissions_core_test.csv

16. Build hadm_id sets for each split:
        - train hadm_id set
        - validation hadm_id set
        - test hadm_id set

17. Split raw MIMIC-III ICD-9 diagnoses by admission split:
        - Load DIAGNOSES_ICD.csv or DIAGNOSES_ICD.csv.gz in chunks.
        - Keep HADM_ID and ICD9_CODE.
        - Convert HADM_ID to integer.
        - Rename columns to hadm_id and icd_code.
        - Normalize ICD-9 codes by uppercasing, removing periods/spaces, and
          stripping whitespace.
        - Drop empty diagnosis codes.
        - Write rows to the split file matching the hadm_id assignment.

18. Write split-specific diagnosis files:
        - diagnoses_icd9_core_train.csv.gz
        - diagnoses_icd9_core_val.csv.gz
        - diagnoses_icd9_core_test.csv.gz

19. Print completion message and final dataset directory.

Split Definition
----------------
The split is deterministic and patient-disjoint. The split assignment is computed
from subject_id:

    h = md5(str(subject_id))
    u = (int(h, 16) % 10000) / 10000.0

Then:

    train if u < 0.70
    val   if 0.70 <= u < 0.80
    test  if u >= 0.80

Because the split is based on subject_id, all admissions for a patient are placed
in the same split.

CORE Cohort Definition
----------------------
The CORE admission cohort is anchored on:

    label_mortality_inhouse_icd9_only.csv

The script keeps subject_id, hadm_id, and age from this file, drops duplicates,
and then merges additional labels and baseline features.

Merged Labels
-------------
The final admission table includes the following clinical labels when available:

    - mortality
    - mortality_30d
    - in_hosp_death
    - post_discharge_30d_death
    - los_days
    - long_stay
    - long_stay_defined
    - icu_any
    - time_to_icu_hours
    - icu_transfer

Missing values are handled differently depending on the field. For event labels
where missing means no recorded event, values are filled with zero. For
length-of-stay-related fields, missingness is preserved and tracked using
long_stay_defined.

Merged Baseline Features
------------------------
The final admission table includes:

    - cci
    - eci_vw_total

Missing comorbidity scores are filled with zero and cast to integer.

Diagnosis Processing
--------------------
The raw MIMIC-III DIAGNOSES_ICD table is split according to the final admission
splits. Diagnosis codes are normalized before writing:

    icd_code = uppercase(ICD9_CODE)
               .replace(".", "")
               .replace(" ", "")
               .strip()

Rows with missing or empty diagnosis codes are dropped. The split-specific
diagnosis files contain only:

    - hadm_id
    - icd_code

Expected Input Directory Layout
-------------------------------
MIMIC-III root:

    <data_root>/
    ├── DIAGNOSES_ICD.csv.gz

or:

    <data_root>/
    ├── DIAGNOSES_ICD.csv

DatasetCreation root:

    <dataset_creation_root>/
    ├── binaryLabels/
    │   ├── label_mortality_inhouse_icd9_only.csv
    │   ├── label_mortality_30d_icd9_only.csv
    │   ├── label_long_stay_icd9_only.csv
    │   └── label_icu_transfer_icd9_only.csv
    ├── charlson/
    │   └── baseline_charlson_mimiciii_icd9.csv
    ├── elixhauser/
    │   └── baseline_elixhauser_vw_mimiciii_icd9.csv
    └── final_datasets/

Required Columns
----------------
From label_mortality_inhouse_icd9_only.csv:
    subject_id, hadm_id, age, mortality

From label_mortality_30d_icd9_only.csv:
    subject_id, hadm_id, mortality_30d, in_hosp_death,
    post_discharge_30d_death

From label_long_stay_icd9_only.csv:
    subject_id, hadm_id, los_days, long_stay

From label_icu_transfer_icd9_only.csv:
    subject_id, hadm_id, icu_any, time_to_icu_hours, icu_transfer

From baseline_charlson_mimiciii_icd9.csv:
    subject_id, hadm_id, cci

From baseline_elixhauser_vw_mimiciii_icd9.csv:
    subject_id, hadm_id, eci_vw_total

From DIAGNOSES_ICD.csv(.gz):
    HADM_ID, ICD9_CODE

Outputs
-------
The script writes final admission split files to:

    <dataset_creation_root>/final_datasets/core/

Admission outputs:

    - admissions_core_train.csv
    - admissions_core_val.csv
    - admissions_core_test.csv

Diagnosis outputs:

    - diagnoses_icd9_core_train.csv.gz
    - diagnoses_icd9_core_val.csv.gz
    - diagnoses_icd9_core_test.csv.gz

The script prints:

    - loaded label and feature status messages
    - CORE cohort size
    - written admission split paths and row counts
    - written diagnosis split paths
    - final dataset directory

Dependencies
------------
    pip install numpy pandas

Running
-------
Run the script with:

    python createFinalDataSet_mimiciii.py \
        --data_root /path/to/mimic-iii \
        --dataset_creation_root /path/to/DatasetCreation

"""

import os
import argparse
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to MIMIC-III root directory containing DIAGNOSES_ICD.csv(.gz).",
    )
    parser.add_argument(
        "--dataset_creation_root",
        type=str,
        required=True,
        help="Path to DatasetCreation directory containing binaryLabels, charlson, elixhauser, and final_datasets.",
    )
    return parser.parse_args()


ARGS = parse_args()


# ============================================================
# CONFIG (MIMIC-III + ICD-9)
# ============================================================

ROOT = Path(ARGS.dataset_creation_root)

LABEL_DIR = ROOT / "binaryLabels"
FINAL_DIR = ROOT / "final_datasets"

# ---- Core ICD-9 label files ----
PATH_MORT_INHOUSE = LABEL_DIR / "label_mortality_inhouse_icd9_only.csv"
PATH_MORT_30D     = LABEL_DIR / "label_mortality_30d_icd9_only.csv"
PATH_LONG_STAY    = LABEL_DIR / "label_long_stay_icd9_only.csv"
PATH_ICU_TRANSFER = LABEL_DIR / "label_icu_transfer_icd9_only.csv"

# ---- Comorbidity scalars (ICD-9 cohort) ----
PATH_CCI = ROOT / "charlson"   / "baseline_charlson_mimiciii_icd9.csv"
PATH_ECI = ROOT / "elixhauser" / "baseline_elixhauser_vw_mimiciii_icd9.csv"

# ---- Diagnoses sources for features ----
# CORE features: raw MIMIC-III diagnoses (ICD-9)
DATA_ROOT = Path(ARGS.data_root)
CORE_DIAGNOSES_RAW = DATA_ROOT / "DIAGNOSES_ICD.csv"

# Patient-disjoint split fractions (hash-based deterministic)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

CHUNK_DX = 2_000_000


# ============================================================
# Helpers
# ============================================================

def stable_split(subject_id: int) -> str:
    """Deterministic split assignment based on hashing subject_id."""
    h = hashlib.md5(str(int(subject_id)).encode("utf-8")).hexdigest()
    u = (int(h, 16) % 10_000) / 10_000.0
    if u < TRAIN_FRAC:
        return "train"
    if u < TRAIN_FRAC + VAL_FRAC:
        return "val"
    return "test"


def read_csv_flexible(path_with_csv: Path, usecols=None, dtype=None) -> pd.DataFrame:
    """Try .csv.gz then .csv."""
    gz = Path(str(path_with_csv) + ".gz") if str(path_with_csv).endswith(".csv") else Path(str(path_with_csv) + ".csv.gz")
    if gz.exists():
        return pd.read_csv(gz, usecols=usecols, dtype=dtype)
    if path_with_csv.exists():
        return pd.read_csv(path_with_csv, usecols=usecols, dtype=dtype)
    raise FileNotFoundError(f"Could not find {gz} or {path_with_csv}")


def write_splits(df: pd.DataFrame, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].copy()
        out_path = out_dir / f"{stem}_{split}.csv"
        part.to_csv(out_path, index=False)
        print(f"✅ wrote {out_path}  (n={len(part):,})")


def split_diagnoses_by_hadm_icd9(
    diagnoses_path: Path,
    hadm_by_split: dict,
    out_dir: Path,
    out_stem: str,
    chunksize: int = CHUNK_DX,
    hadm_col_in: str = "hadm_id",
    code_col_in: str = "icd_code",
    hadm_col_out: str = "hadm_id",
    code_col_out: str = "icd_code",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    out_files = {s: out_dir / f"{out_stem}_{s}.csv.gz" for s in ["train", "val", "test"]}
    for s in out_files:
        if out_files[s].exists():
            out_files[s].unlink()

    wrote_header = {s: False for s in ["train", "val", "test"]}
    usecols = [hadm_col_in, code_col_in]

    dx_path = diagnoses_path
    dx_gz_path = Path(str(diagnoses_path) + ".gz")

    if dx_gz_path.exists():
        dx_path = dx_gz_path
    elif not diagnoses_path.exists():
        raise FileNotFoundError(f"Could not find {diagnoses_path} or {dx_gz_path}")

    for chunk in pd.read_csv(dx_path, usecols=usecols, chunksize=chunksize, dtype={code_col_in: str}):

        # hadm_id -> int
        chunk[hadm_col_in] = pd.to_numeric(chunk[hadm_col_in], errors="coerce")
        chunk = chunk.dropna(subset=[hadm_col_in]).copy()
        chunk[hadm_col_in] = chunk[hadm_col_in].astype(int)

        # rename to standard output names
        chunk = chunk.rename(columns={hadm_col_in: hadm_col_out, code_col_in: code_col_out})

        chunk[code_col_out] = (
            chunk[code_col_out].fillna("")
            .astype(str).str.upper()
            .str.replace(".", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
        )
        chunk = chunk[chunk[code_col_out] != ""]

        for split in ["train", "val", "test"]:
            keep = chunk[hadm_col_out].isin(hadm_by_split[split])
            if not keep.any():
                continue

            chunk.loc[keep, [hadm_col_out, code_col_out]].to_csv(
                out_files[split],
                index=False,
                compression="gzip",
                mode="a",
                header=(not wrote_header[split]),
            )
            wrote_header[split] = True

    for split in ["train", "val", "test"]:
        print(f"✅ wrote diagnoses: {out_files[split]}")


# ============================================================
# 1) Build CORE master admission table (ICD-9 cohort)
# ============================================================

print("\n=== Loading CORE labels (MIMIC-III ICD-9) ===")
mort_in = read_csv_flexible(PATH_MORT_INHOUSE)   # subject_id, hadm_id, age, mortality
mort30  = read_csv_flexible(PATH_MORT_30D)       # subject_id, hadm_id, mortality_30d, ...
los     = read_csv_flexible(PATH_LONG_STAY)      # subject_id, hadm_id, los_days, long_stay
icu     = read_csv_flexible(PATH_ICU_TRANSFER)   # subject_id, hadm_id, icu_transfer, ...

# Anchor cohort on mortality_inhouse file (should cover ICD-9 admissions cohort)
core = mort_in[["subject_id", "hadm_id", "age"]].drop_duplicates().copy()
core["split"] = core["subject_id"].apply(stable_split)

print("\n=== Loading CCI/ECI scalars ===")
cci = read_csv_flexible(PATH_CCI, usecols=["subject_id", "hadm_id", "cci"])
eci = read_csv_flexible(PATH_ECI, usecols=["subject_id", "hadm_id", "eci_vw_total"])

core = core.merge(cci, on=["subject_id", "hadm_id"], how="left")
core = core.merge(eci, on=["subject_id", "hadm_id"], how="left")

core["cci"] = core["cci"].fillna(0).astype(int)
core["eci_vw_total"] = core["eci_vw_total"].fillna(0).astype(int)

# Merge core labels
core = core.merge(
    mort_in[["subject_id", "hadm_id", "mortality"]],
    on=["subject_id", "hadm_id"],
    how="left",
)

core = core.merge(
    mort30[["subject_id", "hadm_id", "mortality_30d", "in_hosp_death", "post_discharge_30d_death"]],
    on=["subject_id", "hadm_id"],
    how="left",
)

core = core.merge(
    los[["subject_id", "hadm_id", "los_days", "long_stay"]],
    on=["subject_id", "hadm_id"],
    how="left",
)

core = core.merge(
    icu[["subject_id", "hadm_id", "icu_any", "time_to_icu_hours", "icu_transfer"]],
    on=["subject_id", "hadm_id"],
    how="left",
)

# Fill where "missing means 0" is correct
for col in ["mortality", "mortality_30d", "icu_any", "icu_transfer"]:
    if col in core.columns:
        core[col] = core[col].fillna(0).astype(int)

# LOS/long_stay: NaN is meaningful (missing dischtime etc.), keep it
core["long_stay_defined"] = core["long_stay"].notna().astype(int)

print("CORE cohort size:", len(core))


# ============================================================
# 2) Write CORE dataset + ICD-9 diagnoses per split
# ============================================================

print("\n=== Writing CORE datasets ===")
core_dir = FINAL_DIR / "core"
write_splits(core, core_dir, stem="admissions_core")

hadm_by_split_core = {
    s: set(core.loc[core["split"] == s, "hadm_id"].astype(int).tolist())
    for s in ["train", "val", "test"]
}

split_diagnoses_by_hadm_icd9(
    CORE_DIAGNOSES_RAW,
    hadm_by_split=hadm_by_split_core,
    out_dir=core_dir,
    out_stem="diagnoses_icd9_core",
    hadm_col_in="HADM_ID",
    code_col_in="ICD9_CODE",
    hadm_col_out="hadm_id",
    code_col_out="icd_code",
)

print("\n✅ DONE. Final dataset written under:", FINAL_DIR)