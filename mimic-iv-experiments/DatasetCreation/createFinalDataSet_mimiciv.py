"""
Final ICD-10 CORE Dataset Builder
=================================

Overview
--------
This script builds the final ICD-10-only CORE dataset used by downstream
comorbidity and clinical outcome experiments. It combines previously generated
binary clinical labels, comorbidity baseline scores, and raw ICD-10 diagnosis
features into deterministic train/validation/test splits.

The script anchors the CORE admission cohort on the ICD-10 in-hospital mortality
label file, which is expected to contain all ICD-10 admissions. It then merges
30-day mortality, long-stay, ICU transfer, Charlson Comorbidity Index (CCI), and
Elixhauser Van Walraven Index (ECI-VW) features. Splits are assigned at the
patient level using a deterministic hash of subject_id, ensuring that admissions
from the same patient do not cross train, validation, and test splits.

Finally, the script writes split-specific admission CSV files and split-specific
ICD-10 diagnosis CSV files under final_datasets/core.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the mimic-iv-experiments/DatasetCreation directory
        - --mimic_root: path to the MIMIC-IV root directory containing
          hosp/diagnoses_icd.csv.gz

2.  Resolve input directories:
        - binaryLabels/
        - charlson/
        - elixhauser/
        - final_datasets/

3.  Define required CORE label files:
        - label_mortality_inhouse_icd10_only.csv
        - label_mortality_30d_icd10_only.csv
        - label_long_stay_icd10_only.csv
        - label_icu_transfer_icd10_only.csv

4.  Define required comorbidity score files:
        - baseline_charlson_mimiciv_icd10.csv
        - baseline_elixhauser_vw_mimiciv_icd10.csv

5.  Define raw diagnosis source:
        - <mimic_root>/hosp/diagnoses_icd.csv.gz

6.  Load CORE label files:
        - in-hospital mortality
        - 30-day mortality
        - long stay
        - ICU transfer

7.  Anchor the CORE cohort:
        - Use subject_id, hadm_id, and age from the in-hospital mortality label
          file.
        - Drop duplicate admission rows.
        - Treat this file as the master ICD-10 admission list.

8.  Assign deterministic patient-disjoint splits:
        - Hash subject_id with MD5.
        - Convert the hash to a value in [0, 1).
        - Assign train if value < 0.70.
        - Assign val if 0.70 <= value < 0.80.
        - Assign test otherwise.
        - Because splitting is based on subject_id, all admissions for the same
          patient are assigned to the same split.

9.  Load comorbidity scalar files:
        - CCI from the Charlson output file.
        - ECI-VW total score from the Elixhauser output file.

10. Merge comorbidity scores into the CORE table:
        - Merge by subject_id and hadm_id.
        - Fill missing cci values with 0.
        - Fill missing eci_vw_total values with 0.
        - Cast both scores to integer.

11. Merge clinical labels into the CORE table:
        - mortality
        - mortality_30d
        - in_hosp_death
        - post_discharge_30d_death
        - los_days
        - long_stay
        - icu_any
        - time_to_icu_hours
        - icu_transfer

12. Fill labels where missing means negative:
        - mortality
        - mortality_30d
        - icu_any
        - icu_transfer

13. Preserve missingness for long-stay:
        - Do not fill missing long_stay or los_days values.
        - Create long_stay_defined = 1 when long_stay is present.
        - Create long_stay_defined = 0 when long_stay is missing.
        - This allows downstream models to mask admissions without valid
          length-of-stay labels.

14. Write split-specific admission CSV files:
        - admissions_core_train.csv
        - admissions_core_val.csv
        - admissions_core_test.csv

15. Build hadm_id sets for each split:
        - Collect train hadm_id values.
        - Collect validation hadm_id values.
        - Collect test hadm_id values.

16. Write split-specific ICD-10 diagnosis files:
        - Read raw MIMIC-IV diagnoses in chunks.
        - Keep only rows where icd_version == 10.
        - Route diagnosis rows into train, validation, or test based on hadm_id.
        - Write gzip-compressed diagnosis files for each split.

17. Print output paths and completion message.

Patient-Disjoint Split Logic
----------------------------
Splits are deterministic and based only on subject_id:

    h = md5(str(subject_id))
    u = (int(h, 16) % 10000) / 10000.0

The split assignment is:

    train if u < 0.70
    val   if 0.70 <= u < 0.80
    test  if u >= 0.80

This produces approximate 70/10/20 splits while ensuring that all admissions for
the same subject_id remain in the same split.

CORE Admission Cohort
---------------------
The CORE cohort is anchored on:

    label_mortality_inhouse_icd10_only.csv

This file is expected to contain the full ICD-10 admission cohort. The final CORE
admission table starts from:

    subject_id, hadm_id, age

and then merges all other labels and baseline scalar features by subject_id and
hadm_id.

Merged Labels and Features
--------------------------
The final CORE admission table includes:

    - subject_id
    - hadm_id
    - age
    - split
    - cci
    - eci_vw_total
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

Missing-Value Handling
----------------------
Missing comorbidity scores are filled as zero:

    cci = 0
    eci_vw_total = 0

Missing values are treated as negative for labels where absence means no event:

    mortality = 0
    mortality_30d = 0
    icu_any = 0
    icu_transfer = 0

Missing long_stay values are preserved because missing long_stay indicates that
length of stay could not be computed or was not available. The script creates:

    long_stay_defined = 1[long_stay is not missing]

Downstream models should use long_stay_defined as the validity mask for
long_stay.

Diagnosis Split Logic
---------------------
Diagnosis rows are split by hadm_id to match the admission splits.

The script reads the raw MIMIC-IV diagnoses table in chunks of 2,000,000 rows.
For CORE features, it keeps only:

    icd_version == 10

and writes rows into the corresponding split file if the diagnosis hadm_id belongs
to that split's admission set.

Expected Directory Layout
--------------------------
Input DatasetCreation directory:

    <data_root>/
    ├── binaryLabels/
    │   ├── label_mortality_inhouse_icd10_only.csv
    │   ├── label_mortality_30d_icd10_only.csv
    │   ├── label_long_stay_icd10_only.csv
    │   └── label_icu_transfer_icd10_only.csv
    ├── charlson/
    │   └── baseline_charlson_mimiciv_icd10.csv
    ├── elixhauser/
    │   └── baseline_elixhauser_vw_mimiciv_icd10.csv
    └── final_datasets/

Input MIMIC-IV directory:

    <mimic_root>/
    └── hosp/
        └── diagnoses_icd.csv.gz

Output Directory Layout
-----------------------
The script writes final datasets under:

    <data_root>/final_datasets/core/

Admission outputs:

    admissions_core_train.csv
    admissions_core_val.csv
    admissions_core_test.csv

Diagnosis outputs:

    diagnoses_icd10_core_train.csv.gz
    diagnoses_icd10_core_val.csv.gz
    diagnoses_icd10_core_test.csv.gz

Required columns
----------------
From label_mortality_inhouse_icd10_only.csv:
    subject_id, hadm_id, age, mortality

From label_mortality_30d_icd10_only.csv:
    subject_id, hadm_id, mortality_30d, in_hosp_death,
    post_discharge_30d_death

From label_long_stay_icd10_only.csv:
    subject_id, hadm_id, los_days, long_stay

From label_icu_transfer_icd10_only.csv:
    subject_id, hadm_id, icu_any, time_to_icu_hours, icu_transfer

From baseline_charlson_mimiciv_icd10.csv:
    subject_id, hadm_id, cci

From baseline_elixhauser_vw_mimiciv_icd10.csv:
    subject_id, hadm_id, eci_vw_total

From MIMIC-IV hosp/diagnoses_icd.csv.gz:
    hadm_id, icd_code, icd_version

Outputs
-------
The script writes:

    - split-specific CORE admission CSV files
    - split-specific CORE ICD-10 diagnosis CSV.GZ files

The script prints:

    - label-loading progress
    - comorbidity-loading progress
    - CORE cohort size
    - admission split output paths and row counts
    - diagnosis split output paths
    - final completion message

Dependencies
------------
    pip install pandas

Running
-------
Run the script with:

    python createFinalDataSet_mimiciv.py \
        --data_root /path/to/mimic-iv-experiments/DatasetCreation \
        --mimic_root /path/to/mimic-iv


"""

import argparse
import os
import hashlib
import pandas as pd
from pathlib import Path


# PLEASE ADJUST THE BASE PATH TO CORRECTLY IMPORT THE DATASETS
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to mimic-iv-experiments/DatasetCreation directory.",
    )
    parser.add_argument(
        "--mimic_root",
        type=str,
        required=True,
        help="Path to MIMIC-IV root directory containing hosp/diagnoses_icd.csv.gz.",
    )
    return parser.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.data_root)
MIMIC_ROOT = Path(ARGS.mimic_root)

LABEL_DIR = ROOT / "binaryLabels"
FINAL_DIR = ROOT / "final_datasets"

# ---- Core ICD-10-only label files ----
PATH_MORT_INHOUSE = LABEL_DIR / "label_mortality_inhouse_icd10_only.csv"
PATH_MORT_30D     = LABEL_DIR / "label_mortality_30d_icd10_only.csv"
PATH_LONG_STAY    = LABEL_DIR / "label_long_stay_icd10_only.csv"
PATH_ICU_TRANSFER = LABEL_DIR / "label_icu_transfer_icd10_only.csv"

# ---- Comorbidity scalars (ICD-10 cohort) ----
PATH_CCI = ROOT / "charlson"   / "baseline_charlson_mimiciv_icd10.csv"        # cols: subject_id, hadm_id, cci
PATH_ECI = ROOT / "elixhauser" / "baseline_elixhauser_vw_mimiciv_icd10.csv"   # cols: subject_id, hadm_id, eci_vw_total


# ---- Diagnoses sources for features ----
# Core features: raw MIMIC diagnoses filtered to ICD-10 inside this script.
CORE_DIAGNOSES_RAW = MIMIC_ROOT / "hosp" / "diagnoses_icd.csv.gz"


# Patient-disjoint split fractions (hash-based deterministic)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9


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


def safe_read_csv(path: Path, usecols=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, usecols=usecols)


def write_splits(df: pd.DataFrame, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].copy()
        out_path = out_dir / f"{stem}_{split}.csv"
        part.to_csv(out_path, index=False)
        print(f"✅ wrote {out_path}  (n={len(part):,})")


def split_diagnoses_by_hadm(
    diagnoses_path: Path,
    hadm_by_split: dict,
    out_dir: Path,
    out_stem: str,
    raw_mimic_keep_icd10_only: bool,
    chunksize: int = 2_000_000,
):
    """
    Writes diagnoses_{split}.csv.gz by filtering rows to hadm_ids in each split.
    Chunked to avoid RAM blowups.
    If raw_mimic_keep_icd10_only=True, filters icd_version==10 and keeps columns hadm_id,icd_code,icd_version.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    out_files = {}
    for split in ["train", "val", "test"]:
        out_files[split] = out_dir / f"{out_stem}_{split}.csv.gz"
        if out_files[split].exists():
            out_files[split].unlink()

    usecols = ["hadm_id", "icd_code", "icd_version"] if raw_mimic_keep_icd10_only else None

    for chunk in pd.read_csv(diagnoses_path, usecols=usecols, chunksize=chunksize):
        if "hadm_id" not in chunk.columns:
            raise ValueError(f"diagnoses file missing hadm_id column: {diagnoses_path}")

        if raw_mimic_keep_icd10_only:
            chunk = chunk[chunk["icd_version"] == 10].copy()

        for split in ["train", "val", "test"]:
            keep = chunk["hadm_id"].isin(hadm_by_split[split])
            if keep.any():
                chunk.loc[keep].to_csv(
                    out_files[split],
                    index=False,
                    compression="gzip",
                    mode="a",
                    header=not out_files[split].exists() or os.path.getsize(out_files[split]) == 0,
                )

    for split in ["train", "val", "test"]:
        print(f"✅ wrote diagnoses: {out_files[split]}")


# ============================================================
# 1) Build CORE master admission table (ICD-10 cohort)
# ============================================================

print("\n=== Loading CORE labels ===")
mort_in = safe_read_csv(PATH_MORT_INHOUSE)   # subject_id, hadm_id, age, mortality
mort30  = safe_read_csv(PATH_MORT_30D)       # subject_id, hadm_id, mortality_30d, ...
los     = safe_read_csv(PATH_LONG_STAY)      # subject_id, hadm_id, los_days, long_stay
icu     = safe_read_csv(PATH_ICU_TRANSFER)   # subject_id, hadm_id, icu_transfer, ...

# Anchor cohort on ICD-10 admissions list (mortality_inhouse_icd10_only should have all ICD-10 admissions)
core = mort_in[["subject_id", "hadm_id", "age"]].drop_duplicates().copy()
core["split"] = core["subject_id"].apply(stable_split)

print("\n=== Loading CCI/ECI scalars ===")
cci = safe_read_csv(PATH_CCI, usecols=["subject_id", "hadm_id", "cci"])
eci = safe_read_csv(PATH_ECI, usecols=["subject_id", "hadm_id", "eci_vw_total"])

core = core.merge(cci, on=["subject_id", "hadm_id"], how="left")
core = core.merge(eci, on=["subject_id", "hadm_id"], how="left")

# If any missing (should be rare if cohorts match), fill with 0
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
# 2) Write CORE dataset + ICD-10 diagnoses per split (raw MIMIC -> ICD10)
# ============================================================

print("\n=== Writing CORE datasets ===")
core_dir = FINAL_DIR / "core"
write_splits(core, core_dir, stem="admissions_core")

hadm_by_split_core = {
    s: set(core.loc[core["split"] == s, "hadm_id"].astype(int).tolist())
    for s in ["train", "val", "test"]
}

split_diagnoses_by_hadm(
    CORE_DIAGNOSES_RAW,
    hadm_by_split=hadm_by_split_core,
    out_dir=core_dir,
    out_stem="diagnoses_icd10_core",
    raw_mimic_keep_icd10_only=True,
)

print("\n✅ DONE. Final datasets written under:", FINAL_DIR)