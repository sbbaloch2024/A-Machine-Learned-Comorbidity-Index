"""
ICD-10-Only Long-Stay Label Creator
===================================

Overview
--------
This script creates a binary long-stay label for hospital admissions in an
ICD-10-only MIMIC-IV cohort. The cohort is restricted to admissions that have at
least one ICD-10 diagnosis code in hosp/diagnoses_icd.csv.gz.

For each eligible admission, the script computes hospital length of stay from
admission and discharge timestamps. Admissions with valid length-of-stay values
are labeled as:

    - long_stay = 1 if LOS > 7 days
    - long_stay = 0 if LOS <= 7 days

The script also computes admission age using MIMIC-IV anchor age/year fields and
caps ages above 89 at 90, following standard MIMIC-IV deidentification
conventions.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-IV root directory containing the hosp/
          subdirectory
        - --out_path: output CSV path for the long-stay label file

2.  Resolve input paths:
        - hosp/admissions.csv.gz
        - hosp/patients.csv.gz
        - hosp/diagnoses_icd.csv.gz

3.  Load required MIMIC-IV tables:
        - admissions
        - patients
        - diagnoses_icd, using only hadm_id and icd_version

4.  Restrict the cohort to ICD-10 admissions:
        - Select hadm_id values from diagnoses_icd where icd_version == 10.
        - Keep only admissions whose hadm_id appears in that ICD-10 set.
        - This ensures consistency with the rest of the ICD-10-only pipeline.

5.  Parse hospital timestamps:
        - Convert admittime to datetime.
        - Convert dischtime to datetime.

6.  Merge patient anchor-age information:
        - Merge admissions with patients on subject_id.
        - Use anchor_age and anchor_year to approximate admission age.

7.  Compute admission age:
        - age = anchor_age + (admittime year - anchor_year)
        - Set age values greater than 89 to 90.

8.  Compute hospital length of stay:
        - los_days is computed as dischtime - admittime in days.
        - Admissions with missing or invalid LOS are dropped.

9.  Create the binary long-stay label:
        - long_stay = 1 if los_days > 7
        - long_stay = 0 otherwise

10. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save selected admission-level columns to --out_path.
        - Print the output path, cohort size, and normalized long_stay counts.

ICD-10 Cohort Definition
------------------------
The ICD-10-only cohort is defined using the diagnoses table:

    diagnoses_icd.icd_version == 10

An admission is included if its hadm_id appears at least once among ICD-10
diagnosis rows. The long-stay label itself does not depend on ICD codes; ICD-10 is
used only as a cohort restriction so that this label file aligns with an
ICD-10-only diagnosis-token modeling pipeline.

Long-Stay Label Definition
--------------------------
Length of stay is computed in days:

    los_days = (dischtime - admittime) / 24 hours

The binary long-stay label is:

    long_stay = 1[los_days > 7]

The threshold is strict. Admissions with exactly 7.0 days of hospital stay are
labeled as non-long-stay.

Validity Handling
-----------------
An admission is retained only if los_days is not missing. Because los_days depends
on both admittime and dischtime, admissions with missing or invalid timestamps are
removed before saving.

The saved output therefore contains only ICD-10 admissions with valid hospital
length of stay.

Age Calculation
---------------
Admission age is approximated using MIMIC-IV anchor fields:

    age = anchor_age + (admittime year - anchor_year)

Ages greater than 89 are set to 90:

    if age > 89:
        age = 90

This preserves the standard top-coding convention for older MIMIC-IV patients.

Expected Directory Layout
--------------------------
    <data_root>/
    └── hosp/
        ├── admissions.csv.gz
        ├── patients.csv.gz
        └── diagnoses_icd.csv.gz

Required columns
----------------
From hosp/admissions.csv.gz:
    subject_id, hadm_id, admittime, dischtime

From hosp/patients.csv.gz:
    subject_id, anchor_age, anchor_year

From hosp/diagnoses_icd.csv.gz:
    hadm_id, icd_version

Outputs
-------
The script writes one CSV file to --out_path.

Saved columns:

    - subject_id
    - hadm_id
    - age
    - los_days
    - long_stay

The script prints:

    - output file path
    - ICD-10 admission cohort size with valid LOS
    - normalized long_stay value counts

Dependencies
------------
    pip install pandas

Running
-------
Run the script with:

    python create_label_long_stay_mimiciv.py \
        --data_root /path/to/mimic-iv \
        --out_path /path/to/label_long_stay_icd10_only.csv


"""

import argparse
from pathlib import Path

import pandas as pd


# ----------------------------
# 1) Paths
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to MIMIC-IV root directory containing hosp/ subdirectory.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for label_long_stay_icd10_only.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

data_root = Path(ARGS.data_root)
base = data_root / "hosp"

out_path = Path(ARGS.out_path)


# ----------------------------
# 2) Load data
# ----------------------------
admissions = pd.read_csv(base / "admissions.csv.gz")
patients = pd.read_csv(base / "patients.csv.gz")

# Load diagnoses ONLY to define ICD-10 cohort
diagnoses = pd.read_csv(
    base / "diagnoses_icd.csv.gz",
    usecols=["hadm_id", "icd_version"],
)


# ----------------------------
# 3) Restrict to ICD-10 admissions
# ----------------------------
hadm_icd10 = diagnoses.loc[
    diagnoses["icd_version"] == 10,
    "hadm_id",
].dropna().unique()

admissions = admissions.loc[admissions["hadm_id"].isin(hadm_icd10)].copy()


# ----------------------------
# 4) Parse timestamps + compute age
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

admissions = admissions.merge(
    patients[["subject_id", "anchor_age", "anchor_year"]],
    on="subject_id",
    how="left",
)

admissions["age"] = admissions["anchor_age"] + (
    admissions["admittime"].dt.year - admissions["anchor_year"]
)
admissions.loc[admissions["age"] > 89, "age"] = 90


# ----------------------------
# 5) Compute LOS + label
# ----------------------------
admissions["los_days"] = (
    (admissions["dischtime"] - admissions["admittime"])
    .dt.total_seconds() / (3600.0 * 24.0)
)

# Keep only valid LOS (needs both times)
admissions = admissions.loc[admissions["los_days"].notna()].copy()

admissions["long_stay"] = (admissions["los_days"] > 7).astype(int)


# ----------------------------
# 6) Save
# ----------------------------
cols = ["subject_id", "hadm_id", "age", "los_days", "long_stay"]

out_path.parent.mkdir(parents=True, exist_ok=True)
admissions[cols].to_csv(out_path, index=False)

print(f"✅ Long-stay (ICD-10-only cohort) labels saved to: {out_path}")
print("Cohort size (ICD-10 admissions with valid LOS):", len(admissions))
print(admissions["long_stay"].value_counts(normalize=True).round(3))