"""
ICD-10-Only In-Hospital Mortality Label Creator
===============================================

Overview
--------
This script creates an in-hospital mortality label for hospital admissions in an
ICD-10-only MIMIC-IV cohort. The cohort is restricted to admissions that have at
least one ICD-10 diagnosis code in hosp/diagnoses_icd.csv.gz.

For each eligible admission, the script labels in-hospital mortality using the
admission-level deathtime field:

    - mortality = 1 if deathtime is present
    - mortality = 0 if deathtime is missing

The script also computes admission age using MIMIC-IV anchor age/year fields and
caps ages above 89 at 90, following standard MIMIC-IV deidentification
conventions.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-IV root directory containing the hosp/
          subdirectory
        - --out_path: output CSV path for the in-hospital mortality label file

2.  Resolve input paths:
        - hosp/diagnoses_icd.csv.gz
        - hosp/admissions.csv.gz
        - hosp/patients.csv.gz

3.  Load ICD-10 cohort identifiers:
        - Read hadm_id and icd_version from diagnoses_icd.
        - Select hadm_id values where icd_version == 10.

4.  Load admissions and patients tables.

5.  Restrict the admissions cohort:
        - Keep only admissions whose hadm_id appears in the ICD-10 admission set.
        - This aligns the mortality label file with an ICD-10-only diagnosis-token
          modeling pipeline.

6.  Create the in-hospital mortality label:
        - mortality = 1 if admissions.deathtime is not null.
        - mortality = 0 if admissions.deathtime is null.

7.  Parse admission time:
        - Convert admittime to datetime.

8.  Merge patient anchor-age information:
        - Merge admissions with patients on subject_id.
        - Include anchor_age and anchor_year.

9.  Compute admission age:
        - age = anchor_age + (admittime year - anchor_year)
        - Set age values greater than 89 to 90.

10. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save subject_id, hadm_id, age, and mortality.
        - Print the output path, cohort size, and normalized mortality counts.

ICD-10 Cohort Definition
------------------------
The ICD-10-only cohort is defined using the diagnoses table:

    diagnoses_icd.icd_version == 10

An admission is included if its hadm_id appears at least once among ICD-10
diagnosis rows. The mortality label itself does not depend on ICD codes; ICD-10 is
used only as a cohort restriction so that this label file aligns with an
ICD-10-only diagnosis-token modeling pipeline.

In-Hospital Mortality Label Definition
--------------------------------------
The in-hospital mortality label is derived from the admission-level deathtime
field:

    mortality = 1[deathtime is not null]

This captures deaths recorded during the hospital admission. It does not include
post-discharge mortality unless that death is reflected in the admission-level
deathtime field.

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
        ├── diagnoses_icd.csv.gz
        ├── admissions.csv.gz
        └── patients.csv.gz

Required columns
----------------
From hosp/diagnoses_icd.csv.gz:
    hadm_id, icd_version

From hosp/admissions.csv.gz:
    subject_id, hadm_id, admittime, deathtime

From hosp/patients.csv.gz:
    subject_id, anchor_age, anchor_year

Outputs
-------
The script writes one CSV file to --out_path.

Saved columns:

    - subject_id
    - hadm_id
    - age
    - mortality

The script prints:

    - output file path
    - ICD-10 cohort size
    - normalized mortality value counts

Dependencies
------------
    pip install pandas

Running
-------
Run the script with:

    python create_label_mortality_inhouse_mimiciv.py \
        --data_root /path/to/mimic-iv \
        --out_path /path/to/label_mortality_inhouse_icd10_only.csv


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
        help="Output CSV path for label_mortality_inhouse_icd10_only.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

data_root = Path(ARGS.data_root)
base = data_root / "hosp"

out_path = Path(ARGS.out_path)


# ----------------------------
# 2) Load data
# ----------------------------
# ICD-10 cohort hadm_ids
diagnoses = pd.read_csv(
    base / "diagnoses_icd.csv.gz",
    usecols=["hadm_id", "icd_version"],
)
hadm_icd10 = diagnoses.loc[
    diagnoses["icd_version"] == 10,
    "hadm_id",
].dropna().unique()

admissions = pd.read_csv(base / "admissions.csv.gz")
patients = pd.read_csv(base / "patients.csv.gz")


# ----------------------------
# 3) Restrict cohort
# ----------------------------
admissions = admissions.loc[admissions["hadm_id"].isin(hadm_icd10)].copy()


# ----------------------------
# 4) Label
# ----------------------------
admissions["mortality"] = admissions["deathtime"].notnull().astype(int)


# ----------------------------
# 5) Age
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")

admissions = admissions.merge(
    patients[["subject_id", "anchor_age", "anchor_year"]],
    on="subject_id",
    how="left",
)

admissions["age"] = (
    admissions["anchor_age"]
    + (admissions["admittime"].dt.year - admissions["anchor_year"])
)
admissions.loc[admissions["age"] > 89, "age"] = 90


# ----------------------------
# 6) Save
# ----------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)

admissions[["subject_id", "hadm_id", "age", "mortality"]].to_csv(
    out_path,
    index=False,
)

print(f"✅ In-hospital mortality (ICD-10 cohort) labels saved to: {out_path}")
print("Cohort size:", len(admissions))
print(admissions["mortality"].value_counts(normalize=True).round(3))