"""
MIMIC-III In-Hospital Mortality Label Creator
=============================================

Overview
--------
This script creates an in-hospital mortality label for MIMIC-III hospital
admissions. It loads ADMISSIONS and PATIENTS tables, computes a binary mortality
indicator from the admission-level DEATHTIME field, computes age at admission
from ADMITTIME and DOB, and saves an admission-level label file.

The in-hospital mortality label is defined as:

    - mortality = 1 if deathtime is present
    - mortality = 0 if deathtime is missing

The script supports flexible input loading from either .csv.gz or .csv files and
normalizes column names to lowercase after loading. Age is computed safely in days
to avoid datetime overflow issues, then converted to years. Ages above 89 are
top-coded to 90, and negative ages are set to missing.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-III root directory containing
          ADMISSIONS.csv(.gz) and PATIENTS.csv(.gz)
        - --out_path: output CSV path for the in-hospital mortality label file

2.  Resolve input/output paths:
        - ADMISSIONS.csv or ADMISSIONS.csv.gz
        - PATIENTS.csv or PATIENTS.csv.gz
        - output CSV path

3.  Load required MIMIC-III tables with flexible CSV loading:
        - Try .csv.gz first.
        - If the compressed file is not found, try .csv.
        - Raise FileNotFoundError if neither exists.

4.  Normalize column names:
        - Convert ADMISSIONS column names to lowercase.
        - Convert PATIENTS column names to lowercase.

5.  Create the in-hospital mortality label:
        - mortality = 1 if admissions.deathtime is not missing.
        - mortality = 0 if admissions.deathtime is missing.

6.  Parse timestamps:
        - Convert admissions.admittime to datetime.
        - Convert patients.dob to datetime.

7.  Merge patient date-of-birth information:
        - Merge admissions with patients on subject_id.
        - Keep dob for age calculation.

8.  Compute age at admission:
        - Use rows where both admittime and dob are present.
        - Normalize timestamps to calendar-day precision.
        - Compute age in days as admittime - dob.
        - Convert days to years using 365.2425 days per year.
        - Set ages greater than 89 to 90.
        - Set negative ages to missing.

9.  Save the output CSV:
        - Create the output directory if it does not exist.
        - Save subject_id, hadm_id, age, and mortality.
        - Print the output path, cohort size, and normalized mortality counts.

Mortality Label Definition
--------------------------
The in-hospital mortality label is derived from the admission-level deathtime
field:

    mortality = 1[deathtime is not missing]

This captures deaths recorded during the hospital admission. It does not use
patient-level date of death and does not include post-discharge mortality unless
that death is reflected in the admission-level deathtime field.

Age Calculation
---------------
Age at admission is computed from ADMITTIME and DOB:

    age_days = admittime_date - dob_date
    age = age_days / 365.2425

The script uses day-level datetime arrays rather than nanosecond datetime
differences to avoid overflow issues that can occur with very old MIMIC-III DOB
values.

Age post-processing:
    - age > 89 is set to 90
    - age < 0 is set to missing

Expected Directory Layout
--------------------------
    <data_root>/
    ├── ADMISSIONS.csv.gz
    └── PATIENTS.csv.gz

or:

    <data_root>/
    ├── ADMISSIONS.csv
    └── PATIENTS.csv

Required columns
----------------
From ADMISSIONS:
    subject_id, hadm_id, admittime, deathtime

From PATIENTS:
    subject_id, dob

Column names may be uppercase in the raw files because the script lowercases all
loaded column names before processing.

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
    - cohort size
    - normalized mortality value counts

Dependencies
------------
    pip install numpy pandas

Running
-------
Run the script with:

    python create_label_mortality_inhouse_mimiciii.py \
        --data_root /path/to/mimic-iii \
        --out_path /path/to/label_mortality_inhouse_icd9_only.csv


"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------
# Paths
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to MIMIC-III root directory containing ADMISSIONS.csv(.gz) and PATIENTS.csv(.gz).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for label_mortality_inhouse_icd9_only.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

base = Path(ARGS.data_root)
out_path = Path(ARGS.out_path)


def read_csv_flexible(path_with_csv: Path):
    """Try .csv.gz then .csv."""
    gz = Path(str(path_with_csv) + ".gz") if str(path_with_csv).endswith(".csv") else Path(str(path_with_csv) + ".csv.gz")
    if gz.exists():
        return pd.read_csv(gz)
    if path_with_csv.exists():
        return pd.read_csv(path_with_csv)
    raise FileNotFoundError(f"Could not find {gz} or {path_with_csv}")


# ----------------------------
# Load tables
# ----------------------------
admissions = read_csv_flexible(base / "ADMISSIONS.csv")
patients   = read_csv_flexible(base / "PATIENTS.csv")

# Normalize column names
admissions.columns = [c.lower() for c in admissions.columns]
patients.columns   = [c.lower() for c in patients.columns]


# ----------------------------
# Label: DEATHTIME only
# ----------------------------
admissions["mortality"] = admissions["deathtime"].notna().astype(int)


# ----------------------------
# Age at admission (MIMIC-III style)
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

df = admissions.merge(
    patients[["subject_id", "dob"]],
    on="subject_id",
    how="left",
)

# --- Age at admission in YEARS (safe: compute in DAYS to avoid ns overflow) ---
mask = df["admittime"].notna() & df["dob"].notna()
df["age"] = pd.NA

admit_d = df.loc[mask, "admittime"].dt.normalize().to_numpy(dtype="datetime64[D]")
dob_d   = df.loc[mask, "dob"].dt.normalize().to_numpy(dtype="datetime64[D]")

age_days  = (admit_d - dob_d).astype("timedelta64[D]").astype("int64")
age_years = age_days / 365.2425

df.loc[mask, "age"] = age_years

df.loc[df["age"] > 89, "age"] = 90
df.loc[df["age"] < 0, "age"] = pd.NA


# ----------------------------
# Save
# ----------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)
df[["subject_id", "hadm_id", "age", "mortality"]].to_csv(out_path, index=False)

print(f"✅ In-hospital mortality (MIMIC-III) labels saved to: {out_path}")
print("Cohort size:", len(df))
print(df["mortality"].value_counts(normalize=True).round(3))