"""
MIMIC-III Long-Stay Label Creator
=================================

Overview
--------
This script creates a binary long-stay label for MIMIC-III hospital admissions.
It loads ADMISSIONS and PATIENTS tables, computes admission age using date of
birth and admission time, computes hospital length of stay from admission and
discharge timestamps, and labels each valid admission as long-stay or non-long-
stay.

The long-stay label is defined as:

    - long_stay = 1 if LOS > 7 days
    - long_stay = 0 if LOS <= 7 days

The script accepts either compressed or uncompressed MIMIC-III CSV files. It first
tries to read .csv.gz files, then falls back to .csv files.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-III root directory containing
          ADMISSIONS.csv(.gz) and PATIENTS.csv(.gz)
        - --out_path: output CSV path for the long-stay label file

2.  Resolve input paths:
        - ADMISSIONS.csv.gz or ADMISSIONS.csv
        - PATIENTS.csv.gz or PATIENTS.csv

3.  Load required MIMIC-III tables:
        - ADMISSIONS
        - PATIENTS

4.  Normalize column names:
        - Convert all ADMISSIONS and PATIENTS column names to lowercase for
          consistent downstream processing.

5.  Parse timestamps:
        - Convert admissions.admittime to datetime.
        - Convert admissions.dischtime to datetime.
        - Convert patients.dob to datetime.

6.  Merge patient date of birth into admissions:
        - Merge admissions with patients on subject_id.
        - Keep dob for age calculation.

7.  Compute admission age:
        - Identify rows with non-missing admittime and dob.
        - Normalize admittime and dob to day-level datetime values.
        - Compute age in days to avoid nanosecond timedelta overflow.
        - Convert age from days to years using 365.2425 days per year.
        - Set ages greater than 89 to 90.
        - Set negative ages to missing.

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

Age Calculation
---------------
Admission age is computed from admission time and date of birth:

    age_days = admittime_date - dob_date
    age = age_days / 365.2425

The script computes age using day-level datetime arrays to avoid nanosecond
timedelta overflow in very old, shifted MIMIC-III dates.

Age cleanup rules:

    - If age > 89, set age to 90.
    - If age < 0, set age to missing.

Flexible CSV Loading
--------------------
The helper read_csv_flexible supports either compressed or uncompressed files.

For a requested path such as:

    ADMISSIONS.csv

the script first tries:

    ADMISSIONS.csv.gz

and then falls back to:

    ADMISSIONS.csv

If neither file exists, a FileNotFoundError is raised.

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
    subject_id, hadm_id, admittime, dischtime

From PATIENTS:
    subject_id, dob

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
    - cohort size among admissions with valid LOS
    - normalized long_stay value counts

Dependencies
------------
    pip install pandas numpy

Running
-------
Run the script with:

    python create_label_long_stay_mimiciii.py \
        --data_root /path/to/mimic-iii \
        --out_path /path/to/label_long_stay_icd9_only.csv

"""

import argparse
from pathlib import Path

import numpy as np
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
        help="Path to MIMIC-III root directory containing ADMISSIONS.csv(.gz) and PATIENTS.csv(.gz).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for label_long_stay_icd9_only.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

base = Path(ARGS.data_root)
out_path = Path(ARGS.out_path)


def read_csv_flexible(path_no_ext: Path):
    """Try .csv.gz then .csv (handy if export differs)."""
    gz = Path(str(path_no_ext) + ".gz") if str(path_no_ext).endswith(".csv") else Path(str(path_no_ext) + ".csv.gz")
    if gz.exists():
        return pd.read_csv(gz)
    if path_no_ext.exists():
        return pd.read_csv(path_no_ext)
    raise FileNotFoundError(f"Could not find {gz} or {path_no_ext}")


# ----------------------------
# 2) Load data
# ----------------------------
admissions = read_csv_flexible(base / "ADMISSIONS.csv")
patients   = read_csv_flexible(base / "PATIENTS.csv")

# Normalize column names to lowercase for consistency
admissions.columns = [c.lower() for c in admissions.columns]
patients.columns   = [c.lower() for c in patients.columns]


# ----------------------------
# 3) Parse timestamps + compute age (MIMIC-III style)
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

admissions = admissions.merge(
    patients[["subject_id", "dob"]],
    on="subject_id",
    how="left",
)

# Compute age in DAYS (avoids ns-timedelta overflow), then convert to years
mask = admissions["admittime"].notna() & admissions["dob"].notna()
admissions["age"] = pd.NA

admit_d = admissions.loc[mask, "admittime"].dt.normalize().to_numpy(dtype="datetime64[D]")
dob_d   = admissions.loc[mask, "dob"].dt.normalize().to_numpy(dtype="datetime64[D]")

age_days  = (admit_d - dob_d).astype("timedelta64[D]").astype("int64")
age_years = age_days / 365.2425

admissions.loc[mask, "age"] = age_years

# Handle de-identification / unrealistic ages
admissions.loc[admissions["age"] > 89, "age"] = 90
admissions.loc[admissions["age"] < 0, "age"] = pd.NA


# ----------------------------
# 4) Compute LOS + label
# ----------------------------
admissions["los_days"] = (
    (admissions["dischtime"] - admissions["admittime"])
    .dt.total_seconds() / (3600.0 * 24.0)
)

# Keep only valid LOS (needs both times)
admissions = admissions.loc[admissions["los_days"].notna()].copy()

admissions["long_stay"] = (admissions["los_days"] > 7).astype(int)


# ----------------------------
# 5) Save
# ----------------------------
cols = ["subject_id", "hadm_id", "age", "los_days", "long_stay"]
out_path.parent.mkdir(parents=True, exist_ok=True)
admissions[cols].to_csv(out_path, index=False)

print(f"✅ Long-stay (MIMIC-III) labels saved to: {out_path}")
print("Cohort size (admissions with valid LOS):", len(admissions))
print(admissions["long_stay"].value_counts(normalize=True).round(3))