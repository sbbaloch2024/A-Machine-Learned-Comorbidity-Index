"""
MIMIC-III ICU Transfer Label Creator
====================================

Overview
--------
This script creates ICU transfer labels for hospital admissions in a MIMIC-III
cohort. It reads admission, patient, and ICU stay tables, computes age at hospital
admission, identifies the first ICU intime for each hospital admission, and
creates an ICU transfer label based on whether the first ICU stay occurs more
than 24 hours after hospital admission.

For each hospital admission, the script generates:

    - icu_any:
          1 if any ICU stay is linked to the admission, otherwise 0

    - icu_transfer:
          1 if the first ICU intime occurs more than 24 hours after hospital
          admission, otherwise 0

The script also computes time_to_icu_hours for inspection and saves it with the
output labels.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-III root directory containing
          ADMISSIONS.csv, PATIENTS.csv, and ICUSTAYS.csv, optionally gzipped
        - --out_path: output CSV path for label_icu_transfer_icd9_only.csv

2.  Load required MIMIC-III tables:
        - ADMISSIONS.csv or ADMISSIONS.csv.gz
        - PATIENTS.csv or PATIENTS.csv.gz
        - ICUSTAYS.csv or ICUSTAYS.csv.gz

3.  Normalize column names:
        - Convert all admissions, patients, and icustays column names to
          lowercase for consistency.

4.  Parse hospital admission timestamps:
        - Convert admittime to datetime.
        - Convert dischtime to datetime.

5.  Parse patient date of birth:
        - Convert dob to datetime.

6.  Merge date of birth into admissions:
        - Merge admissions with patients on subject_id.
        - Keep dob for age calculation.

7.  Compute admission age:
        - Use date-normalized admittime and dob.
        - Compute age in days to avoid nanosecond timedelta overflow.
        - Convert age_days to years using 365.2425 days per year.
        - Set ages greater than 89 to 90.
        - Set negative ages to missing.

8.  Determine first ICU intime per admission:
        - Convert icustays.intime to datetime.
        - Drop ICU rows missing hadm_id or intime.
        - Sort ICU stays by intime.
        - Group by hadm_id.
        - Keep the earliest ICU intime per hadm_id.
        - Rename this timestamp to first_icu_intime.

9.  Merge ICU information into the admission-level label table:
        - Keep subject_id, hadm_id, admittime, and age from admissions.
        - Left-join first_icu_intime by hadm_id.
        - Admissions without linked ICU stays remain in the cohort.

10. Create ICU indicators:
        - icu_any = 1 if first_icu_intime is present, otherwise 0.
        - time_to_icu_hours is the time difference between first ICU intime and
          hospital admittime, measured in hours.
        - icu_transfer = 1 if the admission has an ICU stay and
          time_to_icu_hours > 24.0.

11. Print sanity checks:
        - number of admissions
        - icu_any prevalence
        - number of admissions without first_icu_intime
        - median time_to_icu_hours among ICU admissions
        - percent of ICU admissions with time_to_icu_hours <= 0
        - percent of ICU admissions with time_to_icu_hours < 0
        - percent of ICU admissions reaching ICU within 1 hour
        - percent of ICU admissions reaching ICU within 6 hours
        - percent of ICU admissions reaching ICU within 24 hours
        - percent of ICU admissions reaching ICU after 24 hours
        - icu_transfer prevalence

12. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save selected admission-level columns to --out_path.
        - Print the output path, cohort size, and normalized icu_transfer counts.

Input File Handling
-------------------
The script uses read_csv_flexible to support both plain CSV and gzipped CSV files.

For an input path such as:

    ADMISSIONS.csv

the loader first checks for:

    ADMISSIONS.csv.gz

and then checks for:

    ADMISSIONS.csv

If neither exists, it raises FileNotFoundError.

ICU Label Definitions
---------------------
For each admission, the first ICU stay is identified by the earliest ICU intime
among all ICUSTAYS rows linked to the same hadm_id.

The any-ICU label is:

    icu_any = 1[first_icu_intime is not missing]

The time-to-ICU feature is:

    time_to_icu_hours = (first_icu_intime - admittime) in hours

The late ICU transfer label is:

    icu_transfer = 1[icu_any == 1 and time_to_icu_hours > 24.0]

The comparison is strict. ICU stays occurring exactly 24 hours after hospital
admission are not labeled as transfers.

Age Calculation
---------------
Admission age is computed from MIMIC-III date of birth and admission time:

    age_days = admittime_date - dob_date
    age = age_days / 365.2425

The script computes age in days using datetime64[D] values to avoid timedelta
overflow issues that can occur with deidentified dates.

Age values are cleaned as follows:

    - age > 89 is set to 90
    - age < 0 is set to missing

Expected Directory Layout
--------------------------
    <data_root>/
    ├── ADMISSIONS.csv or ADMISSIONS.csv.gz
    ├── PATIENTS.csv or PATIENTS.csv.gz
    └── ICUSTAYS.csv or ICUSTAYS.csv.gz

Required columns
----------------
From ADMISSIONS:
    subject_id, hadm_id, admittime, dischtime

From PATIENTS:
    subject_id, dob

From ICUSTAYS:
    hadm_id, intime

Outputs
-------
The script writes one CSV file to --out_path.

Saved columns:

    - subject_id
    - hadm_id
    - age
    - icu_any
    - time_to_icu_hours
    - icu_transfer

The script prints:

    - sanity-check prevalence and timing summaries
    - output file path
    - admission cohort size
    - normalized icu_transfer value counts

Dependencies
------------
    pip install numpy pandas

Running
-------
Run the script with:

    python create_label_icu_transfer_mimiciii.py \
        --data_root /path/to/mimic-iii \
        --out_path /path/to/label_icu_transfer_icd9_only.csv

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
        help="Path to MIMIC-III root directory containing ADMISSIONS.csv(.gz), PATIENTS.csv(.gz), and ICUSTAYS.csv(.gz).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for label_icu_transfer_icd9_only.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

base = Path(ARGS.data_root)
out_path = Path(ARGS.out_path)


def read_csv_flexible(path_no_ext: Path):
    gz = Path(str(path_no_ext) + ".gz") if str(path_no_ext).endswith(".csv") else Path(str(path_no_ext) + ".csv.gz")
    if gz.exists():
        return pd.read_csv(gz)
    if path_no_ext.exists():
        return pd.read_csv(path_no_ext)
    raise FileNotFoundError(f"Could not find {gz} or {path_no_ext}")


# ----------------------------
# 2) Load tables
# ----------------------------
admissions = read_csv_flexible(base / "ADMISSIONS.csv")
patients   = read_csv_flexible(base / "PATIENTS.csv")
icustays   = read_csv_flexible(base / "ICUSTAYS.csv")

# Normalize column names to lowercase for consistency
admissions.columns = [c.lower() for c in admissions.columns]
patients.columns   = [c.lower() for c in patients.columns]
icustays.columns   = [c.lower() for c in icustays.columns]


# ----------------------------
# 4) Compute age + parse admission times (MIMIC-III style)
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

admissions = admissions.merge(
    patients[["subject_id", "dob"]],
    on="subject_id", how="left"
)

# Compute age in DAYS (avoids ns-timedelta overflow), then convert to years
mask = admissions["admittime"].notna() & admissions["dob"].notna()

admissions["age"] = pd.NA

admit_d = admissions.loc[mask, "admittime"].dt.normalize().to_numpy(dtype="datetime64[D]")
dob_d   = patients.merge(
    admissions.loc[mask, ["subject_id"]],
    on="subject_id",
    how="right"
)["dob"].dt.normalize().to_numpy(dtype="datetime64[D]")

# Preserve original alignment from the merged admissions dataframe
dob_d = admissions.loc[mask, "dob"].dt.normalize().to_numpy(dtype="datetime64[D]")

age_days = (admit_d - dob_d).astype("timedelta64[D]").astype("int64")
age_years = age_days / 365.2425

admissions.loc[mask, "age"] = age_years

# Handle de-identification / unrealistic ages
admissions.loc[admissions["age"] > 89, "age"] = 90
admissions.loc[admissions["age"] < 0, "age"] = pd.NA


# ----------------------------
# 5) Determine first ICU intime per hadm_id
# ----------------------------
icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")

first_icu = (
    icustays.dropna(subset=["hadm_id", "intime"])
           .sort_values("intime")
           .groupby("hadm_id", as_index=False)["intime"]
           .first()
           .rename(columns={"intime": "first_icu_intime"})
)

labels = admissions[["subject_id", "hadm_id", "admittime", "age"]].merge(
    first_icu, on="hadm_id", how="left"
)


# ----------------------------
# 6) ICU transfer logic (LATE transfer definition)
# ----------------------------
labels["icu_any"] = labels["first_icu_intime"].notna().astype(int)

labels["time_to_icu_hours"] = (
    (labels["first_icu_intime"] - labels["admittime"])
    .dt.total_seconds() / 3600.0
)

LATE_HOURS = 24.0

# Late ICU transfer: ICU occurs >24h after hospital admission
labels["icu_transfer"] = (
    (labels["icu_any"] == 1) &
    (labels["time_to_icu_hours"] > LATE_HOURS)
).astype(int)


# ----------------------------
# 6b) Sanity checks
# ----------------------------
tt = labels.loc[labels["icu_any"] == 1, "time_to_icu_hours"].dropna()

print("\nSanity checks:")
print("  admissions:", len(labels))
print("  icu_any prevalence:", float(labels["icu_any"].mean()))
print("  missing first_icu_intime:", int(labels["first_icu_intime"].isna().sum()))

if len(tt) > 0:
    print("  median time_to_icu_hours (ICU only):", float(tt.median()))
    print("  pct time_to_icu_hours <= 0 (ICU only):", float((tt <= 0).mean()))
    print("  pct time_to_icu_hours < 0  (ICU only):", float((tt < 0).mean()))
    print("  pct ICU within 1h of admit (ICU only):", float((tt <= 1).mean()))
    print("  pct ICU within 6h of admit (ICU only):", float((tt <= 6).mean()))
    print("  pct ICU within 24h of admit (ICU only):", float((tt <= 24).mean()))
    print(f"  pct ICU after {LATE_HOURS:.0f}h (ICU only):", float((tt > LATE_HOURS).mean()))

print("  icu_transfer (>24h) prevalence:", float(labels["icu_transfer"].mean()))


# ----------------------------
# 7) Save
# ----------------------------
save_cols = ["subject_id", "hadm_id", "age", "icu_any", "time_to_icu_hours", "icu_transfer"]
out_path.parent.mkdir(parents=True, exist_ok=True)
labels[save_cols].to_csv(out_path, index=False)

print(f"✅ ICU transfer (ICD-9 cohort) labels saved to: {out_path}")
print("Cohort size (ICD-9 admissions):", len(labels))
print(labels["icu_transfer"].value_counts(normalize=True).round(3))