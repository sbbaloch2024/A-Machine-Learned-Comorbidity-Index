"""
ICD-10-Only ICU Transfer Label Creator
======================================

Overview
--------
This script creates ICU transfer labels for hospital admissions in an ICD-10-only
MIMIC-IV cohort. ICU transfer status is derived from ICU stay timing, while the
cohort itself is restricted to admissions that have at least one ICD-10 diagnosis
code in hosp/diagnoses_icd.csv.gz.

For each ICD-10 hospital admission, the script identifies the first ICU stay
linked to the admission and compares the first ICU intime with the hospital
admission time. The script generates two saved binary ICU labels:

    - icu_any:
          1 if the admission has any linked ICU stay, otherwise 0

    - icu_transfer:
          1 if the first ICU intime occurs strictly after hospital admittime,
          otherwise 0

The script also computes admission age from the MIMIC-IV anchor age/year fields
and caps ages above 89 at 90, following standard MIMIC-IV deidentification
conventions. A stricter inspection-only label, icu_transfer_24h, is computed for
sanity checking but is not saved to the output CSV.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-IV root directory containing hosp/ and
          icu/ subdirectories
        - --out_path: output CSV path for the ICU transfer label file

2.  Resolve input paths:
        - hosp/admissions.csv.gz
        - hosp/patients.csv.gz
        - hosp/diagnoses_icd.csv.gz
        - icu/icustays.csv.gz

3.  Load required MIMIC-IV tables:
        - admissions
        - patients
        - icustays
        - diagnoses_icd, using only hadm_id and icd_version

4.  Restrict the cohort to ICD-10 admissions:
        - Select hadm_id values from diagnoses_icd where icd_version == 10.
        - Keep only admissions whose hadm_id appears in that ICD-10 set.
        - This ensures consistency with the rest of the ICD-10-only pipeline.

5.  Parse hospital admission and discharge times:
        - Convert admittime to datetime.
        - Convert dischtime to datetime.

6.  Merge patient anchor-age information:
        - Merge admissions with patients on subject_id.
        - Use anchor_age and anchor_year to approximate admission age.

7.  Compute admission age:
        - age = anchor_age + (admittime year - anchor_year)
        - Set age values greater than 89 to 90.

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

10. Create ICU labels:
        - icu_any = 1 if first_icu_intime is present, otherwise 0.
        - time_to_icu_hours is the time difference between first ICU intime and
          hospital admittime, measured in hours.
        - icu_transfer = 1 if time_to_icu_hours > 0, otherwise 0.

11. Print sanity checks:
        - icu_any prevalence
        - median time_to_icu_hours among ICU admissions
        - percent of ICU admissions with time_to_icu_hours <= 0
        - percent of ICU admissions with time_to_icu_hours < 0
        - number of admissions with missing admittime
        - number of admissions without first_icu_intime
        - percent of ICU admissions reaching ICU within 1 hour
        - percent of ICU admissions reaching ICU within 6 hours
        - percent of ICU admissions reaching ICU within 24 hours
        - icu_transfer prevalence
        - icu_transfer_24h prevalence

12. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save selected admission-level columns to --out_path.
        - Print the output path, cohort size, and normalized icu_transfer counts.

ICD-10 Cohort Definition
------------------------
The ICD-10-only cohort is defined using the diagnoses table:

    diagnoses_icd.icd_version == 10

An admission is included if its hadm_id appears at least once among ICD-10
diagnosis rows. ICU transfer status itself does not depend on ICD codes; ICD-10 is
used only as a cohort restriction so that this label file aligns with an
ICD-10-only diagnosis-token modeling pipeline.

ICU Label Definitions
---------------------
For each admission, the first ICU stay is identified by the earliest ICU intime
among all icustays rows linked to the same hadm_id.

The any-ICU label is:

    icu_any = 1[first_icu_intime is not missing]

The time-to-ICU feature is:

    time_to_icu_hours = (first_icu_intime - admittime) in hours

The ICU transfer label is:

    icu_transfer = 1[time_to_icu_hours > 0]

The comparison is strict. ICU stays whose first intime is exactly equal to
admittime are not labeled as transfers.

Inspection-Only Label
---------------------
The script also computes:

    icu_transfer_24h = 1[time_to_icu_hours > 24]

This stricter late-transfer label is printed for inspection but is not included in
the saved output CSV.

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
    ├── hosp/
    │   ├── admissions.csv.gz
    │   ├── patients.csv.gz
    │   └── diagnoses_icd.csv.gz
    └── icu/
        └── icustays.csv.gz

Required columns
----------------
From hosp/admissions.csv.gz:
    subject_id, hadm_id, admittime, dischtime

From hosp/patients.csv.gz:
    subject_id, anchor_age, anchor_year

From hosp/diagnoses_icd.csv.gz:
    hadm_id, icd_version

From icu/icustays.csv.gz:
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
    - ICD-10 admission cohort size
    - normalized icu_transfer value counts

Dependencies
------------
    pip install pandas

Running
-------
Run the script with:

    python create_label_icu_transfer_mimiciv.py \
        --data_root /path/to/mimic-iv \
        --out_path /path/to/label_icu_transfer_icd10_only.csv


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
        help="Path to MIMIC-IV root directory containing hosp/ and icu/ subdirectories.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for label_icu_transfer_icd10_only.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

data_root = Path(ARGS.data_root)
base_hosp = data_root / "hosp"
base_icu = data_root / "icu"

out_path = Path(ARGS.out_path)


# ----------------------------
# 2) Load tables
# ----------------------------
admissions = pd.read_csv(base_hosp / "admissions.csv.gz")
patients = pd.read_csv(base_hosp / "patients.csv.gz")
icustays = pd.read_csv(base_icu / "icustays.csv.gz")

# Load diagnoses ONLY to define ICD-10 cohort
diagnoses = pd.read_csv(
    base_hosp / "diagnoses_icd.csv.gz",
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
# 4) Compute age + parse admission times
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

# Merge ICU info
labels = admissions[["subject_id", "hadm_id", "admittime", "age"]].merge(
    first_icu,
    on="hadm_id",
    how="left",
)


# ----------------------------
# 6) ICU transfer logic
# ----------------------------
labels["icu_any"] = labels["first_icu_intime"].notna().astype(int)

labels["time_to_icu_hours"] = (
    (labels["first_icu_intime"] - labels["admittime"])
    .dt.total_seconds() / 3600.0
)

# transfer only if ICU intime strictly after admission time
labels["icu_transfer"] = ((labels["time_to_icu_hours"].fillna(-1) > 0)).astype(int)


# ----------------------------
# 6b) Sanity checks (optional)
# ----------------------------
print("\nSanity checks:")
print("  icu_any prevalence:", float(labels["icu_any"].mean()))

# time_to_icu_hours distribution (only where ICU exists)
tt = labels.loc[labels["icu_any"] == 1, "time_to_icu_hours"]

print("  median time_to_icu_hours (ICU only):", float(tt.median()))
print("  pct time_to_icu_hours <= 0 (ICU only):", float((tt <= 0).mean()))
print("  pct time_to_icu_hours < 0  (ICU only):", float((tt < 0).mean()))

# How many admissions have missing admittime / ICU intime
print("  missing admittime:", int(labels["admittime"].isna().sum()))
print("  missing first_icu_intime:", int(labels["first_icu_intime"].isna().sum()))

# Check if most "transfers" are basically immediate (e.g., within 1 hour)
print("  pct ICU within 1h of admit (ICU only):", float((tt <= 1).mean()))
print("  pct ICU within 6h of admit (ICU only):", float((tt <= 6).mean()))
print("  pct ICU within 24h of admit (ICU only):", float((tt <= 24).mean()))

# Optional: a stricter "late transfer" label for inspection (doesn't change saved output)
labels["icu_transfer_24h"] = (labels["time_to_icu_hours"].fillna(-1) > 24).astype(int)
print("  icu_transfer prevalence:", float(labels["icu_transfer"].mean()))
print("  icu_transfer_24h prevalence:", float(labels["icu_transfer_24h"].mean()))


# ----------------------------
# 7) Save
# ----------------------------
save_cols = [
    "subject_id",
    "hadm_id",
    "age",
    "icu_any",
    "time_to_icu_hours",
    "icu_transfer",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
labels[save_cols].to_csv(out_path, index=False)

print(f"✅ ICU transfer (ICD-10-only cohort) labels saved to: {out_path}")
print("Cohort size (ICD-10 admissions):", len(labels))
print(labels["icu_transfer"].value_counts(normalize=True).round(3))