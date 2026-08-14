"""
ICD-10-Only 30-Day Mortality Label Creator
==========================================

Overview
--------
This script creates an all-cause 30-day mortality label for hospital admissions in
an ICD-10-only MIMIC-IV cohort. The cohort is restricted to admissions that have at
least one ICD-10 diagnosis code in hosp/diagnoses_icd.csv.gz.

For each eligible admission, the script uses the patient-level date of death
field, dod, as the all-cause death time. It compares dod with the hospital
admission time and labels the admission as 30-day mortality positive if death
occurred from admission day through 30 days after admission.

The script also computes admission age using MIMIC-IV anchor age/year fields,
caps ages above 89 at 90, and saves diagnostic columns for in-hospital death and
post-discharge 30-day death.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-IV root directory containing the hosp/
          subdirectory
        - --out_path: output CSV path for the 30-day mortality label file

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

6.  Parse timestamps:
        - Convert admissions.admittime to datetime.
        - Convert admissions.dischtime to datetime.
        - Convert admissions.deathtime to datetime.
        - Convert patients.dod to datetime.
        - If dod is unavailable, create a missing dod column.

7.  Merge patient information into admissions:
        - Merge on subject_id.
        - Include anchor_age, anchor_year, and dod.

8.  Compute admission age:
        - age = anchor_age + (admittime year - anchor_year)
        - Set age values greater than 89 to 90.

9.  Define all-cause death time:
        - death_time_allcause = dod

10. Compute time from admission to death:
        - time_to_death_days is the difference between dod and admittime in days.

11. Create the 30-day mortality label:
        - mortality_30d = 1 if:
              dod is present
              admittime is present
              time_to_death_days >= 0
              time_to_death_days <= 30.0
        - mortality_30d = 0 otherwise

12. Create diagnostic death indicators:
        - in_hosp_death = 1 if deathtime is present, otherwise 0.
        - post_discharge_30d_death = 1 if mortality_30d is positive and
          in_hosp_death is zero.

13. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save admission identifiers, timing columns, mortality labels, and
          diagnostic death indicators.
        - Print output path, cohort size, mortality prevalence, in-hospital death
          prevalence, and the post-discharge share among 30-day deaths.

ICD-10 Cohort Definition
------------------------
The ICD-10-only cohort is defined using the diagnoses table:

    diagnoses_icd.icd_version == 10

An admission is included if its hadm_id appears at least once among ICD-10
diagnosis rows. The mortality label itself does not depend on ICD codes; ICD-10 is
used only as a cohort restriction so that this label file aligns with an
ICD-10-only diagnosis-token modeling pipeline.

30-Day Mortality Label Definition
---------------------------------
The all-cause death time is defined as the patient-level date of death:

    death_time_allcause = dod

Time to death is measured from hospital admission:

    time_to_death_days = (dod - admittime) in days

The 30-day mortality label is:

    mortality_30d = 1[
        dod is not missing
        and admittime is not missing
        and time_to_death_days >= 0
        and time_to_death_days <= 30
    ]

Deaths before the recorded admission time are not counted. Deaths exactly 30 days
after admission are counted as positive.

Diagnostic Indicators
---------------------
The script also computes two diagnostic columns:

    in_hosp_death = 1[deathtime is not missing]

    post_discharge_30d_death = 1[
        mortality_30d == 1 and in_hosp_death == 0
    ]

These columns help distinguish deaths captured during hospitalization from
30-day deaths that occur after discharge.

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
    subject_id, hadm_id, admittime, dischtime, deathtime

From hosp/patients.csv.gz:
    subject_id, anchor_age, anchor_year, dod

Outputs
-------
The script writes one CSV file to --out_path.

Saved columns:

    - subject_id
    - hadm_id
    - age
    - admittime
    - dischtime
    - deathtime
    - dod
    - time_to_death_days
    - mortality_30d
    - in_hosp_death
    - post_discharge_30d_death

The script prints:

    - output file path
    - ICD-10 cohort size
    - mortality_30d prevalence
    - in_hosp_death prevalence
    - share of post-discharge deaths among 30-day deaths

Dependencies
------------
    pip install pandas

Running
-------
Run the script with:

    python create_label_mortality_30d_mimiciv.py \
        --data_root /path/to/mimic-iv \
        --out_path /path/to/label_mortality_30d_icd10_only.csv

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
        help="Output CSV path for label_mortality_30d_icd10_only.csv.",
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
# 4) Parse times
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")
patients["dod"] = pd.to_datetime(
    patients.get("dod", pd.Series([pd.NaT] * len(patients))),
    errors="coerce",
)


# ----------------------------
# 5) Merge
# ----------------------------
df = admissions.merge(
    patients[["subject_id", "anchor_age", "anchor_year", "dod"]],
    on="subject_id",
    how="left",
)


# ----------------------------
# 6) Age
# ----------------------------
df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
df.loc[df["age"] > 89, "age"] = 90


# ----------------------------
# 7) All-cause death time + 30d label
# ----------------------------
df["death_time_allcause"] = df["dod"]

df["time_to_death_days"] = (
    (df["death_time_allcause"] - df["admittime"])
    .dt.total_seconds() / (3600.0 * 24.0)
)

df["mortality_30d"] = (
    df["death_time_allcause"].notna()
    & df["admittime"].notna()
    & (df["time_to_death_days"] >= 0)
    & (df["time_to_death_days"] <= 30.0)
).astype(int)


# ----------------------------
# 8) Diagnostics
# ----------------------------
df["in_hosp_death"] = df["deathtime"].notna().astype(int)
df["post_discharge_30d_death"] = (
    (df["mortality_30d"] == 1)
    & (df["in_hosp_death"] == 0)
).astype(int)


# ----------------------------
# 9) Save
# ----------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)

df[
    [
        "subject_id",
        "hadm_id",
        "age",
        "admittime",
        "dischtime",
        "deathtime",
        "dod",
        "time_to_death_days",
        "mortality_30d",
        "in_hosp_death",
        "post_discharge_30d_death",
    ]
].to_csv(out_path, index=False)

print(f"✅ 30-day mortality (ICD-10 cohort) labels saved to: {out_path}")
print("Cohort size:", len(df))
print("Prevalence mortality_30d:", float(df["mortality_30d"].mean()))
print("Prevalence in_hosp_death:", float(df["in_hosp_death"].mean()))
print(
    "Share post-discharge among 30d deaths:",
    float(df["post_discharge_30d_death"].sum() / max(df["mortality_30d"].sum(), 1)),
)