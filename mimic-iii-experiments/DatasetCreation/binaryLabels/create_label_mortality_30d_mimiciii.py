
"""
MIMIC-III 30-Day Mortality Label Creator
========================================

Overview
--------
This script creates an all-cause 30-day mortality label for MIMIC-III hospital
admissions. It loads ADMISSIONS and PATIENTS tables, computes age at admission,
uses patient-level date of death as the all-cause death time, and labels each
admission as 30-day mortality positive if death occurred from admission day
through 30 days after admission.

The script uses a MIMIC-IV-style mortality definition for consistency with the
ICD-9 MIMIC-IV label pipeline: mortality_30d is based only on patient-level DOD.
Admission-level DEATHTIME is not used to define mortality_30d, but it is retained
for diagnostics through the in_hosp_death and post_discharge_30d_death columns.

The script supports both compressed and uncompressed MIMIC-III CSV exports by
trying .csv.gz first and then .csv.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-III root directory containing
          ADMISSIONS.csv(.gz) and PATIENTS.csv(.gz)
        - --out_path: output CSV path for the 30-day mortality label file

2.  Resolve input paths:
        - ADMISSIONS.csv.gz or ADMISSIONS.csv
        - PATIENTS.csv.gz or PATIENTS.csv

3.  Load required MIMIC-III tables:
        - ADMISSIONS
        - PATIENTS

4.  Normalize column names:
        - Convert all admission and patient column names to lowercase.

5.  Parse timestamps:
        - Convert admittime to datetime.
        - Convert dischtime to datetime.
        - Convert deathtime to datetime.
        - Convert dob to datetime.
        - Convert dod to datetime.

6.  Merge patient information into admissions:
        - Merge admissions with patients on subject_id.
        - Include dob and dod.

7.  Compute age at admission:
        - Use admittime and dob.
        - Normalize both dates to day precision.
        - Compute age in days to avoid nanosecond overflow.
        - Convert age_days to years using 365.2425 days per year.
        - Set ages greater than 89 to 90.
        - Set negative ages to missing.

8.  Define all-cause death time:
        - death_time_allcause = dod
        - This intentionally ignores admission-level deathtime for the
          mortality_30d label.

9.  Compute time from admission to death:
        - time_to_death_days is the difference between dod and admittime in days.

10. Create the 30-day mortality label:
        - mortality_30d = 1 if:
              dod is present
              admittime is present
              time_to_death_days >= 0
              time_to_death_days <= 30.0
        - mortality_30d = 0 otherwise

11. Create diagnostic death indicators:
        - in_hosp_death = 1 if deathtime is present, otherwise 0.
        - post_discharge_30d_death = 1 if mortality_30d is positive and
          in_hosp_death is zero.

12. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save admission identifiers, age, timing columns, mortality label, and
          diagnostic death indicators.
        - Print output path, cohort size, mortality prevalence, in-hospital death
          prevalence, and the post-discharge share among 30-day deaths.

Input File Handling
-------------------
The helper read_csv_flexible supports either compressed or uncompressed CSV files.

For a requested file such as:

    ADMISSIONS.csv

the script tries:

    ADMISSIONS.csv.gz

first. If the compressed file does not exist, it tries:

    ADMISSIONS.csv

If neither file exists, it raises FileNotFoundError.

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

These diagnostics are saved to help compare 30-day deaths with admission-level
in-hospital death indicators. They do not change the mortality_30d label.

Age Calculation
---------------
Age is computed from date of birth and admission time:

    age_days = admittime_date - dob_date
    age = age_days / 365.2425

The script uses day-level datetime conversion to avoid nanosecond overflow during
subtraction.

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
    subject_id, hadm_id, admittime, dischtime, deathtime

From PATIENTS:
    subject_id, dob, dod

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
    - cohort size
    - mortality_30d prevalence
    - in_hosp_death prevalence
    - share of post-discharge deaths among 30-day deaths

Dependencies
------------
    pip install numpy pandas

Running
-------
Run the script with:

    python create_label_mortality_30d_mimiciii.py \
        --data_root /path/to/mimic-iii \
        --out_path /path/to/label_mortality_30d_icd9_only.csv

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
        help="Output CSV path for label_mortality_30d_icd9_only.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

base = Path(ARGS.data_root)
out_path = Path(ARGS.out_path)


def read_csv_flexible(path_with_csv: Path):
    """Try .csv.gz then .csv (handy if export differs)."""
    # if user passes ".../FILE.csv" we try ".../FILE.csv.gz" first
    gz = Path(str(path_with_csv) + ".gz") if str(path_with_csv).endswith(".csv") else Path(str(path_with_csv) + ".csv.gz")
    if gz.exists():
        return pd.read_csv(gz)
    if path_with_csv.exists():
        return pd.read_csv(path_with_csv)
    raise FileNotFoundError(f"Could not find {gz} or {path_with_csv}")


# ----------------------------
# 2) Load tables (MIMIC-III)
# ----------------------------
admissions = read_csv_flexible(base / "ADMISSIONS.csv")
patients   = read_csv_flexible(base / "PATIENTS.csv")

# normalize column names
admissions.columns = [c.lower() for c in admissions.columns]
patients.columns   = [c.lower() for c in patients.columns]


# ----------------------------
# 3) Parse timestamps
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions.get("admittime"), errors="coerce")
admissions["dischtime"] = pd.to_datetime(admissions.get("dischtime"), errors="coerce")
admissions["deathtime"] = pd.to_datetime(admissions.get("deathtime"), errors="coerce")

patients["dob"] = pd.to_datetime(patients.get("dob"), errors="coerce")
patients["dod"] = pd.to_datetime(patients.get("dod"), errors="coerce")


# ----------------------------
# 4) Merge + compute age (MIMIC-III style)
# ----------------------------
df = admissions.merge(
    patients[["subject_id", "dob", "dod"]],
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

# Handle de-identification / unrealistic ages
df.loc[df["age"] > 89, "age"] = 90
df.loc[df["age"] < 0, "age"] = pd.NA


# ----------------------------
# 5) All-cause death time + 30d label  (MIMIC-IV-style: DOD ONLY)
# ----------------------------
# Use ONLY patient-level DOD (day-level), ignore admission DEATHTIME for labeling
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

# diagnostics (you can keep these even though mortality_30d uses DOD only)
df["in_hosp_death"] = df["deathtime"].notna().astype(int)
df["post_discharge_30d_death"] = ((df["mortality_30d"] == 1) & (df["in_hosp_death"] == 0)).astype(int)


# ----------------------------
# 6) Save
# ----------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)

save_cols = [
    "subject_id", "hadm_id", "age",
    "admittime", "dischtime", "deathtime", "dod",
    "time_to_death_days", "mortality_30d", "in_hosp_death", "post_discharge_30d_death",
]

df[save_cols].to_csv(out_path, index=False)

print(f"✅ 30-day mortality (MIMIC-III) labels saved to: {out_path}")
print("Cohort size:", len(df))
print("Prevalence mortality_30d:", float(df["mortality_30d"].mean()))
print("Prevalence in_hosp_death:", float(df["in_hosp_death"].mean()))
print(
    "Share post-discharge among 30d deaths:",
    float(df["post_discharge_30d_death"].sum() / max(df["mortality_30d"].sum(), 1))
)