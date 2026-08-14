
"""
MIMIC-III ICD-9 Charlson Comorbidity Index Creator
==================================================

Overview
--------
This script computes the Charlson Comorbidity Index (CCI) for MIMIC-III hospital
admissions using ICD-9 diagnosis codes from DIAGNOSES_ICD. Each admission is
mapped to Charlson comorbidity categories based on ICD-9 prefix matching. Each
category contributes its Charlson weight at most once per admission, even if
multiple diagnosis codes match the same category.

This script computes the comorbidity-only Charlson score. It does not add age
points to the final CCI score. Age is computed and saved for reference, but the
output cci column is equal to the summed Charlson diagnosis weights only.

The saved cohort is restricted to admissions that actually have at least one
non-empty ICD-9 diagnosis code after normalization.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-III root directory containing
          ADMISSIONS.csv(.gz), PATIENTS.csv(.gz), and DIAGNOSES_ICD.csv(.gz)
        - --out_path: output CSV path for the Charlson score file

2.  Resolve input/output paths:
        - ADMISSIONS.csv or ADMISSIONS.csv.gz
        - PATIENTS.csv or PATIENTS.csv.gz
        - DIAGNOSES_ICD.csv or DIAGNOSES_ICD.csv.gz
        - output CSV path

3.  Load required MIMIC-III tables with flexible CSV loading:
        - Try .csv.gz first.
        - If the compressed file is not found, try .csv.
        - Raise FileNotFoundError if neither exists.

4.  Normalize column names:
        - Convert admissions, patients, and diagnoses column names to lowercase.

5.  Parse admission time and patient date of birth:
        - Convert admissions.admittime to datetime.
        - Convert patients.dob to datetime.

6.  Merge patient date-of-birth information into admissions:
        - Merge admissions with patients on subject_id.
        - Keep dob for age calculation.

7.  Compute age at admission safely:
        - Use rows where both admittime and dob are present.
        - Normalize timestamps to calendar-day precision.
        - Compute age in days as admittime - dob.
        - Convert days to years using 365.2425 days per year.
        - Set ages greater than 89 to 90.
        - Set negative ages to missing.

8.  Normalize ICD-9 diagnosis codes:
        - Convert codes to uppercase.
        - Remove periods.
        - Remove spaces.
        - Strip leading and trailing whitespace.
        - Drop empty normalized codes.

9.  Build the ICD-9 admission cohort:
        - Identify admissions with at least one non-empty normalized ICD-9
          diagnosis code.
        - Restrict admissions to those hadm_id values.

10. Normalize Charlson category prefixes:
        - Normalize each ICD-9 prefix in the Charlson mapping using the same
          uppercase/no-dot/no-space rule.
        - Drop empty prefixes.
        - Deduplicate prefixes while preserving order.
        - Store each category's prefix list and Charlson weight.

11. Build admission-level Charlson category flags:
        - For each Charlson category, identify admissions with at least one
          diagnosis code that starts with any prefix assigned to that category.
        - Store one Boolean flag per category and hadm_id.
        - Each category is counted at most once per admission.

12. Apply Charlson hierarchy overrides:
        - DM_COMP overrides DM_NO_COMP.
        - SEV_LIVER overrides MILD_LIVER.
        - METS overrides MALIGNANCY.
        - These rules prevent double-counting less severe categories when the
          more severe related category is present.

13. Compute the Charlson diagnosis weight:
        - Multiply each category flag by its Charlson category weight.
        - Sum weighted category indicators per admission.
        - Store the result as charlson_weight.

14. Merge Charlson scores back into admissions:
        - Merge by hadm_id.
        - Fill missing Charlson weights with zero.
        - Set cci = charlson_weight.

15. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save subject_id, hadm_id, age, and cci.
        - Print output path, cohort size, mean CCI, and fraction of admissions
          with CCI equal to zero.

Diagnosis Code Normalization
----------------------------
ICD-9 diagnosis codes are normalized before matching:

    normalized_code = uppercase(code).replace(".", "").replace(" ", "").strip()

For example:

    250.40 -> 25040
    428.0  -> 4280

The Charlson mapping prefixes are normalized with the same function so that
dot-form and no-dot-form ICD-9 codes can be matched consistently.

ICD-9 Cohort Definition
-----------------------
The output cohort is defined as admissions whose hadm_id appears in the normalized
diagnosis table with at least one non-empty ICD-9 diagnosis code.

Unlike the MIMIC-IV ICD-9 scripts, this MIMIC-III script does not need an
icd_version filter because MIMIC-III diagnosis codes are handled as ICD-9 through
the DIAGNOSES_ICD icd9_code column.

Charlson Category Matching
--------------------------
For each Charlson category, the script performs prefix matching:

    diagnosis_code starts with any normalized prefix assigned to the category

If any diagnosis code for an admission matches a category, that category flag is
set to True for the admission. Multiple matching codes in the same category do not
increase the score beyond that category's single weight.

Charlson Hierarchy Overrides
----------------------------
The script applies three hierarchy rules before computing the final score:

    - Complicated diabetes overrides uncomplicated diabetes:
          DM_NO_COMP = DM_NO_COMP and not DM_COMP

    - Severe liver disease overrides mild liver disease:
          MILD_LIVER = MILD_LIVER and not SEV_LIVER

    - Metastatic solid tumor overrides malignancy:
          MALIGNANCY = MALIGNANCY and not METS

These overrides reduce double-counting between related Charlson categories.

CCI Score Definition
--------------------
The final comorbidity score is:

    cci = sum(category_present * category_weight)

where each category contributes at most once per admission.

This script does not compute age-adjusted CCI. The saved age column is included
for reference only and is not added to the cci score.

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
    ├── PATIENTS.csv.gz
    └── DIAGNOSES_ICD.csv.gz

or:

    <data_root>/
    ├── ADMISSIONS.csv
    ├── PATIENTS.csv
    └── DIAGNOSES_ICD.csv

Required columns
----------------
From ADMISSIONS:
    subject_id, hadm_id, admittime

From PATIENTS:
    subject_id, dob

From DIAGNOSES_ICD:
    hadm_id, icd9_code

Column names may be uppercase in the raw files because the script lowercases all
loaded column names before processing.

Outputs
-------
The script writes one CSV file to --out_path.

Saved columns:

    - subject_id
    - hadm_id
    - age
    - cci

The script prints:

    - output file path
    - cohort size
    - mean CCI
    - fraction of admissions with CCI equal to zero

Dependencies
------------
    pip install numpy pandas

Running
-------
Run the script with:

    python create_charlson_index_mimiciii.py \
        --data_root /path/to/mimic-iii \
        --out_path /path/to/baseline_charlson_mimiciii_icd9.csv


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
        help="Path to MIMIC-III root directory containing ADMISSIONS.csv(.gz), PATIENTS.csv(.gz), and DIAGNOSES_ICD.csv(.gz).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output CSV path for baseline_charlson_mimiciii_icd9.csv.",
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
# 2) Load data (MIMIC-III)
# ----------------------------
admissions = read_csv_flexible(base / "ADMISSIONS.csv")
patients   = read_csv_flexible(base / "PATIENTS.csv")
diagnoses  = read_csv_flexible(base / "DIAGNOSES_ICD.csv")

# Normalize column names
admissions.columns = [c.lower() for c in admissions.columns]
patients.columns   = [c.lower() for c in patients.columns]
diagnoses.columns  = [c.lower() for c in diagnoses.columns]


# ----------------------------
# 3) Compute age at admission (overflow-safe)
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

admissions = admissions.merge(
    patients[["subject_id", "dob"]],
    on="subject_id",
    how="left",
)

mask = admissions["admittime"].notna() & admissions["dob"].notna()
admissions["age"] = pd.NA

admit_d = admissions.loc[mask, "admittime"].dt.normalize().to_numpy(dtype="datetime64[D]")
dob_d   = admissions.loc[mask, "dob"].dt.normalize().to_numpy(dtype="datetime64[D]")

age_days = (admit_d - dob_d).astype("timedelta64[D]").astype("int64")
admissions.loc[mask, "age"] = age_days / 365.2425

# MIMIC-III de-id convention
admissions.loc[admissions["age"] > 89, "age"] = 90
admissions.loc[admissions["age"] < 0, "age"] = pd.NA


# ----------------------------
# 4) ICD-9 Mapping
# ----------------------------

cci9 = {
    "MI": {"codes": ["410", "412"], "w": 1},
    "CHF": {"codes": ["39891", "40201", "40211", "40291", "40401", "40403",
                      "40411", "40413", "40491", "40493", "4254", "4255","4256","4257","4258","4259", "428"], "w": 1},
    "PVD": {"codes": ["0930", "4373", "440", "441", "4431","4432","4433","4434","4435","4436","4437","4438", "4439", "4471",
                      "5571", "5579", "V434"], "w": 1},
    "CVD": {"codes": ["36234", "430", "431", "432", "433", "434", "435",
                      "436", "437", "438"], "w": 1},
    "DEMENTIA": {"codes": ["290", "2941", "3312"], "w": 1},
    "CPD": {"codes": ["4168", "4169", "490", "491", "492", "493", "494",
                      "495", "496", "500", "501", "502", "503", "504", "505",
                      "5064", "5081", "5088"], "w": 1},
    "RHEUM": {"codes": ["4465", "7100","7101","7102","7103","7104", "7140","7141","7142","7148", "725"], "w": 1},
    "PUD": {"codes": ["531", "532", "533", "534"], "w": 1},
    "MILD_LIVER": {"codes": ["07022", "07023", "07032", "07033", "07044", "07054",
                             "0706", "0709", "570", "571", "5733", "5734", "5738", "5739", "V427"], "w": 1},
    "DM_NO_COMP": {"codes": ["2500", "2501", "2502", "2503", "2508", "2509"], "w": 1},
    "DM_COMP": {"codes": ["2504", "2505", "2506", "2507"], "w": 2},
    "PARA_HEMI": {"codes": ["3341", "342", "343", "3440","3441","3442","3443","3444", "3445","3446", "3449"], "w": 2},
    "RENAL": {"codes": ["40301", "40311", "40391", "40402", "40403", "40412",
                        "40413", "40492", "40493", "582", "5830", "5831", "5832", "5833", "5834","5835",
                        "5836", "5837", "585", "586", "5880", "V420", "V451", "V56"], "w": 2},
    "MALIGNANCY": {"codes": ["140", "141", "142", "143", "144", "145", "146", "147",
                             "148", "149", "150", "151", "152", "153", "154", "155",
                             "156", "157", "158", "159", "160", "161", "162", "163",
                             "164", "165", "166", "167", "168", "169", "170", "171",
                             "172", "174", "175", "176", "179", "180", "181", "182",
                             "183", "184", "185", "186", "187", "188", "189", "190",
                             "191", "192", "193", "194", "195","1951","1952","1953","1954","1955","1956","1957","1958",
                            "200", "201", "202", "203", "204", "205", "206",
                             "207", "208", "2386"], "w": 2},
    "SEV_LIVER": {"codes": ["4560", "4561", "4562", "5722", "5723", "5724", "5725","5726","5727",
                            "5728"], "w": 3},
    "METS": {"codes": ["196", "197", "198", "199"], "w": 6},
    "AIDS": {"codes": ["042", "043", "044"], "w": 6},
}



def norm_icd9(code) -> str:
    """Uppercase + remove dots/spaces (ICD-9 often has decimals)."""
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()


# Normalize diagnosis codes once (MIMIC-III column is icd9_code)
diagnoses["code_norm"] = diagnoses["icd9_code"].map(norm_icd9)
dx = diagnoses.loc[diagnoses["code_norm"] != "", ["hadm_id", "code_norm"]].drop_duplicates()

# Optional: restrict to admissions that actually have ICD-9 diagnoses
hadm_with_dx = dx["hadm_id"].dropna().unique()
admissions = admissions.loc[admissions["hadm_id"].isin(hadm_with_dx)].copy()

# Pre-normalize mapping prefixes once
mapping_norm = {}
for cat, spec in cci9.items():
    prefixes = [norm_icd9(p) for p in spec["codes"]]
    seen = set()
    prefixes = [p for p in prefixes if p and not (p in seen or seen.add(p))]
    mapping_norm[cat] = {"prefixes": prefixes, "w": int(spec["w"])}

# Build hadm-level flags
hadm_index = pd.Index(dx["hadm_id"].unique(), name="hadm_id")
flags = pd.DataFrame(index=hadm_index)

for cat, spec in mapping_norm.items():
    prefixes = spec["prefixes"]
    if not prefixes:
        flags[cat] = False
        continue

    prefixes = tuple(prefixes)
    matched_hadm = dx.loc[dx["code_norm"].str.startswith(prefixes, na=False), "hadm_id"].unique()
    flags[cat] = False
    flags.loc[matched_hadm, cat] = True


# ----------------------------
# 4b) Hierarchy overrides
# ----------------------------
if "DM_COMP" in flags.columns and "DM_NO_COMP" in flags.columns:
    flags["DM_NO_COMP"] = flags["DM_NO_COMP"] & (~flags["DM_COMP"])

if "SEV_LIVER" in flags.columns and "MILD_LIVER" in flags.columns:
    flags["MILD_LIVER"] = flags["MILD_LIVER"] & (~flags["SEV_LIVER"])

if "METS" in flags.columns and "MALIGNANCY" in flags.columns:
    flags["MALIGNANCY"] = flags["MALIGNANCY"] & (~flags["METS"])


# ----------------------------
# 4c) Compute CCI weight per hadm_id
# ----------------------------
cci_weight = pd.Series(0, index=flags.index, dtype="int64")
for cat, spec in mapping_norm.items():
    if cat in flags.columns:
        cci_weight += flags[cat].astype("int64") * int(spec["w"])

cci_scores = cci_weight.rename("charlson_weight").reset_index()


# ----------------------------
# 5) Merge + finalize
# ----------------------------
df = admissions.merge(cci_scores, on="hadm_id", how="left")
df["charlson_weight"] = df["charlson_weight"].fillna(0).astype("int64")
df["cci"] = df["charlson_weight"]


# ----------------------------
# 6) Save
# ----------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)
df[["subject_id", "hadm_id", "age", "cci"]].to_csv(out_path, index=False)

print(f"✅ Saved Charlson (CCI-only, no age points) → {out_path}")
print("Cohort size:", len(df))
print("CCI mean:", float(df["cci"].mean()))
print("CCI pct zero:", float((df["cci"] == 0).mean()))
