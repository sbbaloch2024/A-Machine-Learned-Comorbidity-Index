"""
MIMIC-III ICD-9 Elixhauser Van Walraven Index Creator
=====================================================

Overview
--------
This script computes the Elixhauser comorbidity index with Van Walraven weights
for MIMIC-III hospital admissions using ICD-9 diagnosis codes. It loads
ADMISSIONS, PATIENTS, and DIAGNOSES_ICD tables, normalizes ICD-9 diagnosis codes,
maps each admission to Elixhauser comorbidity categories, applies selected
hierarchy overrides, and computes one admission-level Van Walraven total score.

The final score, eci_vw_total, is the weighted sum of Elixhauser category flags.
Each category contributes at most once per admission, even if multiple diagnosis
codes match the same category.

The script also computes age at admission for internal cohort processing, using an
overflow-safe day-level datetime calculation. Age is not added to the Elixhauser
score and is not saved in the final output.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-III root directory containing
          ADMISSIONS.csv(.gz), PATIENTS.csv(.gz), and DIAGNOSES_ICD.csv(.gz)
        - --out_path: output CSV path for the Elixhauser Van Walraven score file

2.  Resolve input/output paths:
        - ADMISSIONS.csv or ADMISSIONS.csv.gz
        - PATIENTS.csv or PATIENTS.csv.gz
        - DIAGNOSES_ICD.csv or DIAGNOSES_ICD.csv.gz
        - output CSV path

3.  Load required MIMIC-III tables with flexible CSV loading:
        - Try .csv.gz first.
        - If the compressed file is not found, try .csv.
        - Raise FileNotFoundError if neither exists.
        - Read diagnoses with dtype=str to preserve leading zeros in ICD-9 codes.

4.  Normalize column names:
        - Convert ADMISSIONS column names to lowercase.
        - Convert PATIENTS column names to lowercase.
        - Convert DIAGNOSES_ICD column names to lowercase.

5.  Clean hadm_id values:
        - Convert admissions.hadm_id to numeric.
        - Convert diagnoses.hadm_id to numeric.
        - Drop rows with missing hadm_id.
        - Cast hadm_id to integer in both admissions and diagnoses.

6.  Parse timestamps for age calculation:
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

9.  Normalize ICD-9 diagnosis codes:
        - Convert codes to uppercase.
        - Remove periods.
        - Remove spaces.
        - Strip leading and trailing whitespace.
        - Drop empty normalized codes.
        - Keep unique hadm_id/code_norm pairs.

10. Build the diagnosis-based admission cohort:
        - Keep diagnosis rows whose hadm_id appears in the admissions table.
        - Identify admissions with at least one non-empty normalized diagnosis
          code.
        - Restrict admissions to those hadm_id values.

11. Normalize Elixhauser category prefixes:
        - Normalize each category prefix using the same uppercase/no-dot/no-space
          rule used for diagnosis codes.
        - Drop empty prefixes.
        - Deduplicate prefixes while preserving order.

12. Build admission-level Elixhauser category flags:
        - For each Elixhauser category, identify admissions with at least one
          diagnosis code that starts with any prefix assigned to that category.
        - Store one 0/1 flag per category and hadm_id.
        - Each category is counted at most once per admission.

13. Apply hierarchy overrides:
        - DiabetesComplicated overrides DiabetesUncomplicated.
        - HypertensionComplicated overrides HypertensionUncomplicated.
        - MetastaticCancer overrides SolidTumorWithoutMetastasis.
        - These rules prevent double-counting less severe categories when the
          more severe related category is present.

14. Compute the Van Walraven Elixhauser score:
        - Initialize eci_vw_total to zero.
        - For each category with a Van Walraven weight, multiply the category flag
          by its weight.
        - Sum weighted category indicators per admission.

15. Merge Elixhauser scores back into admissions:
        - Merge by hadm_id.
        - Fill missing eci_vw_total values with zero.
        - Cast eci_vw_total to integer.

16. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save subject_id, hadm_id, and eci_vw_total.
        - Print the output path.

Diagnosis Cohort Definition
---------------------------
The output cohort is restricted to admissions that have at least one non-empty
normalized ICD-9 diagnosis code in DIAGNOSES_ICD after matching hadm_id values to
the admissions table.

Diagnosis Code Normalization
----------------------------
ICD-9 diagnosis codes are read as strings to preserve leading zeros. Codes are
then normalized before matching:

    normalized_code = uppercase(code).replace(".", "").replace(" ", "").strip()

For example:

    042 -> 042
    250.40 -> 25040
    V45.0 -> V450

The Elixhauser mapping prefixes are normalized with the same function so that
dot-form and no-dot-form ICD-9 codes can be matched consistently.

Elixhauser Category Matching
----------------------------
For each Elixhauser category, the script performs prefix matching:

    diagnosis_code starts with any normalized prefix assigned to the category

If any diagnosis code for an admission matches a category, that category flag is
set to 1 for the admission. Multiple matching codes in the same category do not
increase the score beyond that category's single indicator.

Hierarchy Overrides
-------------------
The script applies three hierarchy rules before computing the final score:

    - Complicated diabetes overrides uncomplicated diabetes:
          DiabetesUncomplicated = DiabetesUncomplicated and not DiabetesComplicated

    - Complicated hypertension overrides uncomplicated hypertension:
          HypertensionUncomplicated = HypertensionUncomplicated and not HypertensionComplicated

    - Metastatic cancer overrides solid tumor without metastasis:
          SolidTumorWithoutMetastasis = SolidTumorWithoutMetastasis and not MetastaticCancer

These overrides reduce double-counting between related Elixhauser categories.

Van Walraven Score Definition
-----------------------------
The final score is:

    eci_vw_total = sum(category_present * van_walraven_weight)

where each Elixhauser category contributes at most once per admission. Some
categories have positive weights, some have zero weights, and some have negative
weights.

Age Handling
------------
Age at admission is computed from ADMITTIME and DOB:

    age_days = admittime_date - dob_date
    age = age_days / 365.2425

The script uses day-level datetime arrays rather than nanosecond datetime
differences to avoid overflow issues that can occur with very old MIMIC-III DOB
values.

Age post-processing:
    - age > 89 is set to 90
    - age < 0 is set to missing

Age is computed during preprocessing but is not added to eci_vw_total and is not
saved in the output file.

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
    - eci_vw_total

The script prints:

    - output file path for the saved global Elixhauser index

Dependencies
------------
    pip install numpy pandas

Running
-------
Run the script with:

    python create_elixhauser_index_mimiciii.py \
        --data_root /path/to/mimic-iii \
        --out_path /path/to/baseline_elixhauser_vw_mimiciii_icd9.csv


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
        help="Output CSV path for baseline_elixhauser_vw_mimiciii_icd9.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

base = Path(ARGS.data_root)
out_path = Path(ARGS.out_path)


def read_csv_flexible(path_with_csv: Path, **kwargs):
    """Try .csv.gz then .csv."""
    gz = Path(str(path_with_csv) + ".gz") if str(path_with_csv).endswith(".csv") else Path(str(path_with_csv) + ".csv.gz")
    if gz.exists():
        return pd.read_csv(gz, **kwargs)
    if path_with_csv.exists():
        return pd.read_csv(path_with_csv, **kwargs)
    raise FileNotFoundError(f"Could not find {gz} or {path_with_csv}")


# ----------------------------
# 2) Load data (MIMIC-III)
# ----------------------------
admissions = read_csv_flexible(base / "ADMISSIONS.csv")
patients   = read_csv_flexible(base / "PATIENTS.csv")

# IMPORTANT: read diagnoses as strings to preserve leading zeros in ICD-9
diagnoses  = read_csv_flexible(base / "DIAGNOSES_ICD.csv", dtype=str)

# Normalize column names
admissions.columns = [c.lower() for c in admissions.columns]
patients.columns   = [c.lower() for c in patients.columns]
diagnoses.columns  = [c.lower() for c in diagnoses.columns]

# Ensure hadm_id is numeric where needed
admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce")
diagnoses["hadm_id"]  = pd.to_numeric(diagnoses["hadm_id"], errors="coerce")

# Drop rows with missing HADM_ID
admissions = admissions.dropna(subset=["hadm_id"]).copy()
diagnoses  = diagnoses.dropna(subset=["hadm_id"]).copy()
admissions["hadm_id"] = admissions["hadm_id"].astype(int)
diagnoses["hadm_id"]  = diagnoses["hadm_id"].astype(int)


# ----------------------------
# 3) Compute age at admission (MIMIC-III, overflow-safe; mirrors other scripts)
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

admissions = admissions.merge(patients[["subject_id", "dob"]], on="subject_id", how="left")

mask = admissions["admittime"].notna() & admissions["dob"].notna()
admissions["age"] = pd.NA

admit_d = admissions.loc[mask, "admittime"].dt.normalize().to_numpy(dtype="datetime64[D]")
dob_d   = admissions.loc[mask, "dob"].dt.normalize().to_numpy(dtype="datetime64[D]")

age_days = (admit_d - dob_d).astype("timedelta64[D]").astype("int64")
admissions.loc[mask, "age"] = age_days / 365.2425

admissions.loc[admissions["age"] > 89, "age"] = 90
admissions.loc[admissions["age"] < 0, "age"] = pd.NA


# ----------------------------
# 4) ICD-9 normalization + dx table
# ----------------------------
def _norm(code):
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()


# MIMIC-III diagnoses column is icd9_code
diagnoses["code_norm"] = diagnoses["icd9_code"].map(_norm)

# Keep only non-empty codes, unique hadm/code pairs
dx = diagnoses.loc[diagnoses["code_norm"] != "", ["hadm_id", "code_norm"]].drop_duplicates()

# Mirror MIMIC-IV script: keep dx within admissions, then restrict admissions to those hadm_ids
dx = dx[dx["hadm_id"].isin(admissions["hadm_id"])].copy()
hadm_dx = dx["hadm_id"].unique()
admissions = admissions[admissions["hadm_id"].isin(hadm_dx)].copy()




# ----------------------------
# 5) ICD-9 Elixhauser mapping ( keep keys same as VW_WEIGHTS)
# ----------------------------
ECI_MAP = {
    "CongestiveHeartFailure": [
        "39891", "40201", "40211", "40291",
        "40401", "40403", "40411", "40413",
        "40491", "40493", "4254", "4255","4256",
        "4257", "4258", "4259", "428"
    ],

    "CardiacArrhythmias": [
        "4260", "42613", "4267", "4269", "42610", "42612","4270","4271","4272","4273","4274","4276","4277","4278","4279","7850",
        "99601","99604","V450", "V533"
    ],

    "ValvularDisease": [
        "0932", "394", "395", "396", "397",
        "424", "7463", "7464", "7465", "7466",
        "V422", "V433"
    ],

    "PulmonaryCirculationDisorders": [
        "4150", "4151", "416" ,"4170","4178","4179",
    ],

    "PeripheralVascularDisorders": [
        "0930", "4373", "440", "441", "4431","4432", "4433",
        "4434", "4435", "4436", "4437", "4438", "4439",
        "4471", "5571", "5579", "V434"
    ],

    "HypertensionUncomplicated": [
        "401",
    ],

    "HypertensionComplicated": [
        "402", "403", "404", "405"
    ],

    "Paralysis": [
        "3341", "342", "343", "3440", "3441",
        "3442", "3443", "3444", "3445", "3446", "3449"
    ],

    "OtherNeurologicalDisorders": [
        "3319", "3320","3321",  "3334","3335" ,"33392", "334", "335",
        "3362", "340", "341", "345", "3481", "3483",
        "7803", "7843"
    ],

    "ChronicPulmonaryDisease": [
        "4168", "4169", "490", "491", "492", "493",
        "494", "495", "496","497","498", "499","500", "501", "502", "503",
        "504", "505", "5064", "5081", "5088"
    ],

    "DiabetesUncomplicated": [
        "2500", "2501", "2502", "2503"
    ],

    "DiabetesComplicated": [
        "2504", "2505", "2506", "2507", "2508", "2509"
    ],

    "Hypothyroidism": [
        "2409", "243", "244", "2461", "2468"
    ],

    "RenalFailure": [
        "40301", "40311", "40391", "40402", "40403",
        "40412", "40413", "40492", "40493", "585","586", "5880",
        "V420", "V451", "V56"
    ],

    "LiverDisease": [
        "07022", "07023", "07032", "07033",
        "07044", "07054", "0706", "0709","4560","4561","4562",
        "570", "571", "5722", "5723","5724", "5725", "5726", "5727","5728", 
        "5733", "5734", "5738", "5739", "V427"
    ],

    "PepticUlcerDiseaseExcludingBleeding": [
        "5317","5319","5327","5329","5337", "5339","5347","5349"    ],

    "AidsHiv": [
        "042", "043", "044"
    ],

    "Lymphoma": [
        "200", "201", "202", "2030", "2386"
    ],

    "MetastaticCancer": [
        "196", "197", "198", "199"
    ],

    "SolidTumorWithoutMetastasis": [
        "140", "141", "142", "143", "144", "145", "146", "147",
        "148", "149", "150", "151", "152", "153", "154", "155",
        "156", "157", "158", "159", "160", "161", "162", "163",
        "164", "165", "166", "167", "168", "169", "170", "171",
        "172", "174", "175", "176", "177", "178", "179", "180",
        "181", "182", "183", "184", "185", "186", "187", "188",
        "189", "190", "191", "192", "193", "194", "195"
    ],

    "RheumatoidArthritisCollagenVascularDiseases": [
        "446", "7010", "7100", "7101", "7102", "7103",
        "7104",  "7108",  "7109",  "7112", "714", "7193", "720", "725", "7285", "72889", "72930"
    ],

    "Coagulopathy": [
        "286", "2871", "2873", "2874", "2875"
    ],

    "Obesity": [
        "2780"
    ],

    "WeightLoss": [
        "260", "261", "262", "263", "7832","7994"
    ],

    "FluidAndElectrolyteDisorders": [
        "2536", "276"
    ],

    "BloodLossAnemia": [
        "2800"
    ],

    "DeficiencyAnemia": [
        "2801","2802","2803","2804","2805","2806","2807", "2808", "2809", "281"
    ],

    "AlcoholAbuse": [
        "2652", "2911", "2912", "2913","2915","2916","2917","2918","2919",  "3030", "3039", "3050", "3575", "4255",
        "5353", "5710", "5711", "5712", "5713", "980",
        "V113"
    ],

    "DrugAbuse": [
        "292", "304", "3052", "3053", "3054", "3055",
        "3056", "3057", "3058", "3059","V6542"
    ],

    "Psychoses": [
        "2938", "295", "29604", "29614", "29644",
        "29654", "297", "298"
    ],

    "Depression": [
        "2962", "2963", "2965",
        "3004", "309", "311"
    ],
}


# Van Walraven weights (same as MIMIC-IV script)
VW_WEIGHTS = {
    "CongestiveHeartFailure": 7,
    "CardiacArrhythmias": 5,
    "ValvularDisease": -1,
    "PulmonaryCirculationDisorders": 4,
    "PeripheralVascularDisorders": 2,
    "HypertensionUncomplicated": 0,
    "HypertensionComplicated": 0,
    "Paralysis": 7,
    "OtherNeurologicalDisorders": 6,
    "ChronicPulmonaryDisease": 3,
    "DiabetesUncomplicated": 0,
    "DiabetesComplicated": 0,
    "Hypothyroidism": 0,
    "RenalFailure": 5,
    "LiverDisease": 11,
    "PepticUlcerDiseaseExcludingBleeding": 0,
    "AidsHiv": 0,
    "Lymphoma": 9,
    "MetastaticCancer": 12,
    "SolidTumorWithoutMetastasis": 4,
    "RheumatoidArthritisCollagenVascularDiseases": 0,
    "Coagulopathy": 3,
    "Obesity": -4,
    "WeightLoss": 6,
    "FluidAndElectrolyteDisorders": 5,
    "BloodLossAnemia": -2,
    "DeficiencyAnemia": -2,
    "AlcoholAbuse": 0,
    "DrugAbuse": -7,
    "Psychoses": 0,
    "Depression": -3,
}

# Pre-normalize mapping prefixes once (fast), like MIMIC-IV script
eci_norm = {}
for cat, prefixes in ECI_MAP.items():
    cleaned = []
    seen = set()
    for p in prefixes:
        p2 = _norm(p)
        if p2 and p2 not in seen:
            cleaned.append(p2)
            seen.add(p2)
    eci_norm[cat] = tuple(cleaned)


# ----------------------------
# 6) Build hadm-level flags (0/1), like MIMIC-IV script
# ----------------------------
hadm_index = pd.Index(hadm_dx, name="hadm_id")
flags = pd.DataFrame(index=hadm_index)

for cat, prefixes in eci_norm.items():
    if not prefixes:
        flags[cat] = 0
        continue

    matched = dx["code_norm"].str.startswith(prefixes, na=False)
    matched_hadm = dx.loc[matched, "hadm_id"].unique()

    flags[cat] = 0
    flags.loc[matched_hadm, cat] = 1

eci_by_hadm = flags.reset_index()


# ----------------------------
# 7) Hierarchy overrides (same intent as MIMIC-IV, but done safely)
# ----------------------------
def _override(uncomp: str, comp: str):
    if uncomp in eci_by_hadm.columns and comp in eci_by_hadm.columns:
        u = eci_by_hadm[uncomp].astype(bool)
        c = eci_by_hadm[comp].astype(bool)
        eci_by_hadm[uncomp] = (u & ~c).astype(int)


_override("DiabetesUncomplicated", "DiabetesComplicated")
_override("HypertensionUncomplicated", "HypertensionComplicated")
_override("SolidTumorWithoutMetastasis", "MetastaticCancer")


# ----------------------------
# 8) Van Walraven total score (same as MIMIC-IV script)
# ----------------------------
eci_by_hadm["eci_vw_total"] = 0
for k, w in VW_WEIGHTS.items():
    if k in eci_by_hadm.columns:
        eci_by_hadm["eci_vw_total"] += eci_by_hadm[k] * int(w)


# ----------------------------
# 9) Merge and save (match MIMIC-IV output columns)
# ----------------------------
df = admissions.merge(eci_by_hadm[["hadm_id", "eci_vw_total"]], on="hadm_id", how="left")
df["eci_vw_total"] = df["eci_vw_total"].fillna(0).astype(int)

out_path.parent.mkdir(parents=True, exist_ok=True)
df[["subject_id", "hadm_id", "eci_vw_total"]].to_csv(out_path, index=False)

print(f"✅ Saved global Elixhauser index → {out_path}")