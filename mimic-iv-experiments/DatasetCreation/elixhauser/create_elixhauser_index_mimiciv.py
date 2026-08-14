"""
ICD-10-Only Elixhauser Van Walraven Index Creator
=================================================

Overview
--------
This script computes the Elixhauser comorbidity index with Van Walraven weights
for hospital admissions in a MIMIC-IV ICD-10 diagnosis cohort. It uses ICD-10
diagnosis codes from hosp/diagnoses_icd.csv.gz and maps each admission to
Elixhauser comorbidity categories. Each category contributes at most once per
admission, even if multiple diagnosis codes match the same category.

The final score, eci_vw_total, is the weighted sum of Elixhauser category flags
using Van Walraven weights. The script computes admission age for internal cohort
processing, but age is not added to the Elixhauser score and is not saved in the
final output.

The script restricts the saved cohort to admissions that actually have at least
one ICD-10 diagnosis code after filtering and normalization.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-IV root directory containing the hosp/
          subdirectory
        - --out_path: output CSV path for the Elixhauser Van Walraven score file

2.  Resolve input paths:
        - hosp/admissions.csv.gz
        - hosp/patients.csv.gz
        - hosp/diagnoses_icd.csv.gz

3.  Load required MIMIC-IV tables:
        - admissions
        - patients
        - diagnoses_icd

4.  Restrict diagnoses to ICD-10:
        - Keep only diagnoses rows where icd_version == 10.

5.  Parse admission time and compute admission age:
        - Convert admissions.admittime to datetime.
        - Merge admissions with patients on subject_id.
        - Use anchor_age and anchor_year to approximate admission age.
        - Cap ages above 89 at 90.

6.  Normalize diagnosis codes:
        - Convert codes to uppercase.
        - Remove periods.
        - Remove spaces.
        - Strip leading and trailing whitespace.
        - Drop empty normalized codes.

7.  Build the ICD-10 admission cohort:
        - Keep diagnosis rows whose hadm_id appears in the admissions table.
        - Identify admissions with at least one non-empty normalized ICD-10
          diagnosis code.
        - Restrict admissions to those hadm_id values.

8.  Normalize Elixhauser category prefixes:
        - Normalize each category prefix using the same uppercase/no-dot/no-space
          rule used for diagnosis codes.
        - Drop empty prefixes.
        - Deduplicate prefixes while preserving order.

9.  Build admission-level Elixhauser category flags:
        - For each Elixhauser category, identify admissions with at least one
          diagnosis code that starts with any prefix assigned to that category.
        - Store one 0/1 flag per category and hadm_id.
        - Each category is counted at most once per admission.

10. Apply hierarchy overrides:
        - DiabetesComplicated overrides DiabetesUncomplicated.
        - HypertensionComplicated overrides HypertensionUncomplicated.
        - MetastaticCancer overrides SolidTumorWithoutMetastasis.
        - These rules prevent double-counting less severe categories when the
          more severe related category is present.

11. Compute the Van Walraven Elixhauser score:
        - Initialize eci_vw_total to zero.
        - For each category with a Van Walraven weight, multiply the category flag
          by its weight.
        - Sum weighted category indicators per admission.

12. Merge Elixhauser scores back into admissions:
        - Merge by hadm_id.
        - Fill missing values with zero.

13. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save subject_id, hadm_id, and eci_vw_total.
        - Print the output path.

ICD-10 Cohort Definition
------------------------
The ICD-10 cohort is defined using the diagnoses table after filtering:

    diagnoses_icd.icd_version == 10

An admission is included in the final output if its hadm_id has at least one
non-empty normalized ICD-10 diagnosis code.

Diagnosis Code Normalization
----------------------------
Diagnosis codes are normalized before matching:

    normalized_code = uppercase(code).replace(".", "").replace(" ", "").strip()

For example:

    I50.9 -> I509
    E11.9 -> E119
    C78.7 -> C787

The Elixhauser mapping prefixes are normalized with the same function so that
dot-form and no-dot-form ICD-10 codes can be matched consistently.

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
Admission age is approximated using MIMIC-IV anchor fields:

    age = anchor_age + (admittime year - anchor_year)

Ages greater than 89 are set to 90:

    if age > 89:
        age = 90

Age is computed during preprocessing but is not added to eci_vw_total and is not
saved in the output file.

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
    subject_id, hadm_id, admittime

From hosp/patients.csv.gz:
    subject_id, anchor_age, anchor_year

From hosp/diagnoses_icd.csv.gz:
    hadm_id, icd_code, icd_version

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
    pip install pandas numpy

Running
-------
Run the script with:

    python create_elixhauser_index_mimiciv.py \
        --data_root /path/to/mimic-iv \
        --out_path /path/to/baseline_elixhauser_vw_mimiciv_icd10.csv


"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# ----------------------------
# 1. Paths
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
        help="Output CSV path for baseline_elixhauser_vw_mimiciv_icd10.csv.",
    )
    return parser.parse_args()


ARGS = parse_args()

data_root = Path(ARGS.data_root)
base = data_root / "hosp"

out_path = Path(ARGS.out_path)


# ----------------------------
# 2. Load data
# ----------------------------
admissions = pd.read_csv(base / "admissions.csv.gz")
patients = pd.read_csv(base / "patients.csv.gz")
diagnoses = pd.read_csv(base / "diagnoses_icd.csv.gz")
diagnoses = diagnoses[diagnoses["icd_version"] == 10].copy()


# ----------------------------
# 3. Compute age at admission
# ----------------------------
admissions["admittime"] = pd.to_datetime(admissions["admittime"])

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
# 4. Full ICD-10 Elixhauser mapping
# ----------------------------
def _norm(code):
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "").strip()


# Normalize ICD10 codes once + keep only non-empty
diagnoses["code_norm"] = diagnoses["icd_code"].map(_norm)
dx = diagnoses.loc[
    diagnoses["code_norm"] != "",
    ["hadm_id", "code_norm"],
].drop_duplicates()

dx = dx[dx["hadm_id"].isin(admissions["hadm_id"])].copy()
hadm_icd10 = dx["hadm_id"].unique()
admissions = admissions[admissions["hadm_id"].isin(hadm_icd10)].copy()





ECI_MAP = {
    "CongestiveHeartFailure": [
        "I09.9", "I11.0", "I13.0", "I13.2",
        "I25.5", "I42.0", "I42.5", "I42.6", "I42.7", "I42.8", "I42.9",
        "I43",
        "I50",
        "P29.0"
    ],
    
    "CardiacArrhythmias": [
       "I44.1", "I44.2", "I44.3",
        "I45.6", "I45.9",
        "I47",
        "I48",
        "I49", 
        "R00.0", "R00.1", "R00.8", "T82.1",
        "Z45.0", "Z95.0"
    ],
    
    "ValvularDisease": [
        "A52.0",
        "I05", 
        "I06",
        "I07", 
        "I08",
        "I09.1", "I09.8",
        "I34",
        "I35", 
        "I36",
        "I37",
        "I38",
        "I39",
        "Q23.0", "Q23.1", "Q23.2", "Q23.3",
        "Z95.2", "Z95.3", "Z95.4"
    ],
    
    "PulmonaryCirculationDisorders": [
        "I26",
        "I27",
        "I28.0","I28.8", "I28.9"
    ],
    
    "PeripheralVascularDisorders": [
        "I70",
        "I71", 
        "I73.1", "I73.8", "I73.9",
       "I77.1", 
        "I79.0","I79.2",
        "K55.1", "K55.8", "K55.9",
        "Z95.8", "Z95.9"
    ],
    
    "HypertensionUncomplicated": [
        "I10", 
    ],
    
    "HypertensionComplicated": [
        "I11", 
        "I12",
        "I13",
        "I15", 
    ],
    
    "Paralysis": [
        "G04.1",
        "G11.4",
        "G80.1", "G80.2",
        "G81",
        "G82", 
        "G83.0", "G83.1", "G83.2", "G83.3", "G83.4", "G83.9"
    ],
    
    "OtherNeurologicalDisorders": [
        "G10",
        "G11",
        "G12",
        "G13", 
        "G20", 
        "G21", 
        "G22", 
        "G25.4", "G25.5",
        "G31.2", "G31.8", "G31.9",
        "G32",
        "G35",
        "G36", 
        "G37",
        "G40", 
        "G41", 
        "G93.1", "G93.4",
        "R47.0",
        "R56"
    ],
    
    "ChronicPulmonaryDisease": [
        "I27.8", "I27.9",
        "J40",
        "J41", 
        "J42", 
        "J43", 
        "J44", 
        "J45", 
        "J46", 
        "J47",
        "J60", 
        "J61",
        "J62",
        "J63", 
        "J64",
        "J65",
        "J66",
        "J67",
        "J68.4",
        "J70.1", "J70.3"
    ],
    
    "DiabetesUncomplicated": [
        "E10.0", "E10.1", "E10.9",
        "E11.0", "E11.1", "E11.9",
        "E12.0", "E12.1", "E12.9",
        "E13.0", "E13.1", "E13.9",
        "E14.0", "E14.1", "E14.9"
    ],
    
    "DiabetesComplicated": [
        "E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8",
        "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8",
        "E12.2", "E12.3", "E12.4", "E12.5", "E12.6", "E12.7", "E12.8",
        "E13.2", "E13.3", "E13.4", "E13.5", "E13.6", "E13.7", "E13.8",
        "E14.2", "E14.3", "E14.4", "E14.5", "E14.6", "E14.7", "E14.8"
    ],
    
    "Hypothyroidism": [
        "E00", 
        "E01", 
        "E02", 
        "E03", 
        "E89.0"
    ],
    
    "RenalFailure": [
        "I12.0", 
        "I13.1",
        "N18", 
        "N19",
        "N25.0",
        "Z49.0", "Z49.1", "Z49.2",
        "Z94.0",
        "Z99.2"
    ],
    
    "LiverDisease": [
        "B18",
        "I85", 
        "I86.4",
        "I98.2",
        "K70", 
        "K71.1", "K71.3", "K71.4", "K71.5", "K71.7", 
        "K72",
        "K73", 
        "K74", 
        "K76.0", "K76.2", "K76.3", "K76.4", "K76.5", "K76.6", "K76.7", "K76.8", "K76.9",
        "Z94.4"
    ],
    
    "PepticUlcerDiseaseExcludingBleeding": [
        "K25.7", "K25.9",
        "K26.7", "K26.9",
        "K27.7", "K27.9",
        "K28.7", "K28.9"
    ],
    
    "AidsHiv": [
        "B20",
        "B21", 
        "B22",
        "B24", 
    ],
    
    "Lymphoma": [
        "C81",
        "C82",
        "C83", 
        "C84", 
        "C85", 
        "C88",
        "C90.0",  "C90.2", 
        "C96",
    ],
    
    "MetastaticCancer": [
        "C77",
        "C78", 
        "C79", 
        "C80", 
    ],
    
    "SolidTumorWithoutMetastasis": [
        "C00", 
        "C01",
        "C02", 
        "C03",
        "C04", 
        "C05",
        "C06",
        "C07", 
        "C08", 
        "C09",
        "C10", 
        "C11", 
        "C12",
        "C13",
        "C14",
        "C15", 
        "C16", 
        "C17",
        "C18",
        "C19", 
        "C20", 
        "C21", 
        "C22", 
        "C23", 
        "C24" ,
        "C25", 
        "C26",
        "C30",
        "C31", 
        "C32", 
        "C33",
        "C34",
        "C37", 
        "C38", 
        "C39", 
        "C40",
        "C41", 
        "C43",
        "C45", 
    "C46", "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",

    "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76",

        "C97",
    ],
    
    "RheumatoidArthritisCollagenVascularDiseases": [
        "L94.0", "L94.1", "L94.3",
        "M05",
        "M06", 
        "M08", 
        "M12.0", "M12.3",
        "M30", 
        "M31.0", "M31.1", "M31.2", "M31.3",
        "M32", "M33", "M34", "M35",
        "M45",
         "M46.1", "M46.8", "M46.9"
    ],
    
    "Coagulopathy": [
   "D65", "D66", "D67", "D68",
        "D69.1", "D69.3", "D69.4", "D69.5", "D69.6"
    ],
    
    "Obesity": [
        "E66",
    ],
    
    "WeightLoss": [
     "E40", "E41", "E42", "E43", "E44", "E45", "E46",
        "R63.4", "R64"
    ],
    
    "FluidAndElectrolyteDisorders": [
        "E22.2",
        "E86", 
        "E87", 
    ],
    
    "BloodLossAnemia": [
        "D50.0"
    ],
    
    "DeficiencyAnemia": [
        "D50.8", "D50.9",
        "D51", 
        "D52", 
        "D53", 
    ],
    
    "AlcoholAbuse": [
        "F10", 
        "E52", 
        "G62.1",
        "I42.6",
        "K29.2",
        "K70.0", "K70.3", "K70.9",
        "T51",
        "Z50.2",
        "Z71.4", 
        "Z72.1"
    ],
    
    "DrugAbuse": [
       "F11", "F12", "F13", "F14", "F15", "F16",
        "F18",
        "F19", 
        "Z71.5", "Z72.2"
    ],
    
    "Psychoses": [
 "F20", "F22", "F23", "F24", "F25", "F28", "F29",
        "F30.2",
        "F31.2", "F31.5"
    ],
    
    "Depression": [
        "F20.4",
        "F31.3", "F31.4", "F31.5",
        "F32",
        "F33", 
      "F34.1",
        "F41.2",
        "F43.2"
    ]
}

# Van Walraven weights, keyed to ECI_MAP names
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



# Pre-normalize mapping prefixes once (fast)
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


# Build hadm-level flags (0/1) using vectorized startswith
hadm_index = pd.Index(hadm_icd10, name="hadm_id")
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
# 4b. Hierarchy overrides (avoid double-counting)
# ----------------------------
# 1) Diabetes complicated overrides uncomplicated
if (
    "DiabetesComplicated" in eci_by_hadm.columns
    and "DiabetesUncomplicated" in eci_by_hadm.columns
):
    eci_by_hadm["DiabetesUncomplicated"] = (
        eci_by_hadm["DiabetesUncomplicated"]
        & (~eci_by_hadm["DiabetesComplicated"])
    ).astype(int)


# 2) Hypertension complicated overrides uncomplicated
if (
    "HypertensionComplicated" in eci_by_hadm.columns
    and "HypertensionUncomplicated" in eci_by_hadm.columns
):
    eci_by_hadm["HypertensionUncomplicated"] = (
        eci_by_hadm["HypertensionUncomplicated"]
        & (~eci_by_hadm["HypertensionComplicated"])
    ).astype(int)


# 3) Metastatic cancer overrides solid tumor without metastasis
if (
    "MetastaticCancer" in eci_by_hadm.columns
    and "SolidTumorWithoutMetastasis" in eci_by_hadm.columns
):
    eci_by_hadm["SolidTumorWithoutMetastasis"] = (
        eci_by_hadm["SolidTumorWithoutMetastasis"]
        & (~eci_by_hadm["MetastaticCancer"])
    ).astype(int)


# VW score
eci_by_hadm["eci_vw_total"] = 0

for k, w in VW_WEIGHTS.items():
    if k in eci_by_hadm.columns:
        eci_by_hadm["eci_vw_total"] += eci_by_hadm[k] * w


# ----------------------------
# 5. Merge and save
# ----------------------------
df = admissions.merge(eci_by_hadm, on="hadm_id", how="left").fillna(0)

out_path.parent.mkdir(parents=True, exist_ok=True)

df[["subject_id", "hadm_id", "eci_vw_total"]].to_csv(out_path, index=False)

print(f"✅ Saved global Elixhauser index → {out_path}")
