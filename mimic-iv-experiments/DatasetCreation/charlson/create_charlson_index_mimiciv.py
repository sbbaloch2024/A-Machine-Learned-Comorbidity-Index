"""
ICD-10-Only Charlson Comorbidity Index Creator
==============================================

Overview
--------
This script computes the Charlson Comorbidity Index (CCI) for hospital admissions
in a MIMIC-IV ICD-10 diagnosis cohort. It uses ICD-10 diagnosis codes from
hosp/diagnoses_icd.csv.gz and maps each admission to Charlson comorbidity
categories. Each category contributes its Charlson weight at most once per
admission, even if multiple diagnosis codes match the same category.

This script computes the comorbidity-only Charlson score. It does not add an
age-adjustment component to the final CCI score. Age is computed and saved for
reference, but the output cci column is equal to the summed Charlson diagnosis
weights only.

The script restricts the saved cohort to admissions that actually have at least
one ICD-10 diagnosis code after filtering and normalization.

Pipeline Flow
-------------
1.  Parse command-line arguments:
        - --data_root: path to the MIMIC-IV root directory containing the hosp/
          subdirectory
        - --out_path: output CSV path for the Charlson score file

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
        - Drop empty normalized codes.

7.  Build the ICD-10 admission cohort:
        - Keep only admissions whose hadm_id appears in the normalized ICD-10
          diagnosis table.
        - This ensures the output contains only admissions with at least one
          ICD-10 diagnosis code.

8.  Normalize Charlson category prefixes:
        - Normalize each ICD-10 prefix in the Charlson mapping using the same
          uppercase/no-dot/no-space rule.
        - Drop empty prefixes.
        - Deduplicate prefixes while preserving order.

9.  Build admission-level Charlson category flags:
        - For each Charlson category, identify admissions with at least one
          diagnosis code that starts with any prefix assigned to that category.
        - Store one Boolean flag per category and hadm_id.
        - Each category is counted at most once per admission.

10. Apply Charlson hierarchy overrides:
        - DM_COMP overrides DM_NO_COMP.
        - SEV_LIVER overrides MILD_LIVER.
        - METS overrides MALIGNANCY.
        - These rules prevent double-counting less severe categories when the
          more severe related category is present.

11. Compute the Charlson diagnosis weight:
        - Multiply each category flag by its Charlson category weight.
        - Sum weighted category indicators per admission.
        - Store the result as charlson_weight.

12. Merge Charlson scores back into admissions:
        - Merge by hadm_id.
        - Fill missing Charlson weights with zero.
        - Set cci = charlson_weight.

13. Save the output CSV:
        - Create the output directory if it does not exist.
        - Save subject_id, hadm_id, age, and cci.
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

    normalized_code = uppercase(code).replace(".", "").replace(" ", "")

For example:

    I25.2 -> I252
    E11.9 -> E119

The Charlson mapping prefixes are normalized with the same function so that
dot-form and no-dot-form ICD-10 codes can be matched consistently.

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

    - Metastatic solid tumor overrides any malignancy:
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
Admission age is approximated using MIMIC-IV anchor fields:

    age = anchor_age + (admittime year - anchor_year)

Ages greater than 89 are set to 90:

    if age > 89:
        age = 90

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
    - age
    - cci

The script prints:

    - output file path for the saved Charlson file

Dependencies
------------
    pip install pandas numpy

Running
-------
Run the script with:

    python create_charlson_index_mimiciv.py \
        --data_root /path/to/mimic-iv \
        --out_path /path/to/baseline_charlson_mimiciv_icd10.csv


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
        help="Output CSV path for baseline_charlson_mimiciv_icd10.csv.",
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
# 4. Full ICD-10 Charlson map (Quan et al. 2005)
# ----------------------------
cci10 = {'MI': {'codes': ['I21', 'I22', 'I25.2'], 'w': 1},
 'CHF': {'codes': ['I09.9',
                   'I11.0',
                   'I13.0',
                   'I13.2',
                   'I25.5',
                   'I42.0',
                   'I42.5',
                   'I42.6',
                   'I42.7',
                   'I42.8',
                   'I42.9',
                   'I43',
                   'I50',
                   'P29.0'],
         'w': 1},
 'PVD': {'codes': ['I70',
                   'I71',
                   'I73.1',
                   'I73.8',
                   'I73.9',
                   'I77.1',
                   'I79.0',
                   'I79.2',
                   'K55.1',
                   'K55.8',
                   'K55.9',
                   'Z95.8',
                   'Z95.9'],
         'w': 1},
 'CVD': {'codes': ['G45',
                   'G46',
                   'H34.0',
                   'I60',
                   'I61',
                   'I62',
                   'I63',
                   'I64',
                   'I65',
                   'I66',
                   'I67',
                   'I68',
                   'I69'],
         'w': 1},
 'DEMENTIA': {'codes': ['F00', 'F01', 'F02', 'F03', 'F05.1', 'G30', 'G31.1'],
              'w': 1},
 'CPD': {'codes': ['I27.8',
                   'I27.9',
                   'J40',
                   'J41',
                   'J42',
                   'J43',
                   'J44',
                   'J45',
                   'J46',
                   'J47',
                   'J60',
                   'J61',
                   'J62',
                   'J63',
                   'J64',
                   'J65',
                   'J66',
                   'J67',
                   'J68.4',
                   'J70.1',
                   'J70.3'],
         'w': 1},
 'RHEUM': {'codes': ['M05',
                     'M06',
                     'M31.5',
                     'M32',
                     'M33',
                     'M34',
                     'M35.1',
                     'M35.3',
                     'M36.0'],
           'w': 1},
 'PUD': {'codes': ['K25', 'K26', 'K27', 'K28'], 'w': 1},
 'MILD_LIVER': {'codes': ['B18',
                          'K70.0',
                          'K70.1',
                          'K70.2',
                          'K70.3',
                          'K70.9',
                          'K71.3',
                          'K71.4',
                          'K71.5',
                          'K71.7',
                          'K73',
                          'K74',
                          'K76.0',
                          'K76.2',
                          'K76.3',
                          'K76.4',
                          'K76.8',
                          'K76.9',
                          'Z94.4'],
                'w': 1},
 'DM_NO_COMP': {'codes': ['E10.0',
                          'E10.1',
                          'E10.6',
                          'E10.8',
                          'E10.9',
                          'E11.0',
                          'E11.1',
                          'E11.6',
                          'E11.8',
                          'E11.9',
                          'E12.0',
                          'E12.1',
                          'E12.6',
                          'E12.8',
                          'E12.9',
                          'E13.0',
                          'E13.1',
                          'E13.6',
                          'E13.8',
                          'E13.9',
                          'E14.0',
                          'E14.1',
                          'E14.6',
                          'E14.8',
                          'E14.9'],
                 'w': 1},
 'DM_COMP': {'codes': ['E10.2',
                       'E10.3',
                       'E10.4',
                       'E10.5',
                       'E10.7',
                       'E11.2',
                       'E11.3',
                       'E11.4',
                       'E11.5',
                       'E11.7',
                       'E12.2',
                       'E12.3',
                       'E12.4',
                       'E12.5',
                       'E12.7',
                       'E13.2',
                       'E13.3',
                       'E13.4',
                       'E13.5',
                       'E13.7',
                       'E14.2',
                       'E14.3',
                       'E14.4',
                       'E14.5',
                       'E14.7'],
             'w': 2},
 'PARA_HEMI': {'codes': ['G04.1',
                         'G11.4',
                         'G80.1',
                         'G80.2',
                         'G81',
                         'G82',
                         'G83.0',
                         'G83.1',
                         'G83.2',
                         'G83.3',
                         'G83.4',
                         'G83.9'],
               'w': 2},
 'RENAL': {'codes': ['I12.0',
                     'I13.1',
                     'N03.2',
                     'N03.3',
                     'N03.4',
                     'N03.5',
                     'N03.6',
                     'N03.7',
                     'N05.2',
                     'N05.3',
                     'N05.4',
                     'N05.5',
                     'N05.6',
                     'N05.7',
                     'N18',
                     'N19',
                     'N25.0',
                     'Z49.0',
                     'Z49.1',
                     'Z49.2',
                     'Z94.0',
                     'Z99.2'],
           'w': 2},
 'MALIGNANCY': {'codes': ['C00',
                          'C01',
                          'C02',
                          'C03',
                          'C04',
                          'C05',
                          'C06',
                          'C07',
                          'C08',
                          'C09',
                          'C10',
                          'C11',
                          'C12',
                          'C13',
                          'C14',
                          'C15',
                          'C16',
                          'C17',
                          'C18',
                          'C19',
                          'C20',
                          'C21',
                          'C22',
                          'C23',
                          'C24',
                          'C25',
                          'C26',
                          'C30',
                          'C31',
                          'C32',
                          'C33',
                          'C34',
                          'C37',
                          'C38',
                          'C39',
                          'C40',
                          'C41',
                          'C43',
                          'C45',
                          'C46',
                          'C47',
                          'C48',
                          'C49',
                          'C50',
                          'C51',
                          'C52',
                          'C53',
                          'C54',
                          'C55',
                          'C56',
                          'C57',
                          'C58',
                          'C60',
                          'C61',
                          'C62',
                          'C63',
                          'C64',
                          'C65',
                          'C66',
                          'C67',
                          'C68',
                          'C69',
                          'C70',
                          'C71',
                          'C72',
                          'C73',
                          'C74',
                          'C75',
                          'C76',
                          'C81',
                          'C82',
                          'C83',
                          'C84',
                          'C85',
                          'C88',
                          'C90',
                          'C91',
                          'C92',
                          'C93',
                          'C94',
                          'C95',
                          'C96',
                          'C97'],
                'w': 2},
 'SEV_LIVER': {'codes': ['I85.0',
                         'I85.9',
                         'I86.4',
                         'I98.2',
                         'K70.4',
                         'K71.1',
                         'K72.1',
                         'K72.9',
                         'K76.5',
                         'K76.6',
                         'K76.7'],
               'w': 3},
 'METS': {'codes': ['C77', 'C78', 'C79', 'C80'], 'w': 6},
 'AIDS': {'codes': ['B20', 'B21', 'B22', 'B24'], 'w': 6}}




def norm_icd(code) -> str:
    """Uppercase + remove dots/spaces. Safe for MIMIC ICD10 which is often no-dot already."""
    if pd.isna(code):
        return ""
    return str(code).upper().replace(".", "").replace(" ", "")


# Normalize diagnosis codes once
diagnoses["code_norm"] = diagnoses["icd_code"].map(norm_icd)
dx = diagnoses.loc[
    diagnoses["code_norm"] != "",
    ["hadm_id", "code_norm"],
].drop_duplicates()


# ✅ Keep ONLY admissions that actually have ICD-10 diagnoses
hadm_icd10 = dx["hadm_id"].unique()
admissions = admissions[admissions["hadm_id"].isin(hadm_icd10)].copy()


# Pre-normalize mapping prefixes once
mapping_norm = {}
for cat, spec in cci10.items():
    prefixes = [norm_icd(p) for p in spec["codes"]]

    # drop empties + dedupe while preserving order
    seen = set()
    prefixes = [p for p in prefixes if p and not (p in seen or seen.add(p))]

    mapping_norm[cat] = {
        "prefixes": prefixes,
        "w": spec["w"],
    }


# Build hadm-level presence flags (Charlson is ONCE per category per admission)
hadm_index = pd.Index(dx["hadm_id"].unique(), name="hadm_id")
flags = pd.DataFrame(index=hadm_index)

for cat, spec in mapping_norm.items():
    prefixes = spec["prefixes"]
    if not prefixes:
        flags[cat] = False
        continue

    # Quan-style matching: any diagnosis code that starts with any prefix
    pattern = r"^(?:" + "|".join(prefixes) + r")"
    matched_hadm = dx.loc[
        dx["code_norm"].str.match(pattern, na=False),
        "hadm_id",
    ].unique()

    flags[cat] = False
    flags.loc[matched_hadm, cat] = True


# ----------------------------
# 4b. Hierarchy overrides (avoid double counting)
# ----------------------------
# DM_COMP overrides DM_NO_COMP
if "DM_COMP" in flags.columns and "DM_NO_COMP" in flags.columns:
    flags["DM_NO_COMP"] = flags["DM_NO_COMP"] & (~flags["DM_COMP"])

# SEV_LIVER overrides MILD_LIVER
if "SEV_LIVER" in flags.columns and "MILD_LIVER" in flags.columns:
    flags["MILD_LIVER"] = flags["MILD_LIVER"] & (~flags["SEV_LIVER"])

# METS overrides MALIGNANCY
if "METS" in flags.columns and "MALIGNANCY" in flags.columns:
    flags["MALIGNANCY"] = flags["MALIGNANCY"] & (~flags["METS"])


# ----------------------------
# 4c. Compute Charlson weight per hadm_id (once per category)
# ----------------------------
cci_weight = pd.Series(0, index=flags.index, dtype="int64")

for cat, spec in mapping_norm.items():
    if cat in flags.columns:
        cci_weight += flags[cat].astype("int64") * int(spec["w"])

cci_scores = cci_weight.rename("charlson_weight").reset_index()


# ----------------------------
# 5. Merge with admissions + age-adjustment
# ----------------------------
df = admissions.merge(cci_scores, on="hadm_id", how="left")
df["charlson_weight"] = df["charlson_weight"].fillna(0).astype("int64")
df["cci"] = df["charlson_weight"]


# ----------------------------
# 6. Save clean global Charlson file
# ----------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)

df[["subject_id", "hadm_id", "age", "cci"]].to_csv(out_path, index=False)

print(f"✅ Saved Charlson (CCI-only, no age) → {out_path}")

