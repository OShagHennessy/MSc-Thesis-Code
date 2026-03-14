"""
MIMIC-IV Ischaemic Stroke Cohort Extraction — Step 1
======================================================
This script identifies ischaemic stroke patients from MIMIC-IV v3.1,
links them to ICU stays, and provides cohort summary statistics.

ICD-10 codes for ischaemic stroke: I63.x (Cerebral infarction)
MIMIC-IV also contains ICD-9 codes for older records: 433.x, 434.x

Author: Lebohang Mosia
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand\Desktop\mimic-iv-3.1")
HOSP_DIR = BASE_DIR / "hosp"
ICU_DIR = BASE_DIR / "icu"
OUTPUT_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand/Desktop/RQ3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: IDENTIFY ISCHAEMIC STROKE PATIENTS
# ============================================================
print("=" * 60)
print("STEP 1: Identifying ischaemic stroke patients")
print("=" * 60)

# Load diagnoses
diagnoses = pd.read_csv(HOSP_DIR / "diagnoses_icd.csv.gz", compression="gzip")
print(f"Total diagnosis records: {len(diagnoses):,}")

# Filter for ischaemic stroke
# ICD-10: I63.x (cerebral infarction)
# ICD-9: 433.x, 434.x (occlusion of cerebral/precerebral arteries)
icd10_stroke = diagnoses[
    (diagnoses["icd_version"] == 10) & 
    (diagnoses["icd_code"].str.startswith("I63"))
]
icd9_stroke = diagnoses[
    (diagnoses["icd_version"] == 9) & 
    (diagnoses["icd_code"].str.match(r"^43[34]"))
]
stroke_diagnoses = pd.concat([icd10_stroke, icd9_stroke])

stroke_subject_ids = stroke_diagnoses["subject_id"].unique()
stroke_hadm_ids = stroke_diagnoses["hadm_id"].unique()

print(f"\nIschaemic stroke diagnoses found:")
print(f"  ICD-10 (I63.x) records: {len(icd10_stroke):,}")
print(f"  ICD-9 (433.x/434.x) records: {len(icd9_stroke):,}")
print(f"  Unique patients: {len(stroke_subject_ids):,}")
print(f"  Unique hospitalisations: {len(stroke_hadm_ids):,}")

# Show distribution of ICD codes
print(f"\nTop 10 ICD codes:")
print(stroke_diagnoses["icd_code"].value_counts().head(10))

# ============================================================
# STEP 2: LINK TO ADMISSIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Linking to admissions data")
print("=" * 60)

admissions = pd.read_csv(HOSP_DIR / "admissions.csv.gz", compression="gzip")
admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

stroke_admissions = admissions[admissions["hadm_id"].isin(stroke_hadm_ids)].copy()
stroke_admissions["los_hours"] = (
    (stroke_admissions["dischtime"] - stroke_admissions["admittime"]).dt.total_seconds() / 3600
)

print(f"Stroke admissions: {len(stroke_admissions):,}")
print(f"\nAdmission type distribution:")
print(stroke_admissions["admission_type"].value_counts())
print(f"\nLength of stay (hours):")
print(stroke_admissions["los_hours"].describe())

# ============================================================
# STEP 3: LINK TO ICU STAYS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Linking to ICU stays")
print("=" * 60)

icustays = pd.read_csv(ICU_DIR / "icustays.csv.gz", compression="gzip")
icustays["intime"] = pd.to_datetime(icustays["intime"])
icustays["outtime"] = pd.to_datetime(icustays["outtime"])

stroke_icu = icustays[icustays["hadm_id"].isin(stroke_hadm_ids)].copy()
stroke_icu["icu_los_hours"] = (
    (stroke_icu["outtime"] - stroke_icu["intime"]).dt.total_seconds() / 3600
)

print(f"Stroke patients with ICU stays: {stroke_icu['subject_id'].nunique():,}")
print(f"Total ICU stays for stroke patients: {len(stroke_icu):,}")
print(f"\nICU Length of stay (hours):")
print(stroke_icu["icu_los_hours"].describe())

# Keep only first ICU stay per hospitalisation
stroke_icu_first = stroke_icu.sort_values("intime").groupby("hadm_id").first().reset_index()
print(f"\nFirst ICU stays (one per hospitalisation): {len(stroke_icu_first):,}")

# ============================================================
# STEP 4: LINK TO PATIENTS (demographics)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Patient demographics")
print("=" * 60)

patients = pd.read_csv(HOSP_DIR / "patients.csv.gz", compression="gzip")
stroke_patients = patients[patients["subject_id"].isin(stroke_subject_ids)].copy()

print(f"Stroke patients: {len(stroke_patients):,}")
print(f"\nGender distribution:")
print(stroke_patients["gender"].value_counts())
print(f"\nAge at anchor year:")
print(stroke_patients["anchor_age"].describe())

# ============================================================
# STEP 5: BUILD COHORT TABLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Building cohort table")
print("=" * 60)

# Merge: ICU stays + admissions + patients
cohort = stroke_icu_first.merge(
    stroke_admissions[["hadm_id", "admittime", "dischtime", "admission_type", 
                        "hospital_expire_flag", "los_hours"]],
    on="hadm_id",
    how="left"
)
cohort = cohort.merge(
    stroke_patients[["subject_id", "gender", "anchor_age"]],
    on="subject_id",
    how="left"
)

# Add the primary stroke ICD code
primary_stroke_code = stroke_diagnoses.sort_values("seq_num").groupby("hadm_id").first().reset_index()
cohort = cohort.merge(
    primary_stroke_code[["hadm_id", "icd_code", "icd_version"]],
    on="hadm_id",
    how="left"
)

print(f"Final cohort size: {len(cohort):,} ICU stays")
print(f"Unique patients: {cohort['subject_id'].nunique():,}")
print(f"\nIn-hospital mortality: {cohort['hospital_expire_flag'].mean()*100:.1f}%")

# ============================================================
# STEP 6: SAVE COHORT
# ============================================================
cohort.to_csv(OUTPUT_DIR / "stroke_cohort.csv", index=False)
print(f"\nCohort saved to: {OUTPUT_DIR / 'stroke_cohort.csv'}")

# Save key IDs for subsequent extraction steps
cohort_ids = cohort[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]].copy()
cohort_ids.to_csv(OUTPUT_DIR / "stroke_cohort_ids.csv", index=False)
print(f"Cohort IDs saved to: {OUTPUT_DIR / 'stroke_cohort_ids.csv'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("COHORT SUMMARY")
print("=" * 60)
print(f"  Total ischaemic stroke patients in MIMIC-IV: {len(stroke_subject_ids):,}")
print(f"  Patients with ICU stays: {stroke_icu['subject_id'].nunique():,}")
print(f"  Final cohort (first ICU stay per admission): {len(cohort):,}")
print(f"  Median ICU LOS: {cohort['icu_los_hours'].median():.1f} hours")
print(f"  Median hospital LOS: {cohort['los_hours'].median():.1f} hours")
print(f"  In-hospital mortality: {cohort['hospital_expire_flag'].mean()*100:.1f}%")
print(f"\nNext step: Run step2_extract_vitals.py to extract time-series data")
