"""
MIMIC-IV Ischaemic Stroke — Step 3: Extract Medications & Impute Vitals
=========================================================================
Part A: Forward-fill GCS and drop temperature from vitals
Part B: Extract stroke-relevant medications from inputevents and emar
Part C: Merge medications with vitals into a unified hourly dataset

Drug categories for ischaemic stroke:
    - Thrombolytics: alteplase (tPA), tenecteplase
    - Antiplatelets: aspirin, clopidogrel
    - Anticoagulants: heparin, enoxaparin, warfarin
    - Antihypertensives: labetalol, nicardipine, hydralazine

Author: [Your Name]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand\Desktop\mimic-iv-3.1")
HOSP_DIR = BASE_DIR / "hosp"
ICU_DIR = BASE_DIR / "icu"
COHORT_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand/Desktop/RQ3")
OUTPUT_DIR = COHORT_DIR

# ============================================================
# PART A: IMPUTE VITALS
# ============================================================
print("=" * 60)
print("PART A: Imputing vitals (forward-fill GCS, drop temperature)")
print("=" * 60)

vitals = pd.read_csv(OUTPUT_DIR / "stroke_vitals_hourly.csv")
print(f"Loaded vitals: {vitals.shape}")

# Drop temperature (70.6% missing — too sparse for hourly data)
if "temperature" in vitals.columns:
    vitals = vitals.drop(columns=["temperature"])
    print("Dropped: temperature")

# Forward-fill GCS within each stay
gcs_cols = ["gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total"]
existing_gcs_cols = [c for c in gcs_cols if c in vitals.columns]

vitals = vitals.sort_values(["stay_id", "hour_bin"])
for col in existing_gcs_cols:
    before_missing = vitals[col].isna().mean() * 100
    vitals[col] = vitals.groupby("stay_id")[col].ffill()
    after_missing = vitals[col].isna().mean() * 100
    print(f"Forward-filled {col}: {before_missing:.1f}% -> {after_missing:.1f}% missing")

# Forward-fill other vitals (small gaps)
other_vitals = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2"]
for col in other_vitals:
    if col in vitals.columns:
        before_missing = vitals[col].isna().mean() * 100
        vitals[col] = vitals.groupby("stay_id")[col].ffill()
        after_missing = vitals[col].isna().mean() * 100
        if before_missing != after_missing:
            print(f"Forward-filled {col}: {before_missing:.1f}% -> {after_missing:.1f}% missing")

print(f"\nPost-imputation missingness:")
vital_cols = [c for c in vitals.columns if c not in ["stay_id", "hour_bin"]]
for col in vital_cols:
    pct = vitals[col].isna().mean() * 100
    print(f"  {col:<15}: {pct:.1f}%")

vitals.to_csv(OUTPUT_DIR / "stroke_vitals_hourly_imputed.csv", index=False)
print(f"\nImputed vitals saved to: {OUTPUT_DIR / 'stroke_vitals_hourly_imputed.csv'}")

# ============================================================
# PART B: EXTRACT MEDICATIONS
# ============================================================
print("\n" + "=" * 60)
print("PART B: Extracting medications")
print("=" * 60)

# Load cohort IDs
cohort_ids = pd.read_csv(COHORT_DIR / "stroke_cohort_ids.csv")
cohort_ids["intime"] = pd.to_datetime(cohort_ids["intime"])
cohort_ids["outtime"] = pd.to_datetime(cohort_ids["outtime"])
cohort_hadm_ids = set(cohort_ids["hadm_id"].unique())
cohort_stay_ids = set(cohort_ids["stay_id"].unique())

# Drug search terms (case-insensitive matching)
DRUG_CATEGORIES = {
    "thrombolytic": ["alteplase", "tpa", "activase", "tenecteplase", "tnkase"],
    "antiplatelet": ["aspirin", "clopidogrel", "plavix", "asa"],
    "anticoagulant": ["heparin", "enoxaparin", "lovenox", "warfarin", "coumadin"],
    "antihypertensive": ["labetalol", "nicardipine", "cardene", "hydralazine"],
}

def classify_drug(drug_name):
    """Classify a drug name into one of our categories."""
    if pd.isna(drug_name):
        return None
    drug_lower = str(drug_name).lower()
    for category, keywords in DRUG_CATEGORIES.items():
        for keyword in keywords:
            if keyword in drug_lower:
                return category
    return None

def get_specific_drug(drug_name):
    """Get the specific drug name (standardised)."""
    if pd.isna(drug_name):
        return None
    drug_lower = str(drug_name).lower()
    # Thrombolytics
    if any(k in drug_lower for k in ["alteplase", "tpa", "activase"]):
        return "alteplase"
    if any(k in drug_lower for k in ["tenecteplase", "tnkase"]):
        return "tenecteplase"
    # Antiplatelets
    if any(k in drug_lower for k in ["aspirin", "asa"]):
        return "aspirin"
    if any(k in drug_lower for k in ["clopidogrel", "plavix"]):
        return "clopidogrel"
    # Anticoagulants
    if any(k in drug_lower for k in ["enoxaparin", "lovenox"]):
        return "enoxaparin"
    if "heparin" in drug_lower and "enoxaparin" not in drug_lower:
        return "heparin"
    if any(k in drug_lower for k in ["warfarin", "coumadin"]):
        return "warfarin"
    # Antihypertensives
    if "labetalol" in drug_lower:
        return "labetalol"
    if any(k in drug_lower for k in ["nicardipine", "cardene"]):
        return "nicardipine"
    if "hydralazine" in drug_lower:
        return "hydralazine"
    return None

# ---------------------------------------------------------
# Source 1: inputevents (ICU IV medications)
# ---------------------------------------------------------
print("\nReading inputevents.csv.gz...")
inputevents = pd.read_csv(
    ICU_DIR / "inputevents.csv.gz",
    compression="gzip",
    usecols=["stay_id", "starttime", "endtime", "itemid", "amount", "amountuom", "ordercategoryname"],
)
print(f"  Total inputevents rows: {len(inputevents):,}")

# Filter to our cohort
inputevents = inputevents[inputevents["stay_id"].isin(cohort_stay_ids)]
print(f"  After cohort filter: {len(inputevents):,}")

# Load d_items for drug names
d_items = pd.read_csv(ICU_DIR / "d_items.csv.gz", compression="gzip")
inputevents = inputevents.merge(d_items[["itemid", "label"]], on="itemid", how="left")

# Classify drugs
inputevents["drug_category"] = inputevents["label"].apply(classify_drug)
inputevents["drug_name"] = inputevents["label"].apply(get_specific_drug)

# Keep only stroke-relevant drugs
iv_meds = inputevents[inputevents["drug_category"].notna()].copy()
iv_meds["starttime"] = pd.to_datetime(iv_meds["starttime"])
print(f"  Stroke-relevant IV meds: {len(iv_meds):,}")
if len(iv_meds) > 0:
    print(f"  IV drug distribution:")
    print(iv_meds["drug_name"].value_counts().to_string(header=False))

# ---------------------------------------------------------
# Source 2: emar (hospital-wide medication administration)
# ---------------------------------------------------------
print("\nReading emar.csv.gz (this may take a few minutes)...")
start_time = time.time()

emar_chunks = []
CHUNK_SIZE = 500_000

for i, chunk in enumerate(pd.read_csv(
    HOSP_DIR / "emar.csv.gz",
    compression="gzip",
    chunksize=CHUNK_SIZE,
    usecols=["hadm_id", "charttime", "medication", "event_txt"],
    dtype={"hadm_id": "Int64"},
)):
    # Filter to our cohort
    filtered = chunk[chunk["hadm_id"].isin(cohort_hadm_ids)]
    if len(filtered) > 0:
        # Classify drugs
        filtered = filtered.copy()
        filtered["drug_category"] = filtered["medication"].apply(classify_drug)
        filtered["drug_name"] = filtered["medication"].apply(get_specific_drug)
        # Keep only relevant drugs that were actually administered
        relevant = filtered[
            (filtered["drug_category"].notna()) &
            (filtered["event_txt"].isin(["Administered", "Applied"]))
        ]
        if len(relevant) > 0:
            emar_chunks.append(relevant)
    
    if (i + 1) % 20 == 0:
        elapsed = time.time() - start_time
        print(f"  Chunk {i+1}: {elapsed:.0f}s elapsed")

elapsed = time.time() - start_time
print(f"  Done reading emar in {elapsed:.0f}s")

if emar_chunks:
    emar_meds = pd.concat(emar_chunks, ignore_index=True)
    emar_meds["charttime"] = pd.to_datetime(emar_meds["charttime"])
    print(f"  Stroke-relevant eMAR meds: {len(emar_meds):,}")
    print(f"  eMAR drug distribution:")
    print(emar_meds["drug_name"].value_counts().to_string(header=False))
else:
    emar_meds = pd.DataFrame()
    print("  No relevant eMAR medications found")

# ============================================================
# PART C: MERGE MEDICATIONS INTO HOURLY BINS
# ============================================================
print("\n" + "=" * 60)
print("PART C: Merging medications into hourly bins")
print("=" * 60)

# Process IV medications from inputevents
iv_hourly_records = []
if len(iv_meds) > 0:
    iv_meds = iv_meds.merge(
        cohort_ids[["stay_id", "intime"]],
        on="stay_id",
        how="left"
    )
    iv_meds["hours_since_admit"] = (
        (iv_meds["starttime"] - iv_meds["intime"]).dt.total_seconds() / 3600
    )
    iv_meds["hour_bin"] = iv_meds["hours_since_admit"].apply(np.floor).astype(int)
    iv_meds = iv_meds[iv_meds["hour_bin"] >= 0]
    
    for _, row in iv_meds.iterrows():
        iv_hourly_records.append({
            "stay_id": row["stay_id"],
            "hour_bin": row["hour_bin"],
            "drug_category": row["drug_category"],
            "drug_name": row["drug_name"],
            "source": "inputevents"
        })

# Process eMAR medications
emar_hourly_records = []
if len(emar_meds) > 0:
    # Link emar to stay_id via hadm_id
    emar_meds = emar_meds.merge(
        cohort_ids[["hadm_id", "stay_id", "intime"]],
        on="hadm_id",
        how="inner"
    )
    emar_meds["hours_since_admit"] = (
        (emar_meds["charttime"] - emar_meds["intime"]).dt.total_seconds() / 3600
    )
    emar_meds["hour_bin"] = emar_meds["hours_since_admit"].apply(np.floor).astype(int)
    emar_meds = emar_meds[emar_meds["hour_bin"] >= 0]
    
    for _, row in emar_meds.iterrows():
        emar_hourly_records.append({
            "stay_id": row["stay_id"],
            "hour_bin": row["hour_bin"],
            "drug_category": row["drug_category"],
            "drug_name": row["drug_name"],
            "source": "emar"
        })

# Combine all medication records
all_med_records = pd.DataFrame(iv_hourly_records + emar_hourly_records)
print(f"Total medication administration records: {len(all_med_records):,}")

if len(all_med_records) > 0:
    # Create binary flags per (stay_id, hour_bin) for each drug category
    med_flags = all_med_records.pivot_table(
        index=["stay_id", "hour_bin"],
        columns="drug_category",
        values="drug_name",
        aggfunc="count",
        fill_value=0
    ).reset_index()
    med_flags.columns.name = None
    
    # Convert counts to binary (1 = administered, 0 = not)
    drug_cat_cols = [c for c in med_flags.columns if c not in ["stay_id", "hour_bin"]]
    for col in drug_cat_cols:
        med_flags[col] = (med_flags[col] > 0).astype(int)
    
    # Also create specific drug flags
    med_specific = all_med_records.pivot_table(
        index=["stay_id", "hour_bin"],
        columns="drug_name",
        values="source",
        aggfunc="count",
        fill_value=0
    ).reset_index()
    med_specific.columns.name = None
    
    specific_drug_cols = [c for c in med_specific.columns if c not in ["stay_id", "hour_bin"]]
    for col in specific_drug_cols:
        med_specific[col] = (med_specific[col] > 0).astype(int)
    
    print(f"\nMedication category flags shape: {med_flags.shape}")
    print(f"Drug categories: {drug_cat_cols}")
    print(f"\nPatients receiving each category:")
    for col in drug_cat_cols:
        n_stays = med_flags[med_flags[col] > 0]["stay_id"].nunique()
        print(f"  {col:<20}: {n_stays:,} stays")
    
    # Save medication data
    all_med_records.to_csv(OUTPUT_DIR / "stroke_medications_raw.csv", index=False)
    med_flags.to_csv(OUTPUT_DIR / "stroke_medications_hourly.csv", index=False)
    med_specific.to_csv(OUTPUT_DIR / "stroke_medications_specific_hourly.csv", index=False)
    print(f"\nMedication data saved to: {OUTPUT_DIR}")
else:
    med_flags = pd.DataFrame(columns=["stay_id", "hour_bin"])
    print("No medication records found — check drug name matching")

# ============================================================
# MERGE: VITALS + MEDICATIONS → UNIFIED DATASET
# ============================================================
print("\n" + "=" * 60)
print("MERGING: Vitals + Medications → Unified hourly dataset")
print("=" * 60)

# Left join: keep all vitals rows, add medication flags where available
unified = vitals.merge(med_flags, on=["stay_id", "hour_bin"], how="left")

# Fill NaN medication flags with 0 (no drug administered that hour)
if len(drug_cat_cols) > 0:
    for col in drug_cat_cols:
        if col in unified.columns:
            unified[col] = unified[col].fillna(0).astype(int)

print(f"Unified dataset shape: {unified.shape}")
print(f"Unique stays: {unified['stay_id'].nunique():,}")

# Save unified dataset
unified.to_csv(OUTPUT_DIR / "stroke_unified_hourly.csv", index=False)
print(f"\nUnified dataset saved to: {OUTPUT_DIR / 'stroke_unified_hourly.csv'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 SUMMARY")
print("=" * 60)
print(f"  Vitals features: {vital_cols}")
print(f"  Medication categories: {drug_cat_cols if len(all_med_records) > 0 else 'None found'}")
print(f"  Unified dataset: {unified.shape[0]:,} rows × {unified.shape[1]} columns")
print(f"  Unique stays: {unified['stay_id'].nunique():,}")
print(f"\nColumns in unified dataset:")
for col in unified.columns:
    print(f"  - {col}")
print(f"\nNext step: Build HMM on vitals to identify health states,")
print(f"then connect to influence diagram for treatment selection.")