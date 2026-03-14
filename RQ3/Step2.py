"""
MIMIC-IV Ischaemic Stroke — Step 2: Extract Vital Signs
=========================================================
Extracts time-series vital signs from chartevents.csv.gz for the
stroke ICU cohort, bins into 1-hour intervals.

chartevents.csv.gz is very large (~30+ GB), so we read in chunks
and filter to only our cohort's stay_ids and relevant itemids.

Vital signs extracted:
    - Heart Rate (220045)
    - Systolic BP: Arterial (220050) + Non-Invasive (220179)
    - Diastolic BP: Arterial (220051) + Non-Invasive (220180)
    - Mean Arterial Pressure: Arterial (220052) + Non-Invasive (220181)
    - Respiratory Rate (220210, 224690)
    - SpO2 (220277)
    - Temperature Celsius (223762), Fahrenheit (223761)
    - GCS Eye (220739)
    - GCS Verbal (223900)
    - GCS Motor (223901)

Author: Lebohang Mosia
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand\Desktop\mimic-iv-3.1")
ICU_DIR = BASE_DIR / "icu"
COHORT_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand/Desktop/RQ3")
OUTPUT_DIR = COHORT_DIR
CHUNK_SIZE = 500_000  # rows per chunk — adjust based on your RAM
# ============================================================
# ITEM IDS FOR VITAL SIGNS
# ============================================================
# Source: MIT-LCP/mimic-code vitalsign.sql
VITAL_ITEMIDS = {
    # Heart Rate
    220045: "heart_rate",
    # Blood Pressure — Arterial (invasive)
    220050: "sbp",       # Arterial BP systolic
    220051: "dbp",       # Arterial BP diastolic
    220052: "map",       # Arterial BP mean
    # Blood Pressure — Non-Invasive
    220179: "sbp",       # NIBP systolic
    220180: "dbp",       # NIBP diastolic
    220181: "map",       # NIBP mean
    # Arterial line BP (alt labels)
    225309: "sbp",       # ART BP Systolic
    225310: "dbp",       # ART BP Diastolic
    225312: "map",       # ART BP Mean
    # Respiratory Rate
    220210: "resp_rate",
    224690: "resp_rate",  # Respiratory Rate (Total)
    # SpO2
    220277: "spo2",
    # Temperature
    223762: "temp_c",     # Temperature Celsius
    223761: "temp_f",     # Temperature Fahrenheit
    # Glasgow Coma Scale
    220739: "gcs_eye",    # GCS - Eye Opening
    223900: "gcs_verbal", # GCS - Verbal Response
    223901: "gcs_motor",  # GCS - Motor Response
}

ALL_ITEMIDS = set(VITAL_ITEMIDS.keys())

# ============================================================
# LOAD COHORT IDS
# ============================================================
print("=" * 60)
print("STEP 2: Extracting vital signs from chartevents")
print("=" * 60)

cohort_ids = pd.read_csv(COHORT_DIR / "stroke_cohort_ids.csv")
cohort_ids["intime"] = pd.to_datetime(cohort_ids["intime"])
cohort_ids["outtime"] = pd.to_datetime(cohort_ids["outtime"])

cohort_stay_ids = set(cohort_ids["stay_id"].unique())
print(f"Cohort size: {len(cohort_stay_ids):,} ICU stays")

# ============================================================
# READ CHARTEVENTS IN CHUNKS
# ============================================================
print(f"\nReading chartevents.csv.gz in chunks of {CHUNK_SIZE:,} rows...")
print("This may take 20-60 minutes depending on your machine.\n")

chartevents_path = ICU_DIR / "chartevents.csv.gz"
vitals_list = []
total_rows_processed = 0
total_rows_kept = 0
start_time = time.time()

for i, chunk in enumerate(pd.read_csv(
    chartevents_path,
    compression="gzip",
    chunksize=CHUNK_SIZE,
    usecols=["stay_id", "itemid", "charttime", "valuenum"],
    dtype={"stay_id": "Int64", "itemid": int},
)):
    total_rows_processed += len(chunk)
    
    # Filter: only our cohort's stays AND our vital sign itemids
    mask = (chunk["stay_id"].isin(cohort_stay_ids)) & (chunk["itemid"].isin(ALL_ITEMIDS))
    filtered = chunk[mask].copy()
    
    if len(filtered) > 0:
        vitals_list.append(filtered)
        total_rows_kept += len(filtered)
    
    # Progress update every 10 chunks
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  Chunk {i+1}: {total_rows_processed:,} rows processed, "
              f"{total_rows_kept:,} vitals kept, "
              f"{elapsed:.0f}s elapsed")

elapsed = time.time() - start_time
print(f"\nDone reading chartevents: {total_rows_processed:,} total rows in {elapsed:.0f}s")
print(f"Kept {total_rows_kept:,} vital sign records for our cohort")

# Concatenate all filtered chunks
vitals_raw = pd.concat(vitals_list, ignore_index=True)
vitals_raw["charttime"] = pd.to_datetime(vitals_raw["charttime"])

# ============================================================
# MAP ITEMIDS TO VITAL SIGN NAMES
# ============================================================
vitals_raw["vital_name"] = vitals_raw["itemid"].map(VITAL_ITEMIDS)
print(f"\nVital sign counts:")
print(vitals_raw["vital_name"].value_counts())

# ============================================================
# CLEAN: REMOVE PHYSIOLOGICALLY IMPLAUSIBLE VALUES
# ============================================================
print("\nCleaning implausible values...")

# Define plausible ranges
plausible_ranges = {
    "heart_rate": (0, 300),
    "sbp": (0, 400),
    "dbp": (0, 300),
    "map": (0, 350),
    "resp_rate": (0, 70),
    "spo2": (0, 100),
    "temp_c": (26, 45),
    "temp_f": (78.8, 113),
    "gcs_eye": (1, 4),
    "gcs_verbal": (1, 5),
    "gcs_motor": (1, 6),
}

before_clean = len(vitals_raw)
for vital, (low, high) in plausible_ranges.items():
    mask = vitals_raw["vital_name"] == vital
    out_of_range = mask & ((vitals_raw["valuenum"] < low) | (vitals_raw["valuenum"] > high))
    vitals_raw = vitals_raw[~out_of_range]

# Convert Fahrenheit to Celsius
f_mask = vitals_raw["vital_name"] == "temp_f"
vitals_raw.loc[f_mask, "valuenum"] = (vitals_raw.loc[f_mask, "valuenum"] - 32) * 5 / 9
vitals_raw.loc[f_mask, "vital_name"] = "temp_c"
# Rename to unified "temperature"
vitals_raw.loc[vitals_raw["vital_name"] == "temp_c", "vital_name"] = "temperature"

after_clean = len(vitals_raw)
print(f"Removed {before_clean - after_clean:,} implausible values ({(before_clean - after_clean)/before_clean*100:.1f}%)")

# ============================================================
# COMPUTE HOURS SINCE ICU ADMISSION
# ============================================================
print("\nComputing hours since ICU admission...")

vitals_raw = vitals_raw.merge(
    cohort_ids[["stay_id", "intime", "outtime"]],
    on="stay_id",
    how="left"
)

# Keep only vitals recorded during ICU stay
vitals_raw = vitals_raw[
    (vitals_raw["charttime"] >= vitals_raw["intime"]) &
    (vitals_raw["charttime"] <= vitals_raw["outtime"])
]

# Hours since ICU admission
vitals_raw["hours_since_admit"] = (
    (vitals_raw["charttime"] - vitals_raw["intime"]).dt.total_seconds() / 3600
)

# Create 1-hour bins
vitals_raw["hour_bin"] = vitals_raw["hours_since_admit"].apply(np.floor).astype(int)

print(f"Vitals within ICU stay: {len(vitals_raw):,}")

# ============================================================
# PIVOT: BIN INTO 1-HOUR INTERVALS
# ============================================================
print("\nBinning vitals into 1-hour intervals...")

# Average multiple measurements within each hour bin
vitals_binned = vitals_raw.groupby(
    ["stay_id", "hour_bin", "vital_name"]
)["valuenum"].mean().reset_index()

# Pivot to wide format: one row per (stay_id, hour_bin)
vitals_wide = vitals_binned.pivot_table(
    index=["stay_id", "hour_bin"],
    columns="vital_name",
    values="valuenum"
).reset_index()

# Flatten column names
vitals_wide.columns.name = None

# Compute GCS total if components are available
if all(c in vitals_wide.columns for c in ["gcs_eye", "gcs_verbal", "gcs_motor"]):
    vitals_wide["gcs_total"] = (
        vitals_wide["gcs_eye"].fillna(0) +
        vitals_wide["gcs_verbal"].fillna(0) +
        vitals_wide["gcs_motor"].fillna(0)
    )
    # Set to NaN where all components are missing
    all_gcs_missing = (
        vitals_wide["gcs_eye"].isna() &
        vitals_wide["gcs_verbal"].isna() &
        vitals_wide["gcs_motor"].isna()
    )
    vitals_wide.loc[all_gcs_missing, "gcs_total"] = np.nan

# Sort by stay_id and hour
vitals_wide = vitals_wide.sort_values(["stay_id", "hour_bin"]).reset_index(drop=True)

print(f"\nFinal vitals table shape: {vitals_wide.shape}")
print(f"Unique ICU stays: {vitals_wide['stay_id'].nunique():,}")
print(f"Hour bins per stay:")
print(vitals_wide.groupby("stay_id")["hour_bin"].count().describe())

# ============================================================
# MISSINGNESS REPORT
# ============================================================
print("\n" + "=" * 60)
print("MISSINGNESS REPORT")
print("=" * 60)
vital_cols = [c for c in vitals_wide.columns if c not in ["stay_id", "hour_bin"]]
for col in vital_cols:
    missing_pct = vitals_wide[col].isna().mean() * 100
    print(f"  {col:<15}: {missing_pct:.1f}% missing")

# ============================================================
# SAVE
# ============================================================
vitals_wide.to_csv(OUTPUT_DIR / "stroke_vitals_hourly.csv", index=False)
print(f"\nVitals saved to: {OUTPUT_DIR / 'stroke_vitals_hourly.csv'}")

# Also save the raw (long format) for reference
vitals_raw[["stay_id", "charttime", "vital_name", "valuenum", "hours_since_admit", "hour_bin"]].to_csv(
    OUTPUT_DIR / "stroke_vitals_raw.csv", index=False
)
print(f"Raw vitals saved to: {OUTPUT_DIR / 'stroke_vitals_raw.csv'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 SUMMARY")
print("=" * 60)
print(f"  ICU stays with vitals: {vitals_wide['stay_id'].nunique():,}")
print(f"  Total hourly records: {len(vitals_wide):,}")
print(f"  Median hours per stay: {vitals_wide.groupby('stay_id')['hour_bin'].count().median():.0f}")
print(f"  Vital signs extracted: {', '.join(vital_cols)}")
print(f"\nNext step: Run step3_extract_medications.py to extract drug administrations")
