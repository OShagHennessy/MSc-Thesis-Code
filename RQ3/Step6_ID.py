"""
MIMIC-IV Ischaemic Stroke — Step 6: Influence Diagram for Treatment Selection
===============================================================================
Builds an influence diagram that recommends optimal drug treatment
based on the patient's current health state (from the HMM).

Structure:
    - Chance Node: "HealthState" (Stable, Moderate, Critical)
      → CPT set from HMM posterior at each patient-hour
    - Decision Node: "Treatment" (Anticoagulant, Antiplatelet, Antihypertensive, Thrombolytic)
      → The drug category to administer
    - Utility Node: "PatientOutcome"
      → Expected clinical benefit of each treatment in each state

The utility table is derived from:
    1. Clinical guidelines for ischaemic stroke treatment
    2. Observed treatment-outcome associations in the MIMIC-IV cohort
    3. Domain knowledge about drug mechanisms of action

Inference uses Shafer-Shenoy LIMID with No Forgetting Assumption
and the Maximum Expected Utility (MEU) criterion.

Key pyAgrum fixes applied (from SADCBO debugging):
    - addNoForgettingAssumption requires explicit decision node list
    - optimalDecision returns a Potential — use toarray().flatten() + argmax
    - Variable names must match exactly across arcs, CPTs, and utility tables

Author: [Your Name]
"""

import pyagrum as gum
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math
from pathlib import Path
from matplotlib.gridspec import GridSpec

# ============================================================
# CONFIGURATION
# ============================================================
COHORT_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand\Desktop\RQ3")
FIG_DIR = COHORT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. BUILD THE INFLUENCE DIAGRAM
# ============================================================

def build_influence_diagram(state_labels):
    """
    Construct the clinical influence diagram.
    
    The number of health states is determined by the HMM (via BIC).
    The treatment options are fixed: the 4 drug categories from MIMIC-IV.
    
    Args:
        state_labels: dict mapping state index → label name
                      e.g. {0: "Critical", 1: "Moderate", 2: "Severe", 3: "Mild", 4: "Stable"}
    
    Returns:
        model (gum.InfluenceDiagram): The constructed influence diagram.
        ordered_labels (list): Labels in the order used by pyAgrum (index 0, 1, 2, ...)
    """
    model = gum.InfluenceDiagram()
    
    n_states = len(state_labels)
    
    # Create ordered labels: sort by severity
    # Severity ordering: Stable < Mild < Moderate < Severe < Critical
    severity_order = {"Stable": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    sorted_labels = sorted(state_labels.values(), key=lambda x: severity_order.get(x, 5))
    
    # ----------------------------------------------------------
    # Chance Node: Patient Health State (from HMM)
    # ----------------------------------------------------------
    health = gum.LabelizedVariable("HealthState", "Patient Health State", n_states)
    for i, label in enumerate(sorted_labels):
        health.changeLabel(i, label)
    model.addChanceNode(health)
    
    # ----------------------------------------------------------
    # Decision Node: Treatment Selection
    # ----------------------------------------------------------
    treatment = gum.LabelizedVariable("Treatment", "Drug Treatment", 4)
    treatment.changeLabel(0, "Anticoagulant")
    treatment.changeLabel(1, "Antiplatelet")
    treatment.changeLabel(2, "Antihypertensive")
    treatment.changeLabel(3, "Thrombolytic")
    model.addDecisionNode(treatment)
    
    # ----------------------------------------------------------
    # Utility Node: Patient Outcome Quality
    # pyAgrum utility nodes require exactly 1 modality
    # ----------------------------------------------------------
    utility = gum.LabelizedVariable("PatientOutcome", "Patient Outcome Quality", 1)
    model.addUtilityNode(utility)
    
    # ----------------------------------------------------------
    # Arcs
    # ----------------------------------------------------------
    # Health state informs the treatment decision
    model.addArc("HealthState", "Treatment")
    # Outcome depends on both health state and chosen treatment
    model.addArc("HealthState", "PatientOutcome")
    model.addArc("Treatment", "PatientOutcome")
    
    # ----------------------------------------------------------
    # Utility Table: U(HealthState, Treatment)
    # ----------------------------------------------------------
    # Set the utility table based on clinical rationale
    set_utility_table(model, state_labels)
    
    return model


# ============================================================
# 2. UTILITY TABLE — CLINICAL RATIONALE
# ============================================================

def set_utility_table(model, state_labels):
    """
    Set the utility values for each (HealthState, Treatment) combination.
    
    Clinical rationale for ischaemic stroke:
    
    STABLE patients (mild stroke, GCS 13-15, stable vitals):
        - Antiplatelet (aspirin/clopidogrel): BEST — standard of care for
          stable ischaemic stroke, prevents recurrent events
        - Anticoagulant (heparin): Good — DVT prophylaxis, some benefit
        - Antihypertensive: Moderate — permissive hypertension preferred
          in stable acute stroke (lowering BP can reduce perfusion)
        - Thrombolytic: LOW — risk outweighs benefit if patient is stable
          and likely outside the tPA window
    
    MODERATE patients (worsening, GCS 9-12, fluctuating vitals):
        - Anticoagulant: Good — prevents clot extension
        - Antiplatelet: Good — standard therapy continues
        - Antihypertensive: BEST — BP control becomes critical to
          prevent haemorrhagic transformation
        - Thrombolytic: Moderate — may still be within window, but risk
          increases with deterioration
    
    CRITICAL patients (severe, GCS 3-8, haemodynamic instability):
        - Thrombolytic: BEST — if within window, aggressive reperfusion
          is the only chance of meaningful recovery
        - Antihypertensive: Good — urgent BP control to prevent
          further damage
        - Anticoagulant: Moderate — cautious use, bleeding risk
        - Antiplatelet: Low — insufficient for acute critical stroke
    
    Scale: 1-10 where 10 = maximum expected clinical benefit
    """
    
    n_states = len(state_labels)
    
    # Define utility tables for different model sizes
    # The labels list is ordered by severity (ascending)
    labels_list = [state_labels[i] for i in range(n_states)]
    
    if n_states == 2:
        # Stable, Critical
        utilities = {
            ("Stable", "Anticoagulant"):     7.0,
            ("Stable", "Antiplatelet"):      9.0,
            ("Stable", "Antihypertensive"):  4.0,
            ("Stable", "Thrombolytic"):      2.0,
            ("Critical", "Anticoagulant"):   5.0,
            ("Critical", "Antiplatelet"):    3.0,
            ("Critical", "Antihypertensive"):7.0,
            ("Critical", "Thrombolytic"):    9.0,
        }
    elif n_states == 3:
        # Stable, Moderate, Critical
        utilities = {
            ("Stable", "Anticoagulant"):      7.0,
            ("Stable", "Antiplatelet"):       9.0,
            ("Stable", "Antihypertensive"):   4.0,
            ("Stable", "Thrombolytic"):       2.0,
            ("Moderate", "Anticoagulant"):    7.0,
            ("Moderate", "Antiplatelet"):     6.0,
            ("Moderate", "Antihypertensive"): 9.0,
            ("Moderate", "Thrombolytic"):     5.0,
            ("Critical", "Anticoagulant"):    5.0,
            ("Critical", "Antiplatelet"):     3.0,
            ("Critical", "Antihypertensive"): 7.0,
            ("Critical", "Thrombolytic"):     9.0,
        }
    elif n_states == 4:
        # Stable, Mild, Moderate, Critical
        utilities = {
            ("Stable", "Anticoagulant"):      7.0,
            ("Stable", "Antiplatelet"):       9.0,
            ("Stable", "Antihypertensive"):   3.0,
            ("Stable", "Thrombolytic"):       1.0,
            ("Mild", "Anticoagulant"):        7.0,
            ("Mild", "Antiplatelet"):         8.0,
            ("Mild", "Antihypertensive"):     5.0,
            ("Mild", "Thrombolytic"):         3.0,
            ("Moderate", "Anticoagulant"):    6.0,
            ("Moderate", "Antiplatelet"):     5.0,
            ("Moderate", "Antihypertensive"): 9.0,
            ("Moderate", "Thrombolytic"):     6.0,
            ("Critical", "Anticoagulant"):    4.0,
            ("Critical", "Antiplatelet"):     3.0,
            ("Critical", "Antihypertensive"): 7.0,
            ("Critical", "Thrombolytic"):     9.0,
        }
    elif n_states == 5:
        # Stable, Mild, Moderate, Severe, Critical
        utilities = {
            ("Stable", "Anticoagulant"):       7.0,
            ("Stable", "Antiplatelet"):        9.0,
            ("Stable", "Antihypertensive"):    3.0,
            ("Stable", "Thrombolytic"):        1.0,
            ("Mild", "Anticoagulant"):         7.0,
            ("Mild", "Antiplatelet"):          8.0,
            ("Mild", "Antihypertensive"):      4.0,
            ("Mild", "Thrombolytic"):          2.0,
            ("Moderate", "Anticoagulant"):     6.0,
            ("Moderate", "Antiplatelet"):      6.0,
            ("Moderate", "Antihypertensive"):  8.0,
            ("Moderate", "Thrombolytic"):      5.0,
            ("Severe", "Anticoagulant"):       5.0,
            ("Severe", "Antiplatelet"):        4.0,
            ("Severe", "Antihypertensive"):    9.0,
            ("Severe", "Thrombolytic"):        7.0,
            ("Critical", "Anticoagulant"):     4.0,
            ("Critical", "Antiplatelet"):      3.0,
            ("Critical", "Antihypertensive"):  7.0,
            ("Critical", "Thrombolytic"):      9.0,
        }
    else:
        # Generic fallback: linearly interpolate
        treatments = ["Anticoagulant", "Antiplatelet", "Antihypertensive", "Thrombolytic"]
        utilities = {}
        for i in range(n_states):
            severity = i / (n_states - 1)  # 0 = stable, 1 = critical
            utilities[(labels_list[i], "Anticoagulant")]     = 7.0 - 2.0 * severity
            utilities[(labels_list[i], "Antiplatelet")]      = 9.0 - 6.0 * severity
            utilities[(labels_list[i], "Antihypertensive")]  = 4.0 + 3.0 * severity
            utilities[(labels_list[i], "Thrombolytic")]      = 2.0 + 7.0 * severity
    
    # Apply to model using explicit dict indexing
    for (state_label, treatment_label), value in utilities.items():
        model.utility("PatientOutcome")[{
            "HealthState": state_label,
            "Treatment": treatment_label
        }] = value


# ============================================================
# 3. SET HEALTH STATE FROM HMM POSTERIOR
# ============================================================

def set_state_from_hmm(model, state_probs, state_labels):
    """
    Update the chance node CPT with HMM posterior probabilities.
    
    Maps from HMM state ordering (arbitrary) to pyAgrum ordering
    (sorted by severity: Stable, Mild, Moderate, Severe, Critical).
    
    Args:
        model: The influence diagram
        state_probs: array of shape (n_states,) — posterior from HMM
                     indexed by HMM state index (0, 1, 2, ...)
        state_labels: dict mapping HMM state index → label name
    """
    severity_order = {"Stable": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    sorted_labels = sorted(state_labels.values(), key=lambda x: severity_order.get(x, 5))
    
    # Build reverse map: label → HMM index
    label_to_hmm_idx = {v: k for k, v in state_labels.items()}
    
    # Build the full CPT array in pyAgrum order, then set at once
    cpt_array = np.array([state_probs[label_to_hmm_idx[label]] for label in sorted_labels])
    
    # Ensure it sums to 1 (numerical safety)
    cpt_sum = cpt_array.sum()
    if cpt_sum > 0:
        cpt_array = cpt_array / cpt_sum
    
    model.cpt("HealthState").fillWith(cpt_array.tolist())


# ============================================================
# 4. INFERENCE: SELECT OPTIMAL TREATMENT
# ============================================================

def select_treatment(model, debug=False):
    """
    Run MEU inference and return the optimal treatment.
    """
    treatment_labels = ["Anticoagulant", "Antiplatelet", "Antihypertensive", "Thrombolytic"]
    
    ie = gum.ShaferShenoyLIMIDInference(model)
    ie.addNoForgettingAssumption(["Treatment"])
    ie.makeInference()
    
    # Get the optimal decision
    decision = ie.optimalDecision("Treatment")
    decision_array = decision.toarray()
    
    if debug:
        print(f"  DEBUG select_treatment — decision.toarray() shape: {decision_array.shape}")
        print(f"  DEBUG select_treatment — decision.toarray():\n{decision_array}")
        # Print the variable ordering in the decision potential
        for i in range(decision.nbrDim()):
            v = decision.variable(i)
            labels = [v.label(j) for j in range(v.domainSize())]
            print(f"  DEBUG select_treatment — dim {i}: {v.name()} = {labels}")
    
    # The optimalDecision for a decision node observed after a chance node
    # returns a CONDITIONAL policy: P(Treatment | HealthState)
    # Shape may be (n_treatments, n_states) or (n_states, n_treatments)
    # We need to marginalise over HealthState using the current CPT
    
    # Get current state probabilities from the model's CPT
    state_probs = model.cpt("HealthState").toarray().flatten()
    
    if decision_array.ndim > 1:
        # Determine which axis is Treatment (size 4) vs HealthState (size 5)
        if decision_array.shape[0] == len(treatment_labels):
            # Shape is (treatments, states) — compute expected decision
            # For each treatment, sum policy(t|s) * P(s) over states
            expected = decision_array @ state_probs
        else:
            # Shape is (states, treatments)
            expected = state_probs @ decision_array
        
        if debug:
            print(f"  DEBUG select_treatment — state_probs: {state_probs}")
            print(f"  DEBUG select_treatment — expected utilities per treatment: {expected}")
        
        best_idx = int(np.argmax(expected))
    else:
        # Already flat
        best_idx = int(np.argmax(decision_array.flatten()))
        expected = decision_array.flatten()
    
    best_treatment = treatment_labels[best_idx]
    meu = ie.MEU()
    
    return {
        "chosen_treatment": best_treatment,
        "meu": meu,
        "decision_distribution": expected,
        "treatment_labels": treatment_labels,
    }


# ============================================================
# 5. END-TO-END PIPELINE: VITALS → HMM → ID → TREATMENT
# ============================================================

def recommend_treatment(hmm_model, scaler, state_labels, observation, diagram=None):
    """
    Full pipeline: raw vitals → normalised → HMM state → ID → treatment.
    
    Args:
        hmm_model: Trained hmmlearn GaussianHMM
        scaler: StandardScaler fitted on training data
        state_labels: dict of state index → label
        observation: Raw vital signs array (1, n_features)
        diagram: Pre-built influence diagram (optional)
    
    Returns:
        dict with treatment recommendation and supporting info
    """
    obs = np.atleast_2d(observation)
    
    # Normalise
    obs_scaled = scaler.transform(obs)
    
    # Get HMM state posterior
    state_probs = hmm_model.predict_proba(obs_scaled)[-1]
    predicted_state = int(np.argmax(state_probs))
    
    # Build or reuse diagram
    if diagram is None:
        diagram = build_influence_diagram(state_labels)
    
    # Set state distribution
    set_state_from_hmm(diagram, state_probs, state_labels)
    
    # Get recommendation
    result = select_treatment(diagram)
    result["state_probs"] = state_probs
    result["predicted_state"] = state_labels[predicted_state]
    
    return result


# ============================================================
# 6. BATCH EVALUATION ON DATASET
# ============================================================

def evaluate_on_dataset(hmm_model, scaler, state_labels, unified_df, cohort_df):
    """
    Run the full pipeline on every patient-hour in the dataset.
    Compare recommended treatments with actual administrations.
    
    Args:
        hmm_model: Trained HMM
        scaler: Fitted StandardScaler
        state_labels: dict of state labels
        unified_df: The unified hourly dataset
        cohort_df: Cohort with outcomes
    
    Returns:
        results_df: DataFrame with recommendations per patient-hour
    """
    HMM_FEATURES = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2"]
    drug_cats = ["anticoagulant", "antihypertensive", "antiplatelet", "thrombolytic"]
    treatment_map = {
        "Anticoagulant": "anticoagulant",
        "Antiplatelet": "antiplatelet",
        "Antihypertensive": "antihypertensive",
        "Thrombolytic": "thrombolytic",
    }
    
    # Build diagram once
    diagram = build_influence_diagram(state_labels)
    
    # Prepare data
    df = unified_df.dropna(subset=HMM_FEATURES).copy()
    
    # Normalise all observations
    X = scaler.transform(df[HMM_FEATURES].values)
    
    # Get HMM posteriors for all observations (per-stay sequences)
    stay_ids = df["stay_id"].unique()
    all_posteriors = []
    all_predictions = []
    
    for stay_id in stay_ids:
        mask = df["stay_id"] == stay_id
        stay_X = X[mask.values]
        if len(stay_X) == 0:
            continue
        posteriors = hmm_model.predict_proba(stay_X)
        predictions = hmm_model.predict(stay_X)
        all_posteriors.append(posteriors)
        all_predictions.append(predictions)
    
    posteriors_concat = np.concatenate(all_posteriors, axis=0)
    predictions_concat = np.concatenate(all_predictions, axis=0)
    
    # Run inference for each observation
    n_states = len(state_labels)
    recommendations = []
    
    print(f"Running influence diagram inference on {len(df):,} observations...")
    
    for i in range(len(df)):
        state_probs = posteriors_concat[i]
        set_state_from_hmm(diagram, state_probs, state_labels)
        result = select_treatment(diagram, debug=False)
        recommendations.append(result["chosen_treatment"])
        
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1:,}/{len(df):,}")
    
    df = df.copy()
    df["hmm_state"] = predictions_concat
    df["state_label"] = [state_labels[int(s)] for s in predictions_concat]
    df["recommended_treatment"] = recommendations
    
    # Check if recommended matches actual
    df["actual_treatment"] = "None"
    for drug in drug_cats:
        df.loc[df[drug] == 1, "actual_treatment"] = drug
    
    # For patients receiving multiple drugs, pick the one matching recommendation
    df["recommended_col"] = df["recommended_treatment"].map(treatment_map)
    df["recommendation_matches"] = df.apply(
        lambda row: row.get(row["recommended_col"], 0) == 1 if pd.notna(row["recommended_col"]) else False,
        axis=1
    )
    
    # Add outcome
    outcome_map = cohort_df.set_index("stay_id")["hospital_expire_flag"].to_dict()
    df["outcome"] = df["stay_id"].map(outcome_map)
    
    print(f"  Done. {len(df):,} recommendations generated.")
    
    return df


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("STEP 6: INFLUENCE DIAGRAM FOR TREATMENT SELECTION")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # Load HMM model and data
    # ----------------------------------------------------------
    print("\nLoading HMM model and data...")
    
    hmm_model = pickle.load(open(COHORT_DIR / "hmm_model.pkl", "rb"))
    scaler = pickle.load(open(COHORT_DIR / "hmm_scaler.pkl", "rb"))
    state_labels_raw = pickle.load(open(COHORT_DIR / "hmm_state_labels.pkl", "rb"))
    unified = pd.read_csv(COHORT_DIR / "stroke_unified_hourly.csv")
    cohort = pd.read_csv(COHORT_DIR / "stroke_cohort.csv")
    
    # CRITICAL FIX: Convert np.int64 keys to plain int
    state_labels = {int(k): v for k, v in state_labels_raw.items()}
    
    # If labels are generic (State_0, State_1, ...), remap to clinical names
    # The severity ordering from Step 5 ensures State_0 = least severe, State_N = most severe
    clinical_names = {
        2: ["Stable", "Critical"],
        3: ["Stable", "Moderate", "Critical"],
        4: ["Stable", "Mild", "Moderate", "Critical"],
        5: ["Stable", "Mild", "Moderate", "Severe", "Critical"],
        6: ["Stable", "Mild", "Moderate", "Severe", "Critical", "Terminal"],
    }
    
    if any(v.startswith("State_") for v in state_labels.values()):
        n = len(state_labels)
        if n in clinical_names:
            # State_0 = least severe (Stable), State_N = most severe (Critical)
            # Sort by the State_X index to get severity order
            sorted_by_generic = sorted(state_labels.items(), key=lambda x: x[1])
            new_labels = {}
            for idx, (hmm_key, _) in enumerate(sorted_by_generic):
                new_labels[hmm_key] = clinical_names[n][idx]
            state_labels = new_labels
            print(f"  Remapped generic labels to clinical names: {state_labels}")
    
    n_states = len(state_labels)
    print(f"  HMM states: {state_labels}")
    print(f"  Dataset: {unified.shape}")
    
    # ----------------------------------------------------------
    # Build and display the influence diagram
    # ----------------------------------------------------------
    print("\nBuilding influence diagram...")
    diagram = build_influence_diagram(state_labels)
    
    # Save diagram structure
    diagram.saveBIFXML(str(COHORT_DIR / "stroke_influence_diagram.bifxml"))
    print(f"  Saved: stroke_influence_diagram.bifxml")
    
    # Print utility table
    print(f"\nUtility Table:")
    treatments = ["Anticoagulant", "Antiplatelet", "Antihypertensive", "Thrombolytic"]
    severity_order = {"Stable": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    sorted_labels = sorted(state_labels.values(), key=lambda x: severity_order.get(x, 5))
    
    header = f"  {'State':<12}" + "".join(f"{t:>18}" for t in treatments)
    print(header)
    print("  " + "-" * (12 + 18 * 4))
    for label in sorted_labels:
        row = f"  {label:<12}"
        for t in treatments:
            try:
                val = diagram.utility("PatientOutcome")[{"HealthState": label, "Treatment": t}]
                v = float(np.array(val).flat[0]) if hasattr(val, '__len__') else float(val)
            except Exception:
                v = 0.0
            row += f"{v:>18.1f}"
        print(row)
    
    # ----------------------------------------------------------
    # Demonstration: test scenarios
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Treatment recommendations by state")
    print("=" * 60)
    
    # Build reverse map: label → HMM index
    label_to_hmm_idx = {v: k for k, v in state_labels.items()}
    
    for label in sorted_labels:
        hmm_idx = label_to_hmm_idx[label]
        probs = np.zeros(n_states)
        probs[hmm_idx] = 1.0
        
        set_state_from_hmm(diagram, probs, state_labels)
        result = select_treatment(diagram)
        
        print(f"\n  State: {label} (100% certainty)")
        print(f"  -> Recommended: {result['chosen_treatment']}")
        meu_val = result['meu']['mean'] if isinstance(result['meu'], dict) else result['meu']
        print(f"  -> MEU: {meu_val:.2f}")
    
    # Mixed state scenario
    print(f"\n  State: Mixed uncertainty")
    mixed_probs = np.ones(n_states) / n_states
    
    print(f"  Posteriors: {dict(zip([state_labels[i] for i in range(n_states)], mixed_probs.round(2)))}")
    set_state_from_hmm(diagram, mixed_probs, state_labels)
    result = select_treatment(diagram)
    print(f"  -> Recommended: {result['chosen_treatment']}")
    
    # ----------------------------------------------------------
    # Full dataset evaluation
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("FULL DATASET EVALUATION")
    print("=" * 60)
    
    results = evaluate_on_dataset(hmm_model, scaler, state_labels, unified, cohort)
    
    # ----------------------------------------------------------
    # Analysis of recommendations
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    # Overall recommendation distribution
    print("\nRecommendation distribution:")
    rec_counts = results["recommended_treatment"].value_counts()
    for treatment, count in rec_counts.items():
        print(f"  {treatment:<20}: {count:,} ({count/len(results)*100:.1f}%)")
    
    # Recommendation by state
    print("\nRecommendation by HMM state:")
    for i in range(n_states):
        state_name = state_labels[i]
        subset = results[results["state_label"] == state_name]
        print(f"\n  {state_name} (n={len(subset):,}):")
        for treatment in treatments:
            n = (subset["recommended_treatment"] == treatment).sum()
            pct = n / len(subset) * 100 if len(subset) > 0 else 0
            print(f"    {treatment:<20}: {pct:.1f}%")
    
    # Agreement with actual treatment (ONLY for hours where a drug was administered)
    print("\nAgreement with actual clinical decisions:")
    print("  (Only evaluated on hours where at least one drug was given)")
    drug_cols_present = [c for c in ["anticoagulant", "antihypertensive", "antiplatelet", "thrombolytic"] 
                         if c in results.columns]
    any_drug = results[drug_cols_present].sum(axis=1) > 0
    treated_hours = results[any_drug]
    untreated_hours = results[~any_drug]
    print(f"  Hours WITH drug administration: {len(treated_hours):,} ({len(treated_hours)/len(results)*100:.1f}%)")
    print(f"  Hours WITHOUT drug administration: {len(untreated_hours):,} ({len(untreated_hours)/len(results)*100:.1f}%)")
    if len(treated_hours) > 0:
        agreement = treated_hours["recommendation_matches"].mean() * 100
        print(f"  Agreement rate (treated hours only): {agreement:.1f}%")
    
    # Recommendation by outcome
    print("\nRecommendation distribution by outcome:")
    for outcome, olabel in [(0, "Survived"), (1, "Died")]:
        subset = results[results["outcome"] == outcome]
        print(f"\n  {olabel} (n={len(subset):,} hours):")
        for treatment in treatments:
            n = (subset["recommended_treatment"] == treatment).sum()
            pct = n / len(subset) * 100 if len(subset) > 0 else 0
            print(f"    {treatment:<20}: {pct:.1f}%")
    
    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    try:
        results.to_csv(COHORT_DIR / "treatment_recommendations.csv", index=False)
        print(f"\nResults saved to: {COHORT_DIR / 'treatment_recommendations.csv'}")
    except PermissionError:
        alt_path = COHORT_DIR / "treatment_recommendations_v2.csv"
        results.to_csv(alt_path, index=False)
        print(f"\nPermission denied on original file (close Excel?). Saved to: {alt_path}")
    
    # ----------------------------------------------------------
    # Visualisation: Recommendation by state
    # ----------------------------------------------------------
    print("\nGenerating visualisations...")
    
    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    
    drug_colors = {
        "Anticoagulant": "#4CAF50",
        "Antiplatelet": "#2196F3",
        "Antihypertensive": "#FF9800",
        "Thrombolytic": "#E53935",
    }
    
    # LaTeX-safe treatment labels for x-axis
    drug_latex = {
        "Anticoagulant": r"Anticoagulant",
        "Antiplatelet": r"Antiplatelet",
        "Antihypertensive": r"Antihypertensive",
        "Thrombolytic": r"Thrombolytic",
    }
    
    # fig, axes = plt.subplots(1, n_states, figsize=(5 * n_states, 5))
    # if n_states == 1:
    #     axes = [axes]
    # # fig.suptitle(r"\textbf{Treatment Recommendations by HMM Health State}",
    # #              fontsize=15, y=1.02)
    
    # for i in range(n_states):
    #     ax = axes[i]
    #     state_name = state_labels[i]
    #     subset = results[results["state_label"] == state_name]
        
    #     counts = subset["recommended_treatment"].value_counts()
    #     pcts = counts / len(subset) * 100
        
    #     bars = ax.bar(range(len(treatments)), [pcts.get(t, 0) for t in treatments],
    #                   color=[drug_colors[t] for t in treatments],
    #                   edgecolor="white", alpha=0.8)
    #     ax.set_xticks(range(len(treatments)))
    #     ax.set_xticklabels([drug_latex[t] for t in treatments], fontsize=9)
    #     ax.set_ylabel(r"\% of hours", fontsize='xx-large')
    #     ax.set_title(r"\textbf{" + state_name + r"}", fontsize='xx-large')
    #     ax.set_ylim(0, 105)
        
    #     for bar, t in zip(bars, treatments):
    #         pct = pcts.get(t, 0)
    #         if pct > 0:
    #             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
    #                    f"{pct:.0f}" + r"\%", ha="center", fontsize=10)
    
    # plt.tight_layout()
    # plt.savefig(FIG_DIR / "fig17_treatment_recommendations_by_state.pdf")
    # plt.close()
    # print("Saved: fig17_treatment_recommendations_by_state.pdf")
    # Reorder states by severity: Stable → Mild → Moderate → Severe → Critical
    # Reorder states by severity: Stable → Mild → Moderate → Severe → Critical
    severity_order = {"Stable": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    sorted_state_indices = sorted(range(n_states), key=lambda i: severity_order.get(state_labels[i], 5))

    fig = plt.figure(figsize=(22, 14))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 6, figure=fig, hspace=0.7, wspace=0.9)

    # Top row: 3 plots
    axes_top = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]

    # Bottom row: 2 plots centered
    axes_bot = [
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    axes = axes_top + axes_bot

    for plot_idx, state_idx in enumerate(sorted_state_indices):
        ax = axes[plot_idx]
        state_name = state_labels[state_idx]
        subset = results[results["state_label"] == state_name]

        counts = subset["recommended_treatment"].value_counts()
        pcts = counts / len(subset) * 100

        bars = ax.bar(range(len(treatments)), [pcts.get(t, 0) for t in treatments],
                      color=[drug_colors[t] for t in treatments],
                      edgecolor="white", alpha=0.8)

        ax.set_xticks(range(len(treatments)))
        ax.set_xticklabels(
            [drug_latex[t] for t in treatments],
            fontsize='x-large',
            rotation=30,
            ha="right"
        )

        ax.tick_params(axis='y', labelsize='xx-large')

        # Show y-axis label for every plot
        ax.set_ylabel(r"\% of hours", fontsize='xx-large')

        ax.set_title(r"\textbf{" + state_name + r"}", fontsize='x-large')
        ax.set_ylim(0, 115)

        for bar, t in zip(bars, treatments):
            pct = pcts.get(t, 0)
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f"{pct:.1f}" + r"\%", ha="center", fontsize='x-large', fontweight="bold")

    plt.savefig(FIG_DIR / "fig17_treatment_recommendations_by_state.pdf")
    plt.close()
    print("Saved: fig17_treatment_recommendations_by_state.pdf")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 SUMMARY")
    print("=" * 60)
    print(f"  Health states: {state_labels}")
    print(f"  Treatment options: {treatments}")
    print(f"  Total recommendations: {len(results):,}")
    print(f"  Diagram saved: stroke_influence_diagram.bifxml")
    print(f"  Results saved: treatment_recommendations.csv")
    print(f"\n  Next step: Learn Bayesian network for synthetic data generation")
    print(f"  and evaluate model on both real and synthetic datasets.")
