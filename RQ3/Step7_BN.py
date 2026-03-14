"""
MIMIC-IV Ischaemic Stroke — Step 7: Bayesian Network for Synthetic Data
========================================================================
Part A: Learn a Bayesian network (BN) from the real MIMIC-IV data
Part B: Generate synthetic patient-hour observations from the learned BN
Part C: Evaluate the HMM + Influence Diagram pipeline on both datasets
Part D: Compare real vs synthetic evaluation results

Inputs:
    - stroke_unified_hourly.csv      (real unified dataset)
    - stroke_cohort.csv              (cohort with outcomes)
    - hmm_model.pkl, hmm_scaler.pkl, hmm_state_labels.pkl
    - stroke_influence_diagram.bifxml

Outputs:
    - bn_model.pkl                   (learned Bayesian network)
    - synthetic_patients.csv         (synthetic dataset)
    - evaluation_real.csv            (recommendations on real test set)
    - evaluation_synthetic.csv       (recommendations on synthetic data)
    - fig18_bn_structure.pdf
    - fig19_real_vs_synthetic_vitals.pdf
    - fig20_evaluation_comparison.pdf

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# pyAgrum for BN learning and influence diagram inference
import pyagrum as gum

# ============================================================
# CONFIGURATION
# ============================================================
COHORT_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand\Desktop\RQ3")
FIG_DIR = Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand\Desktop\RQ3")

HMM_FEATURES = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2"]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# HELPER: Import Step 6 functions
# ============================================================
# We need the influence diagram functions from Step 6
import importlib.util
import sys

step6_path = FIG_DIR / "Step6_ID.py"
if not step6_path.exists():
    # Try alternative paths
    for candidate in [
        Path(r"C:\Users\leboh\OneDrive - University of Witwatersrand\Desktop\RQ3\step6_influence_diagram.py"),
        COHORT_DIR / "Step6_ID.py",
    ]:
        if candidate.exists():
            step6_path = candidate
            break

spec = importlib.util.spec_from_file_location("step6", step6_path)
step6 = importlib.util.module_from_spec(spec)
sys.modules["step6"] = step6
spec.loader.exec_module(step6)


# ============================================================
# PART A: LEARN BAYESIAN NETWORK FROM REAL DATA
# ============================================================

def prepare_bn_data(unified_df, cohort_df):
    """
    Prepare data for BN learning.
    
    We discretise continuous vitals into clinically meaningful bins
    and include drug administration flags and outcome.
    
    Variables in the BN:
        - heart_rate_bin:   {Bradycardia, Normal, Tachycardia}
        - sbp_bin:          {Hypotensive, Normal, Hypertensive, Crisis}
        - spo2_bin:         {Critical, Low, Normal}
        - resp_rate_bin:    {Low, Normal, Tachypnoea}
        - map_bin:          {Low, Normal, High}
        - anticoagulant, antiplatelet, antihypertensive, thrombolytic: {0, 1}
        - outcome:          {survived, died}
    """
    df = unified_df.dropna(subset=HMM_FEATURES).copy()
    
    # Merge outcome
    outcome_map = cohort_df.set_index("stay_id")["hospital_expire_flag"].to_dict()
    df["outcome"] = df["stay_id"].map(outcome_map).map({0: "survived", 1: "died"})
    df = df.dropna(subset=["outcome"])
    
    # Discretise vitals into clinical categories
    df["heart_rate_bin"] = pd.cut(
        df["heart_rate"],
        bins=[0, 60, 100, 300],
        labels=["Bradycardia", "Normal", "Tachycardia"],
        include_lowest=True
    )
    
    df["sbp_bin"] = pd.cut(
        df["sbp"],
        bins=[0, 90, 140, 180, 400],
        labels=["Hypotensive", "Normal", "Hypertensive", "Crisis"],
        include_lowest=True
    )
    
    df["spo2_bin"] = pd.cut(
        df["spo2"],
        bins=[0, 90, 94, 100],
        labels=["Critical", "Low", "Normal"],
        include_lowest=True
    )
    
    df["resp_rate_bin"] = pd.cut(
        df["resp_rate"],
        bins=[0, 12, 20, 100],
        labels=["Low", "Normal", "Tachypnoea"],
        include_lowest=True
    )
    
    df["map_bin"] = pd.cut(
        df["map"],
        bins=[0, 65, 110, 300],
        labels=["Low", "Normal", "High"],
        include_lowest=True
    )
    
    # Drug flags (ensure they exist and are string for BN)
    drug_cols = ["anticoagulant", "antiplatelet", "antihypertensive", "thrombolytic"]
    for col in drug_cols:
        if col in df.columns:
            df[col] = df[col].astype(int).astype(str)
        else:
            df[col] = "0"
    
    # Select BN variables (using actual DataFrame column names)
    bn_vars = [
        "heart_rate_bin", "sbp_bin", "spo2_bin", "resp_rate_bin", "map_bin",
        "anticoagulant", "antiplatelet", "antihypertensive", "thrombolytic",
        "outcome"
    ]
    
    bn_data = df[bn_vars].dropna().copy()
    
    # Convert all to string for pyAgrum
    for col in bn_data.columns:
        bn_data[col] = bn_data[col].astype(str)
    
    # Rename columns to clean display names for the BN diagram
    display_names = {
        "heart_rate_bin": "Heart Rate",
        "sbp_bin": "Systolic BP",
        "spo2_bin": "SpO2",
        "resp_rate_bin": "Resp Rate",
        "map_bin": "MAP",
        "anticoagulant": "Anticoagulant",
        "antiplatelet": "Antiplatelet",
        "antihypertensive": "Antihypertensive",
        "thrombolytic": "Thrombolytic",
        "outcome": "Outcome",
    }
    bn_data = bn_data.rename(columns=display_names)
    
    print(f"  BN data shape: {bn_data.shape}")
    print(f"  Variables: {list(bn_data.columns)}")
    for col in bn_data.columns:
        print(f"    {col}: {sorted(bn_data[col].unique())}")
    
    return bn_data, df


def learn_bayesian_network(bn_data):
    """
    Learn BN structure and parameters from discretised data.
    
    Uses pyAgrum's BN learner with the Greedy Hill Climbing algorithm
    and BIC scoring. The structure captures dependencies between
    vitals, treatments, and outcome.
    """
    # Save to CSV for pyAgrum learner
    tmp_csv = COHORT_DIR / "_bn_temp_data.csv"
    bn_data.to_csv(tmp_csv, index=False)
    
    # Learn structure using Greedy Hill Climbing with BIC
    learner = gum.BNLearner(str(tmp_csv))
    learner.useGreedyHillClimbing()
    learner.useScoreBIC()
    
    # Add Laplace smoothing to handle zero-count conditioning sets
    learner.useSmoothingPrior(1.0)
    
    # Domain knowledge constraints:
    # 1. Outcome cannot cause vitals/treatments (it's a future event)
    vitals_and_drugs = [c for c in bn_data.columns if c != "Outcome"]
    for var in vitals_and_drugs:
        learner.addForbiddenArc("Outcome", var)
    
    # 2. Mandatory arcs — clinical relationships that MUST be in the BN
    mandatory_arcs = [
        # Vitals → Outcome (severity predicts mortality)
        ("SpO2", "Outcome"),
        ("Heart Rate", "Outcome"),
        ("Systolic BP", "Outcome"),
        # Vitals → Thrombolytic (clinical decision based on severity)
        ("SpO2", "Thrombolytic"),
        ("Systolic BP", "Thrombolytic"),
        # Treatment → Outcome (drugs affect survival)
        ("Antihypertensive", "Outcome"),
        ("Thrombolytic", "Outcome"),
        ("Anticoagulant", "Outcome"),
        ("Antiplatelet", "Outcome"),
    ]
    
    for parent, child in mandatory_arcs:
        try:
            learner.addMandatoryArc(parent, child)
        except Exception as e:
            print(f"    Warning: Could not add mandatory arc {parent} → {child}: {e}")
    
    bn = learner.learnBN()
    
    # Clean up temp file
    tmp_csv.unlink(missing_ok=True)
    
    print(f"\n  Learned BN:")
    print(f"    Nodes: {bn.size()}")
    print(f"    Arcs:  {bn.sizeArcs()}")
    print(f"    Arcs list:")
    for arc in bn.arcs():
        tail = bn.variable(arc[0]).name()
        head = bn.variable(arc[1]).name()
        print(f"      {tail} → {head}")
    
    return bn


def plot_bn_structure(bn):
    """Save a visualisation of the learned BN structure."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
    })
    
    # Use pyAgrum's dot export to get structure, then plot with matplotlib
    try:
        import pydot
        dot_str = bn.toDot()
        graphs = pydot.graph_from_dot_data(dot_str)
        if graphs:
            graph = graphs[0]
            # Fix "no_name" label that appears below the diagram
            graph.set_name("")
            graph.set_label("")
            graph.write_pdf(str(FIG_DIR / "fig18_bn_structure2.pdf"))
            print("  Saved: fig18_bn_structure2.pdf")
            return
    except ImportError:
        pass
    
    # Fallback: manual adjacency plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    var_names = [bn.variable(i).name() for i in range(bn.size())]
    n = len(var_names)
    
    # Arrange nodes in a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Draw arcs
    for arc in bn.arcs():
        tail_name = bn.variable(arc[0]).name()
        head_name = bn.variable(arc[1]).name()
        ti = var_names.index(tail_name)
        hi = var_names.index(head_name)
        ax.annotate("",
                     xy=(x_pos[hi], y_pos[hi]),
                     xytext=(x_pos[ti], y_pos[ti]),
                     arrowprops=dict(arrowstyle="->", color="#555555",
                                     connectionstyle="arc3,rad=0.1", lw=1.5))
    
    # Draw nodes
    for i, name in enumerate(var_names):
        color = "#4CAF50" if "bin" in name else ("#2196F3" if name == "outcome" else "#FF9800")
        ax.scatter(x_pos[i], y_pos[i], s=800, c=color, zorder=5, edgecolors="white", linewidths=2)
        # LaTeX-safe label
        label = name.replace("_", r"\_")
        ax.text(x_pos[i], y_pos[i], label, ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=6)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"\textbf{Learned Bayesian Network Structure}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig18_bn_structure2.pdf")
    plt.close()
    print("  Saved: fig18_bn_structure2.pdf")


# ============================================================
# PART B: GENERATE SYNTHETIC DATA FROM LEARNED BN
# ============================================================

def generate_synthetic_data(bn, n_samples=50000):
    """
    Generate synthetic patient-hour observations by sampling from the BN.
    
    The BN produces discrete categories. We then convert these back
    to continuous vitals by sampling uniformly within each bin's range,
    producing realistic synthetic vital signs.
    """
    print(f"\n  Generating {n_samples:,} synthetic observations...")
    
    # Sample from BN using pyAgrum's forward sampling
    generator = gum.BNDatabaseGenerator(bn)
    generator.setTopologicalVarOrder()
    generator.drawSamples(n_samples)
    
    # Save to temp CSV and read back
    tmp_synth = COHORT_DIR / "_synth_temp.csv"
    generator.toCSV(str(tmp_synth))
    synth_df = pd.read_csv(tmp_synth)
    tmp_synth.unlink(missing_ok=True)
    
    print(f"  Raw synthetic shape: {synth_df.shape}")
    print(f"  Columns: {list(synth_df.columns)}")
    
    # Convert discrete bins back to continuous vitals
    # Sample uniformly within each clinical bin
    bin_ranges = {
        "Heart Rate": {
            "Bradycardia": (40, 60), "Normal": (60, 100), "Tachycardia": (100, 150)
        },
        "Systolic BP": {
            "Hypotensive": (60, 90), "Normal": (90, 140),
            "Hypertensive": (140, 180), "Crisis": (180, 220)
        },
        "SpO2": {
            "Critical": (75, 90), "Low": (90, 94), "Normal": (94, 100)
        },
        "Resp Rate": {
            "Low": (8, 12), "Normal": (12, 20), "Tachypnoea": (20, 35)
        },
        "MAP": {
            "Low": (40, 65), "Normal": (65, 110), "High": (110, 150)
        },
    }
    
    feature_map = {
        "Heart Rate": "heart_rate",
        "Systolic BP": "sbp",
        "SpO2": "spo2",
        "Resp Rate": "resp_rate",
        "MAP": "map",
    }
    
    for bin_col, feat_col in feature_map.items():
        if bin_col in synth_df.columns:
            values = []
            for _, row in synth_df.iterrows():
                label = str(row[bin_col])
                if label in bin_ranges[bin_col]:
                    lo, hi = bin_ranges[bin_col][label]
                    values.append(np.random.uniform(lo, hi))
                else:
                    values.append(np.nan)
            synth_df[feat_col] = values
    
    # Generate DBP from MAP and SBP: MAP ≈ DBP + (SBP - DBP)/3
    # So DBP ≈ (3*MAP - SBP) / 2
    if "map" in synth_df.columns and "sbp" in synth_df.columns:
        synth_df["dbp"] = (3 * synth_df["map"] - synth_df["sbp"]) / 2
        synth_df["dbp"] = synth_df["dbp"].clip(30, 120)
    
    # Convert drug columns to int (BN uses display names)
    drug_col_map = {
        "Anticoagulant": "anticoagulant",
        "Antiplatelet": "antiplatelet",
        "Antihypertensive": "antihypertensive",
        "Thrombolytic": "thrombolytic",
    }
    for bn_name, feat_name in drug_col_map.items():
        if bn_name in synth_df.columns:
            synth_df[feat_name] = synth_df[bn_name].astype(int)
    
    # Rename Outcome column
    if "Outcome" in synth_df.columns:
        synth_df["outcome"] = synth_df["Outcome"]
    
    # Add synthetic stay_id and hour_bin for compatibility
    synth_df["stay_id"] = np.arange(len(synth_df)) // 100 + 900000  # synthetic IDs
    synth_df["hour_bin"] = np.arange(len(synth_df)) % 100
    
    print(f"  Synthetic dataset with continuous vitals: {synth_df.shape}")
    print(f"  Outcome distribution: {synth_df['outcome'].value_counts().to_dict()}")
    
    return synth_df


def plot_real_vs_synthetic(real_df, synth_df):
    """Compare distributions of vitals between real and synthetic data."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
    })

    features = ["heart_rate", "sbp", "map", "resp_rate", "spo2"]
    latex_labels = {
        "heart_rate": r"Heart Rate (bpm)",
        "sbp": r"Systolic BP (mmHg)",
        "map": r"MAP (mmHg)",
        "resp_rate": r"Respiratory Rate (br/min)",
        "spo2": r"SpO$_2$ (\%)",
    }

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.9)

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

    for i, feat in enumerate(features):
        ax = axes[i]

        real_vals = real_df[feat].dropna()
        synth_vals = synth_df[feat].dropna()

        lo = min(real_vals.quantile(0.01), synth_vals.quantile(0.01))
        hi = max(real_vals.quantile(0.99), synth_vals.quantile(0.99))
        bins = np.linspace(lo, hi, 50)

        ax.hist(real_vals, bins=bins, density=True, alpha=0.6,
                color="#2196F3", label="Real (MIMIC-IV)", edgecolor="white")
        ax.hist(synth_vals, bins=bins, density=True, alpha=0.6,
                color="#FF9800", label="Synthetic (BN)", edgecolor="white")

        ax.set_xlabel(latex_labels.get(feat, feat), fontsize='xx-large')
        ax.set_ylabel(r"Density", fontsize='xx-large')
        ax.tick_params(axis='both', labelsize='x-large')
        ax.legend(fontsize='x-large')
        ax.set_title(r"\textbf{" + latex_labels.get(feat, feat) + r"}", fontsize='xx-large')

    fig.suptitle(r"\textbf{Real vs Synthetic Vital Sign Distributions}", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig19_real_vs_synthetic_vitals.pdf")
    plt.close()
    print("  Saved: fig19_real_vs_synthetic_vitals.pdf")


# ============================================================
# PART C: EVALUATE HMM + ID PIPELINE ON BOTH DATASETS
# ============================================================

def evaluate_pipeline(data_df, hmm_model, scaler, state_labels, diagram, label=""):
    """
    Run the full HMM → Influence Diagram pipeline on a dataset.
    
    Returns a DataFrame with state labels and treatment recommendations.
    """
    df = data_df.dropna(subset=HMM_FEATURES).copy()
    
    if len(df) == 0:
        print(f"  WARNING: No valid observations in {label} data")
        return pd.DataFrame()
    
    # Normalise
    X = scaler.transform(df[HMM_FEATURES].values)
    
    # HMM inference (per-stay sequences for real data, batch for synthetic)
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
    
    # Influence diagram inference
    recommendations = []
    print(f"  Running ID inference on {len(df):,} {label} observations...")
    
    for i in range(len(df)):
        state_probs = posteriors_concat[i]
        step6.set_state_from_hmm(diagram, state_probs, state_labels)
        result = step6.select_treatment(diagram)
        recommendations.append(result["chosen_treatment"])
        
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i+1:,}/{len(df):,}")
    
    df = df.copy()
    df["hmm_state"] = predictions_concat
    df["state_label"] = [state_labels[int(s)] for s in predictions_concat]
    df["recommended_treatment"] = recommendations
    
    print(f"  Done. {len(df):,} recommendations generated.")
    return df


def compute_metrics(eval_df, label=""):
    """Compute evaluation metrics for a dataset."""
    print(f"\n  {label} Evaluation Metrics:")
    print(f"  {'='*50}")
    
    n_total = len(eval_df)
    
    # Recommendation distribution
    rec_dist = eval_df["recommended_treatment"].value_counts()
    print(f"\n  Recommendation distribution:")
    for treatment, count in rec_dist.items():
        print(f"    {treatment:<20}: {count:>8,} ({count/n_total*100:.1f}%)")
    
    # Recommendation by state
    severity_order = {"Stable": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    sorted_states = sorted(eval_df["state_label"].unique(),
                           key=lambda x: severity_order.get(x, 5))
    
    print(f"\n  Recommendations by HMM state:")
    state_rec = {}
    for state in sorted_states:
        subset = eval_df[eval_df["state_label"] == state]
        dist = subset["recommended_treatment"].value_counts(normalize=True) * 100
        state_rec[state] = dist.to_dict()
        print(f"    {state} (n={len(subset):,}):")
        for t, pct in sorted(dist.items(), key=lambda x: -x[1]):
            if pct > 0:
                print(f"      {t:<20}: {pct:.1f}%")
    
    # Agreement with actual (only for real data with drug columns)
    drug_cols = ["anticoagulant", "antiplatelet", "antihypertensive", "thrombolytic"]
    has_drugs = all(c in eval_df.columns for c in drug_cols)
    
    agreement_rate = None
    if has_drugs:
        any_drug = eval_df[drug_cols].astype(float).sum(axis=1) > 0
        treated = eval_df[any_drug]
        if len(treated) > 0:
            treatment_map = {
                "Anticoagulant": "anticoagulant",
                "Antiplatelet": "antiplatelet",
                "Antihypertensive": "antihypertensive",
                "Thrombolytic": "thrombolytic",
            }
            matches = 0
            for _, row in treated.iterrows():
                rec_col = treatment_map.get(row["recommended_treatment"])
                if rec_col and float(row.get(rec_col, 0)) == 1:
                    matches += 1
            agreement_rate = matches / len(treated) * 100
            print(f"\n  Agreement with actual clinical decisions:")
            print(f"    Treated hours: {len(treated):,}")
            print(f"    Agreement rate: {agreement_rate:.1f}%")
    
    # State distribution
    state_dist = eval_df["state_label"].value_counts(normalize=True) * 100
    print(f"\n  HMM state distribution:")
    for state in sorted_states:
        print(f"    {state:<12}: {state_dist.get(state, 0):.1f}%")
    
    return {
        "n_total": n_total,
        "rec_dist": rec_dist.to_dict(),
        "state_rec": state_rec,
        "state_dist": state_dist.to_dict(),
        "agreement_rate": agreement_rate,
    }


def plot_evaluation_comparison(real_metrics, synth_metrics, state_labels):
    """Compare recommendation distributions between real and synthetic."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
    })

    treatments = ["Antiplatelet", "Antihypertensive", "Thrombolytic", "Anticoagulant"]
    severity_order = {"Stable": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
    sorted_states = sorted(state_labels.values(), key=lambda x: severity_order.get(x, 5))
    n_states = len(sorted_states)

    drug_colors = {
        "Anticoagulant": "#4CAF50",
        "Antiplatelet": "#2196F3",
        "Antihypertensive": "#FF9800",
        "Thrombolytic": "#E53935",
    }

    drug_latex = {
        "Anticoagulant": "Anticoagulant",
        "Antiplatelet": "Antiplatelet",
        "Antihypertensive": "Antihypertensive",
        "Thrombolytic": "Thrombolytic",
    }

    # Layout: n_states rows x 2 columns (Real | Synthetic)
    fig, axes = plt.subplots(
        n_states, 2,
        figsize=(16, 4 * n_states)
    )
    if n_states == 1:
        axes = axes.reshape(1, 2)

    datasets = [("Real (MIMIC-IV)", real_metrics), ("Synthetic (BN)", synth_metrics)]

    # Column headers
    for col_idx, (ds_label, _) in enumerate(datasets):
        axes[0, col_idx].set_title(
            r"\textbf{" + ds_label + r"}",
            fontsize='xx-large', pad=15
        )

    for row_idx, state in enumerate(sorted_states):
        for col_idx, (ds_label, metrics) in enumerate(datasets):
            ax = axes[row_idx, col_idx]

            state_data = metrics["state_rec"].get(state, {})
            pcts = [state_data.get(t, 0) for t in treatments]

            bars = ax.bar(range(len(treatments)), pcts,
                          color=[drug_colors[t] for t in treatments],
                          edgecolor="white", alpha=0.8)

            ax.set_xticks(range(len(treatments)))
            ax.set_xticklabels(
                [drug_latex[t] for t in treatments],
                fontsize='xx-large',
                rotation=30,
                ha="right"
            )
            ax.tick_params(axis='y', labelsize='xx-large')
            ax.set_ylim(0, 115)
            ax.set_ylabel(r"\% of hours", fontsize='xx-large')

            # Row label on the left side only
            if col_idx == 0:
                ax.set_ylabel(r"\% of hours", fontsize='xx-large')
                ax.annotate(
                    r"\textbf{" + state + r"}",
                    xy=(-0.35, 0.5), xycoords="axes fraction",
                    fontsize='xx-large', ha="center", va="center", rotation=90
                )

            for bar, pct in zip(bars, pcts):
                if pct > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{pct:.1f}" + r"\%", ha="center",
                            fontsize='x-large', fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, hspace=0.6, wspace=0.4)
    plt.savefig(FIG_DIR / "fig20_evaluation_comparison.pdf")
    plt.close()
    print("  Saved: fig20_evaluation_comparison.pdf")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print("STEP 7: BAYESIAN NETWORK & SYNTHETIC DATA EVALUATION")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # Load data and models
    # ----------------------------------------------------------
    print("\nLoading data and models...")
    
    unified = pd.read_csv(COHORT_DIR / "stroke_unified_hourly.csv")
    cohort = pd.read_csv(COHORT_DIR / "stroke_cohort.csv")
    hmm_model = pickle.load(open(COHORT_DIR / "hmm_model.pkl", "rb"))
    scaler = pickle.load(open(COHORT_DIR / "hmm_scaler.pkl", "rb"))
    state_labels_raw = pickle.load(open(COHORT_DIR / "hmm_state_labels.pkl", "rb"))
    
    state_labels = {int(k): v for k, v in state_labels_raw.items()}
    
    # Remap generic labels if needed
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
            sorted_by_generic = sorted(state_labels.items(), key=lambda x: x[1])
            state_labels = {k: clinical_names[n][i] for i, (k, _) in enumerate(sorted_by_generic)}
            print(f"  Remapped to clinical labels: {state_labels}")
    
    print(f"  Unified dataset: {unified.shape}")
    print(f"  HMM states: {state_labels}")
    
    # Build influence diagram
    diagram = step6.build_influence_diagram(state_labels)
    
    # ----------------------------------------------------------
    # Part A: Train/test split + Learn BN
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART A: Train/Test Split & Learn Bayesian Network")
    print("=" * 60)
    
    # Split by stay_id (patient-level split)
    stay_ids = unified["stay_id"].unique()
    train_stays, test_stays = train_test_split(
        stay_ids, test_size=0.2, random_state=RANDOM_STATE
    )
    
    train_df = unified[unified["stay_id"].isin(train_stays)]
    test_df = unified[unified["stay_id"].isin(test_stays)]
    
    print(f"  Train: {len(train_stays):,} stays, {len(train_df):,} hours")
    print(f"  Test:  {len(test_stays):,} stays, {len(test_df):,} hours")
    
    # Prepare BN data from training set
    print("\n  Preparing data for BN learning...")
    bn_data, train_enriched = prepare_bn_data(train_df, cohort)
    
    # Learn BN
    print("\n  Learning Bayesian network structure...")
    bn = learn_bayesian_network(bn_data)
    
    # Save BN
    pickle.dump(bn, open(COHORT_DIR / "bn_model.pkl", "wb"))
    gum.saveBN(bn, str(COHORT_DIR / "bn_model.bifxml"))
    print(f"  Saved: bn_model.pkl, bn_model.bifxml")
    
    # Plot BN structure
    print("\n  Plotting BN structure...")
    plot_bn_structure(bn)
    
    # ----------------------------------------------------------
    # Part B: Generate synthetic data
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART B: Generate Synthetic Data")
    print("=" * 60)
    
    # Generate similar number of observations as test set
    n_synthetic = len(test_df)
    synth_df = generate_synthetic_data(bn, n_samples=n_synthetic)
    
    # Save
    synth_df.to_csv(COHORT_DIR / "synthetic_patients.csv", index=False)
    print(f"  Saved: synthetic_patients.csv ({len(synth_df):,} rows)")
    
    # Plot real vs synthetic distributions
    print("\n  Comparing distributions...")
    plot_real_vs_synthetic(test_df, synth_df)
    
    # ----------------------------------------------------------
    # Part C: Evaluate on both datasets
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART C: Evaluate HMM + ID Pipeline")
    print("=" * 60)
    
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Evaluate on real test data
    print("\n  [1/2] Evaluating on REAL test data...")
    real_eval = evaluate_pipeline(test_df, hmm_model, scaler, state_labels, diagram, label="real")
    
    # Evaluate on synthetic data
    print("\n  [2/2] Evaluating on SYNTHETIC data...")
    synth_eval = evaluate_pipeline(synth_df, hmm_model, scaler, state_labels, diagram, label="synthetic")
    
    # Save evaluation results
    try:
        real_eval.to_csv(COHORT_DIR / "evaluation_real.csv", index=False)
    except PermissionError:
        real_eval.to_csv(COHORT_DIR / "evaluation_real_v2.csv", index=False)
    
    try:
        synth_eval.to_csv(COHORT_DIR / "evaluation_synthetic.csv", index=False)
    except PermissionError:
        synth_eval.to_csv(COHORT_DIR / "evaluation_synthetic_v2.csv", index=False)
    
    print(f"  Saved evaluation results.")
    
    # ----------------------------------------------------------
    # Part D: Compare results
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART D: Comparison — Real vs Synthetic")
    print("=" * 60)
    
    real_metrics = compute_metrics(real_eval, label="Real (MIMIC-IV test set)")
    synth_metrics = compute_metrics(synth_eval, label="Synthetic (BN-generated)")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n  {'Metric':<35} {'Real':>12} {'Synthetic':>12}")
    print(f"  {'-'*59}")
    print(f"  {'Total observations':<35} {real_metrics['n_total']:>12,} {synth_metrics['n_total']:>12,}")
    
    for treatment in ["Antiplatelet", "Antihypertensive", "Thrombolytic", "Anticoagulant"]:
        real_pct = real_metrics["rec_dist"].get(treatment, 0) / real_metrics["n_total"] * 100
        synth_pct = synth_metrics["rec_dist"].get(treatment, 0) / synth_metrics["n_total"] * 100
        print(f"  {treatment + ' (%)':<35} {real_pct:>11.1f}% {synth_pct:>11.1f}%")
    
    if real_metrics["agreement_rate"] is not None:
        print(f"  {'Agreement with clinicians (%)':<35} {real_metrics['agreement_rate']:>11.1f}%  {'N/A':>12}")
    
    # Plot comparison
    print("\n  Generating comparison plot...")
    plot_evaluation_comparison(real_metrics, synth_metrics, state_labels)
    
    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 SUMMARY")
    print("=" * 60)
    print(f"  BN learned: {bn.size()} nodes, {bn.sizeArcs()} arcs")
    print(f"  Synthetic data: {len(synth_df):,} observations")
    print(f"  Real test evaluation: {len(real_eval):,} recommendations")
    print(f"  Synthetic evaluation: {len(synth_eval):,} recommendations")
    print(f"  Outputs: bn_model.pkl, synthetic_patients.csv,")
    print(f"           evaluation_real.csv, evaluation_synthetic.csv")
    print(f"  Figures: fig18_bn_structure2.pdf, fig19_real_vs_synthetic_vitals.pdf,")
    print(f"           fig20_evaluation_comparison.pdf")