# cohort_correlations.py
"""
Cohort-Level Pillar <-> CGM Spike Correlation Analysis
------------------------------------------------------
Pools all patients' day-level data, applies within-patient z-scoring,
and computes cohort-wide correlations between lifestyle pillars and
glucose spike metrics.

Outputs:
    - Console summary with ranked pillar associations
    - Correlation heatmap (PNG)
    - Scatter plots for top predictors (PNG)
    - Forest plot of effect sizes with CIs (PNG)
    - Mixed-effects model summary (if statsmodels available)
    - CSV exports of all results

Usage:
    python cohort_correlations.py

Requires:
    - spike_analysis.py
    - pillar_analysis.py
    - CGM CSV (wide-format)
    - Pillars CSV (self-reported gauges)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

from spike_analysis import reshape_wide_to_long, detect_spikes
from pillar_analysis import (
    clean_pillars,
    PILLAR_NAMES,
    GAUGE_COLS,
    NEGATIVE_COL,
    PILLAR_COLORS,
    zscore_within_patient,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================
# CONFIGURATION
# ==============================================================

CGM_CSV_PATH    = Path("data/cgm_matrix.csv")
PILLAR_CSV_PATH = Path("data/patient_reported_metrics.csv")

# Spike detection thresholds
AMP_THRESH  = 1.0
RISE_RATE   = 0.02
END_BAND    = 0.3
MIN_ABOVE   = 15

# Column names in CGM CSV
ID_COL  = "Client ID"
DAY_COL = "Day"

# Minimum days per patient to include in analysis
MIN_DAYS_PER_PATIENT = 5

# Output
OUTPUT_DIR = Path("output_cohort")


# ==============================================================
# HELPER: bootstrap confidence interval for Spearman r
# ==============================================================

def bootstrap_spearman_ci(x, y, n_boot=2000, ci=0.95, seed=42):
    """
    Compute a bootstrap confidence interval for Spearman r.
    Returns (r, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)

    if n < 5:
        return np.nan, np.nan, np.nan

    r_obs = stats.spearmanr(x, y).statistic

    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_rs[i] = stats.spearmanr(x[idx], y[idx]).statistic

    alpha = (1 - ci) / 2
    ci_lo = np.nanpercentile(boot_rs, 100 * alpha)
    ci_hi = np.nanpercentile(boot_rs, 100 * (1 - alpha))

    return r_obs, ci_lo, ci_hi


# ==============================================================
# 1. LOAD DATA
# ==============================================================

print("=" * 65)
print("  Cohort-Level Pillar <-> Spike Correlation Analysis")
print("=" * 65)
print()

cgm_wide = pd.read_csv(CGM_CSV_PATH)
pillars_raw = pd.read_csv(PILLAR_CSV_PATH)

print("CGM data:     {} rows".format(len(cgm_wide)))
print("Pillars data: {} rows".format(len(pillars_raw)))


# ==============================================================
# 2. IDENTIFY ALL PATIENTS WITH BOTH DATA SOURCES
# ==============================================================

print()
print("-" * 65)
print("  Step 2: Patient Identification")
print("-" * 65)
print()

cgm_patients = set(cgm_wide[ID_COL].dropna().astype(str).unique())
pillars_clean = clean_pillars(pillars_raw)
pillar_patients = set(pillars_clean[ID_COL].dropna().astype(str).unique())

shared_patients = sorted(cgm_patients & pillar_patients)

print("CGM patients:       {}".format(len(cgm_patients)))
print("Pillar patients:    {}".format(len(pillar_patients)))
print("Shared (both data): {}".format(len(shared_patients)))

if len(shared_patients) == 0:
    print()
    print("[!] No patients have both CGM and pillar data. Cannot proceed.")
    exit(1)


# ==============================================================
# 3. RUN SPIKE DETECTION FOR ALL SHARED PATIENTS
# ==============================================================

print()
print("-" * 65)
print("  Step 3: Spike Detection (all shared patients)")
print("-" * 65)
print()

long_df = reshape_wide_to_long(cgm_wide, id_col=ID_COL, day_col=DAY_COL)

all_daily_rows = []
patients_processed = 0
patients_skipped = 0

for pid in shared_patients:
    try:
        events_df, annotated_df = detect_spikes(
            long_df,
            person_id=pid,
            amp_thresh=AMP_THRESH,
            rise_rate_thresh=RISE_RATE,
            end_band=END_BAND,
            min_above_min=MIN_ABOVE,
        )
    except Exception as e:
        print("  [!] Skipping {} -- spike detection error: {}".format(pid, e))
        patients_skipped += 1
        continue

    # Determine day grouping
    if "study_day" in annotated_df.columns:
        spike_day_col = "study_day"
        annotated_df[spike_day_col] = pd.to_numeric(annotated_df[spike_day_col], errors="coerce")
    else:
        annotated_df["datetime"] = pd.to_datetime(annotated_df["datetime"])
        spike_day_col = "date"
        annotated_df[spike_day_col] = annotated_df["datetime"].dt.date

    # Daily CGM summary
    daily = (
        annotated_df.groupby(spike_day_col).agg(
            mean_glucose=("glucose_mmol_l", "mean"),
            std_glucose=("glucose_mmol_l", "std"),
            min_glucose=("glucose_mmol_l", "min"),
            max_glucose=("glucose_mmol_l", "max"),
        ).reset_index()
    )

    # Daily spike counts
    if not events_df.empty and "Day" in events_df.columns:
        if spike_day_col == "study_day":
            events_df["_day_key"] = pd.to_numeric(events_df["Day"], errors="coerce")
        else:
            events_df["_day_key"] = events_df["Day"]

        agg_dict = {
            "spike_count": ("peak_glucose", "size"),
            "max_peak_glucose": ("peak_glucose", "max"),
            "mean_peak_glucose": ("peak_glucose", "mean"),
        }

        spike_daily = (
            events_df.groupby("_day_key").agg(**agg_dict).reset_index().rename(columns={"_day_key": spike_day_col})
        )

        daily = daily.merge(spike_daily, on=spike_day_col, how="left")
    
    daily["spike_count"] = daily.get("spike_count", pd.Series(0, index=daily.index)).fillna(0).astype(int)
    if "max_peak_glucose" not in daily.columns:
        daily["max_peak_glucose"] = np.nan
    if "mean_peak_glucose" not in daily.columns:
        daily["mean_peak_glucose"] = np.nan

    daily = daily.sort_values(spike_day_col).reset_index(drop=True)
    daily["day_index"] = range(1, len(daily) + 1)
    daily[ID_COL] = pid

    all_daily_rows.append(daily)
    patients_processed += 1

print("Processed: {}  |  Skipped: {}".format(patients_processed, patients_skipped))

if not all_daily_rows:
    print("[!] No patient data produced. Cannot proceed.")
    exit(1)

cgm_daily_all = pd.concat(all_daily_rows, ignore_index=True)
print("Total CGM day-rows: {}".format(len(cgm_daily_all)))


# ==============================================================
# 4. PREPARE PILLAR DATA & JOIN
# ==============================================================

print()
print("-" * 65)
print("  Step 4: Join CGM + Pillar Data")
print("-" * 65)
print()

# Build pillar daily table with gauge columns + day_index + ID
gauge_cols_present = [GAUGE_COLS[p] for p in PILLAR_NAMES if GAUGE_COLS[p] in pillars_clean.columns]
if NEGATIVE_COL in pillars_clean.columns:
    gauge_cols_present.append(NEGATIVE_COL)

pillar_daily_all = pillars_clean[[ID_COL, "day_index"] + gauge_cols_present].copy()

# Merge on (Client ID, day_index)
combined = cgm_daily_all.merge(pillar_daily_all, on=[ID_COL, "day_index"], how="inner")

print("Combined rows (before filtering): {}".format(len(combined)))

# Filter to patients with enough matched days
day_counts = combined.groupby(ID_COL).size()
valid_patients = day_counts[day_counts >= MIN_DAYS_PER_PATIENT].index
combined = combined[combined[ID_COL].isin(valid_patients)].reset_index(drop=True)

n_patients_final = combined[ID_COL].nunique()
n_rows_final = len(combined)

print("Patients with >= {} matched days: {}".format(MIN_DAYS_PER_PATIENT, n_patients_final))
print("Final data points: {}".format(n_rows_final))

if n_rows_final < 10:
    print("[!] Too few data points for meaningful cohort analysis.")
    exit(1)


# ==============================================================
# 5. WITHIN-PATIENT Z-SCORING
# ==============================================================

print()
print("-" * 65)
print("  Step 5: Within-Patient Z-Scoring")
print("-" * 65)
print()

combined_z = zscore_within_patient(combined, id_col=ID_COL)

# List the z-scored columns created
z_cols = [c for c in combined_z.columns if c.endswith("_z")]
print("Z-scored columns: {}".format(", ".join(z_cols)))
print("Sample (first 5 rows):")
print(combined_z[[ID_COL, "day_index", "spike_count"] + z_cols].head().to_string(index=False))


# ==============================================================
# 6. COHORT-WIDE CORRELATIONS (z-scored)
# ==============================================================

print()
print("-" * 65)
print("  Step 6: Cohort-Wide Spearman Correlations (z-scored)")
print("-" * 65)
print()

# Build mapping from pillar name to z-scored column name
pillar_z_map = {}
for pillar in PILLAR_NAMES:
    gcol = GAUGE_COLS[pillar]
    z_col = gcol.replace(" ", "_").lower() + "_z"
    if z_col in combined_z.columns:
        pillar_z_map[pillar] = z_col

if NEGATIVE_COL in combined_z.columns:
    neg_z = NEGATIVE_COL.replace(" ", "_").lower() + "_z"
    if neg_z in combined_z.columns:
        pillar_z_map["Negative"] = neg_z

spike_z_col = "spike_count_z"
glucose_z_col = "mean_glucose_z"

correlation_results = []

for pillar, z_col in pillar_z_map.items():
    # vs z-scored spike count
    valid_spike = combined_z[[z_col, spike_z_col]].dropna()
    if len(valid_spike) >= 10:
        r_spike, ci_lo_spike, ci_hi_spike = bootstrap_spearman_ci(
            valid_spike[z_col].values, valid_spike[spike_z_col].values
        )
        _, p_spike = stats.spearmanr(valid_spike[z_col], valid_spike[spike_z_col])
    else:
        r_spike, ci_lo_spike, ci_hi_spike, p_spike = np.nan, np.nan, np.nan, np.nan

    # vs z-scored mean glucose
    valid_gluc = combined_z[[z_col, glucose_z_col]].dropna()
    if len(valid_gluc) >= 10:
        r_gluc, ci_lo_gluc, ci_hi_gluc = bootstrap_spearman_ci(
            valid_gluc[z_col].values, valid_gluc[glucose_z_col].values
        )
        _, p_gluc = stats.spearmanr(valid_gluc[z_col], valid_gluc[glucose_z_col])
    else:
        r_gluc, ci_lo_gluc, ci_hi_gluc, p_gluc = np.nan, np.nan, np.nan, np.nan

    # Also compute raw (un-z-scored) correlation for comparison
    raw_gcol = GAUGE_COLS.get(pillar, NEGATIVE_COL)
    valid_raw = combined[[raw_gcol, "spike_count"]].dropna()
    if len(valid_raw) >= 10:
        r_raw, p_raw = stats.spearmanr(valid_raw[raw_gcol], valid_raw["spike_count"])
    else:
        r_raw, p_raw = np.nan, np.nan

    sig = "***" if p_spike < 0.001 else "**" if p_spike < 0.01 else "*" if p_spike < 0.05 else ""

    correlation_results.append({
        "Pillar": pillar,
        "r_spike_z": round(r_spike, 4) if not np.isnan(r_spike) else np.nan,
        "ci_lo_spike": round(ci_lo_spike, 4) if not np.isnan(ci_lo_spike) else np.nan,
        "ci_hi_spike": round(ci_hi_spike, 4) if not np.isnan(ci_hi_spike) else np.nan,
        "p_spike_z": round(p_spike, 6) if not np.isnan(p_spike) else np.nan,
        "r_glucose_z": round(r_gluc, 4) if not np.isnan(r_gluc) else np.nan,
        "ci_lo_glucose": round(ci_lo_gluc, 4) if not np.isnan(ci_lo_gluc) else np.nan,
        "ci_hi_glucose": round(ci_hi_gluc, 4) if not np.isnan(ci_hi_gluc) else np.nan,
        "p_glucose_z": round(p_gluc, 6) if not np.isnan(p_gluc) else np.nan,
        "r_spike_raw": round(r_raw, 4) if not np.isnan(r_raw) else np.nan,
        "p_spike_raw": round(p_raw, 6) if not np.isnan(p_raw) else np.nan,
        "n_datapoints": len(valid_spike),
    })

    print("  {:12s}  r_z={:+.4f} [{:+.4f}, {:+.4f}]  p={:.6f} {}  (raw r={:+.4f})".format(
        pillar,
        r_spike if not np.isnan(r_spike) else 0,
        ci_lo_spike if not np.isnan(ci_lo_spike) else 0,
        ci_hi_spike if not np.isnan(ci_hi_spike) else 0,
        p_spike if not np.isnan(p_spike) else 1,
        sig,
        r_raw if not np.isnan(r_raw) else 0,
    ))

corr_df = pd.DataFrame(correlation_results)

# Rank by absolute z-scored spike correlation
if not corr_df.empty:
    corr_df = corr_df.sort_values("r_spike_z", key=lambda x: x.abs(), ascending=False)
    print()
    print("  Ranked by |r| (z-scored vs spike count):")
    for i, row in corr_df.iterrows():
        sig = "*" if row["p_spike_z"] < 0.05 else ""
        print("    {}. {:12s}  r={:+.4f}  {}".format(
            corr_df.index.get_loc(i) + 1, row["Pillar"], row["r_spike_z"], sig))


# ==============================================================
# 7. MULTIPLE COMPARISON CORRECTION (Bonferroni + FDR)
# ==============================================================

# ==============================================================
# 7. MULTIPLE COMPARISON CORRECTION (Bonferroni + FDR)
# ==============================================================

print()
print("-" * 65)
print("  Step 7: Multiple Comparison Correction")
print("-" * 65)
print()

n_tests = len(corr_df)
alpha = 0.05

if n_tests > 0 and corr_df["p_spike_z"].notna().any():
    # Bonferroni
    corr_df["p_bonferroni"] = np.minimum(corr_df["p_spike_z"] * n_tests, 1.0)
    corr_df["sig_bonferroni"] = corr_df["p_bonferroni"] < alpha

    # Benjamini-Hochberg FDR
    p_vals = corr_df["p_spike_z"].values.copy()
    n_p = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]

    # Compute FDR-adjusted p-values
    fdr_p = np.empty(n_p)
    for i in range(n_p):
        rank = i + 1
        fdr_p[i] = sorted_p[i] * n_p / rank

    # Enforce monotonicity (step-up): walk backwards, take cumulative min
    fdr_p = np.minimum.accumulate(fdr_p[::-1])[::-1]
    fdr_p = np.clip(fdr_p, 0, 1.0)

    # Map back to original order
    p_fdr_original = np.empty(n_p)
    for i in range(n_p):
        p_fdr_original[sorted_indices[i]] = fdr_p[i]

    corr_df["p_fdr"] = p_fdr_original
    corr_df["sig_fdr"] = corr_df["p_fdr"] < alpha

    print("  {:12s}  {:>10s}  {:>10s}  {:>10s}  {:>5s}  {:>5s}".format(
        "Pillar", "p_raw", "p_bonf", "p_fdr", "Bonf", "FDR"))
    print("  " + "-" * 58)
    for _, row in corr_df.iterrows():
        print("  {:12s}  {:10.6f}  {:10.6f}  {:10.6f}  {:>5s}  {:>5s}".format(
            row["Pillar"],
            row["p_spike_z"],
            row["p_bonferroni"],
            row["p_fdr"],
            "Yes" if row["sig_bonferroni"] else "No",
            "Yes" if row["sig_fdr"] else "No",
        ))

# ==============================================================
# 8. MIXED-EFFECTS MODEL (optional)
# ==============================================================

print()
print("-" * 65)
print("  Step 8: Mixed-Effects Model")
print("-" * 65)
print()

try:
    import statsmodels.formula.api as smf

    # Build formula with all available z-scored pillar columns
    predictor_cols = []
    for pillar, z_col in pillar_z_map.items():
        if z_col in combined_z.columns and combined_z[z_col].notna().sum() > 20:
            predictor_cols.append(z_col)

    if predictor_cols and spike_z_col in combined_z.columns:
        # Drop rows with any NaN in the model columns
        model_df = combined_z[[ID_COL, spike_z_col] + predictor_cols].dropna()

        if len(model_df) >= 30 and model_df[ID_COL].nunique() >= 3:
            formula = "{} ~ {}".format(spike_z_col, " + ".join(predictor_cols))
            print("  Formula: {}".format(formula))
            print("  Groups:  {} (patients)".format(model_df[ID_COL].nunique()))
            print("  N:       {}".format(len(model_df)))
            print()

            model = smf.mixedlm(
                formula,
                data=model_df,
                groups=model_df[ID_COL],
            )
            result = model.fit(reml=True)
            print(result.summary())

            # Extract fixed effects for export
            fe = result.fe_params
            fe_pvals = result.pvalues
            mixed_results = pd.DataFrame({
                "Predictor": fe.index,
                "Coefficient": fe.values.round(4),
                "p_value": fe_pvals.values.round(6),
            })
            mixed_results["significant"] = mixed_results["p_value"] < 0.05
            print()
            print("  Fixed effects summary:")
            for _, row in mixed_results.iterrows():
                sig = "*" if row["significant"] else ""
                print("    {:30s}  coef={:+.4f}  p={:.6f} {}".format(
                    row["Predictor"], row["Coefficient"], row["p_value"], sig))
        else:
            print("  [!] Insufficient data for mixed-effects model.")
            mixed_results = None
    else:
        print("  [!] No valid predictor columns for mixed-effects model.")
        mixed_results = None

except ImportError:
    print("  [!] statsmodels not installed. Skipping mixed-effects model.")
    print("      Install with: pip install statsmodels")
    mixed_results = None


# ==============================================================
# 9. VISUALISATIONS
# ==============================================================

print()
print("-" * 65)
print("  Step 9: Generating Figures")
print("-" * 65)
print()

sns.set_theme(style="whitegrid", font_scale=0.95)
OUTPUT_DIR.mkdir(exist_ok=True)


# -- 9a. Correlation heatmap (z-scored) --
if not corr_df.empty:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    hm = corr_df.set_index("Pillar")[["r_spike_z", "r_glucose_z"]].copy()
    hm.columns = ["Spike Count (z)", "Mean Glucose (z)"]

    sns.heatmap(
        hm, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
        vmin=-0.5, vmax=0.5, linewidths=0.5,
        cbar_kws={"label": "Spearman r (within-patient z-scored)"},
        ax=ax,
    )

    # Add significance markers
    for i, (_, row) in enumerate(corr_df.iterrows()):
        for j, p_col in enumerate(["p_spike_z", "p_glucose_z"]):
            p = row[p_col]
            if pd.notna(p) and p < 0.05:
                marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                ax.text(j + 0.5, i + 0.78, marker, ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")

    ax.set_title("Cohort Pillar <-> Glucose Correlations\n(within-patient z-scored, n={})".format(
        n_rows_final), fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cohort_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] cohort_correlation_heatmap.png")


# -- 9b. Forest plot: effect sizes with 95% CIs --
if not corr_df.empty:
    plot_df = corr_df.dropna(subset=["r_spike_z", "ci_lo_spike", "ci_hi_spike"]).copy()
    plot_df = plot_df.sort_values("r_spike_z")

    fig, ax = plt.subplots(figsize=(8, max(3, len(plot_df) * 0.6)))

    y_pos = range(len(plot_df))
    colors = [PILLAR_COLORS.get(p, "#666") for p in plot_df["Pillar"]]

    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = PILLAR_COLORS.get(row["Pillar"], "#666")
        ax.plot(
            [row["ci_lo_spike"], row["ci_hi_spike"]], [i, i],
            color=color, linewidth=2.5, solid_capstyle="round",
        )
        ax.scatter(row["r_spike_z"], i, color=color, s=80, zorder=5, edgecolors="white", linewidths=0.8)

        sig = ""
        if pd.notna(row["p_spike_z"]):
            if row["p_spike_z"] < 0.001:
                sig = " ***"
            elif row["p_spike_z"] < 0.01:
                sig = " **"
            elif row["p_spike_z"] < 0.05:
                sig = " *"

        ax.text(row["ci_hi_spike"] + 0.01, i, "{:+.3f}{}".format(row["r_spike_z"], sig),
                va="center", fontsize=9, fontweight="bold", color=color)

    ax.axvline(0, color="#999", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["Pillar"].tolist(), fontsize=10)
    ax.set_xlabel("Spearman r (95% bootstrap CI)", fontsize=10)
    ax.set_title("Cohort Pillar -> Spike Count Associations\n(within-patient z-scored, n={})".format(
        n_rows_final), fontsize=13, fontweight="bold", pad=12)
    sns.despine(left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cohort_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] cohort_forest_plot.png")


# -- 9c. Scatter plots: top 3 pillars vs spike count (z-scored) --
if not corr_df.empty and len(corr_df) >= 2:
    top3 = corr_df.head(3)  # already sorted by |r|

    n_plots = len(top3)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top3.iterrows()):
        pillar = row["Pillar"]
        z_col = pillar_z_map.get(pillar)
        if not z_col:
            continue

        color = PILLAR_COLORS.get(pillar, "#4a90d9")
        valid = combined_z[[z_col, spike_z_col, ID_COL]].dropna()

        ax.scatter(
            valid[z_col], valid[spike_z_col],
            color=color, alpha=0.3, s=30, edgecolors="none",
        )

        # Trend line
        if len(valid) >= 5:
            z = np.polyfit(valid[z_col], valid[spike_z_col], 1)
            poly = np.poly1d(z)
            x_range = np.linspace(valid[z_col].min(), valid[z_col].max(), 50)
            ax.plot(x_range, poly(x_range), color=color, linewidth=2, linestyle="-", alpha=0.8)

        sig = ""
        if row["p_spike_z"] < 0.001:
            sig = "***"
        elif row["p_spike_z"] < 0.01:
            sig = "**"
        elif row["p_spike_z"] < 0.05:
            sig = "*"

        ax.set_title("{}\nr={:+.3f} (p={:.4f}) {}".format(
            pillar, row["r_spike_z"], row["p_spike_z"], sig),
            fontsize=11, fontweight="bold")
        ax.set_xlabel("{} (z-scored)".format(pillar), fontsize=10)
        ax.set_ylabel("Spike Count (z-scored)", fontsize=10)
        ax.axhline(0, color="#ccc", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#ccc", linewidth=0.5, zorder=0)

    fig.suptitle("Top Correlated Pillars vs Spike Count (cohort, n={})".format(n_rows_final),
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cohort_scatter_top_pillars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] cohort_scatter_top_pillars.png")


# -- 9d. Z-scored vs raw correlation comparison --
if not corr_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(corr_df))
    width = 0.35

    bars_raw = ax.bar(x - width / 2, corr_df["r_spike_raw"].fillna(0), width,
                      label="Raw (no z-score)", color="#94a3b8", alpha=0.8)
    bars_z = ax.bar(x + width / 2, corr_df["r_spike_z"].fillna(0), width,
                    label="Within-patient z-scored", color="#4a90d9", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(corr_df["Pillar"].tolist(), fontsize=10)
    ax.set_ylabel("Spearman r vs Spike Count", fontsize=10)
    ax.set_title("Effect of Within-Patient Z-Scoring on Correlations", fontsize=13, fontweight="bold", pad=12)
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.legend(fontsize=9)
    sns.despine(left=False, bottom=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cohort_raw_vs_zscore.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] cohort_raw_vs_zscore.png")


# ==============================================================
# 10. CLINICAL SUMMARY
# ==============================================================

print()
print("=" * 65)
print("  COHORT CLINICAL SUMMARY")
print("=" * 65)
print()

print("  Patients analysed:  {}".format(n_patients_final))
print("  Total data points:  {}".format(n_rows_final))
print("  Mean days/patient:  {:.1f}".format(n_rows_final / n_patients_final))
print("  Total spikes:       {}".format(int(combined["spike_count"].sum())))
print("  Mean spikes/day:    {:.2f} (+/-{:.2f})".format(
    combined["spike_count"].mean(), combined["spike_count"].std()))
print("  Mean glucose:       {:.2f} mmol/L".format(combined["mean_glucose"].mean()))

if not corr_df.empty:
    print()
    print("  Pillar associations with spike count (z-scored, ranked):")
    print("  " + "-" * 55)
    for rank, (_, row) in enumerate(corr_df.iterrows(), 1):
        sig = ""
        if pd.notna(row.get("sig_fdr")) and row["sig_fdr"]:
            sig = " (FDR significant)"
        elif pd.notna(row.get("sig_bonferroni")) and row["sig_bonferroni"]:
            sig = " (Bonferroni significant)"
        elif pd.notna(row["p_spike_z"]) and row["p_spike_z"] < 0.05:
            sig = " (nominally significant)"

        print("    {}. {:12s}  r={:+.4f}  p={:.6f}{}".format(
            rank, row["Pillar"], row["r_spike_z"], row["p_spike_z"], sig))

    best = corr_df.iloc[0]
    direction = "Lower" if best["r_spike_z"] < 0 else "Higher"
    print()
    print("  > Primary cohort-level finding:")
    print("    {} self-reported {} is most strongly associated".format(direction, best["Pillar"]))
    print("    with increased glucose spike frequency across the cohort.")
    print("    (r={:+.4f}, 95% CI [{:+.4f}, {:+.4f}], p={:.6f})".format(
        best["r_spike_z"], best["ci_lo_spike"], best["ci_hi_spike"], best["p_spike_z"]))

if mixed_results is not None:
    sig_predictors = mixed_results[mixed_results["significant"] & (mixed_results["Predictor"] != "Intercept")]
    if not sig_predictors.empty:
        print()
        print("  > Mixed-effects model significant predictors:")
        for _, row in sig_predictors.iterrows():
            # Map z-col name back to pillar name
            pname = row["Predictor"].replace("_gauge_z", "").replace("_", " ").title()
            print("    {:20s}  coef={:+.4f}  p={:.6f}".format(pname, row["Coefficient"], row["p_value"]))


# ==============================================================
# 11. EXPORT
# ==============================================================

print()
print("-" * 65)
print("  Step 11: Exporting Results")
print("-" * 65)
print()

corr_df.to_csv(OUTPUT_DIR / "cohort_correlations.csv", index=False)
print("  [OK] cohort_correlations.csv")

combined.to_csv(OUTPUT_DIR / "cohort_combined_daily.csv", index=False)
print("  [OK] cohort_combined_daily.csv")

combined_z.to_csv(OUTPUT_DIR / "cohort_combined_zscored.csv", index=False)
print("  [OK] cohort_combined_zscored.csv")

if mixed_results is not None:
    mixed_results.to_csv(OUTPUT_DIR / "cohort_mixed_effects.csv", index=False)
    print("  [OK] cohort_mixed_effects.csv")

print()
print("  All outputs saved to: {}/".format(OUTPUT_DIR.resolve()))
print()
print("=" * 65)
print("  Pipeline complete.")
print("=" * 65)