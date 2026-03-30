# pillar_analysis.py
"""
Lifestyle Pillars Analysis Module
──────────────────────────────────
Provides functions for parsing, summarising, and visualising
self-reported lifestyle pillar data from the SPIKES study.

Pillars:
    Health · Sleep · Nutrition · Movement · Stress · Connection

Each pillar has a gauge (1–10) and a trend ("improving" / "staying the same" / "declining").
An additional "Negative Gauge" (1–10) captures negative affect.

Expected CSV columns:
    Client ID |
    Health Gauge | Health Gauge Trend |
    Sleep Gauge  | Sleep Gauge Trend  |
    Nutrition Gauge | Nutrition Gauge Trend |
    Movement Gauge  | Movement Gauge Trend  |
    Stress Gauge | Stress Gauge Trend |
    Connection Gauge | Connection Gauge Trend |
    Negative Gauge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Optional

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

PILLAR_NAMES = ["Health", "Sleep", "Nutrition", "Movement", "Stress", "Connection"]

GAUGE_COLS = {p: f"{p} Gauge" for p in PILLAR_NAMES}
TREND_COLS = {p: f"{p} Gauge Trend" for p in PILLAR_NAMES}
NEGATIVE_COL = "Negative Gauge"
ID_COL = "Client ID"

TREND_MAP = {
    "improving": 1,
    "staying the same": 0,
    "declining": -1,
}

PILLAR_COLORS = {
    "Health":     "#4a90d9",
    "Sleep":      "#7c5cbf",
    "Nutrition":  "#22c55e",
    "Movement":   "#f59e0b",
    "Stress":     "#ef4444",
    "Connection": "#ec4899",
}

TREND_COLORS = {
    "improving":        "#22c55e",
    "staying the same": "#6b7280",
    "declining":        "#ef4444",
}


# ──────────────────────────────────────────────
# 1. CLEANING & VALIDATION
# ──────────────────────────────────────────────

def clean_pillars(
    df: pd.DataFrame,
    id_col: str = ID_COL,
) -> pd.DataFrame:
    """
    Clean and validate a raw pillars DataFrame.

    - Strips whitespace from string columns
    - Coerces gauge columns to numeric
    - Normalises trend strings to lowercase
    - Drops rows with no Client ID
    - Adds a ``day_index`` (0-based) per patient

    Returns a cleaned copy.
    """
    d = df.copy()

    # Strip column names
    d.columns = d.columns.str.strip()

    # Drop rows without an ID
    d = d.dropna(subset=[id_col])
    d[id_col] = d[id_col].astype(str).str.strip()

    # Coerce gauge columns
    for pillar, col in GAUGE_COLS.items():
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    if NEGATIVE_COL in d.columns:
        d[NEGATIVE_COL] = pd.to_numeric(d[NEGATIVE_COL], errors="coerce")

    # Normalise trend strings
    for pillar, col in TREND_COLS.items():
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip().str.lower()
            # Fix common partial matches
            d[col] = d[col].replace({
                "staying the": "staying the same",
                "staying the s": "staying the same",
                "staying the sa": "staying the same",
                "staying the sam": "staying the same",
                "staying the same": "staying the same",
            })

    # Add day index per patient
    d["day_index"] = d.groupby(id_col).cumcount() + 1

    d = d.reset_index(drop=True)
    return d


# ──────────────────────────────────────────────
# 2. PER-PATIENT SUMMARY
# ──────────────────────────────────────────────

def patient_summary(
    df: pd.DataFrame,
    patient_id: str,
    id_col: str = ID_COL,
) -> dict:
    """
    Compute a summary dict for one patient across all pillars.

    Returns
    -------
    dict with keys:
        patient_id : str
        n_days : int
        pillars : dict[str, dict]
            Per-pillar sub-dict with:
                mean, median, std, min, max,
                first, last, delta (last - first),
                trend_counts (dict), dominant_trend (str)
        negative : dict
            mean, median, std, min, max
    """
    p = df.loc[df[id_col].astype(str) == str(patient_id)].copy()

    if p.empty:
        return {"patient_id": patient_id, "n_days": 0, "pillars": {}, "negative": {}}

    n_days = len(p)
    result = {"patient_id": patient_id, "n_days": n_days, "pillars": {}}

    for pillar in PILLAR_NAMES:
        gcol = GAUGE_COLS[pillar]
        tcol = TREND_COLS[pillar]

        vals = p[gcol].dropna() if gcol in p.columns else pd.Series(dtype=float)
        trends = p[tcol].dropna() if tcol in p.columns else pd.Series(dtype=str)

        pinfo = {}
        if not vals.empty:
            pinfo["mean"]   = round(float(vals.mean()), 2)
            pinfo["median"] = round(float(vals.median()), 2)
            pinfo["std"]    = round(float(vals.std()), 2)
            pinfo["min"]    = int(vals.min())
            pinfo["max"]    = int(vals.max())
            pinfo["first"]  = float(vals.iloc[0])
            pinfo["last"]   = float(vals.iloc[-1])
            pinfo["delta"]  = round(float(vals.iloc[-1] - vals.iloc[0]), 2)
        else:
            pinfo = {k: None for k in ["mean", "median", "std", "min", "max", "first", "last", "delta"]}

        # Trend distribution
        if not trends.empty:
            tc = trends.value_counts().to_dict()
            pinfo["trend_counts"] = tc
            pinfo["dominant_trend"] = max(tc, key=tc.get)
        else:
            pinfo["trend_counts"] = {}
            pinfo["dominant_trend"] = None

        result["pillars"][pillar] = pinfo

    # Negative gauge
    if NEGATIVE_COL in p.columns:
        nvals = p[NEGATIVE_COL].dropna()
        if not nvals.empty:
            result["negative"] = {
                "mean":   round(float(nvals.mean()), 2),
                "median": round(float(nvals.median()), 2),
                "std":    round(float(nvals.std()), 2),
                "min":    int(nvals.min()),
                "max":    int(nvals.max()),
            }
        else:
            result["negative"] = {}
    else:
        result["negative"] = {}

    return result


# ──────────────────────────────────────────────
# 3. VISUALISATION: gauge time-series
# ──────────────────────────────────────────────

def plot_gauge_timeseries(
    df: pd.DataFrame,
    patient_id: str,
    id_col: str = ID_COL,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Line chart of all pillar gauges over the study days for one patient.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)

    p = df.loc[df[id_col].astype(str) == str(patient_id)].copy()
    p = p.sort_values("day_index")

    fig, ax = plt.subplots(figsize=figsize)

    for pillar in PILLAR_NAMES:
        gcol = GAUGE_COLS[pillar]
        if gcol not in p.columns:
            continue
        subset = p[["day_index", gcol]].dropna()
        if subset.empty:
            continue
        ax.plot(
            subset["day_index"],
            subset[gcol],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=PILLAR_COLORS[pillar],
            label=pillar,
            alpha=0.85,
        )

    ax.set_xlabel("Study Day", fontsize=10)
    ax.set_ylabel("Self-Reported Gauge (1–10)", fontsize=10)
    ax.set_title(f"{patient_id}  ·  Lifestyle Pillars Over Time", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=3)
    sns.despine(left=False, bottom=False)
    fig.tight_layout()

    return fig


# ──────────────────────────────────────────────
# 4. VISUALISATION: trend distribution (stacked bar)
# ──────────────────────────────────────────────

def plot_trend_distribution(
    df: pd.DataFrame,
    patient_id: str,
    id_col: str = ID_COL,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Horizontal stacked bar chart showing the proportion of
    improving / staying the same / declining for each pillar.
    """
    sns.set_theme(style="whitegrid", font_scale=0.95)

    p = df.loc[df[id_col].astype(str) == str(patient_id)].copy()

    data = []
    for pillar in PILLAR_NAMES:
        tcol = TREND_COLS[pillar]
        if tcol not in p.columns:
            continue
        valid = p[tcol].dropna()
        valid = valid[valid.isin(["improving", "staying the same", "declining"])]
        if valid.empty:
            continue
        counts = valid.value_counts()
        total = counts.sum()
        for trend_label in ["improving", "staying the same", "declining"]:
            data.append({
                "Pillar": pillar,
                "Trend": trend_label,
                "Count": counts.get(trend_label, 0),
                "Pct": counts.get(trend_label, 0) / total * 100 if total > 0 else 0,
            })

    # Handle empty data gracefully
    if not data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No trend data available",
            ha="center", va="center", fontsize=12, color="#6b7280",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.tight_layout()
        return fig

    td = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Only plot pillars that actually have data
    pillars_with_data = [p for p in reversed(PILLAR_NAMES) if p in td["Pillar"].values]
    y_pos = np.arange(len(pillars_with_data))

    for trend_label in ["improving", "staying the same", "declining"]:
        widths = []
        lefts = []
        for pname in pillars_with_data:
            w = td.loc[(td["Pillar"] == pname) & (td["Trend"] == trend_label), "Pct"]
            widths.append(float(w.values[0]) if not w.empty else 0)

            left = 0
            for prev_trend in ["improving", "staying the same", "declining"]:
                if prev_trend == trend_label:
                    break
                prev_sub = td.loc[(td["Pillar"] == pname) & (td["Trend"] == prev_trend), "Pct"]
                left += float(prev_sub.values[0]) if not prev_sub.empty else 0
            lefts.append(left)

        ax.barh(
            y_pos, widths, left=lefts, height=0.6,
            color=TREND_COLORS[trend_label],
            label=trend_label.title(),
            alpha=0.85,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pillars_with_data, fontsize=10)
    ax.set_xlabel("% of Days", fontsize=10)
    ax.set_title(f"{patient_id}  ·  Trend Distribution", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    sns.despine(left=False, bottom=False)
    fig.tight_layout()

    return fig


# ──────────────────────────────────────────────
# 5. VISUALISATION: radar / spider chart (averages)
# ──────────────────────────────────────────────

def plot_pillar_radar(
    df: pd.DataFrame,
    patient_id: str,
    id_col: str = ID_COL,
    figsize: tuple = (6, 6),
) -> plt.Figure:
    """
    Radar chart of mean gauge values across the six pillars.
    """
    p = df.loc[df[id_col].astype(str) == str(patient_id)].copy()

    means = []
    labels = []
    for pillar in PILLAR_NAMES:
        gcol = GAUGE_COLS[pillar]
        if gcol in p.columns:
            val = p[gcol].mean()
            means.append(val if pd.notna(val) else 0)
        else:
            means.append(0)
        labels.append(pillar)

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    means += means[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.fill(angles, means, color="#4a90d9", alpha=0.15)
    ax.plot(angles, means, color="#4a90d9", linewidth=2, marker="o", markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="600")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color="#999")
    ax.set_title(
        f"{patient_id}  ·  Average Pillar Scores",
        fontsize=13, fontweight="bold", pad=20, y=1.08,
    )

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# 6. VISUALISATION: negative gauge time-series
# ──────────────────────────────────────────────

def plot_negative_gauge(
    df: pd.DataFrame,
    patient_id: str,
    id_col: str = ID_COL,
    figsize: tuple = (14, 3),
) -> Optional[plt.Figure]:
    """
    Simple line chart of the Negative Gauge over study days.
    Returns None if no data available.
    """
    if NEGATIVE_COL not in df.columns:
        return None

    sns.set_theme(style="whitegrid", font_scale=0.95)

    p = df.loc[df[id_col].astype(str) == str(patient_id)].copy()
    p = p.sort_values("day_index")
    subset = p[["day_index", NEGATIVE_COL]].dropna()

    if subset.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(
        subset["day_index"], subset[NEGATIVE_COL],
        color="#ef4444", alpha=0.12,
    )
    ax.plot(
        subset["day_index"], subset[NEGATIVE_COL],
        color="#ef4444", linewidth=1.5, marker="o", markersize=4,
    )

    ax.set_xlabel("Study Day", fontsize=10)
    ax.set_ylabel("Negative Gauge", fontsize=10)
    ax.set_title(f"{patient_id}  ·  Negative Affect Over Time", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    sns.despine(left=False, bottom=False)
    fig.tight_layout()

    return fig


# ──────────────────────────────────────────────
# 7. COHORT-LEVEL SUMMARY
# ──────────────────────────────────────────────

def cohort_summary(
    df: pd.DataFrame,
    id_col: str = ID_COL,
) -> pd.DataFrame:
    """
    Compute per-patient mean gauges across all pillars.

    Returns a DataFrame with one row per patient and columns:
        Client ID | Health | Sleep | Nutrition | Movement | Stress | Connection | Negative
    """
    patients = df[id_col].unique()
    rows = []
    for pid in patients:
        p = df.loc[df[id_col] == pid]
        row = {id_col: pid}
        for pillar in PILLAR_NAMES:
            gcol = GAUGE_COLS[pillar]
            if gcol in p.columns:
                row[pillar] = round(float(p[gcol].mean()), 2) if p[gcol].notna().any() else None
            else:
                row[pillar] = None
        if NEGATIVE_COL in p.columns:
            row["Negative"] = round(float(p[NEGATIVE_COL].mean()), 2) if p[NEGATIVE_COL].notna().any() else None
        else:
            row["Negative"] = None
        rows.append(row)

    return pd.DataFrame(rows)

def zscore_within_patient(df, id_col="Client ID"):
    """
    Z-score all pillar gauge columns within each patient.
    Also z-scores spike_count and mean_glucose for consistency.
    
    Adds new columns with '_z' suffix. Originals are preserved.
    """
    result = df.copy()
    
    cols_to_zscore = [
        "Health Gauge", "Sleep Gauge", "Nutrition Gauge",
        "Movement Gauge", "Stress Gauge", "Connection Gauge",
        "Negative Gauge", "spike_count", "mean_glucose",
    ]
    
    for col in cols_to_zscore:
        if col not in result.columns:
            continue
        z_col = col.replace(" ", "_").lower() + "_z"
        result[z_col] = result.groupby(id_col)[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
    
    return result