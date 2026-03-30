# app.py
"""
CGM Spike & Dip Analysis Dashboard
─────────────────────────────────────
A Streamlit app that lets users select a patient, toggle detection layers
(high-confidence / sensitive × spikes / dips), and view annotated plots.

Launch with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from io import BytesIO



# ──────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────

# # Import from your analysis module — detect_events replaces detect_spikes
# from spike_analysis import reshape_wide_to_long, detect_events
# from pillar_analysis import (
#     clean_pillars, patient_summary,
#     plot_gauge_timeseries, plot_trend_distribution,
#     plot_pillar_radar, plot_negative_gauge,
#     PILLAR_NAMES, PILLAR_COLORS, TREND_COLORS,
# )



# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Sidebar branding */.sidebar-brand {
        font-size: 1.35rem;
        font-weight: 800;
        color: #1a1a2e;
        letter-spacing: 0.02em;
        padding-bottom: 0.25rem;
        margin-bottom: 0.25rem;
        border-bottom: 3px solid #4a90d9;
        display: inline-block;
    }.sidebar-brand span {
        color: #4a90d9;
    }

    /* Header bar */.main-header {
        background: #ffffff;
        border: 1px solid #e2e6ec;
        border-left: 4px solid #4a90d9;
        padding: 1.25rem 1.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }.main-header h1 {
        color: #1a1a2e;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }.main-header p {
        color: #6b7280;
        margin: 0.3rem 0 0 0;
        font-size: 0.88rem;
        font-weight: 400;
    }

    /* Metric cards */.metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }.metric-card {
        flex: 1;
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
    }.metric-card.spike   { border-left-color: #e94560; }.metric-card.dip     { border-left-color: #3B7DD8; }.metric-card.days    { border-left-color: #0f3460; }.metric-card.conf    { border-left-color: #e9a045; }.metric-card.glucose { border-left-color: #45e980; }.metric-card.label  { font-size: 0.8rem; color: #666; text-transform: uppercase; }.metric-card.value  { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }

    /* Day section */.day-header {
        background: #f0f2f6;
        padding: 0.6rem 1rem;
        border-radius: 0.4rem;
        margin: 1.5rem 0 0.75rem 0;
        font-weight: 600;
        color: #1a1a2e;
    }

    /* WIP badge */.wip-badge {
        display: inline-block;
        background: #e9a045;
        color: #fff;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 0.3rem;
        vertical-align: middle;
        margin-left: 0.4rem;
        letter-spacing: 0.03em;
    }

    /* Toggle section label */.toggle-section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
        margin-top: 0.5rem;
    }

    /* Patient Overview */.patient-profile {
        background: #ffffff;
        border: 1px solid #e2e6ec;
        border-radius: 0.5rem;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }.patient-profile h3 {
        color: #1a1a2e;
        margin: 0 0 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #eef0f4;
        padding-bottom: 0.6rem;
    }.patient-field {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        font-size: 0.9rem;
    }.patient-field.field-label {
        color: #6b7280;
        font-weight: 500;
    }.patient-field.field-value {
        color: #1a1a2e;
        font-weight: 600;
        text-align: right;
    }.health-goal-quote {
        background: #f8f9fb;
        border-left: 4px solid #4a90d9;
        border-radius: 0 0.4rem 0.4rem 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        font-style: italic;
        color: #1a1a2e;
        line-height: 1.6;
    }.health-goal-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }.overview-header {
        background: #ffffff;
        border: 1px solid #e2e6ec;
        border-left: 4px solid #4a90d9;
        padding: 1.25rem 1.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }.overview-header h1 {
        color: #1a1a2e;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }.overview-header p {
        color: #6b7280;
        margin: 0.3rem 0 0 0;
        font-size: 0.88rem;
        font-weight: 400;
    }

    /* Pillar cards */.pillar-card {
        background: #ffffff;
        border: 1px solid #e2e6ec;
        border-top: 3px solid #4a90d9;
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        text-align: center;
    }.pillar-card.pillar-name {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }.pillar-card.pillar-score {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }.pillar-card.pillar-trend {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }.pillar-card.pillar-delta {
        font-size: 0.72rem;
        color: #6b7280;
        margin-top: 0.15rem;
    }.trend-improving  { color: #22c55e; }.trend-staying    { color: #6b7280; }.trend-declining  { color: #ef4444; }.pillars-header {
        background: #ffffff;
        border: 1px solid #e2e6ec;
        border-left: 4px solid #4a90d9;
        padding: 1.25rem 1.75rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }.pillars-header h1 {
        color: #1a1a2e;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }.pillars-header p {
        color: #6b7280;
        margin: 0.3rem 0 0 0;
        font-size: 0.88rem;
        font-weight: 400;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# EVENT VISUAL STYLES
# ──────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
# EVENT STYLES — used by build_day_figure
# ══════════════════════════════════════════════════════════════

_EVENT_STYLES = {
    ("up", "high_confidence"): {
        "shade": "#D32F2F",
        "shade_alpha": 0.45,
        "marker_color": "#D32F2F",
        "marker": "^",
        "marker_size": 130,
        "border_color": "#D32F2F",
        "label": "Hard Spike (>2 SD)",
    },
    ("up", "sensitive"): {
        "shade": "#F57C00",
        "shade_alpha": 0.35,
        "marker_color": "#F57C00",
        "marker": "^",
        "marker_size": 80,
        "border_color": "#F57C00",
        "label": "Soft Spike (1–2 SD)",
    },
    ("down", "high_confidence"): {
        "shade": "#0D47A1",
        "shade_alpha": 0.45,
        "marker_color": "#0D47A1",
        "marker": "v",
        "marker_size": 130,
        "border_color": "#0D47A1",
        "label": "Hard Dip (>2 SD)",
    },
    ("down", "sensitive"): {
        "shade": "#42A5F5",
        "shade_alpha": 0.35,
        "marker_color": "#42A5F5",
        "marker": "v",
        "marker_size": 80,
        "border_color": "#42A5F5",
        "label": "Soft Dip (1–2 SD)",
    },
}


def build_day_figure(
    day_df: pd.DataFrame,
    ev_day: pd.DataFrame,
    person_id: str,
    day,
    active_filters: set,
    show_baseline: bool = True,
    show_event_component: bool = False,
    glucose_col: str = "glucose_mmol_l",
    dt_col: str = "datetime",
) -> plt.Figure:
    """
    Return a matplotlib Figure for one day's CGM trace with event annotations.
    """

    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)

    has_ec = "event_component" in day_df.columns and show_event_component

    if has_ec:
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, 7),
            height_ratios=[3.5, 1],
            sharex=True,
            gridspec_kw={"hspace": 0.06},
        )
    else:
        fig, ax1 = plt.subplots(figsize=(14, 4.5))
        ax2 = None

    # ── Raw glucose trace (faint) ──
    ax1.plot(
        day_df[dt_col], day_df[glucose_col],
        color="#C0C0C0", lw=0.7, alpha=0.5, zorder=2,
    )

    # ── Denoised glucose (primary trace) ──
    if "denoised" in day_df.columns:
        ax1.plot(
            day_df[dt_col], day_df["denoised"],
            color="#1F2937", lw=1.5, zorder=3, label="Glucose (denoised)",
        )
    else:
        ax1.plot(
            day_df[dt_col], day_df[glucose_col],
            color="#1F2937", lw=1.5, zorder=3, label="Glucose",
        )

    # ── Baseline (check both column names for compatibility) ──
    baseline_col = None
    if "baseline" in day_df.columns:
        baseline_col = "baseline"
    elif "baseline_fft" in day_df.columns:
        baseline_col = "baseline_fft"

    if show_baseline and baseline_col is not None:
        ax1.plot(
            day_df[dt_col], day_df[baseline_col],
            color="#7F8C8D", lw=1.6, ls="--", alpha=0.6,
            zorder=2, label="Baseline",
        )

    # ── Hypo reference line ──
    ax1.axhline(3.9, color="#3B7DD8", ls=":", lw=0.7, alpha=0.35)
    if not day_df.empty:
        ax1.text(
            day_df[dt_col].iloc[0], 3.92, " Hypo 3.9",
            fontsize=7, color="#3B7DD8", alpha=0.5, va="bottom",
        )

    # ── Event shading, boundaries, and markers ──
    legend_added = set()
    n_spikes_day = 0
    n_dips_day = 0

    if not ev_day.empty:
        for _, e in ev_day.iterrows():
            dirn = e.get("direction", "up")
            tier = e.get("tier", "high_confidence")
            key = (dirn, tier)

            if key not in active_filters:
                continue

            if dirn == "up":
                n_spikes_day += 1
            else:
                n_dips_day += 1

            sty = _EVENT_STYLES.get(key, _EVENT_STYLES[("up", "sensitive")])

            # ── Shaded region ──
            ax1.axvspan(
                e["start_time"], e["end_time"],
                color=sty["shade"], alpha=sty["shade_alpha"], zorder=1,
            )

            # ── Start/end boundary lines ──
            ax1.axvline(
                e["start_time"],
                color=sty["border_color"], ls=(0, (4, 2)), lw=0.9,
                alpha=0.45, zorder=4,
            )
            ax1.axvline(
                e["end_time"],
                color=sty["border_color"], ls=(0, (1, 2)), lw=0.9,
                alpha=0.45, zorder=4,
            )

            # ── Start/end tick marks ──
            for btime in [e["start_time"], e["end_time"]]:
                nearest_idx = (day_df[dt_col] - btime).abs().idxmin()
                nearest = day_df.loc[nearest_idx]
                ax1.scatter(
                    nearest[dt_col], nearest[glucose_col],
                    color=sty["border_color"], marker="|",
                    s=80, zorder=7, linewidths=1.5, alpha=0.7,
                )

            # ── Peak/trough marker ──
            if "peak_time" in e and pd.notna(e["peak_time"]):
                pt = pd.to_datetime(e["peak_time"])
                nearest_idx = (day_df[dt_col] - pt).abs().idxmin()
                nearest = day_df.loc[nearest_idx]

                lbl = sty["label"] if key not in legend_added else None
                legend_added.add(key)

                ax1.scatter(
                    nearest[dt_col], nearest[glucose_col],
                    color=sty["marker_color"], marker=sty["marker"],
                    s=sty["marker_size"], zorder=6,
                    edgecolors="white", linewidths=0.8, label=lbl,
                )

                # Annotation
                offset_y = -20 if dirn == "down" else 16
                glucose_val = nearest[glucose_col]

                if tier == "high_confidence":
                    label_text = f"{glucose_val:.1f} mmol/L"
                    ax1.annotate(
                        label_text,
                        xy=(nearest[dt_col], glucose_val),
                        xytext=(0, offset_y),
                        textcoords="offset points",
                        fontsize=8, fontweight="bold", ha="center",
                        color=sty["marker_color"],
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            fc="white", ec=sty["marker_color"],
                            alpha=0.88, lw=0.6,
                        ),
                    )
                else:
                    ax1.annotate(
                        f"{glucose_val:.1f}",
                        xy=(nearest[dt_col], glucose_val),
                        xytext=(0, offset_y * 0.7),
                        textcoords="offset points",
                        fontsize=7, ha="center",
                        color=sty["marker_color"], alpha=0.75,
                    )

    # ── Title ──
    parts = []
    if n_spikes_day:
        parts.append(f"🔴 {n_spikes_day} spike{'s' if n_spikes_day != 1 else ''}")
    if n_dips_day:
        parts.append(f"🔵 {n_dips_day} dip{'s' if n_dips_day != 1 else ''}")
    count_str = f"   —   {', '.join(parts)}" if parts else "   —   ✅ No events"

    ax1.set_title(
        f"{person_id}  ·  {day}{count_str}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax1.set_ylabel("Glucose (mmol/L)", fontsize=10)
    ax1.legend(
        loc="upper right", fontsize=8, framealpha=0.9,
        ncol=3, handlelength=1.5, columnspacing=1.0,
    )
    ax1.grid(True, alpha=0.12)
    sns.despine(ax=ax1, left=False, bottom=False)

    # ── Event component panel ──
    if ax2 is not None and has_ec:
        ec = day_df["event_component"].to_numpy()
        times = day_df[dt_col]

        ax2.fill_between(
            times, ec, 0, where=ec > 0,
            color="#D32F2F", alpha=0.25, label="Spike energy",
        )
        ax2.fill_between(
            times, ec, 0, where=ec < 0,
            color="#0D47A1", alpha=0.25, label="Dip energy",
        )
        ax2.plot(times, ec, color="#374151", lw=0.7)
        ax2.axhline(0, color="gray", lw=0.4)
        ax2.set_ylabel("Event\nComponent", fontsize=8)
        ax2.set_xlabel("Time of Day", fontsize=10)
        ax2.legend(loc="upper right", fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.12)
        sns.despine(ax=ax2, left=False, bottom=False)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")
    else:
        ax1.set_xlabel("Time of Day", fontsize=10)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha="center")

    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────
# FIXED COLUMN NAMES
# ──────────────────────────────────────────────

id_col = "Client ID"
day_col = "Day"


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:

    # ── Branding ──
    st.markdown(
        '<div class="sidebar-brand"><span>SPIKES</span> Demo</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Tab navigation ──
    active_tab = st.radio(
        "Select view",
        ["🔬 CGM Spikes", "🧑‍⚕️ Patient Overview", "🌿 Lifestyle Pillars"],
        index=0,
        label_visibility="collapsed",
    )

    is_overview = active_tab == "🧑‍⚕️ Patient Overview"
    is_pillars  = active_tab == "🌿 Lifestyle Pillars"
    is_spikes   = active_tab == "🔬 CGM Spikes"

    if is_overview:
        st.markdown(
            '<span class="wip-badge">WORK IN PROGRESS</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ──────────────────────────────────────────
    # DATA UPLOAD — collapsible toggle at bottom
    # ──────────────────────────────────────────

    st.markdown("")
    st.markdown("")

    with st.expander("📂 Data Upload", expanded=False):

        # ── CGM Data ──
        st.markdown("**CGM Data**")
        cgm_file = st.file_uploader(
            "Upload CGM CSV",
            type=["csv"],
            key="cgm_uploader",
            help="CGM Data `Client ID | Day | 00:00 | 00:05 |... | 23:55` ",
        )
        if cgm_file is not None:
            wide_df = pd.read_csv(cgm_file)
            st.success(f"CGM: {len(wide_df)} rows loaded")
        else:
            wide_df = None

        st.markdown("---")

        # ── Patient Information ──
        st.markdown("**Patient Information**")
        meta_file = st.file_uploader(
            "Upload patient metadata CSV",
            type=["csv"],
            key="meta_uploader",
            help=(
                "Expected columns: Client_ID, Age, Sex Assigned, "
                "Ethnicity, Height, Weight, BMI, Waist, "
                "Blood Pressure, HbA1c, Medication, Health Goal, etc."
            ),
        )
        if meta_file is not None:
            meta_df = pd.read_csv(meta_file)
            st.success(f"Patient Info: {len(meta_df)} patients loaded")
        else:
            meta_df = None

        st.markdown("---")

        # ── Self Reported Pillars ──
        st.markdown("**Self Reported Pillars**")
        pillars_file = st.file_uploader(
            "Upload self-reported pillars CSV",
            type=["csv"],
            key="pillars_uploader",
            help="Self-reported lifestyle and wellness pillar data.",
        )
        if pillars_file is not None:
            pillars_df = pd.read_csv(pillars_file)
            st.success(f"Pillars: {len(pillars_df)} rows loaded")
        else:
            pillars_df = None


# ──────────────────────────────────────────────
# POST-SIDEBAR: resolve CGM data
# ──────────────────────────────────────────────

cgm_available = False
patients = []

if wide_df is not None:
    missing_cols = [c for c in [id_col, day_col] if c not in wide_df.columns]
    if not missing_cols:
        patients = sorted(wide_df[id_col].dropna().unique().astype(str))
        cgm_available = True


# ──────────────────────────────────────────────
# PATIENT OVERVIEW TAB
# ──────────────────────────────────────────────

if is_overview:
    st.markdown(
        """
        <div class="overview-header">
            <h1>🧑‍⚕️ Patient Overview</h1>
            <p>Participant metadata and health profile</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if meta_df is None:
        st.info(
            "📋 Open **Data Upload** in the sidebar and upload a "
            "**Patient Information** CSV to view profiles."
        )
        st.stop()

    # ── Resolve Client_ID column ──
    meta_id_col = "Client_ID"
    if meta_id_col not in meta_df.columns:
        candidates = [c for c in meta_df.columns if "client" in c.lower() or "id" in c.lower()]
        if candidates:
            meta_id_col = candidates[0]
        else:
            st.error(
                f"Could not find a **Client_ID** column. "
                f"Available: {', '.join(meta_df.columns[:10])}…"
            )
            st.stop()

    meta_patients = sorted(meta_df[meta_id_col].dropna().unique().astype(str))

    with st.sidebar:
        st.markdown("### 🩺 Patient")
        overview_patient = st.selectbox(
            "Select Patient",
            meta_patients,
            key="overview_patient_select",
            label_visibility="collapsed",
        )

    patient_row = meta_df.loc[meta_df[meta_id_col].astype(str) == overview_patient]

    if patient_row.empty:
        st.warning(f"No metadata found for **{overview_patient}**.")
        st.stop()

    p = patient_row.iloc[0]

    def field(col_name, fmt=None, suffix=""):
        val = p.get(col_name, np.nan)
        if pd.isna(val) or str(val).strip() == "":
            return "—"
        if fmt:
            try:
                return f"{fmt.format(float(val))}{suffix}"
            except (ValueError, TypeError):
                return f"{val}{suffix}"
        return f"{val}{suffix}"

    health_goal = field("Health Goal")
    if health_goal != "—":
        st.markdown(
            f"""
            <div class="health-goal-label">Health Goal</div>
            <div class="health-goal-quote">"{health_goal}"</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="health-goal-label">Health Goal</div>
            <div class="health-goal-quote" style="color:#9ca3af;">
                No health goal recorded for this patient.
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="patient-profile">
                <h3>Demographics</h3>
                <div class="patient-field">
                    <span class="field-label">Age</span>
                    <span class="field-value">{field("Age")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Sex Assigned</span>
                    <span class="field-value">{field("Sex Assigned")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Ethnicity</span>
                    <span class="field-value">{field("Ethnicity")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="patient-profile">
                <h3>Body Measurements</h3>
                <div class="patient-field">
                    <span class="field-label">Height</span>
                    <span class="field-value">{field("Height", "{:.2f}", " m")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Weight</span>
                    <span class="field-value">{field("Weight", "{:.1f}", " kg")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">BMI</span>
                    <span class="field-value">{field("BMI", "{:.1f}")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Waist</span>
                    <span class="field-value">{field("Waist", "{:.0f}", " cm")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Waist:Height Ratio</span>
                    <span class="field-value">{field("Waist:Height Ratio", "{:.3f}")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="patient-profile">
                <h3>Clinical</h3>
                <div class="patient-field">
                    <span class="field-label">Blood Pressure</span>
                    <span class="field-value">{field("Blood Pressure")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">HbA1c</span>
                    <span class="field-value">{field("HbA1c")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">HbA1c Date</span>
                    <span class="field-value">{field("HbA1c Date")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Medication</span>
                    <span class="field-value">{field("Medication")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            f"""
            <div class="patient-profile">
                <h3>Lifestyle</h3>
                <div class="patient-field">
                    <span class="field-label">Wearable Type</span>
                    <span class="field-value">{field("Wearable Type")}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Chronotype</span>
                    <span class="field-value">{field("Night Owl / Morning Lark")}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        bmi_val = p.get("BMI", np.nan)
        if pd.notna(bmi_val):
            try:
                bmi_num = float(bmi_val)
                if bmi_num < 18.5:
                    bmi_cat, bmi_color = "Underweight", "#3b82f6"
                elif bmi_num < 25:
                    bmi_cat, bmi_color = "Normal", "#22c55e"
                elif bmi_num < 30:
                    bmi_cat, bmi_color = "Overweight", "#f59e0b"
                else:
                    bmi_cat, bmi_color = "Obese", "#ef4444"
            except (ValueError, TypeError):
                bmi_cat, bmi_color = "—", "#6b7280"
        else:
            bmi_cat, bmi_color = "—", "#6b7280"

        st.markdown(
            f"""
            <div class="patient-profile">
                <h3>Quick Summary</h3>
                <div class="patient-field">
                    <span class="field-label">BMI Category</span>
                    <span class="field-value" style="color:{bmi_color};">{bmi_cat}</span>
                </div>
                <div class="patient-field">
                    <span class="field-label">Data Source</span>
                    <span class="field-value">CGM + Metadata</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ──────────────────────────────────────────────
# LIFESTYLE PILLARS TAB
# ──────────────────────────────────────────────

if is_pillars:
    st.markdown(
        """
        <div class="pillars-header">
            <h1>🌿 Lifestyle Pillars</h1>
            <p>Self-reported daily gauges · Health · Sleep · Nutrition · Movement · Stress · Connection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pillars_df is None:
        st.info(
            "📂 Open **Data Upload** in the sidebar and upload a "
            "**Self Reported Pillars** CSV to view lifestyle data."
        )
        st.stop()

    clean_df = clean_pillars(pillars_df)

    pillars_id_col = "Client ID"
    if pillars_id_col not in clean_df.columns:
        candidates = [c for c in clean_df.columns if "client" in c.lower() or "id" in c.lower()]
        if candidates:
            pillars_id_col = candidates[0]
        else:
            st.error(
                f"Could not find a **Client ID** column. "
                f"Available: {', '.join(clean_df.columns[:10])}…"
            )
            st.stop()

    pillar_patients = sorted(clean_df[pillars_id_col].dropna().unique().astype(str))

    with st.sidebar:
        st.markdown("### 🩺 Patient")
        pillars_patient = st.selectbox(
            "Select Patient",
            pillar_patients,
            key="pillars_patient_select",
            label_visibility="collapsed",
        )

    summary = patient_summary(clean_df, pillars_patient, id_col=pillars_id_col)

    if summary["n_days"] == 0:
        st.warning(f"No pillar data found for **{pillars_patient}**.")
        st.stop()

    st.markdown(f"**{pillars_patient}** · {summary['n_days']} days of self-reported data")

    cols = st.columns(6)
    for i, pillar in enumerate(PILLAR_NAMES):
        pdata = summary["pillars"].get(pillar, {})
        mean_val = pdata.get("mean")
        dominant = pdata.get("dominant_trend")
        delta = pdata.get("delta")

        if dominant == "improving":
            trend_class = "trend-improving"
            trend_icon = "↑"
        elif dominant == "declining":
            trend_class = "trend-declining"
            trend_icon = "↓"
        else:
            trend_class = "trend-staying"
            trend_icon = "→"

        score_str = f"{mean_val:.1f}" if mean_val is not None else "—"
        trend_str = f"{trend_icon} {dominant.title()}" if dominant else "—"
        delta_str = f"Δ {delta:+.1f}" if delta is not None else ""

        color = PILLAR_COLORS.get(pillar, "#4a90d9")

        with cols[i]:
            st.markdown(
                f"""
                <div class="pillar-card" style="border-top-color: {color};">
                    <div class="pillar-name">{pillar}</div>
                    <div class="pillar-score">{score_str}</div>
                    <div class="pillar-trend {trend_class}">{trend_str}</div>
                    <div class="pillar-delta">{delta_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    neg = summary.get("negative", {})
    if neg:
        neg_mean = neg.get("mean")
        if neg_mean is not None:
            neg_color = "#ef4444" if neg_mean >= 6 else "#f59e0b" if neg_mean >= 4 else "#22c55e"
            st.markdown(
                f"""
                <div class="pillar-card" style="border-top-color: #ef4444; max-width: 200px;">
                    <div class="pillar-name">Negative Affect</div>
                    <div class="pillar-score" style="color: {neg_color};">{neg_mean:.1f}</div>
                    <div class="pillar-delta">range {neg.get('min', '—')}–{neg.get('max', '—')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    chart_col1, chart_col2 = st.columns([1, 1.4])

    with chart_col1:
        st.markdown("### Pillar Balance")
        fig_radar = plot_pillar_radar(clean_df, pillars_patient, id_col=pillars_id_col)
        st.pyplot(fig_radar, use_container_width=True)
        plt.close(fig_radar)

    with chart_col2:
        st.markdown("### Trend Distribution")
        fig_trends = plot_trend_distribution(clean_df, pillars_patient, id_col=pillars_id_col)
        st.pyplot(fig_trends, use_container_width=True)
        plt.close(fig_trends)

    st.markdown("---")

    st.markdown("### 📈 Gauges Over Time")
    fig_ts = plot_gauge_timeseries(clean_df, pillars_patient, id_col=pillars_id_col)
    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    fig_neg = plot_negative_gauge(clean_df, pillars_patient, id_col=pillars_id_col)
    if fig_neg is not None:
        st.markdown("### 😟 Negative Affect Over Time")
        st.pyplot(fig_neg, use_container_width=True)
        plt.close(fig_neg)

    with st.expander("📋 Raw Pillar Data", expanded=False):
        patient_raw = clean_df.loc[clean_df[pillars_id_col].astype(str) == pillars_patient]
        st.dataframe(patient_raw, use_container_width=True, hide_index=True)

    st.stop()


## ──────────────────────────────────────────────
# CGM SPIKES TAB
# ──────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <h1>📈 CGM Event Analysis</h1>
        <p>Continuous Glucose Monitoring · Spike & Dip Detection with Dual Sensitivity Pipelines</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not cgm_available:
    st.info("📂 Open **Data Upload** in the sidebar and load **CGM Data** to get started.")
    st.stop()

# ── Patient selector in sidebar ──
with st.sidebar:
    st.markdown("### 🩺 Patient")
    selected_patient = st.selectbox(
        "Select Patient",
        patients,
        label_visibility="collapsed",
    )
    st.markdown("---")


# ══════════════════════════════════════════════════════════════
# DETECTION CONTROLS — CENTRED IN MAIN AREA
# ══════════════════════════════════════════════════════════════

_, ctrl_centre, _ = st.columns([0.5, 4, 0.5])

with ctrl_centre:

    # ── Row 1: Toggles (left) + Advanced Parameters (right, collapsible) ──
    toggle_col, spacer_col, param_col = st.columns([1.8, 0.2, 2])

    with toggle_col:
        st.markdown(
            '<div class="toggle-section-label" style="text-align:center;">'
            'Detection Layers</div>',
            unsafe_allow_html=True,
        )

        t1, t2 = st.columns(2)
        with t1:
            show_spike_hc = st.toggle(
                "🔴 Hard Spikes",
                value=True,
                help="High-confidence glucose spikes (>2 SD from baseline)",
            )
            show_dip_hc = st.toggle(
                "🔵 Hard Dips",
                value=True,
                help="High-confidence glucose dips (>2 SD below baseline)",
            )
        with t2:
            show_spike_sens = st.toggle(
                "🟠 Soft Spikes",
                value=False,
                help="Sensitive spike detection (1–2 SD from baseline)",
            )
            show_dip_sens = st.toggle(
                "🔹 Soft Dips",
                value=False,
                help="Sensitive dip detection (1–2 SD below baseline)",
            )

    with param_col:
        with st.expander("⚙️ Advanced Parameters", expanded=False):
            fft_cutoff = st.slider(
                "FFT Denoise Cutoff (hours)",
                0.25, 2.0, 0.5, 0.25,
                help="Low-pass filter cutoff. Oscillations faster than this are removed as jitter.",
            )
            baseline_window = st.slider(
                "Baseline Window (hours)",
                2.0, 8.0, 5.0, 0.5,
                help="Rolling median window for the stable baseline. Larger = smoother.",
            )
            sample_step = st.selectbox(
                "Sample Interval (min)",
                [5, 10, 15],
                index=0,
                help="Expected CGM sampling interval.",
            )

    st.markdown("")

    # ── Row 2: Show Baseline + Analyse (centred) ──
    _, bottom_centre, _ = st.columns([1, 2, 1])

    with bottom_centre:
        bl_col, analyse_col = st.columns([1, 1])
        with bl_col:
            show_baseline = st.toggle("📏 Show Baseline", value=True)
        with analyse_col:
            analyse = st.button(
                "🔬 Analyse", type="primary", use_container_width=True,
            )

    st.markdown("---")


# ── Handle presets ──
if st.session_state.pop("_preset_hc", False):
    show_spike_hc = True
    show_spike_sens = False
    show_dip_hc = True
    show_dip_sens = False

if st.session_state.pop("_preset_all", False):
    show_spike_hc = True
    show_spike_sens = True
    show_dip_hc = True
    show_dip_sens = True

# Build active filter set
active_filters = set()
if show_spike_hc:
    active_filters.add(("up", "high_confidence"))
if show_spike_sens:
    active_filters.add(("up", "sensitive"))
if show_dip_hc:
    active_filters.add(("down", "high_confidence"))
if show_dip_sens:
    active_filters.add(("down", "sensitive"))

detect_mode = "both"
detect_direction = "both"
show_event_component = False

if not analyse:
    st.markdown(
        """
        Select a patient from the sidebar, configure detection layers,
        and press **Analyse** to begin.
        """
    )
    st.stop()


# ══════════════════════════════════════════════════════════════
# RUN DETECTION
# ══════════════════════════════════════════════════════════════

with st.spinner(f"Running spike detection for {selected_patient} …"):
    long_df = reshape_wide_to_long(wide_df, id_col=id_col, day_col=day_col)

    events_df, annotated_df = detect_events(
        long_df,
        person_id=selected_patient,
        mode=detect_mode,
        direction=detect_direction,
        fft_cutoff_hours=fft_cutoff,
        baseline_window_hours=baseline_window,
        sample_step_min=sample_step,
    )

# ── Filter events for display based on active toggles ──
if not events_df.empty:
    display_mask = events_df.apply(
        lambda r: (r["direction"], r["tier"]) in active_filters, axis=1
    )
    filtered_events = events_df[display_mask].copy()
else:
    filtered_events = events_df.copy()


# ══════════════════════════════════════════════════════════════
# TIME-OF-DAY HELPER
# ══════════════════════════════════════════════════════════════

def classify_time_of_day(hour: int) -> str:
    """Classify an hour (0-23) into a period of day."""
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"


PERIOD_ORDER = ["Morning", "Afternoon", "Evening", "Night"]
PERIOD_COLOURS = {
    "Morning": "#F59E0B",
    "Afternoon": "#EF4444",
    "Evening": "#6366F1",
    "Night": "#1E3A5F",
}


# ══════════════════════════════════════════════════════════════
# SUMMARY METRICS
# ══════════════════════════════════════════════════════════════

annotated_df["datetime"] = pd.to_datetime(annotated_df["datetime"])
annotated_df["_cal_date"] = annotated_df["datetime"].dt.date

# Ensure filtered_events has proper datetime types
if not filtered_events.empty:
    filtered_events["start_time"] = pd.to_datetime(filtered_events["start_time"])
    filtered_events["end_time"] = pd.to_datetime(filtered_events["end_time"])
    if "peak_time" in filtered_events.columns:
        filtered_events["peak_time"] = pd.to_datetime(filtered_events["peak_time"])
    if "Day" in filtered_events.columns:
        filtered_events["_ev_date"] = pd.to_datetime(
            filtered_events["Day"], errors="coerce"
        ).dt.date

n_spikes = len(filtered_events[filtered_events["direction"] == "up"]) if not filtered_events.empty else 0
n_dips = len(filtered_events[filtered_events["direction"] == "down"]) if not filtered_events.empty else 0
n_total = n_spikes + n_dips
n_days = annotated_df["_cal_date"].nunique()
mean_conf = filtered_events["p_val"].mean() if n_total > 0 else 0.0
max_glucose = (
    filtered_events.loc[filtered_events["direction"] == "up", "extreme_glucose"].max()
    if n_spikes > 0 else 0.0
)
min_glucose = (
    filtered_events.loc[filtered_events["direction"] == "down", "extreme_glucose"].min()
    if n_dips > 0 else 0.0
)

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card spike">
            <div class="label">Spikes Detected</div>
            <div class="value">{n_spikes}</div>
        </div>
        <div class="metric-card dip">
            <div class="label">Dips Detected</div>
            <div class="value">{n_dips}</div>
        </div>
        <div class="metric-card days">
            <div class="label">Days Analysed</div>
            <div class="value">{n_days}</div>
        </div>
        <div class="metric-card conf">
            <div class="label">Mean Confidence</div>
            <div class="value">{mean_conf:.2f}</div>
        </div>
        <div class="metric-card glucose">
            <div class="label">Highest Spike (mmol/L)</div>
            <div class="value">{max_glucose:.1f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if n_dips > 0:
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card dip" style="max-width: 250px;">
                <div class="label">Lowest Dip (mmol/L)</div>
                <div class="value">{min_glucose:.1f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Active layers indicator ──
active_labels = []
if show_spike_hc:
    active_labels.append("🔴 Hard Spikes")
if show_spike_sens:
    active_labels.append("🟠 Soft Spikes")
if show_dip_hc:
    active_labels.append("🔵 Hard Dips")
if show_dip_sens:
    active_labels.append("🔹 Soft Dips")

if active_labels:
    st.caption(f"Active layers: {' · '.join(active_labels)}")
else:
    st.warning("No detection layers are active. Toggle at least one layer above.")
    st.stop()


# ══════════════════════════════════════════════════════════════
# TIME-OF-DAY PIE CHARTS
# ══════════════════════════════════════════════════════════════

st.markdown("## 🕐 Events by Time of Day")
st.caption(f"Average events per day across {n_days} day(s)")

if not filtered_events.empty and "peak_time" in filtered_events.columns:
    filtered_events["_peak_hour"] = pd.to_datetime(
        filtered_events["peak_time"]
    ).dt.hour
    filtered_events["_period"] = filtered_events["_peak_hour"].apply(classify_time_of_day)

    # ── Build counts per period ──
    spike_events = filtered_events[filtered_events["direction"] == "up"]
    dip_events = filtered_events[filtered_events["direction"] == "down"]

    spike_counts = spike_events.groupby("_period").size().reindex(PERIOD_ORDER, fill_value=0)
    dip_counts = dip_events.groupby("_period").size().reindex(PERIOD_ORDER, fill_value=0)
    total_counts = filtered_events.groupby("_period").size().reindex(PERIOD_ORDER, fill_value=0)

    # Average per day
    spike_avg = spike_counts / max(n_days, 1)
    dip_avg = dip_counts / max(n_days, 1)
    total_avg = total_counts / max(n_days, 1)

    pie_col1, pie_col2, pie_col3 = st.columns(3)

    def _make_pie(data, title, ax):
        """Create a clean pie chart for time-of-day distribution."""
        colours = [PERIOD_COLOURS[p] for p in PERIOD_ORDER]
        values = data.values

        if values.sum() == 0:
            ax.text(
                0.5, 0.5, "No events",
                ha="center", va="center", fontsize=11, color="#9CA3AF",
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
            ax.axis("off")
            return

        labels = [
            f"{p}\n({v:.1f}/day)" if v > 0 else ""
            for p, v in zip(PERIOD_ORDER, values)
        ]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colours,
            autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=1.5, edgecolor="white"),
            textprops=dict(fontsize=8),
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")
            at.set_color("white")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    with pie_col1:
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        _make_pie(spike_avg, "Spikes by Time of Day", ax_pie)
        fig_pie.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    with pie_col2:
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        _make_pie(dip_avg, "Dips by Time of Day", ax_pie)
        fig_pie.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    with pie_col3:
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        _make_pie(total_avg, "All Events by Time of Day", ax_pie)
        fig_pie.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    # ── Summary table under pie charts ──
    summary_data = pd.DataFrame({
        "Period": PERIOD_ORDER,
        "Time Range": ["06:00 – 11:59", "12:00 – 17:59", "18:00 – 23:59", "00:00 – 05:59"],
        "Total Spikes": spike_counts.values.astype(int),
        "Total Dips": dip_counts.values.astype(int),
        "Avg Spikes/Day": [f"{v:.2f}" for v in spike_avg.values],
        "Avg Dips/Day": [f"{v:.2f}" for v in dip_avg.values],
    })

    with st.expander("📊 Time-of-Day Breakdown Table"):
        st.dataframe(summary_data, use_container_width=True, hide_index=True)

else:
    st.info("No events detected — pie charts will appear once events are found.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# DAILY PLOTS
# ══════════════════════════════════════════════════════════════

st.markdown("## 📊 Daily Glucose Traces")

study_day_col = "study_day"
has_study_day = study_day_col in annotated_df.columns

if has_study_day:
    group_col = study_day_col
    annotated_df[group_col] = pd.to_numeric(annotated_df[study_day_col], errors="coerce")
else:
    group_col = "date"
    annotated_df[group_col] = annotated_df["_cal_date"]

# Build study_day → calendar date mapping
if has_study_day:
    day_to_dates = (
        annotated_df.groupby(study_day_col)["_cal_date"].apply(lambda x: set(x.dropna())).to_dict()
    )
else:
    day_to_dates = None

days = sorted(annotated_df[group_col].dropna().unique())

for day_val in days:
    day_df = annotated_df.loc[annotated_df[group_col] == day_val].sort_values("datetime")

    # Match events to this day
    ev_day = pd.DataFrame()

    if not filtered_events.empty and "_ev_date" in filtered_events.columns:
        if has_study_day and day_to_dates is not None:
            cal_dates = day_to_dates.get(day_val, set())
            if cal_dates:
                ev_day = filtered_events.loc[
                    filtered_events["_ev_date"].isin(cal_dates)
                ]
        else:
            ev_day = filtered_events.loc[
                filtered_events["_ev_date"] == day_val
            ]

    # Fallback: time-range matching
    if ev_day.empty and not filtered_events.empty and not day_df.empty:
        day_start = day_df["datetime"].min()
        day_end = day_df["datetime"].max()
        ev_day = filtered_events.loc[
            (filtered_events["start_time"] >= day_start)
            & (filtered_events["start_time"] <= day_end)
        ]

    n_day_spikes = len(ev_day[ev_day["direction"] == "up"]) if not ev_day.empty else 0
    n_day_dips = len(ev_day[ev_day["direction"] == "down"]) if not ev_day.empty else 0

    badge_parts = []
    if n_day_spikes > 0:
        badge_parts.append(f"🔴 {n_day_spikes} spike{'s' if n_day_spikes != 1 else ''}")
    if n_day_dips > 0:
        badge_parts.append(f"🔵 {n_day_dips} dip{'s' if n_day_dips != 1 else ''}")
    badge = ", ".join(badge_parts) if badge_parts else "✅ No events"

    day_label = f"Day {int(day_val)}" if has_study_day else str(day_val)

    st.markdown(
        f'<div class="day-header">📅 {day_label}    {badge}</div>',
        unsafe_allow_html=True,
    )

    fig = build_day_figure(
        day_df, ev_day, selected_patient, day_label,
        active_filters=active_filters,
        show_baseline=show_baseline,
        show_event_component=show_event_component,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if not ev_day.empty:
        with st.expander(f"Details for {day_label}"):
            detail = ev_day.copy()
            detail["start_time"] = detail["start_time"].dt.strftime("%H:%M")
            detail["end_time"] = detail["end_time"].dt.strftime("%H:%M")
            if "peak_time" in detail.columns:
                detail["peak_time"] = detail["peak_time"].dt.strftime("%H:%M")

            display_cols = [c for c in [
                "direction", "tier", "start_time", "end_time", "peak_time",
                "p_val", "range", "extreme_glucose", "baseline_at_extreme",
                "closure_reason",
            ] if c in detail.columns]

            rename_map = {
                "direction": "Type",
                "tier": "Pipeline",
                "start_time": "Start",
                "end_time": "End",
                "peak_time": "Peak",
                "p_val": "Confidence",
                "range": "Range (mmol/L)",
                "extreme_glucose": "Extreme (mmol/L)",
                "baseline_at_extreme": "Baseline (mmol/L)",
                "closure_reason": "Closure",
            }

            st.dataframe(
                detail[display_cols].rename(columns=rename_map),
                use_container_width=True,
                hide_index=True,
            )


# ── Full event table ──
with st.expander(f"📋 Event Table  ({n_total} events)", expanded=n_total <= 20):
    if filtered_events.empty:
        st.info("No events detected with current settings and active layers.")
    else:
        display_df = filtered_events.copy()
        display_df["start_time"] = display_df["start_time"].dt.strftime("%H:%M")
        display_df["end_time"] = display_df["end_time"].dt.strftime("%H:%M")
        if "peak_time" in display_df.columns:
            display_df["peak_time"] = display_df["peak_time"].dt.strftime("%H:%M")

        display_cols = [c for c in [
            "Day", "direction", "tier", "start_time", "end_time", "peak_time",
            "p_val", "range", "extreme_glucose", "baseline_at_extreme",
            "closure_reason",
        ] if c in display_df.columns]

        rename_map = {
            "direction": "Type",
            "tier": "Pipeline",
            "start_time": "Start",
            "end_time": "End",
            "peak_time": "Peak",
            "p_val": "Confidence",
            "range": "Range (mmol/L)",
            "extreme_glucose": "Extreme (mmol/L)",
            "baseline_at_extreme": "Baseline (mmol/L)",
            "closure_reason": "Closure",
        }

        st.dataframe(
            display_df[display_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )