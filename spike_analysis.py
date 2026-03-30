"""
Spike Detection and Visualization Module

This module provides functions for detecting glucose spikes in continuous
glucose monitoring (CGM) data and visualizing the results.

Pipeline:
    1. reshape_wide_to_long()  – Convert wide-format CGM → long-format
    2. detect_spikes()         – Detect clinically interpretable spike events
    3. visualize_spikes()      – Seaborn visualisation with annotated peaks

Expected wide-format input:
    Client ID | Day | 00:00 | 00:05 | 00:10 |... | 23:55

Output of detect_spikes():
    events_df:    Day | start_time | end_time | peak_time | p_val | range | peak_glucose | baseline_at_peak
    annotated_df: original long-format trace with spike annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. RESHAPE: wide → long
# ---------------------------------------------------------------------------

def reshape_wide_to_long(
    df: pd.DataFrame,
    id_col: str = "Client ID",
    day_col: str = "Day",
    glucose_col_out: str = "glucose_mmol_l",
    dt_col_out: str = "datetime",
    id_col_out: str = "person_id",
    date_col_out: str = "date",
    day_col_out: str = "study_day",          # ← NEW: preserve integer day
    reference_date: str = "2025-01-01",      # ← NEW: anchor for building datetimes
) -> pd.DataFrame:
    """
    Melt wide-format CGM data into long format.

    Wide:  Client ID | Day | 00:00 | 00:05 | … | 23:55
           (Day is an integer 1–14, not a date)

    Long:  person_id | study_day | date | datetime | glucose_mmol_l
    """
    time_cols = [c for c in df.columns if c not in (id_col, day_col)]

    # for tc in time_cols:
    #     if not _is_time_like(tc):
    #         raise ValueError(
    #             f"Column '{tc}' does not look like HH:MM. "
    #             f"Check id_col='{id_col}' and day_col='{day_col}'."
    #         )

    long = df.melt(
        id_vars=[id_col, day_col],
        value_vars=time_cols,
        var_name="_time_str",
        value_name=glucose_col_out,
    )

    # ── Handle integer Day column ──
    long[day_col] = pd.to_numeric(long[day_col], errors="coerce").astype("Int64")
    long[day_col_out] = long[day_col]  # preserve original integer

    # Build synthetic calendar dates: Day 1 → reference_date, Day 2 → +1, etc.
    ref = pd.Timestamp(reference_date)
    long["_synth_date"] = long[day_col].apply(
        lambda d: ref + pd.Timedelta(days=int(d) - 1) if pd.notna(d) else pd.NaT
    )

    # Build full datetime from synthetic date + time string
    long[dt_col_out] = pd.to_datetime(
        long["_synth_date"].dt.strftime("%Y-%m-%d") + " " + long["_time_str"],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    long[date_col_out] = long[dt_col_out].dt.date

    # Tidy up
    long = long.rename(columns={id_col: id_col_out})
    long = long.drop(columns=[day_col, "_time_str", "_synth_date"])
    long[glucose_col_out] = pd.to_numeric(long[glucose_col_out], errors="coerce")
    long = long.dropna(subset=[glucose_col_out])
    long = long.sort_values([id_col_out, dt_col_out]).reset_index(drop=True)

    return long


# ---------------------------------------------------------------------------
# 2. SPIKE DETECTION
# ---------------------------------------------------------------------------

'''def detect_spikes(
    df: pd.DataFrame,
    person_id: str = "",
    glucose_col: str = "glucose_mmol_l",
    dt_col: str = "datetime",
    id_col: str = "person_id",
    date_col: str = "date",
    # ── Baseline parameters ──
    baseline_lookback_min: int = 60,
    baseline_exclude_last_min: int = 10,
    # ── Spike entry / exit thresholds ──
    amp_thresh: float = 1.0,
    end_band: float = 0.3,
    rise_window_min: int = 15,
    rise_rate_thresh: float = 0.02,
    min_above_min: int = 15,
    refractory_min: int = 30,
    # ── Sampling ──
    sample_step_min: int = 5,
    # ── Confidence scoring ──
    expected_noise_mad_floor: float = 0.05,
    w_amp: float = 0.55,
    w_slope: float = 0.25,
    w_dur: float = 0.10,
    w_return: float = 0.10,
):
    """
    Detect clinically interpretable glucose spike *events* for one client.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format CGM data (as produced by ``reshape_wide_to_long``).
    person_id : str
        The person_id to analyse.
    glucose_col, dt_col, id_col, date_col : str
        Column names.
    baseline_lookback_min : int
        Minutes to look back for rolling-median baseline.
    baseline_exclude_last_min : int
        Minutes to exclude from the end of the lookback window so the
        current rise doesn't contaminate the baseline.
    amp_thresh : float
        Minimum glucose rise above baseline (mmol/L) to trigger a spike.
    end_band : float
        Glucose must return to within this band above baseline for the
        spike to be considered resolved.
    rise_window_min : int
        Window for calculating the instantaneous rise rate.
    rise_rate_thresh : float
        Minimum rise rate (mmol/L per min) required alongside amplitude.
    min_above_min : int
        Minimum contiguous duration above ``amp_thresh`` to qualify.
    refractory_min : int
        Cool-down period after a spike ends before a new one can start.
    sample_step_min : int
        Expected sampling interval (minutes).
    expected_noise_mad_floor : float
        Floor for MAD-based noise estimate used in confidence scoring.
    w_amp, w_slope, w_dur, w_return : float
        Weights for the four confidence-score components (should sum to 1).

    Returns
    -------
    events_df : pd.DataFrame
        One row per detected spike with columns:
        ``Day | start_time | end_time | peak_time | p_val | range |
        peak_glucose | baseline_at_peak``
    annotated_df : pd.DataFrame
        The client's long-format trace with added annotation columns:
        ``baseline | slope_15m | residual | event_id | in_spike_event |
        is_event_start | is_event_end | is_event_peak | p_val_event``

    Notes
    -----
    *  ``p_val`` is a composite *confidence score* in [0, 1], **not** a
       classical statistical p-value.
    *  The function assumes approximately regular 5-min sampling.
       Pre-impute short gaps for best results.
    """
    # ── Column validation ──
    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' not found. Available: {df.columns.tolist()}")

    # ── Filter for this person ──
    d = df.loc[df[id_col].astype(str) == str(person_id)].copy()

    _event_cols = [
        "Day", "start_time", "end_time", "peak_time",
        "p_val", "range", "peak_glucose", "baseline_at_peak",
    ]
    _annot_extra = [
        "baseline", "slope_15m", "residual", "event_id",
        "in_spike_event", "is_event_start", "is_event_end",
        "is_event_peak", "p_val_event",
    ]

    if d.empty:
        return (
            pd.DataFrame(columns=_event_cols),
            pd.DataFrame(columns=[glucose_col, dt_col, id_col, date_col] + _annot_extra),
        )

    # ── Datetime housekeeping ──
    d[dt_col] = pd.to_datetime(d[dt_col])
    d[date_col] = (
        pd.to_datetime(d[date_col]).dt.date
        if date_col in d.columns
        else d[dt_col].dt.date
    )
    d = d.sort_values(dt_col).reset_index(drop=True)
    study_day_col = "study_day"
    d = d.dropna(subset=[glucose_col]).reset_index(drop=True)

    g = d[glucose_col].to_numpy(dtype=float)
    t = d[dt_col].to_numpy()
    day = d[date_col].to_numpy()
    n = len(g)

    # ── Convert minute-based params to sample counts ──
    spm = 1.0 / sample_step_min                       # samples per minute
    lookback_n = int(baseline_lookback_min * spm)
    excl_n     = int(baseline_exclude_last_min * spm)
    rise_n     = int(rise_window_min * spm)
    min_above_n = int(min_above_min * spm)
    refractory_n = int(refractory_min * spm)

    # ── Guard: not enough data ──
    if n < lookback_n + rise_n + 1:
        annotated = d.copy()
        for col, val in [
            ("baseline", np.nan), ("slope_15m", np.nan), ("residual", np.nan),
            ("event_id", pd.NA), ("in_spike_event", False),
            ("is_event_start", False), ("is_event_end", False),
            ("is_event_peak", False), ("p_val_event", np.nan),
        ]:
            annotated[col] = val
        if col == "event_id":
            annotated["event_id"] = annotated["event_id"].astype("Int64")
        return pd.DataFrame(columns=_event_cols), annotated

    # ── Baseline: rolling median of [t − lookback … t − excl] ──
    B = np.full(n, np.nan)
    for i in range(lookback_n, n):
        win_start = i - lookback_n
        win_end   = i - excl_n
        if win_end > win_start:
            B[i] = np.nanmedian(g[win_start:win_end])

    # ── Rise rate over rise_window_min ──
    slope = np.full(n, np.nan)
    for i in range(rise_n, n):
        slope[i] = (g[i] - g[i - rise_n]) / rise_window_min

    # ── Noise estimate (MAD of short-window residuals) ──
    rolling_med = (
        pd.Series(g).rolling(window=7, center=True, min_periods=3).median().to_numpy()
    )
    resid_short = g - rolling_med
    mad = np.nanmedian(np.abs(resid_short - np.nanmedian(resid_short)))
    noise = max(1.4826 * mad, expected_noise_mad_floor)

    # ── Prepare annotation columns ──
    annotated = d.copy()
    annotated["baseline"]  = B
    annotated["slope_15m"] = slope
    annotated["residual"]  = g - B

    annotated["event_id"]       = pd.array([pd.NA] * n, dtype="Int64")
    annotated["in_spike_event"] = False
    annotated["is_event_start"] = False
    annotated["is_event_end"]   = False
    annotated["is_event_peak"]  = False
    annotated["p_val_event"]    = np.nan

    # ── Main sweep ──
    events: list[dict] = []
    i = 0
    in_refractory_until = -1
    event_id = 0

    while i < n:
        # Refractory enforcement (allow exit if glucose already near baseline)
        if i < in_refractory_until:
            if not (np.isfinite(B[i]) and g[i] <= B[i] + end_band):
                i += 1
                continue

        # Need valid baseline & slope
        if not (np.isfinite(B[i]) and np.isfinite(slope[i])):
            i += 1
            continue

        # ── Entry conditions ──
        if (g[i] - B[i]) < amp_thresh or slope[i] < rise_rate_thresh:
            i += 1
            continue

        # Walk back to first sample above amp_thresh
        start_i = i
        while (
            start_i - 1 >= 0
            and np.isfinite(B[start_i - 1])
            and (g[start_i - 1] - B[start_i - 1]) >= amp_thresh
        ):
            start_i -= 1

        # Minimum duration above threshold
        above = 0
        k = start_i
        while k < n and np.isfinite(B[k]) and (g[k] - B[k]) >= amp_thresh:
            above += 1
            k += 1
            if above >= min_above_n:
                break
        if above < min_above_n:
            i += 1
            continue

        # ── Walk forward to find peak and return-to-baseline ──
        peak_i = i
        max_g  = g[i]
        min_g  = g[i]
        j = i
        returned = False
        end_i = None

        while j < n:
            if g[j] > max_g:
                max_g  = g[j]
                peak_i = j
            if g[j] < min_g:
                min_g = g[j]
            if np.isfinite(B[j]) and g[j] <= B[j] + end_band:
                returned = True
                end_i = j
                break
            j += 1

        if not returned:
            # Spike never resolved within the data — skip
            i += 1
            continue

        event_range = float(max_g - min_g)

        # ── Confidence sub-scores ──
        peak_baseline = B[peak_i] if np.isfinite(B[peak_i]) else np.nanmedian(B)
        peak_amp = float(max_g - peak_baseline)

        # Amplitude score
        amp_excess = max(0.0, peak_amp - amp_thresh)
        amp_score  = 1.0 - np.exp(-amp_excess / (noise + 1e-9))

        # Slope score (max slope in first 30 min of event)
        slope_end = min(end_i, start_i + int(30 * spm))
        local_slopes = slope[start_i : slope_end + 1]
        local_max_slope = float(np.nanmax(local_slopes)) if len(local_slopes) > 0 else float(slope[i])
        slope_excess = max(0.0, local_max_slope - rise_rate_thresh)
        slope_score  = 1.0 - np.exp(-slope_excess / (noise / max(rise_window_min, 1) + 1e-9))

        # Duration score
        dur_above = sum(
            1 for q in range(start_i, end_i + 1)
            if np.isfinite(B[q]) and (g[q] - B[q]) >= amp_thresh
        )
        dur_min   = dur_above * sample_step_min
        dur_score = min(1.0, dur_min / 60.0)

        # Return-to-baseline quality
        end_gap = float(g[end_i] - (B[end_i] if np.isfinite(B[end_i]) else np.nanmedian(B)))
        return_score = float(np.clip(1.0 - max(0.0, end_gap) / max(end_band, 1e-9), 0.0, 1.0))

        p_val = float(np.clip(
            w_amp * amp_score
            + w_slope * slope_score
            + w_dur * dur_score
            + w_return * return_score,
            0.0, 1.0,
        ))

        # ── Record event ──
        events.append({
            "Day":              int(d.iloc[start_i][study_day_col]) if study_day_col in d.columns else day[start_i],
            "start_time":       pd.Timestamp(t[start_i]),
            "end_time":         pd.Timestamp(t[end_i]),
            "peak_time":        pd.Timestamp(t[peak_i]),
            "p_val":            round(p_val, 4),
            "range":            round(event_range, 2),
            "peak_glucose":     round(float(max_g), 2),
            "baseline_at_peak": round(float(peak_baseline), 2),
        })

        # ── Annotate trace ──
        idx = annotated.index[start_i : end_i + 1]
        annotated.loc[idx, "event_id"]       = event_id
        annotated.loc[idx, "in_spike_event"] = True
        annotated.loc[idx, "p_val_event"]    = p_val
        annotated.loc[annotated.index[start_i], "is_event_start"] = True
        annotated.loc[annotated.index[end_i],   "is_event_end"]   = True
        annotated.loc[annotated.index[peak_i],  "is_event_peak"]  = True

        in_refractory_until = end_i + refractory_n
        i = end_i + 1
        event_id += 1

    events_df = pd.DataFrame(events, columns=_event_cols)
    if not events_df.empty:
        events_df = events_df.sort_values(["Day", "start_time"]).reset_index(drop=True)

    return events_df, annotated


# ---------------------------------------------------------------------------
# 3. VISUALISATION (Seaborn + annotated peaks)
# ---------------------------------------------------------------------------

def visualize_spikes(
    annotated_df: pd.DataFrame,
    events_df: pd.DataFrame,
    person_id: str,
    out_dir: str = "spike_plots",
    glucose_col: str = "glucose_mmol_l",
    dt_col: str = "datetime",
    day_col: str = "date",
    dpi: int = 180,
    style: str = "whitegrid",
    palette: str = "deep",
):
    """
    Create polished Seaborn-styled daily CGM plots with spike annotations.

    For each day in the data, produces a PNG showing:
      • Glucose trace (dark line)
      • Rolling baseline (blue dashed line)
      • Spike intervals (orange shading)
      • Peak markers (red) annotated with peak glucose and confidence score

    Parameters
    ----------
    annotated_df : pd.DataFrame
        Annotated trace from ``detect_spikes()``.
    events_df : pd.DataFrame
        Events table from ``detect_spikes()``.
    person_id : str
        Identifier used in titles and filenames.
    out_dir : str
        Directory for saved PNGs (created if absent).
    glucose_col, dt_col, day_col : str
        Column names.
    dpi : int
        Resolution of saved figures.
    style : str
        Seaborn theme (e.g. ``"whitegrid"``, ``"darkgrid"``).
    palette : str
        Seaborn colour palette name.

    Returns
    -------
    list[str]
        Paths to all saved PNG files.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style=style, palette=palette, font_scale=1.05)

    a = annotated_df.copy()
    a[dt_col] = pd.to_datetime(a[dt_col])
    a[day_col] = (
        pd.to_datetime(a[day_col]).dt.date
        if day_col in a.columns
        else a[dt_col].dt.date
    )

    ev = events_df.copy()
    if not ev.empty:
        for c in ("start_time", "end_time", "peak_time"):
            if c in ev.columns:
                ev[c] = pd.to_datetime(ev[c])
        if "Day" in ev.columns:
            ev["Day"] = pd.to_datetime(ev["Day"]).dt.date

    saved_paths: list[str] = []

    for day in sorted(a[day_col].dropna().unique()):
        day_df = a.loc[a[day_col] == day].sort_values(dt_col)
        if day_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(15, 4.5))

        # ── Glucose trace ──
        sns.lineplot(
            data=day_df, x=dt_col, y=glucose_col,
            color="0.15", linewidth=1.3, ax=ax, label="Glucose",
        )

        # ── Baseline ──
        if "baseline" in day_df.columns:
            ax.plot(
                day_df[dt_col], day_df["baseline"],
                color="dodgerblue", linewidth=1.0, linestyle="--",
                alpha=0.75, label="Baseline",
            )

        # ── Spike shading + peak annotations ──
        if not ev.empty:
            ev_day = (
                ev.loc[ev["Day"] == day]
                if "Day" in ev.columns
                else ev.loc[ev["start_time"].dt.date == day]
            )

            for _, e in ev_day.iterrows():
                ax.axvspan(
                    e["start_time"], e["end_time"],
                    color="orange", alpha=0.20, zorder=0,
                )

        # ── Peak markers with annotation ──
        if "is_event_peak" in day_df.columns:
            peaks = day_df.loc[day_df["is_event_peak"] == True]
            if not peaks.empty:
                ax.scatter(
                    peaks[dt_col], peaks[glucose_col],
                    color="crimson", s=50, zorder=5,
                    edgecolors="white", linewidths=0.6,
                    label="Spike peak",
                )
                for _, pk in peaks.iterrows():
                    pval_str = (
                        f"{pk['p_val_event']:.2f}"
                        if pd.notna(pk.get("p_val_event"))
                        else ""
                    )
                    label_text = f"{pk[glucose_col]:.1f} mmol/L\nconf {pval_str}"
                    ax.annotate(
                        label_text,
                        xy=(pk[dt_col], pk[glucose_col]),
                        xytext=(0, 14),
                        textcoords="offset points",
                        fontsize=7.5,
                        fontweight="bold",
                        color="crimson",
                        ha="center",
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            fc="white", ec="crimson",
                            alpha=0.85, lw=0.6,
                        ),
                    )

        # ── Axes formatting ──
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate(rotation=0, ha="center")

        ax.set_title(
            f"{person_id}  ·  {day}",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Time of day", fontsize=10)
        ax.set_ylabel("Glucose (mmol/L)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        sns.despine(left=False, bottom=False)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{person_id}_{day}_spikes.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    print(f"✓ Saved {len(saved_paths)} daily plot(s) to: {out_dir}/")
    return saved_paths


# ---------------------------------------------------------------------------
# 4. CONVENIENCE RUNNER
# ---------------------------------------------------------------------------

def run_pipeline(
    wide_df: pd.DataFrame,
    person_id: str,
    id_col: str = "Client ID",
    day_col: str = "Day",
    out_dir: str = "spike_plots",
    **spike_kwargs,
):
    """
    End-to-end convenience function: reshape → detect → visualise.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Raw wide-format CGM data.
    person_id : str
        Client to analyse.
    id_col, day_col : str
        Column names in the wide data.
    out_dir : str
        Output directory for plots.
    **spike_kwargs
        Forwarded to ``detect_spikes()`` (thresholds, weights, etc.).

    Returns
    -------
    events_df : pd.DataFrame
    annotated_df : pd.DataFrame
    saved_paths : list[str]
    """
    long_df = reshape_wide_to_long(wide_df, id_col=id_col, day_col=day_col)
    events_df, annotated_df = detect_spikes(
        long_df, person_id=person_id, **spike_kwargs,
    )
    saved_paths = visualize_spikes(
        annotated_df, events_df,
        person_id=person_id, out_dir=out_dir,
    )
    return events_df, annotated_df, saved_paths


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── Example with synthetic wide-format data ──
    np.random.seed(42)

    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
    n_times = len(times)  # 288 readings per day

    # Simulate 3 days for one client
    rows = []
    for day_offset in range(3):
        base = 5.0 + 0.3 * np.random.randn(n_times)
        # Inject a spike around 08:00–09:30 (indices ~96–114)
        spike_centre = 102 + day_offset * 2
        for k in range(-6, 18):
            idx = spike_centre + k
            if 0 <= idx < n_times:
                base[idx] += 3.5 * np.exp(-0.5 * ((k - 4) / 4) ** 2)
        # Inject a second spike around 18:00
        spike2 = 216
        for k in range(-5, 15):
            idx = spike2 + k
            if 0 <= idx < n_times:
                base[idx] += 2.8 * np.exp(-0.5 * ((k - 3) / 3.5) ** 2)

        row = {
            "Client ID": "DEMO-001",
            "Day": pd.Timestamp("2025-03-01") + pd.Timedelta(days=day_offset),
        }
        for i, tc in enumerate(times):
            row[tc] = round(float(base[i]), 2)
        rows.append(row)

    wide_df = pd.DataFrame(rows)

    print("Wide-format shape:", wide_df.shape)
    print(wide_df.head(), "\n")

    # Run full pipeline
    events, annotated, paths = run_pipeline(
        wide_df,
        person_id="DEMO-001",
        out_dir="spike_plots",
    )

    print("\n── Detected spikes ──")
    print(events.to_string(index=False))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# def mean_glucose(
#         cgm_data: pd.DataFrame,
#         level = c
# )
"""
Function to provide a mean glucose metric on a given level: cohort, per patient, or per day
""" '''
'''# ---------------------------------------------------------------------------
# spike_detection.py — CGM Spike & Dip Detection (FFT-Enhanced, Dual Pipeline)
# ---------------------------------------------------------------------------
#
# Evolution of the original rolling-median spike detector.
# Key additions:
#   • FFT decomposition replaces rolling-median baseline
#   • Bidirectional detection (spikes up + dips down)
#   • Two sensitivity tiers: high_confidence & sensitive
#   • Confidence scoring with SNR component
#   • Visually polished plots with start/end highlighting
#
# The public API mirrors the original where possible:
#   detect_spikes()  → now detect_events()  (superset)
#   visualize_spikes() → now visualize_events()
#   run_pipeline()   → updated to use new functions
#
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.fft import fft, ifft, fftfreq
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
sns.set_context("notebook")


# ═══════════════════════════════════════════════════════════════
#  PRESETS — thresholds for each (direction × sensitivity) combo
# ═══════════════════════════════════════════════════════════════

PRESETS_SPIKE = {
    "high_confidence": dict(
        amp_thresh=1.2,
        rise_rate_thresh=0.025,
        rise_window_min=15,
        min_above_min=20,
        end_band=0.3,
        refractory_min=45,
        gate_mode="and",
        require_return=True,
        max_event_duration_min=300,
    ),
    "sensitive": dict(
        amp_thresh=0.55,
        rise_rate_thresh=0.012,
        rise_window_min=20,
        min_above_min=10,
        end_band=0.4,
        refractory_min=20,
        gate_mode="and",
        require_return=False,
        max_event_duration_min=180,
    ),
}

PRESETS_DIP = {
    "high_confidence": dict(
        amp_thresh=0.9,
        rise_rate_thresh=0.02,
        rise_window_min=15,
        min_above_min=15,
        end_band=0.3,
        refractory_min=45,
        gate_mode="and",
        require_return=True,
        max_event_duration_min=300,
        absolute_thresh=3.9,
    ),
    "sensitive": dict(
        amp_thresh=0.4,
        rise_rate_thresh=0.008,
        rise_window_min=20,
        min_above_min=10,
        end_band=0.4,
        refractory_min=20,
        gate_mode="and",
        require_return=False,
        max_event_duration_min=180,
        absolute_thresh=4.2,
    ),
}


# ═══════════════════════════════════════════════════════════════
#  FFT DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

def fft_decompose_cgm(
    glucose: np.ndarray,
    sample_step_min: int = 5,
    baseline_cutoff_hours: float = 1.5,
    noise_cutoff_min: float = 12.0,
):
    """
    Decompose a CGM glucose signal into three frequency bands via FFT.

    Instead of the original rolling-median baseline, this uses spectral
    separation to produce a smooth baseline that tracks the slow drift,
    an event component capturing spike/dip-scale oscillations, and a
    high-frequency noise component that gets discarded.

    Parameters
    ----------
    glucose : np.ndarray
        Raw glucose values (mmol/L), may contain NaNs.
    sample_step_min : int
        Sampling interval in minutes (default 5).
    baseline_cutoff_hours : float
        Periods longer than this go into the baseline band.
        1.5 h keeps enough natural variation so the baseline actually
        follows the glucose trend rather than being a flat line.
    noise_cutoff_min : float
        Periods shorter than this are classified as noise (default 12 min).

    Returns
    -------
    dict with keys:
        baseline, events, noise, denoised, freqs, power, g_interpolated
    """
    n = len(glucose)

    # Interpolate NaNs for FFT (FFT needs a complete signal)
    g = glucose.copy().astype(float)
    nans = np.isnan(g)
    if nans.any():
        not_nan = np.where(~nans)[0]
        if len(not_nan) >= 2:
            g[nans] = np.interp(np.where(nans)[0], not_nan, g[not_nan])
        elif len(not_nan) == 1:
            g[nans] = g[not_nan[0]]
        else:
            g[:] = 0.0

    g_mean = np.mean(g)
    g_centered = g - g_mean

    G = fft(g_centered)
    freqs = fftfreq(n, d=sample_step_min / 60.0)  # cycles per hour

    baseline_cutoff_freq = 1.0 / baseline_cutoff_hours
    noise_cutoff_freq = 60.0 / noise_cutoff_min
    abs_freqs = np.abs(freqs)

    # Three non-overlapping masks
    baseline_mask = abs_freqs <= baseline_cutoff_freq
    noise_mask = abs_freqs >= noise_cutoff_freq
    event_mask = ~baseline_mask & ~noise_mask

    # Inverse-transform each band
    G_baseline = np.zeros_like(G)
    G_baseline[baseline_mask] = G[baseline_mask]
    baseline = np.real(ifft(G_baseline)) + g_mean

    G_events = np.zeros_like(G)
    G_events[event_mask] = G[event_mask]
    events_component = np.real(ifft(G_events))

    G_noise = np.zeros_like(G)
    G_noise[noise_mask] = G[noise_mask]
    noise_component = np.real(ifft(G_noise))

    denoised = baseline + events_component

    power = np.abs(G[: n // 2]) ** 2 / n

    return {
        "baseline": baseline,
        "events": events_component,
        "noise": noise_component,
        "denoised": denoised,
        "freqs": freqs[: n // 2],
        "power": power,
        "g_interpolated": g,
    }


# ═══════════════════════════════════════════════════════════════
#  CORE SINGLE-DIRECTION DETECTOR (replaces the while-loop in
#  the original detect_spikes, now generalised for up/down)
# ═══════════════════════════════════════════════════════════════

def _detect_single_direction(
    g_raw,
    g_denoised,
    fft_baseline,
    event_component,
    t,
    day_arr,
    n,
    idx_per_min,
    noise_std,
    dirn,
    *,
    amp_thresh,
    rise_rate_thresh,
    rise_window_min,
    min_above_min,
    end_band,
    refractory_min,
    gate_mode,
    require_return,
    max_event_duration_min,
    sample_step_min,
    w_amp,
    w_slope,
    w_dur,
    w_return,
    w_snr,
    absolute_thresh=None,
    **_extra,
):
    """
    Core event detection for one direction (up or down).

    This is the direct evolution of the original ``detect_spikes`` while-loop.
    Key changes from the original:
      • Works on FFT-denoised signal instead of raw glucose
      • Uses FFT baseline instead of rolling median
      • Supports downward detection by flipping the signal
      • AND-gate logic prevents over-triggering (original used simple AND)
      • Adds SNR and prominence to the confidence score
      • Respects max_event_duration_min timeout
    """

    # For dip detection, flip the signal so we can reuse the same
    # "find peaks above baseline" logic
    if dirn == "down":
        g_work = -g_denoised.copy()
        B = -fft_baseline.copy()
    else:
        g_work = g_denoised.copy()
        B = fft_baseline.copy()

    rise_n = int(rise_window_min * idx_per_min)
    min_above_n = max(1, int(min_above_min * idx_per_min))
    refractory_n = int(refractory_min * idx_per_min)
    max_event_n = int(max_event_duration_min * idx_per_min)

    if n < (rise_n + 1):
        return []

    # ── Rise rate on denoised signal (replaces original slope calc) ──
    slope = np.full(n, np.nan)
    for i in range(rise_n, n):
        slope[i] = (g_work[i] - g_work[i - rise_n]) / rise_window_min

    # ── Local prominence (for confidence scoring, NOT gating) ──
    prom_window = int(60 * idx_per_min)
    prominence = np.full(n, 0.0)
    for i in range(n):
        l_start = max(0, i - prom_window)
        r_end = min(n, i + prom_window + 1)
        l_min = np.nanmin(g_work[l_start : i + 1])
        r_min = np.nanmin(g_work[i : r_end])
        prominence[i] = g_work[i] - max(l_min, r_min)

    # ── Main sweep (mirrors original structure) ──
    events = []
    i = 0
    in_refractory_until = -1

    while i < n:
        # Refractory enforcement (same logic as original)
        if i < in_refractory_until:
            if not (np.isfinite(B[i]) and g_work[i] <= B[i] + end_band):
                i += 1
                continue

        if not np.isfinite(B[i]):
            i += 1
            continue

        # ── Entry conditions ──
        amp_ok = (g_work[i] - B[i]) >= amp_thresh
        slope_ok = np.isfinite(slope[i]) and slope[i] >= rise_rate_thresh

        # Absolute threshold — dips only (e.g. glucose < 3.9)
        if dirn == "down" and absolute_thresh is not None:
            absolute_ok = (-g_work[i]) <= absolute_thresh
        else:
            absolute_ok = False

        # Gate logic
        if gate_mode == "or":
            triggered = amp_ok or slope_ok or absolute_ok
        else:  # "and" — the default, same as original
            triggered = (amp_ok and slope_ok) or absolute_ok

        if not triggered:
            i += 1
            continue

        # ── Walk back to find event start (same as original) ──
        walkback_thresh = amp_thresh * 0.4
        start_i = i
        while (
            start_i - 1 >= 0
            and np.isfinite(B[start_i - 1])
            and (g_work[start_i - 1] - B[start_i - 1]) >= walkback_thresh
        ):
            start_i -= 1

        # ── Minimum duration above threshold (same as original) ──
        above_count = 0
        k = start_i
        while k < n:
            if np.isfinite(B[k]) and (g_work[k] - B[k]) >= amp_thresh * 0.6:
                above_count += 1
            else:
                break
            k += 1
            if above_count >= min_above_n:
                break

        if above_count < min_above_n:
            i += 1
            continue

        # ── Walk forward for peak and end (extended from original) ──
        peak_i = i
        max_g = g_work[i]
        min_g = g_work[i]
        j = i
        closure_reason = "end_of_data"
        end_i = n - 1

        while j < n:
            if g_work[j] > max_g:
                max_g = g_work[j]
                peak_i = j
            if g_work[j] < min_g:
                min_g = g_work[j]

            # Return-to-baseline check (same concept as original)
            if np.isfinite(B[j]) and g_work[j] <= B[j] + end_band:
                end_i = j
                closure_reason = "return_to_baseline"
                break

            # NEW: timeout guard (original just skipped unresolved spikes)
            if (j - start_i) >= max_event_n:
                end_i = j
                closure_reason = "timeout"
                break

            j += 1

        # Original required return; now configurable per tier
        if require_return and closure_reason != "return_to_baseline":
            i += 1
            continue

        event_range = float(max_g - min_g)

        # ── Confidence sub-scores (extended from original 4 → 6) ──
        peak_baseline = B[peak_i] if np.isfinite(B[peak_i]) else np.nanmedian(B)
        peak_amp = float(max_g - peak_baseline)

        # 1. Amplitude score (same formula as original)
        amp_excess = max(0.0, peak_amp - amp_thresh)
        amp_score = 1.0 - np.exp(-(amp_excess / (noise_std + 1e-9)))

        # 2. Slope score (same as original)
        slope_end = min(end_i, start_i + int(30 * idx_per_min))
        if slope_end >= start_i:
            local_slopes = slope[start_i : slope_end + 1]
            local_max_slope = (
                float(np.nanmax(local_slopes))
                if np.any(np.isfinite(local_slopes))
                else 0.0
            )
        else:
            local_max_slope = float(slope[i]) if np.isfinite(slope[i]) else 0.0
        slope_excess = max(0.0, local_max_slope - rise_rate_thresh)
        slope_score = 1.0 - np.exp(
            -(slope_excess / (noise_std / max(rise_window_min, 1) + 1e-9))
        )

        # 3. Duration score (same as original)
        above = sum(
            1
            for q in range(start_i, end_i + 1)
            if np.isfinite(B[q]) and (g_work[q] - B[q]) >= amp_thresh * 0.6
        )
        dur_min = above * sample_step_min
        dur_score = min(1.0, dur_min / 60.0)

        # 4. Return-to-baseline score (extended: handles timeout)
        end_gap = float(
            g_work[end_i]
            - (B[end_i] if np.isfinite(B[end_i]) else np.nanmedian(B))
        )
        if closure_reason == "return_to_baseline" and np.isfinite(end_gap):
            return_score = float(
                np.clip(1.0 - (max(0.0, end_gap) / max(end_band, 1e-9)), 0.0, 1.0)
            )
        elif closure_reason == "timeout":
            return_score = 0.25
        else:
            return_score = 0.15

        # 5. NEW: SNR score
        snr = peak_amp / (noise_std + 1e-9)
        snr_score = float(np.clip(snr / 5.0, 0.0, 1.0))

        # 6. NEW: Prominence bonus (small weight, not a gate)
        prom_score = float(np.clip(prominence[peak_i] / 2.0, 0.0, 1.0))

        p_val = float(
            np.clip(
                w_amp * amp_score
                + w_slope * slope_score
                + w_dur * dur_score
                + w_return * return_score
                + w_snr * snr_score
                + 0.05 * prom_score,
                0.0,
                1.0,
            )
        )

        # ── Un-flip for dip direction ──
        if dirn == "down":
            extreme_glucose = float(-max_g)
            baseline_at_extreme = (
                float(-B[peak_i]) if np.isfinite(B[peak_i]) else np.nan
            )
        else:
            extreme_glucose = float(max_g)
            baseline_at_extreme = (
                float(B[peak_i]) if np.isfinite(B[peak_i]) else np.nan
            )

        events.append(
            {
                "Day": day_arr[start_i],
                "start_time": pd.to_datetime(t[start_i]),
                "end_time": pd.to_datetime(t[end_i]),
                "peak_time": pd.to_datetime(t[peak_i]),
                "p_val": round(p_val, 4),
                "range": round(event_range, 3),
                "extreme_glucose": round(extreme_glucose, 3),
                "baseline_at_extreme": (
                    round(baseline_at_extreme, 3)
                    if not np.isnan(baseline_at_extreme)
                    else np.nan
                ),
                "closure_reason": closure_reason,
            }
        )

        in_refractory_until = end_i + refractory_n
        i = end_i + 1

    return events


# ═══════════════════════════════════════════════════════════════
#  DEDUPLICATION — when running both tiers, HC takes priority
# ═══════════════════════════════════════════════════════════════

def _deduplicate_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Remove overlapping events, preferring high_confidence over sensitive."""
    if events_df.empty:
        return events_df

    tier_priority = {"high_confidence": 0, "sensitive": 1}
    events_df["_priority"] = events_df["tier"].map(tier_priority).fillna(2)
    events_df = events_df.sort_values(
        ["direction", "_priority", "start_time"]
    ).reset_index(drop=True)

    keep = []
    for dirn in events_df["direction"].unique():
        dir_events = (
            events_df[events_df["direction"] == dirn].copy().sort_values(["_priority", "start_time"]).reset_index(drop=True)
        )
        kept_intervals = []

        for _, row in dir_events.iterrows():
            st = pd.to_datetime(row["start_time"])
            et = pd.to_datetime(row["end_time"])
            overlaps = any(st <= ke and et >= ks for ks, ke in kept_intervals)
            if not overlaps:
                keep.append(row)
                kept_intervals.append((st, et))

    result = pd.DataFrame(keep)
    if "_priority" in result.columns:
        result = result.drop(columns=["_priority"])
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  2. MAIN DETECTION FUNCTION (replaces detect_spikes)
# ═══════════════════════════════════════════════════════════════

def detect_events(
    df: pd.DataFrame,
    person_id: str = "",
    glucose_col: str = "glucose_mmol_l",
    dt_col: str = "datetime",
    id_col: str = "person_id",
    date_col: str = "date",
    # ── What to detect ──
    mode: str = "both",           # "high_confidence", "sensitive", or "both"
    direction: str = "both",      # "up" (spikes), "down" (dips), or "both"
    # ── FFT parameters (replace rolling-median baseline) ──
    sample_step_min: int = 5,
    baseline_cutoff_hours: float = 1.5,
    noise_cutoff_min: float = 12.0,
    # ── Confidence scoring weights ──
    w_amp: float = 0.35,
    w_slope: float = 0.20,
    w_dur: float = 0.15,
    w_return: float = 0.10,
    w_snr: float = 0.15,
    expected_noise_mad_floor: float = 0.05,
    # ── Override any preset parameter ──
    **override_kwargs,
):
    """
    Detect glucose spike and/or dip *events* for one client using
    FFT-based signal decomposition.

    This replaces the original ``detect_spikes()`` function. Key changes:

    1. **FFT baseline** instead of rolling median — the signal is
       decomposed into baseline (slow drift), event-scale oscillations,
       and high-frequency noise. The denoised signal (baseline + events)
       is what gets analysed.

    2. **Bidirectional** — detects both upward spikes and downward dips.
       The ``direction`` parameter controls which.

    3. **Dual sensitivity** — ``mode="both"`` runs high-confidence AND
       sensitive pipelines, then deduplicates (HC wins on overlap).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format CGM data.
    person_id : str
        The person_id to analyse.
    mode : str
        ``"high_confidence"``, ``"sensitive"``, or ``"both"``.
    direction : str
        ``"up"`` (spikes only), ``"down"`` (dips only), or ``"both"``.
    baseline_cutoff_hours : float
        FFT baseline cutoff. Periods > this → baseline band.
    noise_cutoff_min : float
        FFT noise cutoff. Periods < this → noise (discarded).
    w_amp, w_slope, w_dur, w_return, w_snr : float
        Confidence score weights.
    **override_kwargs
        Override any preset parameter (e.g. ``amp_thresh=1.5``).

    Returns
    -------
    events_df : pd.DataFrame
        One row per detected event. Columns:
        ``Day | start_time | end_time | peak_time | p_val | range |
        extreme_glucose | baseline_at_extreme | closure_reason |
        direction | tier``
    annotated_df : pd.DataFrame
        The client's trace with annotation columns:
        ``baseline_fft | event_component | denoised | slope |
        residual | event_id | in_event | is_event_start |
        is_event_end | is_event_extreme | p_val_event |
        event_direction | event_tier``
    """

    EVENT_COLS = [
        "Day", "start_time", "end_time", "peak_time", "p_val",
        "range", "extreme_glucose", "baseline_at_extreme",
        "closure_reason", "direction", "tier",
    ]

    # ── Column validation (same as original) ──
    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' not found. Available: {df.columns.tolist()}")

    d = df.loc[df[id_col].astype(str) == str(person_id)].copy()

    if d.empty:
        return (pd.DataFrame(columns=EVENT_COLS), pd.DataFrame())

    # ── Datetime housekeeping (same as original) ──
    d[dt_col] = pd.to_datetime(d[dt_col])
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col]).dt.date
    else:
        d[date_col] = d[dt_col].dt.date

    d = d.sort_values(dt_col).reset_index(drop=True)
    d = d.dropna(subset=[glucose_col]).reset_index(drop=True)

    g = d[glucose_col].to_numpy(dtype=float)
    t = d[dt_col].to_numpy()
    day_arr = d[date_col].to_numpy()
    n = len(g)
    idx_per_min = 1.0 / sample_step_min

    # ── FFT decomposition (replaces rolling-median baseline) ──
    fft_result = fft_decompose_cgm(
        g,
        sample_step_min=sample_step_min,
        baseline_cutoff_hours=baseline_cutoff_hours,
        noise_cutoff_min=noise_cutoff_min,
    )

    fft_baseline = fft_result["baseline"]
    event_component = fft_result["events"]
    denoised = fft_result["denoised"]
    noise_component = fft_result["noise"]

    # Noise estimate from the FFT noise band (replaces MAD estimate)
    noise_std = max(np.std(noise_component), expected_noise_mad_floor)

    # ── Resolve which runs to do ──
    directions = []
    if direction in ("up", "both"):
        directions.append("up")
    if direction in ("down", "both"):
        directions.append("down")

    modes = []
    if mode in ("high_confidence", "both"):
        modes.append("high_confidence")
    if mode in ("sensitive", "both"):
        modes.append("sensitive")

    # ── Run detection for each (direction × tier) combination ──
    all_events = []

    for dirn in directions:
        for m in modes:
            params = (PRESETS_SPIKE if dirn == "up" else PRESETS_DIP)[m].copy()
            params.update(override_kwargs)

            events = _detect_single_direction(
                g_raw=g,
                g_denoised=denoised,
                fft_baseline=fft_baseline,
                event_component=event_component,
                t=t,
                day_arr=day_arr,
                n=n,
                idx_per_min=idx_per_min,
                noise_std=noise_std,
                dirn=dirn,
                sample_step_min=sample_step_min,
                w_amp=w_amp,
                w_slope=w_slope,
                w_dur=w_dur,
                w_return=w_return,
                w_snr=w_snr,
                **params,
            )
            for ev in events:
                ev["direction"] = dirn
                ev["tier"] = m
            all_events.extend(events)

    events_df = pd.DataFrame(all_events, columns=EVENT_COLS)
    if not events_df.empty:
        events_df = _deduplicate_events(events_df)
        events_df = events_df.sort_values(["Day", "start_time"]).reset_index(drop=True)

    # ── Build annotated trace (extended from original) ──
    rise_n = int(20 * idx_per_min)
    slope = np.full(n, np.nan)
    for i in range(rise_n, n):
        slope[i] = (denoised[i] - denoised[i - rise_n]) / 20.0

    annotated = d.copy()
    annotated["baseline_fft"] = fft_baseline
    annotated["event_component"] = event_component
    annotated["denoised"] = denoised
    annotated["slope"] = slope
    annotated["residual"] = g - fft_baseline

    # Event annotation columns (extended from original)
    annotated["event_id"] = pd.array([pd.NA] * n, dtype="Int64")
    annotated["in_event"] = False
    annotated["is_event_start"] = False
    annotated["is_event_end"] = False
    annotated["is_event_extreme"] = False
    annotated["p_val_event"] = np.nan
    annotated["event_direction"] = ""
    annotated["event_tier"] = ""

    if not events_df.empty:
        for eid, row in events_df.iterrows():
            st = pd.to_datetime(row["start_time"])
            et = pd.to_datetime(row["end_time"])
            pt = pd.to_datetime(row["peak_time"])

            mask = (annotated[dt_col] >= st) & (annotated[dt_col] <= et)
            annotated.loc[mask, "event_id"] = eid
            annotated.loc[mask, "in_event"] = True
            annotated.loc[mask, "p_val_event"] = row["p_val"]
            annotated.loc[mask, "event_direction"] = row["direction"]
            annotated.loc[mask, "event_tier"] = row["tier"]

            annotated.loc[
                mask & (annotated[dt_col] == st), "is_event_start"
            ] = True
            annotated.loc[
                mask & (annotated[dt_col] == et), "is_event_end"
            ] = True

            extreme_mask = mask & (
                (annotated[dt_col] - pt).abs()
                == (annotated[dt_col] - pt).abs().min()
            )
            annotated.loc[extreme_mask, "is_event_extreme"] = True

    return events_df, annotated


# ═══════════════════════════════════════════════════════════════
#  3. VISUALISATION (Seaborn + event highlighting)
# ═══════════════════════════════════════════════════════════════

# Colour and style definitions for each (direction, tier) pair
_VIS_STYLES = {
    ("up", "high_confidence"): {
        "shade": "#E8453C",
        "shade_alpha": 0.18,
        "marker_color": "#E8453C",
        "marker": "^",
        "marker_size": 60,
        "border_color": "#E8453C",
        "label": "Spike (HC)",
    },
    ("up", "sensitive"): {
        "shade": "#F5A623",
        "shade_alpha": 0.12,
        "marker_color": "#F5A623",
        "marker": "^",
        "marker_size": 40,
        "border_color": "#F5A623",
        "label": "Spike (Sensitive)",
    },
    ("down", "high_confidence"): {
        "shade": "#3B7DD8",
        "shade_alpha": 0.18,
        "marker_color": "#3B7DD8",
        "marker": "v",
        "marker_size": 60,
        "border_color": "#3B7DD8",
        "label": "Dip (HC)",
    },
    ("down", "sensitive"): {
        "shade": "#7EC8E3",
        "shade_alpha": 0.12,
        "marker_color": "#7EC8E3",
        "marker": "v",
        "marker_size": 40,
        "border_color": "#7EC8E3",
        "label": "Dip (Sensitive)",
    },
}


def visualize_events(
    annotated_df: pd.DataFrame,
    events_df: pd.DataFrame,
    person_id: str,
    *,
    # ── Toggle which layers are visible ──
    show_spike_hc: bool = True,
    show_spike_sens: bool = False,
    show_dip_hc: bool = True,
    show_dip_sens: bool = False,
    # ── Display options ──
    show_baseline: bool = True,
    show_event_component: bool = True,
    # ── Output ──
    out_dir: str = "event_plots",
    glucose_col: str = "glucose_mmol_l",
    dt_col: str = "datetime",
    day_col: str = "date",
    dpi: int = 180,
    style: str = "whitegrid",
    palette: str = "deep",
    return_figs: bool = False,
):
    """
    Create polished daily CGM plots with spike/dip annotations.

    Replaces the original ``visualize_spikes()`` with:
      • Dual-panel layout (glucose + event component)
      • Toggle-controlled event layers
      • Start/end boundary markers on highlighted regions
      • Cleaner typography and reduced clutter

    Parameters
    ----------
    annotated_df, events_df : pd.DataFrame
        Outputs from ``detect_events()``.
    person_id : str
        Identifier for titles and filenames.
    show_spike_hc, show_spike_sens : bool
        Toggle high-confidence / sensitive spike layers.
    show_dip_hc, show_dip_sens : bool
        Toggle high-confidence / sensitive dip layers.
    show_baseline : bool
        Show the FFT baseline curve.
    show_event_component : bool
        Show the bottom panel with FFT event-band energy.
    out_dir : str
        Directory for saved PNGs.
    return_figs : bool
        If True, return list of (day, fig) tuples instead of saving.

    Returns
    -------
    list[str] or list[tuple]
        Saved file paths, or (day, fig) tuples if ``return_figs=True``.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style=style, palette=palette, font_scale=1.0)

    # Build the active filter set from toggles
    active_filters = set()
    if show_spike_hc:
        active_filters.add(("up", "high_confidence"))
    if show_spike_sens:
        active_filters.add(("up", "sensitive"))
    if show_dip_hc:
        active_filters.add(("down", "high_confidence"))
    if show_dip_sens:
        active_filters.add(("down", "sensitive"))

    a = annotated_df.copy()
    a[dt_col] = pd.to_datetime(a[dt_col])
    a[day_col] = (
        pd.to_datetime(a[day_col]).dt.date
        if day_col in a.columns
        else a[dt_col].dt.date
    )

    ev = events_df.copy()
    if not ev.empty:
        for c in ("start_time", "end_time", "peak_time"):
            if c in ev.columns:
                ev[c] = pd.to_datetime(ev[c])
        if "Day" in ev.columns:
            ev["Day"] = pd.to_datetime(ev["Day"]).dt.date

    results = []

    for day in sorted(a[day_col].dropna().unique()):
        day_df = a.loc[a[day_col] == day].sort_values(dt_col)
        if day_df.empty:
            continue

        # ── Decide layout: 1 or 2 panels ──
        has_ec = "event_component" in day_df.columns and show_event_component

        if has_ec:
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(16, 7.5),
                height_ratios=[3.5, 1],
                sharex=True,
                gridspec_kw={"hspace": 0.06},
            )
        else:
            fig, ax1 = plt.subplots(figsize=(16, 5))
            ax2 = None

        # ── Raw glucose trace (faint) ──
        ax1.plot(
            day_df[dt_col],
            day_df[glucose_col],
            color="#C0C0C0",
            lw=0.7,
            alpha=0.5,
            zorder=2,
        )

        # ── Denoised glucose (primary trace) ──
        if "denoised" in day_df.columns:
            ax1.plot(
                day_df[dt_col],
                day_df["denoised"],
                color="#1F2937",
                lw=1.5,
                zorder=3,
                label="Glucose (denoised)",
            )
        else:
            ax1.plot(
                day_df[dt_col],
                day_df[glucose_col],
                color="#1F2937",
                lw=1.5,
                zorder=3,
                label="Glucose",
            )

        # ── FFT Baseline ──
        if show_baseline and "baseline_fft" in day_df.columns:
            ax1.plot(
                day_df[dt_col],
                day_df["baseline_fft"],
                color="dodgerblue",
                lw=1.3,
                ls="--",
                alpha=0.6,
                zorder=2,
                label="FFT Baseline",
            )

        # ── Hypo reference line ──
        ax1.axhline(3.9, color="#3B7DD8", ls=":", lw=0.7, alpha=0.35)
        ax1.text(
            day_df[dt_col].iloc[0],
            3.92,
            " Hypo 3.9",
            fontsize=7,
            color="#3B7DD8",
            alpha=0.5,
            va="bottom",
        )

        # ── Event shading with start/end highlighting ──
        legend_added = set()
        n_spikes = 0
        n_dips = 0

        if not ev.empty:
            ev_day = (
                ev.loc[ev["Day"] == day]
                if "Day" in ev.columns
                else ev.loc[ev["start_time"].dt.date == day]
            )

            for _, e in ev_day.iterrows():
                dirn = e.get("direction", "up")
                tier = e.get("tier", "sensitive")
                key = (dirn, tier)

                # Skip if this layer is toggled off
                if key not in active_filters:
                    continue

                if dirn == "up":
                    n_spikes += 1
                else:
                    n_dips += 1

                sty = _VIS_STYLES.get(key, _VIS_STYLES[("up", "sensitive")])

                # ── Shaded region (start → end) ──
                ax1.axvspan(
                    e["start_time"],
                    e["end_time"],
                    color=sty["shade"],
                    alpha=sty["shade_alpha"],
                    zorder=1,
                )

                # ── Start/end boundary lines ──
                for boundary_time, boundary_ls in [
                    (e["start_time"], (0, (4, 2))),   # dashed
                    (e["end_time"], (0, (1, 2))),      # dotted
                ]:
                    ax1.axvline(
                        boundary_time,
                        color=sty["border_color"],
                        ls=boundary_ls,
                        lw=0.9,
                        alpha=0.45,
                        zorder=4,
                    )

                # ── Start/end tick marks on the glucose trace ──
                for btime, bmarker in [
                    (e["start_time"], "|"),
                    (e["end_time"], "|"),
                ]:
                    nearest_idx = (day_df[dt_col] - btime).abs().idxmin()
                    nearest = day_df.loc[nearest_idx]
                    ax1.scatter(
                        nearest[dt_col],
                        nearest[glucose_col],
                        color=sty["border_color"],
                        marker=bmarker,
                        s=80,
                        zorder=7,
                        linewidths=1.5,
                        alpha=0.7,
                    )

                # ── Peak/trough marker + label ──
                if "peak_time" in e:
                    pt = e["peak_time"]
                    nearest_idx = (day_df[dt_col] - pt).abs().idxmin()
                    nearest = day_df.loc[nearest_idx]

                    lbl = sty["label"] if key not in legend_added else None
                    legend_added.add(key)

                    ax1.scatter(
                        nearest[dt_col],
                        nearest[glucose_col],
                        color=sty["marker_color"],
                        marker=sty["marker"],
                        s=sty["marker_size"],
                        zorder=6,
                        edgecolors="white",
                        linewidths=0.8,
                        label=lbl,
                    )

                    # Annotation label — HC gets full label, sensitive gets minimal
                    if tier == "high_confidence":
                        offset_y = -20 if dirn == "down" else 16
                        label_text = (
                            f"{nearest[glucose_col]:.1f} mmol/L\n"
                            f"p={e['p_val']:.2f}"
                        )
                        ax1.annotate(
                            label_text,
                            xy=(nearest[dt_col], nearest[glucose_col]),
                            xytext=(0, offset_y),
                            textcoords="offset points",
                            fontsize=7.5,
                            fontweight="bold",
                            ha="center",
                            color=sty["marker_color"],
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                fc="white",
                                ec=sty["marker_color"],
                                alpha=0.88,
                                lw=0.6,
                            ),
                        )
                    else:
                        # Sensitive: just a small p-value tag
                        offset_y = -14 if dirn == "down" else 12
                        ax1.annotate(
                            f"p={e['p_val']:.2f}",
                            xy=(nearest[dt_col], nearest[glucose_col]),
                            xytext=(0, offset_y),
                            textcoords="offset points",
                            fontsize=6.5,
                            ha="center",
                            color=sty["marker_color"],
                            alpha=0.75,
                        )

        # ── Title with event counts ──
        parts = []
        if n_spikes:
            parts.append(f"{n_spikes} spike{'s' if n_spikes != 1 else ''}")
        if n_dips:
            parts.append(f"{n_dips} dip{'s' if n_dips != 1 else ''}")
        count_str = f"   —   {', '.join(parts)}" if parts else ""

        ax1.set_title(
            f"{person_id}  ·  {day}{count_str}",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax1.set_ylabel("Glucose (mmol/L)", fontsize=10)
        ax1.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            ncol=3,
            handlelength=1.5,
            columnspacing=1.0,
        )
        ax1.grid(True, alpha=0.12)
        sns.despine(ax=ax1, left=False, bottom=False)

        # ── Event component panel ──
        if ax2 is not None and has_ec:
            ec = day_df["event_component"].to_numpy()
            times = day_df[dt_col]

            ax2.fill_between(
                times, ec, 0, where=ec > 0,
                color="#E8453C", alpha=0.25, label="Spike energy",
            )
            ax2.fill_between(
                times, ec, 0, where=ec < 0,
                color="#3B7DD8", alpha=0.25, label="Dip energy",
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
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha="center")

        plt.tight_layout()

        if return_figs:
            results.append((day, fig))
        else:
            out_path = os.path.join(
                out_dir, f"{person_id}_{day}_events.png"
            )
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            results.append(out_path)

    if not return_figs:
        print(f"✓ Saved {len(results)} daily plot(s) to: {out_dir}/")

    return results


# ═══════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY — detect_spikes wrapper
# ═══════════════════════════════════════════════════════════════

def detect_spikes(df, person_id="", **kwargs):
    """
    Backward-compatible wrapper. Calls detect_events with
    direction='up', mode='high_confidence' to match the original
    behaviour of only detecting upward spikes at strict thresholds.

    Returns the same (events_df, annotated_df) tuple. The events_df
    columns are a superset of the original; 'peak_glucose' is now
    'extreme_glucose' and 'baseline_at_peak' is 'baseline_at_extreme'.
    """
    return detect_events(
        df, person_id=person_id,
        mode="high_confidence", direction="up",
        **kwargs,
    )


def visualize_spikes(annotated_df, events_df, person_id, **kwargs):
    """Backward-compatible wrapper for visualize_events (spikes only)."""
    return visualize_events(
        annotated_df, events_df, person_id,
        show_spike_hc=True, show_spike_sens=False,
        show_dip_hc=False, show_dip_sens=False,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════
#  4. CONVENIENCE RUNNER
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    long_df: pd.DataFrame,
    person_id: str,
    *,
    mode: str = "both",
    direction: str = "both",
    show_spike_hc: bool = True,
    show_spike_sens: bool = False,
    show_dip_hc: bool = True,
    show_dip_sens: bool = False,
    show_baseline: bool = True,
    show_event_component: bool = True,
    out_dir: str = "event_plots",
    id_col: str = "person_id",
    **detect_kwargs,
):
    """
    End-to-end: detect → visualise.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format CGM data (output of your preprocessing).
    person_id : str
        Client to analyse.
    mode : str
        "high_confidence", "sensitive", or "both".
    direction : str
        "up", "down", or "both".
    show_spike_hc, show_spike_sens, show_dip_hc, show_dip_sens : bool
        Toggle visibility of each layer in the plots.
    out_dir : str
        Output directory for plots.
    **detect_kwargs
        Forwarded to detect_events().

    Returns
    -------
    events_df, annotated_df, saved_paths
    """
    events_df, annotated_df = detect_events(
        long_df,
        person_id=person_id,
        mode=mode,
        direction=direction,
        id_col=id_col,
        **detect_kwargs,
    )

    saved_paths = visualize_events(
        annotated_df,
        events_df,
        person_id=person_id,
        show_spike_hc=show_spike_hc,
        show_spike_sens=show_spike_sens,
        show_dip_hc=show_dip_hc,
        show_dip_sens=show_dip_sens,
        show_baseline=show_baseline,
        show_event_component=show_event_component,
        out_dir=out_dir,
    )

    return events_df, annotated_df, saved_paths


# ═══════════════════════════════════════════════════════════════
#  BATCH RUNNER — all clients
# ═══════════════════════════════════════════════════════════════

def run_all_events(
    df,
    *,
    id_col="person_id",
    mode="both",
    direction="both",
    output_events_csv="events_all_clients.csv",
    output_annotated_csv="cgm_annotated_all_clients.csv",
    **kwargs,
):
    """Run detection for every person in the DataFrame."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    person_ids = df[id_col].dropna().astype(str).unique()
    all_events, all_annotated = [], []

    for pid in person_ids:
        ev, ann = detect_events(
            df, person_id=pid,
            mode=mode, direction=direction,
            id_col=id_col, **kwargs,
        )
        ev.insert(0, id_col, pid)
        ann[id_col] = pid
        all_events.append(ev)
        all_annotated.append(ann)

    events_all = pd.concat(all_events, ignore_index=True)
    annotated_all = pd.concat(all_annotated, ignore_index=True)

    events_all.to_csv(output_events_csv, index=False)
    annotated_all.to_csv(output_annotated_csv, index=False)

    print(f"✓ Events saved to {output_events_csv}")
    print(f"✓ Annotated traces saved to {output_annotated_csv}")

    return events_all, annotated_all


# ═══════════════════════════════════════════════════════════════
#  MAIN — demo with synthetic data
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    np.random.seed(42)

    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]
    n_times = len(times)  # 288 readings per day

    # Simulate 3 days for one client with spikes AND dips
    rows = []
    for day_offset in range(3):
        base = 5.5 + 0.3 * np.random.randn(n_times)

        # Inject a spike around 08:00–09:30
        spike_centre = 102 + day_offset * 2
        for k in range(-6, 18):
            idx = spike_centre + k
            if 0 <= idx < n_times:
                base[idx] += 3.5 * np.exp(-0.5 * ((k - 4) / 4) ** 2)

        # Inject a second spike around 18:00
        spike2 = 216
        for k in range(-5, 15):
            idx = spike2 + k
            if 0 <= idx < n_times:
                base[idx] += 2.8 * np.exp(-0.5 * ((k - 3) / 3.5) ** 2)

        # Inject a dip around 03:00 (nocturnal hypo)
        dip_centre = 36
        for k in range(-8, 12):
            idx = dip_centre + k
            if 0 <= idx < n_times:
                base[idx] -= 2.0 * np.exp(-0.5 * ((k - 2) / 3) ** 2)

        row_data = {
            "person_id": "DEMO-001",
            "date": pd.Timestamp("2025-03-01") + pd.Timedelta(days=day_offset),
        }
        for i, tc in enumerate(times):
            row_data[tc] = round(float(max(base[i], 2.5)), 2)
        rows.append(row_data)

    wide_df = pd.DataFrame(rows)

    # Reshape to long format (manual, since we're self-contained)
    time_cols = [c for c in wide_df.columns if re.fullmatch(r"\d{2}:\d{2}", c)]
    long_df = wide_df.melt(
        id_vars=["person_id", "date"],
        value_vars=time_cols,
        var_name="time",
        value_name="glucose_mmol_l",
    )
    long_df["datetime"] = pd.to_datetime(
        long_df["date"].astype(str) + " " + long_df["time"]
    )
    long_df = long_df.sort_values(["person_id", "datetime"]).reset_index(drop=True)
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.date

    print("Long-format shape:", long_df.shape)
    print()

    # ── Run full pipeline: all events, all layers visible ──
    events, annotated, paths = run_pipeline(
        long_df,
        person_id="DEMO-001",
        mode="both",
        direction="both",
        show_spike_hc=True,
        show_spike_sens=True,
        show_dip_hc=True,
        show_dip_sens=True,
        out_dir="event_plots",
    )

    print("\n── Detected events ──")
    if not events.empty:
        print(events.to_string(index=False))
    else:
        print("No events detected.")'''
''' 2ND CODE HERE'''
"""
spike_detection.py
==================
Gillvray Health Hackathon — Spike Detection Module

v5: SD-based labelling algorithm (from pipeline v4) wrapped in the
    existing app API. Drop-in replacement for the old spike_detection.py.

PUBLIC API (unchanged):
    detect_events(df, person_id,...) → (events_df, annotated_df)
    visualize_events(annotated_df, events_df, person_id,...) → list[str]
    run_pipeline(long_df, person_id,...) → (events_df, annotated_df, paths)
    run_all_events(df,...) → (events_all, annotated_all)
    detect_spikes(df, person_id,...)   — backward compat
    visualize_spikes(...)               — backward compat
"""

import os
import re
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft, fftfreq, ifft

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
sns.set_context("notebook")


# ═══════════════════════════════════════════════════════════════
#  INTERNAL — Signal Processing (from pipeline v4)
# ═══════════════════════════════════════════════════════════════

def _fft_denoise(
    glucose: np.ndarray,
    sampling_interval_min: float = 5.0,
    cutoff_hours: float = 0.5,
) -> np.ndarray:
    """FFT low-pass filter — removes sensor jitter only."""
    n = len(glucose)
    g = glucose.copy().astype(float)

    nans = np.isnan(g)
    if nans.any():
        not_nan = np.where(~nans)[0]
        if len(not_nan) >= 2:
            g[nans] = np.interp(np.where(nans)[0], not_nan, g[not_nan])
        elif len(not_nan) == 1:
            g[nans] = g[not_nan[0]]
        else:
            g[:] = 0.0

    freqs = fftfreq(n, d=sampling_interval_min / 60.0)
    spectrum = fft(g)
    spectrum[np.abs(freqs) > (1.0 / cutoff_hours)] = 0.0
    return np.real(ifft(spectrum))


def _rolling_median_baseline(
    glucose_denoised: np.ndarray,
    window_hours: float = 5.0,
    sampling_interval_min: float = 5.0,
) -> np.ndarray:
    """Long-window rolling median — stable glycaemic floor."""
    samples = int((window_hours * 60) / sampling_interval_min)
    return (
        pd.Series(glucose_denoised).rolling(window=samples, center=True, min_periods=1).median().values
    )


def _decompose_signal(
    glucose: np.ndarray,
    sample_step_min: float = 5.0,
    fft_cutoff_hours: float = 0.5,
    baseline_window_hours: float = 5.0,
) -> dict:
    """Full signal decomposition → denoised, baseline, deviation, SD, freq/power."""
    n = len(glucose)
    g = glucose.copy().astype(float)

    nans = np.isnan(g)
    if nans.any():
        not_nan = np.where(~nans)[0]
        if len(not_nan) >= 2:
            g[nans] = np.interp(np.where(nans)[0], not_nan, g[not_nan])
        elif len(not_nan) == 1:
            g[nans] = g[not_nan[0]]
        else:
            g[:] = 0.0

    denoised = _fft_denoise(g, sampling_interval_min=sample_step_min,
                            cutoff_hours=fft_cutoff_hours)

    freqs = fftfreq(n, d=sample_step_min / 60.0)
    spectrum = fft(g)
    power = np.abs(spectrum) ** 2

    baseline = _rolling_median_baseline(
        denoised, window_hours=baseline_window_hours,
        sampling_interval_min=sample_step_min,
    )
    deviation = denoised - baseline
    sd = float(np.std(deviation, ddof=1))

    return {
        "denoised": denoised,
        "baseline": baseline,
        "deviation": deviation,
        "sd": sd,
        "g_interpolated": g,
        "freqs": freqs,
        "power": power,
    }


# ═══════════════════════════════════════════════════════════════
#  INTERNAL — SD-Based Spike Labelling (from pipeline v4)
# ═══════════════════════════════════════════════════════════════

def _merge_soft_into_hard(
    labels: np.ndarray,
    soft_label: str,
    hard_label: str,
) -> np.ndarray:
    """Promote soft regions adjacent to hard regions → hard."""
    out = labels.copy()
    n = len(out)

    soft_mask = (out == soft_label).astype(int)
    diff = np.diff(np.concatenate([[0], soft_mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        left_is_hard = (s > 0) and (out[s - 1] == hard_label)
        right_is_hard = (e < n) and (out[e] == hard_label)
        if left_is_hard or right_is_hard:
            out[s:e] = hard_label

    return out


def _label_spikes(deviation: np.ndarray, sd: float) -> np.ndarray:
    """Two-tier SD labelling with contiguous-region merging."""
    labels = np.full(len(deviation), "normal", dtype=object)
    labels[deviation > 1 * sd] = "soft_spike"
    labels[deviation > 2 * sd] = "hard_spike"
    labels[deviation < -1 * sd] = "soft_dip"
    labels[deviation < -2 * sd] = "hard_dip"

    labels = _merge_soft_into_hard(labels, "soft_spike", "hard_spike")
    labels = _merge_soft_into_hard(labels, "soft_dip", "hard_dip")
    return labels


def _filter_labels(
    labels: np.ndarray,
    show_hard_spikes: bool = True,
    show_soft_spikes: bool = True,
    show_hard_dips: bool = True,
    show_soft_dips: bool = True,
) -> np.ndarray:
    """Toggle visibility of spike/dip tiers. Returns filtered copy."""
    out = labels.copy()
    if not show_hard_spikes:
        out[out == "hard_spike"] = "normal"
    if not show_soft_spikes:
        out[out == "soft_spike"] = "normal"
    if not show_hard_dips:
        out[out == "hard_dip"] = "normal"
    if not show_soft_dips:
        out[out == "soft_dip"] = "normal"
    return out


def _find_peaks_in_regions(
    glucose: np.ndarray, labels: np.ndarray, categories: tuple,
) -> np.ndarray:
    """Find the single highest point in each contiguous spike region."""
    mask = np.isin(labels, categories).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return np.array([], dtype=int)
    return np.array(
        [s + np.argmax(glucose[s:e]) for s, e in zip(starts, ends)], dtype=int
    )


def _find_troughs_in_regions(
    glucose: np.ndarray, labels: np.ndarray, categories: tuple,
) -> np.ndarray:
    """Find the single lowest point in each contiguous dip region."""
    mask = np.isin(labels, categories).astype(int)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return np.array([], dtype=int)
    return np.array(
        [s + np.argmin(glucose[s:e]) for s, e in zip(starts, ends)], dtype=int
    )


# ═══════════════════════════════════════════════════════════════
#  INTERNAL — Event Extraction from SD Labels
# ═══════════════════════════════════════════════════════════════

# Maps SD labels → (direction, tier) to match the old API's schema
_LABEL_TO_DIRECTION_TIER = {
    "hard_spike": ("up", "high_confidence"),
    "soft_spike": ("up", "sensitive"),
    "hard_dip": ("down", "high_confidence"),
    "soft_dip": ("down", "sensitive"),
}


def _extract_events(
    datetimes: np.ndarray,
    glucose: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    day_date,
) -> pd.DataFrame:
    """
    Extract structured event records from SD-labelled regions.

    Returns DataFrame with columns matching the OLD API:
        Day, start_time, end_time, peak_time, p_val, range,
        extreme_glucose, baseline_at_extreme, closure_reason,
        direction, tier, spike_label
    """
    events = []

    for tier_label, (direction, tier_name) in _LABEL_TO_DIRECTION_TIER.items():
        mask = (labels == tier_label).astype(int)
        diff = np.diff(np.concatenate([[0], mask, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            region_glucose = glucose[s:e]
            region_baseline = baseline[s:e]

            if direction == "up":
                extreme_idx = s + np.argmax(region_glucose)
            else:
                extreme_idx = s + np.argmin(region_glucose)

            extreme_val = float(glucose[extreme_idx])
            baseline_val = float(baseline[extreme_idx])
            amplitude = abs(extreme_val - baseline_val)

            # Compute a pseudo p_val for backward compatibility:
            # Scale amplitude relative to the day's SD.
            # Hard events (>2 SD) get p_val ≥ 0.7; soft (1-2 SD) get 0.3-0.7
            dev_at_extreme = abs(glucose[extreme_idx] - baseline[extreme_idx])
            day_sd = np.std(glucose - baseline, ddof=1)
            if day_sd > 0:
                sd_ratio = dev_at_extreme / day_sd
                p_val = float(np.clip(sd_ratio / 4.0, 0.0, 1.0))
            else:
                p_val = 0.5

            event_range = float(np.max(region_glucose) - np.min(region_glucose))

            events.append({
                "Day": day_date,
                "start_time": pd.to_datetime(datetimes[s]),
                "end_time": pd.to_datetime(datetimes[e - 1]),
                "peak_time": pd.to_datetime(datetimes[extreme_idx]),
                "p_val": round(p_val, 4),
                "range": round(event_range, 3),
                "extreme_glucose": round(extreme_val, 2),
                "baseline_at_extreme": round(baseline_val, 2),
                "closure_reason": "sd_label",
                "direction": direction,
                "tier": tier_name,
                "spike_label": tier_label,
            })

    if not events:
        return pd.DataFrame()

    return pd.DataFrame(events).sort_values("start_time").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  INTERNAL — Colour Scheme
# ═══════════════════════════════════════════════════════════════

SPIKE_COLOURS = {
    "hard_spike": "#D32F2F",
    "soft_spike": "#F57C00",
    "hard_dip": "#0D47A1",
    "soft_dip": "#42A5F5",
}

SPIKE_ALPHAS = {
    "hard_spike": 0.45,
    "soft_spike": 0.35,
    "hard_dip": 0.45,
    "soft_dip": 0.35,
}

# Old-API style mapping for visualize_events compatibility
_VIS_STYLES = {
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


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API — detect_events()
# ═══════════════════════════════════════════════════════════════

def detect_events(
    df: pd.DataFrame,
    person_id: str = "",
    glucose_col: str = "glucose_mmol_l",
    dt_col: str = "datetime",
    id_col: str = "person_id",
    date_col: str = "date",
    # ── What to detect ──
    mode: str = "both",
    direction: str = "both",
    # ── Signal processing parameters ──
    sample_step_min: float = 5.0,
    fft_cutoff_hours: float = 0.5,
    baseline_window_hours: float = 5.0,
    # ── Accepted but ignored (backward compat with old callers) ──
    baseline_cutoff_hours: float = None,
    noise_cutoff_min: float = None,
    w_amp: float = None,
    w_slope: float = None,
    w_dur: float = None,
    w_return: float = None,
    w_snr: float = None,
    expected_noise_mad_floor: float = None,
    **_ignored_kwargs,
) -> tuple:
    """
    Detect glucose spike and/or dip events for one client.

    Uses SD-based labelling (v4 algorithm) internally.
    Returns the same (events_df, annotated_df) tuple as the old API.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format CGM data.
    person_id : str
        The person_id to analyse.
    mode : str
        "high_confidence", "sensitive", or "both".
    direction : str
        "up" (spikes), "down" (dips), or "both".
    sample_step_min : float
        Sampling interval in minutes.
    fft_cutoff_hours : float
        FFT low-pass cutoff for denoising.
    baseline_window_hours : float
        Rolling median window for baseline.

    Returns
    -------
    events_df : pd.DataFrame
        One row per detected event. Columns:
        Day, start_time, end_time, peak_time, p_val, range,
        extreme_glucose, baseline_at_extreme, closure_reason,
        direction, tier, spike_label
    annotated_df : pd.DataFrame
        Full trace with added columns:
        denoised, baseline_fft, event_component, deviation, sd,
        spike_label, slope, residual, event_id, in_event,
        is_event_start, is_event_end, is_event_extreme,
        p_val_event, event_direction, event_tier
    """
    EVENT_COLS = [
        "Day", "start_time", "end_time", "peak_time", "p_val",
        "range", "extreme_glucose", "baseline_at_extreme",
        "closure_reason", "direction", "tier", "spike_label",
    ]

    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' not found. Available: {df.columns.tolist()}")

    d = df.loc[df[id_col].astype(str) == str(person_id)].copy()
    if d.empty:
        return pd.DataFrame(columns=EVENT_COLS), pd.DataFrame()

    d[dt_col] = pd.to_datetime(d[dt_col])
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col]).dt.date
    else:
        d[date_col] = d[dt_col].dt.date

    d = d.sort_values(dt_col).dropna(subset=[glucose_col]).reset_index(drop=True)

    g = d[glucose_col].to_numpy(dtype=float)
    n = len(g)

    # ── Signal decomposition ──
    sig = _decompose_signal(g, sample_step_min, fft_cutoff_hours,
                            baseline_window_hours)
    denoised = sig["denoised"]
    baseline = sig["baseline"]
    deviation = sig["deviation"]
    sd = sig["sd"]
    labels = _label_spikes(deviation, sd)

    # ── Filter labels by mode and direction ──
    show_hard_spikes = direction in ("up", "both") and mode in ("high_confidence", "both")
    show_soft_spikes = direction in ("up", "both") and mode in ("sensitive", "both")
    show_hard_dips = direction in ("down", "both") and mode in ("high_confidence", "both")
    show_soft_dips = direction in ("down", "both") and mode in ("sensitive", "both")

    filtered = _filter_labels(
        labels,
        show_hard_spikes=show_hard_spikes,
        show_soft_spikes=show_soft_spikes,
        show_hard_dips=show_hard_dips,
        show_soft_dips=show_soft_dips,
    )

    # ── Extract events per day ──
    all_events = []
    for day in sorted(d[date_col].unique()):
        day_mask = d[date_col] == day
        day_data = d.loc[day_mask]
        day_idx = day_data.index

        day_events = _extract_events(
            datetimes=day_data[dt_col].values,
            glucose=denoised[day_idx],
            baseline=baseline[day_idx],
            labels=filtered[day_idx],
            day_date=day,
        )
        if not day_events.empty:
            all_events.append(day_events)

    events_df = (
        pd.concat(all_events, ignore_index=True) if all_events
        else pd.DataFrame(columns=EVENT_COLS)
    )

    # ── Build annotated trace (backward-compatible columns) ──
    rise_n = int(20 * (1.0 / sample_step_min))
    slope = np.full(n, np.nan)
    for i in range(rise_n, n):
        slope[i] = (denoised[i] - denoised[i - rise_n]) / 20.0

    annotated = d.copy()
    # New algorithm columns
    annotated["denoised"] = denoised
    annotated["baseline"] = baseline
    annotated["deviation"] = deviation
    annotated["sd"] = sd
    annotated["spike_label"] = filtered

    # Old API columns (for backward compat with app code)
    annotated["baseline_fft"] = baseline
    annotated["event_component"] = deviation  # closest equivalent
    annotated["slope"] = slope
    annotated["residual"] = g - baseline

    # Event annotation columns (old API)
    annotated["event_id"] = pd.array([pd.NA] * n, dtype="Int64")
    annotated["in_event"] = False
    annotated["is_event_start"] = False
    annotated["is_event_end"] = False
    annotated["is_event_extreme"] = False
    annotated["p_val_event"] = np.nan
    annotated["event_direction"] = ""
    annotated["event_tier"] = ""

    if not events_df.empty:
        for eid, row in events_df.iterrows():
            st = pd.to_datetime(row["start_time"])
            et = pd.to_datetime(row["end_time"])
            pt = pd.to_datetime(row["peak_time"])

            mask = (annotated[dt_col] >= st) & (annotated[dt_col] <= et)
            annotated.loc[mask, "event_id"] = eid
            annotated.loc[mask, "in_event"] = True
            annotated.loc[mask, "p_val_event"] = row["p_val"]
            annotated.loc[mask, "event_direction"] = row["direction"]
            annotated.loc[mask, "event_tier"] = row["tier"]

            annotated.loc[
                mask & (annotated[dt_col] == st), "is_event_start"
            ] = True
            annotated.loc[
                mask & (annotated[dt_col] == et), "is_event_end"
            ] = True

            extreme_mask = mask & (
                (annotated[dt_col] - pt).abs()
                == (annotated[dt_col] - pt).abs().min()
            )
            annotated.loc[extreme_mask, "is_event_extreme"] = True

    return events_df, annotated


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API — visualize_events()
# ═══════════════════════════════════════════════════════════════

def visualize_events(
    annotated_df: pd.DataFrame,
    events_df: pd.DataFrame,
    person_id: str,
    *,
    show_spike_hc: bool = True,
    show_spike_sens: bool = True,
    show_dip_hc: bool = True,
    show_dip_sens: bool = True,
    show_baseline: bool = True,
    show_event_component: bool = False,
    out_dir: str = "event_plots",
    glucose_col: str = "glucose_mmol_l",
    dt_col: str = "datetime",
    day_col: str = "date",
    dpi: int = 150,
    style: str = "whitegrid",
    palette: str = "deep",
    return_figs: bool = False,
) -> list:
    """
    Create polished daily CGM plots with spike/dip annotations.

    Uses the v4 fill_between shading style with glucose-value
    annotations. Same function signature as the old API.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style=style, palette=palette, font_scale=1.0)

    a = annotated_df.copy()
    a[dt_col] = pd.to_datetime(a[dt_col])
    if day_col in a.columns:
        a[day_col] = pd.to_datetime(a[day_col]).dt.date
    else:
        a[day_col] = a[dt_col].dt.date

    # Ensure spike_label exists
    if "spike_label" not in a.columns:
        a["spike_label"] = "normal"

    # Ensure denoised exists
    denoised_col = "denoised" if "denoised" in a.columns else glucose_col
    baseline_col = (
        "baseline" if "baseline" in a.columns
        else "baseline_fft" if "baseline_fft" in a.columns
        else None
    )

    results = []

    for day in sorted(a[day_col].dropna().unique()):
        day_df = a.loc[a[day_col] == day].sort_values(dt_col).reset_index(drop=True)
        if day_df.empty:
            continue

        times = day_df[dt_col].values
        glucose = day_df[denoised_col].values
        baseline = day_df[baseline_col].values if baseline_col else None
        labels = day_df["spike_label"].values
        sd_val = day_df["sd"].iloc[0] if "sd" in day_df.columns else 0.0

        # Apply visibility toggles
        visible = _filter_labels(
            labels,
            show_hard_spikes=show_spike_hc,
            show_soft_spikes=show_spike_sens,
            show_hard_dips=show_dip_hc,
            show_soft_dips=show_dip_sens,
        )

        fig, ax = plt.subplots(figsize=(16, 6))

        # ── Spike/dip region shading (fill_between) ──
        if baseline is not None:
            for category in ["soft_spike", "hard_spike", "soft_dip", "hard_dip"]:
                colour = SPIKE_COLOURS[category]
                alpha = SPIKE_ALPHAS[category]
                cat_mask = visible == category
                if not cat_mask.any():
                    continue
                ax.fill_between(
                    times, glucose, baseline,
                    where=cat_mask, color=colour, alpha=alpha,
                    linewidth=0, zorder=2,
                )

        # ── Baseline ──
        if show_baseline and baseline is not None:
            ax.plot(times, baseline, color="#7F8C8D", lw=1.6, ls="--",
                    label="Baseline", zorder=3)

        # ── Glucose trace ──
        ax.plot(times, glucose, color="#1C1C1C", lw=2.0,
                label="Glucose (denoised)", zorder=5)

        # ── Hypo line ──
        ax.axhline(3.9, color="#0D47A1", ls=":", lw=0.8, alpha=0.4)
        ax.text(times[0], 3.92, " Hypo 3.9",
                fontsize=7, color="#0D47A1", alpha=0.6, va="bottom")

        # ── Find peaks and troughs ──
        spike_cats = tuple(
            c for c, show in [
                ("hard_spike", show_spike_hc),
                ("soft_spike", show_spike_sens),
            ] if show
        )
        dip_cats = tuple(
            c for c, show in [
                ("hard_dip", show_dip_hc),
                ("soft_dip", show_dip_sens),
            ] if show
        )

        all_peak_idxs = (
            _find_peaks_in_regions(glucose, visible, spike_cats)
            if spike_cats else np.array([], dtype=int)
        )
        all_trough_idxs = (
            _find_troughs_in_regions(glucose, visible, dip_cats)
            if dip_cats else np.array([], dtype=int)
        )

        global_highest_idx = (
            all_peak_idxs[np.argmax(glucose[all_peak_idxs])]
            if len(all_peak_idxs) > 0 else None
        )
        global_lowest_idx = (
            all_trough_idxs[np.argmin(glucose[all_trough_idxs])]
            if len(all_trough_idxs) > 0 else None
        )

        # ── Annotate peaks ──
        for idx in all_peak_idxs:
            colour = SPIKE_COLOURS.get(visible[idx], "#D32F2F")
            is_global = (idx == global_highest_idx)
            marker_size = 130 if is_global else 80
            font_size = 10 if is_global else 8
            edge_width = 2.0 if is_global else 1.0

            ax.scatter(
                times[idx], glucose[idx],
                color=colour, s=marker_size, zorder=7, marker="^",
                edgecolors="white", linewidths=edge_width,
            )

            label_text = f"▲ {glucose[idx]:.1f}" if is_global else f"{glucose[idx]:.1f}"
            offset_y = 18 if is_global else 14

            ax.annotate(
                label_text,
                xy=(times[idx], glucose[idx]),
                xytext=(0, offset_y),
                textcoords="offset points",
                ha="center", fontsize=font_size, fontweight="bold",
                color=colour,
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white",
                    ec=colour, alpha=0.92,
                    lw=1.2 if is_global else 0.6,
                ),
            )

        # ── Annotate troughs ──
        for idx in all_trough_idxs:
            colour = SPIKE_COLOURS.get(visible[idx], "#0D47A1")
            is_global = (idx == global_lowest_idx)
            marker_size = 130 if is_global else 80
            font_size = 10 if is_global else 8
            edge_width = 2.0 if is_global else 1.0

            ax.scatter(
                times[idx], glucose[idx],
                color=colour, s=marker_size, zorder=7, marker="v",
                edgecolors="white", linewidths=edge_width,
            )

            label_text = f"▼ {glucose[idx]:.1f}" if is_global else f"{glucose[idx]:.1f}"
            offset_y = -20 if is_global else -16

            ax.annotate(
                label_text,
                xy=(times[idx], glucose[idx]),
                xytext=(0, offset_y),
                textcoords="offset points",
                ha="center", fontsize=font_size, fontweight="bold",
                color=colour,
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white",
                    ec=colour, alpha=0.92,
                    lw=1.2 if is_global else 0.6,
                ),
            )

        # ── Legend ──
        legend_elements = [
            mpatches.Patch(color="#1C1C1C", label="Glucose (denoised)"),
            mpatches.Patch(color="#7F8C8D", label="Baseline"),
        ]
        if show_spike_hc:
            legend_elements.append(
                mpatches.Patch(color="#D32F2F", alpha=0.7, label="Hard Spike (>2 SD)")
            )
        if show_spike_sens:
            legend_elements.append(
                mpatches.Patch(color="#F57C00", alpha=0.7, label="Soft Spike (1–2 SD)")
            )
        if show_dip_hc:
            legend_elements.append(
                mpatches.Patch(color="#0D47A1", alpha=0.7, label="Hard Dip (>2 SD)")
            )
        if show_dip_sens:
            legend_elements.append(
                mpatches.Patch(color="#42A5F5", alpha=0.7, label="Soft Dip (1–2 SD)")
            )

        ax.legend(
            handles=legend_elements, loc="upper right",
            fontsize=9, framealpha=0.95, edgecolor="#CCCCCC",
            fancybox=True, shadow=False,
        )

        # ── Formatting ──
        n_spikes = len(all_peak_idxs)
        n_dips = len(all_trough_idxs)

        ax.set_xlabel("Time of Day", fontsize=11)
        ax.set_ylabel("Glucose (mmol/L)", fontsize=11)
        ax.set_title(
            f"{person_id}  ·  {day}    "
            f"{n_spikes} spike(s), {n_dips} dip(s)",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.grid(axis="x", linestyle=":", alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()

        if return_figs:
            results.append((day, fig))
        else:
            out_path = os.path.join(out_dir, f"{person_id}_{day}_events.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            results.append(out_path)

    if not return_figs:
        print(f"✓ Saved {len(results)} daily plot(s) to: {out_dir}/")

    return results


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API — run_pipeline()
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    long_df: pd.DataFrame,
    person_id: str,
    *,
    mode: str = "both",
    direction: str = "both",
    show_spike_hc: bool = True,
    show_spike_sens: bool = True,
    show_dip_hc: bool = True,
    show_dip_sens: bool = True,
    show_baseline: bool = True,
    show_event_component: bool = False,
    out_dir: str = "event_plots",
    id_col: str = "person_id",
    return_figs: bool = False,
    **detect_kwargs,
) -> tuple:
    """
    End-to-end: detect → visualise.
    Same signature as old API.

    Returns
    -------
    events_df, annotated_df, saved_paths
    """
    events_df, annotated_df = detect_events(
        long_df,
        person_id=person_id,
        mode=mode,
        direction=direction,
        id_col=id_col,
        **detect_kwargs,
    )

    saved_paths = visualize_events(
        annotated_df,
        events_df,
        person_id=person_id,
        show_spike_hc=show_spike_hc,
        show_spike_sens=show_spike_sens,
        show_dip_hc=show_dip_hc,
        show_dip_sens=show_dip_sens,
        show_baseline=show_baseline,
        show_event_component=show_event_component,
        out_dir=out_dir,
        return_figs=return_figs,
    )

    return events_df, annotated_df, saved_paths


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API — run_all_events()
# ═══════════════════════════════════════════════════════════════

def run_all_events(
    df,
    *,
    id_col="person_id",
    mode="both",
    direction="both",
    output_events_csv="events_all_clients.csv",
    output_annotated_csv="cgm_annotated_all_clients.csv",
    **kwargs,
):
    """Run detection for every person in the DataFrame."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    person_ids = df[id_col].dropna().astype(str).unique()
    all_events, all_annotated = [], []

    for pid in person_ids:
        ev, ann = detect_events(
            df, person_id=pid,
            mode=mode, direction=direction,
            id_col=id_col, **kwargs,
        )
        if not ev.empty:
            ev.insert(0, id_col, pid)
        ann[id_col] = pid
        all_events.append(ev)
        all_annotated.append(ann)

    events_all = pd.concat(all_events, ignore_index=True)
    annotated_all = pd.concat(all_annotated, ignore_index=True)

    events_all.to_csv(output_events_csv, index=False)
    annotated_all.to_csv(output_annotated_csv, index=False)

    print(f"✓ Events saved to {output_events_csv}")
    print(f"✓ Annotated traces saved to {output_annotated_csv}")

    return events_all, annotated_all


# ═══════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY WRAPPERS
# ═══════════════════════════════════════════════════════════════

def detect_spikes(df, person_id="", **kwargs):
    """Backward-compatible wrapper — spikes only, high confidence."""
    return detect_events(
        df, person_id=person_id,
        mode="high_confidence", direction="up",
        **kwargs,
    )


def visualize_spikes(annotated_df, events_df, person_id, **kwargs):
    """Backward-compatible wrapper — spikes only."""
    return visualize_events(
        annotated_df, events_df, person_id,
        show_spike_hc=True, show_spike_sens=False,
        show_dip_hc=False, show_dip_sens=False,
        **kwargs,
    )
