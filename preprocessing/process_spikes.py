"""
Fixed SPIKES processing script with numeric indexing

Outputs:
1. patient_metadata.csv  — one row per patient (24 patients)
2. cgm_matrix.csv        — CGM data with time columns 00:00–23:55 in 5-min intervals
3. patient_reported_metrics.csv — patient-reported gauge data per day
"""

import pandas as pd
from pathlib import Path
import datetime

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "../data/raw_spikes.xlsx"   # adjust if needed
OUTPUT_DIR = BASE_DIR / "../data/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Read labelling rows (used only for output column names, not for indexing) ─
#    Row index 1 (skiprows=1) → readable column headers
#    Row index 3 (skiprows=3) → 5-min time labels for CGM (00:00, 00:05, …)
header_row1 = pd.read_excel(INPUT_FILE, sheet_name=0, skiprows=1, nrows=1, header=None).iloc[0].tolist()
header_row3 = pd.read_excel(INPUT_FILE, sheet_name=0, skiprows=3, nrows=1, header=None).iloc[0].tolist()

# ── Read data rows ─────────────────────────────────────────────────────────────
# IMPORTANT: do NOT reassign df.columns from any header row.
# Rows 0–4 of the workbook contain multiple header/metadata rows whose first 5
# columns are all blank; assigning them as column names creates duplicate NaN
# labels that break label-based operations (drop_duplicates, dropna, etc.).
# Keeping pandas' default integer labels (0, 1, 2, …) means iloc and
# label-based operations refer to the same column safely.
df = pd.read_excel(INPUT_FILE, sheet_name=0, skiprows=5, header=None)

# ── Column positions (0-indexed) ──────────────────────────────────────────────
PATIENT_ID_COL = 0   # 'Client ID'
DAY_NUM_COL    = 2   # Day number 1–14  (was wrongly set to 5 = Data Import Key)

# ── Define data slices ────────────────────────────────────────────────────────
PATIENT_META_COLS     = [0] + list(range(6, 22))  # Client ID, then Age → Health Goal
                                                  # Skips cols 1–5: Day_Label, Day, Date, Weekday, Data Import Key
CGM_COLS              = slice(22, 310)     # 288 × 5-min CGM readings, 00:00–23:55 ✓
PATIENT_REPORTED_COLS = slice(1252, 1265)  # Health Gauge → Negative Gauge (13 columns) ✓

# ── Processing ────────────────────────────────────────────────────────────────

# Fill down patient IDs — only populated on the first day row of each patient
df.iloc[:, PATIENT_ID_COL] = df.iloc[:, PATIENT_ID_COL].ffill()

# 1. Patient metadata — one row per patient ───────────────────────────────────
patient_meta = df.drop_duplicates(subset=[PATIENT_ID_COL], keep='first')
patient_meta = patient_meta.iloc[:, PATIENT_META_COLS].copy()

meta_col_names = [header_row1[i] for i in PATIENT_META_COLS]
patient_meta.columns = meta_col_names
patient_meta.to_csv(OUTPUT_DIR / 'patient_metadata.csv', index=False)

# 2. CGM matrix — all rows, time columns 00:00–23:55 ─────────────────────────
cgm_data = df.iloc[:, CGM_COLS].copy()

if len(cgm_data.columns) == 288:
    start = datetime.datetime.strptime("00:00", "%H:%M")
    times = [(start + datetime.timedelta(minutes=5 * i)).strftime("%H:%M") for i in range(288)]
    cgm_data.columns = times
else:
    print(f"Warning: CGM column count is {len(cgm_data.columns)}, expected 288. Keeping numeric names.")

cgm_data.insert(0, 'Client ID', df.iloc[:, PATIENT_ID_COL].values)
cgm_data.insert(1, 'Day',        df.iloc[:, DAY_NUM_COL].values)
cgm_data.to_csv(OUTPUT_DIR / 'cgm_matrix.csv', index=False)

# 3. Patient-reported metrics — all rows ──────────────────────────────────────
patient_reported = df.iloc[:, PATIENT_REPORTED_COLS].copy()
patient_reported.columns = header_row1[1252:1265]

patient_reported.insert(0, 'Client ID', df.iloc[:, PATIENT_ID_COL].values)
patient_reported.to_csv(OUTPUT_DIR / 'patient_reported_metrics.csv', index=False)

print("Done — saved 3 CSV files to", OUTPUT_DIR)
print(f"  patient_metadata.csv          : {len(patient_meta)} rows × {len(patient_meta.columns)} cols")
print(f"  cgm_matrix.csv                : {len(cgm_data)} rows × {len(cgm_data.columns)} cols")
print(f"  patient_reported_metrics.csv  : {len(patient_reported)} rows × {len(patient_reported.columns)} cols")