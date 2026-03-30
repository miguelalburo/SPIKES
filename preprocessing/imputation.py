"""Demo imputation script for SPIKES output

Inputs (CSV):
- spikes_output/cgm_matrix.csv
- spikes_output/patient_metadata.csv
- spikes_output/patient_reported_metrics.csv

Outputs (CSV) in imputed/ folder:
- cgm_matrix_imputed.csv
- patient_metadata_imputed.csv
- patient_reported_metrics_imputed.csv

Imputation strategy: numeric columns -> median; categorical -> mode; forward/backfill for per-day CGM / patient reports; no heavy external libs needed beyond pandas.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR.parent / "imputed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CGM_FILE = INPUT_DIR / "cgm_matrix.csv"
META_FILE = INPUT_DIR / "patient_metadata.csv"
PRM_FILE = INPUT_DIR / "patient_reported_metrics.csv"

print(f"Loading {CGM_FILE}")
cgm = pd.read_csv(CGM_FILE)
print(f"Loading {META_FILE}")
meta = pd.read_csv(META_FILE)
print(f"Loading {PRM_FILE}")
prm = pd.read_csv(PRM_FILE)


def impute_df(df, key_cols=None, strategy=None):
    """Impute missing values using one of configured strategies.

    Parameters:
      df: pandas DataFrame
      key_cols: list of id columns to keep untouched during strategy-based fill
      strategy: dict with keys 'numeric' and 'categorical', values:
        - 'median', 'mean', 'zero', 'forward', 'backward' for numeric
        - 'mode', 'constant', 'ffill', 'bfill' for categorical

    Returns:
      DataFrame with imputed values.
    """
    if key_cols is None:
        key_cols = []
    if strategy is None:
        strategy = {"numeric": "median", "categorical": "mode"}

    result = df.copy()

    # numeric columns (exclude key columns)
    numeric_cols = result.select_dtypes(include=["number"]).columns.difference(key_cols)
    for col in numeric_cols:
        if strategy.get("numeric") == "median":
            fill_value = result[col].median(skipna=True)
        elif strategy.get("numeric") == "mean":
            fill_value = result[col].mean(skipna=True)
        elif strategy.get("numeric") == "zero":
            fill_value = 0
        elif strategy.get("numeric") == "forward":
            result[col] = result[col].ffill()
            continue
        elif strategy.get("numeric") == "backward":
            result[col] = result[col].bfill()
            continue
        else:
            raise ValueError(f"Unknown numeric strategy '{strategy.get('numeric')}'")

        result[col] = result[col].fillna(fill_value)

    # categorical columns
    cat_cols = result.select_dtypes(include=["object", "category"]).columns.difference(key_cols)
    for col in cat_cols:
        if strategy.get("categorical") == "mode":
            mode_vals = result[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "NA"
            result[col] = result[col].fillna(fill_value)
        elif strategy.get("categorical") == "constant":
            result[col] = result[col].fillna("NA")
        elif strategy.get("categorical") == "ffill":
            result[col] = result[col].ffill()
        elif strategy.get("categorical") == "bfill":
            result[col] = result[col].bfill()
        else:
            raise ValueError(f"Unknown categorical strategy '{strategy.get('categorical')}'")

    return result

print("Imputing patient_metadata")
meta_strategy = {"numeric": "median", "categorical": "mode"}
meta_imputed = impute_df(meta, key_cols=["Patient_ID"] if "Patient_ID" in meta.columns else [], strategy=meta_strategy)
meta_out = OUTPUT_DIR / "patient_metadata_imputed.csv"
meta_imputed.to_csv(meta_out, index=False)
print(f"Wrote {meta_out}")

print("Imputing cgm_matrix")
if "Patient_ID" in cgm.columns and "Day" in cgm.columns:
    key_cols = ["Patient_ID", "Day"]
else:
    key_cols = []

cgm_imputed = cgm.copy()
if key_cols and set(key_cols).issubset(cgm_imputed.columns):
    cgm_imputed = cgm_imputed.groupby(key_cols).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
else:
    cgm_imputed = cgm_imputed.ffill().bfill()

# choose CGM-specific imputation strategy; can be changed quickly here
cgm_strategy = {"numeric": "median", "categorical": "ffill"}
cgm_imputed = impute_df(cgm_imputed, key_cols=key_cols, strategy=cgm_strategy)

for col in key_cols:
    if col in cgm_imputed.columns:
        cgm_imputed[col] = cgm_imputed[col].fillna(method="ffill").fillna(method="bfill")

cgm_out = OUTPUT_DIR / "cgm_matrix_imputed.csv"
cgm_imputed.to_csv(cgm_out, index=False)
print(f"Wrote {cgm_out}")

print("Imputing patient_reported_metrics")
prm_imputed = prm.copy()
if key_cols and set(key_cols).issubset(prm_imputed.columns):
    prm_imputed = prm_imputed.groupby(key_cols).apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
else:
    prm_imputed = prm_imputed.ffill().bfill()

# choose patient-reported strategy here
prm_strategy = {"numeric": "mean", "categorical": "mode"}
prm_imputed = impute_df(prm_imputed, key_cols=key_cols, strategy=prm_strategy)
prm_out = OUTPUT_DIR / "patient_reported_metrics_imputed.csv"
prm_imputed.to_csv(prm_out, index=False)
print(f"Wrote {prm_out}")

print("Imputation complete.")