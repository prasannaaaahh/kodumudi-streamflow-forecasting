"""Data loading, validation and temporal aggregation."""
from pathlib import Path

import pandas as pd

# Repository root = two levels above src/kodumudi/
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "kodumudi_daily.csv"

PARAMS = ["rainfall_mm", "discharge_m3s", "temperature_c", "soil_moisture_kgm2"]
LABELS = {
    "rainfall_mm": "Rainfall (mm)",
    "discharge_m3s": "Discharge (m³/s)",
    "temperature_c": "Temperature (°C)",
    "soil_moisture_kgm2": "Soil moisture (kg/m²)",
}

# Chronological split on *target months* (shared by the ARIMA and ML notebooks
# so that every model is scored on exactly the same test months).
VAL_START = "2008-10-01"   # validation: 2008-10 .. 2012-04
TEST_START = "2012-05-01"  # test:       2012-05 .. 2015-12

# Rainfall accumulates over a period (sum); the others are state variables (mean).
_AGG = {
    "rainfall_mm": "sum",
    "discharge_m3s": "mean",
    "temperature_c": "mean",
    "soil_moisture_kgm2": "mean",
}


def load_daily(path=DATA_PATH) -> pd.DataFrame:
    """Load the daily CSV with a validated, gap-free DatetimeIndex."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    df = df[PARAMS]
    assert df.index.is_unique, "duplicate dates found"
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    assert len(full) == len(df), "missing calendar days found"
    df.index.freq = "D"
    return df


def aggregate(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate daily data. rule: 'MS' (monthly) or 'YS' (annual)."""
    return df.resample(rule).agg(_AGG).dropna()


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-variable quality summary."""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "variable": c,
            "missing": int(s.isna().sum()),
            "min": s.min(), "median": s.median(), "max": s.max(),
            "negative_values": int((s < 0).sum()),
            "zero_share_%": round(100 * (s == 0).mean(), 1),
            "skew": round(s.skew(), 2),
        })
    return pd.DataFrame(rows).set_index("variable")
