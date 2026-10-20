"""Leakage-safe feature engineering for one-month-ahead discharge forecasting.

Forecast origin = month t. Every feature uses information from month t or
earlier. The target is mean discharge in month t+1.
"""
import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    "discharge_memory": ["q_t", "q_t-1", "q_t-2", "q_t-5", "q_t-11"],
    "rainfall": ["rain_t", "rain_t-1", "rain_t-2", "rain_3m_sum"],
    "catchment_state": ["soil_t", "temp_t"],
    "seasonality": ["target_month_sin", "target_month_cos"],
}


def build_supervised(monthly: pd.DataFrame):
    """Return (X, y) indexed by TARGET month (t+1).

    q_t-11 is the discharge in the same calendar month as the target, one year
    earlier (a seasonal-naive signal).
    """
    q = monthly["discharge_m3s"]
    r = monthly["rainfall_mm"]
    X = pd.DataFrame(index=monthly.index)
    X["q_t"] = q
    X["q_t-1"] = q.shift(1)
    X["q_t-2"] = q.shift(2)
    X["q_t-5"] = q.shift(5)
    X["q_t-11"] = q.shift(11)
    X["rain_t"] = r
    X["rain_t-1"] = r.shift(1)
    X["rain_t-2"] = r.shift(2)
    X["rain_3m_sum"] = r.rolling(3).sum()
    X["soil_t"] = monthly["soil_moisture_kgm2"]
    X["temp_t"] = monthly["temperature_c"]

    target_month = (monthly.index.month % 12) + 1          # month of t+1
    X["target_month_sin"] = np.sin(2 * np.pi * target_month / 12)
    X["target_month_cos"] = np.cos(2 * np.pi * target_month / 12)

    y = q.shift(-1)                                         # discharge at t+1
    # re-index by target month so train/val/test splits are on forecast dates
    X.index = X.index + pd.offsets.MonthBegin(1)
    y.index = y.index + pd.offsets.MonthBegin(1)
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    return data.drop(columns="y"), data["y"]
