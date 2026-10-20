"""Pure plotting/metric helpers for the dashboard (no Streamlit imports -> unit-testable)."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

COLORS = {"rainfall_mm": "#1f77b4", "discharge_m3s": "#008080",
          "temperature_c": "#d62728", "soil_moisture_kgm2": "#8b4513"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def kpis(daily: pd.DataFrame) -> dict:
    q = daily["discharge_m3s"]
    annual_rain = daily["rainfall_mm"].resample("YS").sum()
    return {
        "Mean discharge (m³/s)": q.mean(),
        "Q10 high flow (m³/s)": float(np.percentile(q, 90)),
        "Q50 median flow (m³/s)": float(np.percentile(q, 50)),
        "Q90 low flow (m³/s)": float(np.percentile(q, 10)),
        "Peak daily flow (m³/s)": q.max(),
        "Peak date": q.idxmax().date().isoformat(),
        "Mean annual rainfall (mm)": annual_rain.mean(),
    }


def timeseries_fig(df: pd.DataFrame, var: str, label: str, smooth: int = 0):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(df.index, df[var], color=COLORS[var], lw=1, alpha=.8 if smooth else 1, label=label)
    if smooth:
        ax.plot(df.index, df[var].rolling(smooth, center=True).mean(), color="black", lw=2, label=f"{smooth}-step rolling mean")
        ax.legend()
    ax.set_ylabel(label); ax.grid(alpha=.4)
    fig.tight_layout()
    return fig


def climatology_fig(monthly: pd.DataFrame, var: str, label: str):
    g = monthly.groupby(monthly.index.month)[var]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(range(1, 13), g.mean(), marker="o", color=COLORS[var], lw=2, label="mean")
    ax.fill_between(range(1, 13), g.quantile(.25), g.quantile(.75), color=COLORS[var], alpha=.2, label="25–75th percentile")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS); ax.set_ylabel(label); ax.legend()
    fig.tight_layout()
    return fig


def lag_corr(monthly: pd.DataFrame, max_lag: int = 4) -> pd.Series:
    return pd.Series({k: monthly["rainfall_mm"].corr(monthly["discharge_m3s"].shift(-k)) for k in range(max_lag + 1)})


def lag_fig(lc: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar([f"+{k}" for k in lc.index], lc.values, color="#4c72b0")
    ax.set(xlabel="lag k (months)", ylabel="Pearson r", title="Rainfall(t) vs Discharge(t+k)")
    fig.tight_layout()
    return fig


def fdc_fig(daily: pd.DataFrame):
    q = np.sort(daily["discharge_m3s"].values)[::-1]
    exceed = np.arange(1, len(q) + 1) / len(q) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(exceed, q, color=COLORS["discharge_m3s"]); ax.set_yscale("log")
    ax.set(xlabel="% of days flow is equalled or exceeded", ylabel="Discharge (m³/s, log)")
    ax.grid(alpha=.4); fig.tight_layout()
    return fig


def forecast_fig(pred: pd.DataFrame, models: list):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pred.index, pred["observed"], color="black", lw=2.2, label="Observed")
    for m in models:
        ax.plot(pred.index, pred[m], lw=1.5, marker="o", ms=3, label=m)
    ax.set_ylabel("Mean monthly discharge (m³/s)"); ax.legend(fontsize=8); ax.grid(alpha=.4)
    fig.tight_layout()
    return fig
