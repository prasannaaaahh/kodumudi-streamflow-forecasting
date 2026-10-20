"""Forecast accuracy metrics (MAE, MSE, RMSE, Nash-Sutcliffe efficiency)."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def rmse(y, p):
    return float(np.sqrt(mse(y, p)))


def nse(y, p):
    """Nash-Sutcliffe efficiency (hydrology's name for R^2 vs. the observed mean).
    1 = perfect, 0 = no better than predicting the mean, <0 = worse than the mean."""
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def evaluate(y, p) -> dict:
    return {"MAE": mae(y, p), "MSE": mse(y, p), "RMSE": rmse(y, p), "NSE": nse(y, p)}
