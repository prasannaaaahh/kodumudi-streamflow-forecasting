"""Guards against the most damaging bug in forecasting projects: data leakage."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import kodumudi as kd  # noqa: E402


def _data():
    daily = kd.load_daily()
    monthly = kd.aggregate(daily, "MS")
    X, y = kd.build_supervised(monthly)
    return monthly, X, y


def test_daily_data_is_complete():
    daily = kd.load_daily()
    assert len(daily) == 13149
    assert daily.isna().sum().sum() == 0


def test_target_is_next_month_and_features_are_past_only():
    monthly, X, y = _data()
    for target_month in [X.index[0], X.index[100], X.index[-1]]:
        origin = target_month - __import__("pandas").offsets.MonthBegin(1)
        assert np.isclose(y.loc[target_month], monthly.loc[target_month, "discharge_m3s"])
        assert np.isclose(X.loc[target_month, "q_t"], monthly.loc[origin, "discharge_m3s"])
        assert np.isclose(X.loc[target_month, "rain_t"], monthly.loc[origin, "rainfall_mm"])


def test_seasonal_lag_is_same_calendar_month_one_year_earlier():
    monthly, X, _ = _data()
    t = X.index[50]
    same_month_last_year = t - __import__("pandas").DateOffset(years=1)
    assert np.isclose(X.loc[t, "q_t-11"], monthly.loc[same_month_last_year, "discharge_m3s"])


def test_splits_do_not_overlap_and_cover_everything():
    _, X, _ = _data()
    tr = X.index < kd.VAL_START
    va = (X.index >= kd.VAL_START) & (X.index < kd.TEST_START)
    te = X.index >= kd.TEST_START
    assert (tr.astype(int) + va.astype(int) + te.astype(int) == 1).all()
    assert te.sum() == 44


def test_rainfall_is_summed_not_averaged():
    daily = kd.load_daily()
    monthly = kd.aggregate(daily, "MS")
    jan80 = daily.loc["1980-01", "rainfall_mm"]
    assert np.isclose(monthly.loc["1980-01-01", "rainfall_mm"], jan80.sum())
    assert np.isclose(monthly.loc["1980-01-01", "discharge_m3s"], daily.loc["1980-01", "discharge_m3s"].mean())
