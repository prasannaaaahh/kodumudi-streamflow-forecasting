"""Interactive dashboard: Kodumudi (Cauvery) hydrology, 1980-2015.

Run from the repository root:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import kodumudi as kd          # noqa: E402
import charts                  # noqa: E402

st.set_page_config(page_title="Kodumudi Streamflow Analytics", page_icon="🌊", layout="wide")


@st.cache_data
def load():
    daily = kd.load_daily()
    return daily


daily_all = load()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
y0, y1 = st.sidebar.slider("Years", 1980, 2015, (1980, 2015))
scale = st.sidebar.radio("Time scale", ["Monthly", "Annual", "Daily"], index=0)
var = st.sidebar.selectbox("Variable", kd.PARAMS, format_func=lambda v: kd.LABELS[v], index=1)
smooth = st.sidebar.slider("Rolling mean window (0 = off)", 0, 60, 0)

daily = daily_all[(daily_all.index.year >= y0) & (daily_all.index.year <= y1)]
monthly = kd.aggregate(daily, "MS")
annual = kd.aggregate(daily, "YS")

# ---------------------------------------------------------------- header
st.title("🌊 Kodumudi (Cauvery) Streamflow Analytics")
st.caption(f"Gauge 5008 · daily data 1980–2015 · selected {y0}–{y1} ({len(daily):,} days). "
           "Data provided by the course instructor.")

k = charts.kpis(daily)
cols = st.columns(5)
cols[0].metric("Mean discharge", f"{k['Mean discharge (m³/s)']:.0f} m³/s")
cols[1].metric("High flow (Q10)", f"{k['Q10 high flow (m³/s)']:.0f} m³/s")
cols[2].metric("Median flow (Q50)", f"{k['Q50 median flow (m³/s)']:.0f} m³/s")
cols[3].metric("Low flow (Q90)", f"{k['Q90 low flow (m³/s)']:.0f} m³/s")
cols[4].metric("Peak daily flow", f"{k['Peak daily flow (m³/s)']:.0f} m³/s", k["Peak date"], delta_color="off")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Time series", "Seasonality", "Rainfall → flow", "Flow duration", "Forecast evaluation"])

with tab1:
    df = {"Monthly": monthly, "Annual": annual, "Daily": daily}[scale]
    st.pyplot(charts.timeseries_fig(df, var, kd.LABELS[var], smooth))
    st.caption("Rainfall is summed over each period; all other variables are averaged.")

with tab2:
    if len(monthly) >= 24:
        st.pyplot(charts.climatology_fig(monthly, var, kd.LABELS[var]))
    else:
        st.info("Select at least two years to see seasonality.")

with tab3:
    if len(monthly) >= 24:
        c1, c2 = st.columns(2)
        c1.pyplot(charts.lag_fig(charts.lag_corr(monthly)))
        c2.markdown(
            "Rainfall in a month tends to be more strongly associated with the **next** month's "
            "river flow than with the same month's flow. This is why the forecasting models use "
            "lagged rainfall features.\n\n"
            "_Correlation, not causation: monthly correlations also include the shared seasonal cycle "
            "(see notebook 01)._")
    else:
        st.info("Select at least two years.")

with tab4:
    st.pyplot(charts.fdc_fig(daily))
    st.caption("Q10 = flow exceeded 10% of the time (high flows); Q90 = exceeded 90% of the time (low flows).")

with tab5:
    pred_path = ROOT / "reports" / "tables" / "ml_test_predictions.csv"
    arima_path = ROOT / "reports" / "tables" / "arima_test_predictions.csv"
    if not pred_path.exists():
        st.warning("Run notebook 04 to generate forecast results.")
    else:
        pred = pd.read_csv(pred_path, index_col="month", parse_dates=True)
        if arima_path.exists():
            arima = pd.read_csv(arima_path, index_col="month", parse_dates=True).drop(columns="observed")
            pred = pred.join(arima)
        options = [c for c in pred.columns if c != "observed"]
        chosen = st.multiselect("Models to plot", options, default=[o for o in ["Gradient Boosting", "Climatology"] if o in options])
        st.pyplot(charts.forecast_fig(pred, chosen))
        rows = {m: kd.evaluate(pred["observed"], pred[m]) for m in options}
        st.subheader("Held-out test period: May 2012 – Dec 2015 (one-month-ahead)")
        st.dataframe(pd.DataFrame(rows).T.sort_values("RMSE").round(2))
        st.caption("44 test months only — differences between close models are within noise "
                   "(see the bootstrap analysis in notebook 04).")
