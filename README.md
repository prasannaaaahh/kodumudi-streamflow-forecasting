# Streamflow Analytics \& Forecasting: Kodumudi (Cauvery) Catchment, 1980-2015

Exploratory analysis, statistical testing, classical time-series models and machine-learning forecasts of monthly river discharge, built on 36 years of daily rainfall, soil-moisture, temperature and discharge data. Models are compared against simple baselines on months they never saw, and the results are explored in an interactive dashboard.

Originally a university course project, substantially extended with lag-feature ML forecasting, robust statistics, baselines and a dashboard.

## Problem statement

How does rainfall translate into river flow in this catchment, is the climate shifting, and can next month's flow be forecast better than simple rules of thumb?

## Data

13,149 consecutive daily records (1980–2015) for the Kodumudi gauge on the Cauvery, supplied by the course instructor. No gaps or missing values. See [`data/README.md`](data/README.md) for the data dictionary and cleaning steps.

## Key findings

* River flow is bimodal: it peaks in Jul–Aug and again in Oct–Nov, with a dry-season low in April. High flows (Q10) are about 5× low flows (Q90).
* Rainfall acts with a delay: monthly rainfall correlates r = 0.71 with next-month discharge versus 0.51 with same-month discharge. Adding lagged rainfall raises the adjusted R² of a robust-error regression from 0.50 to 0.65.
* No annual climate trend survives multiple-testing correction. A seasonal test on monthly data does suggest rising temperature, but this is suggestive evidence only.
* For one-month-ahead forecasts, every model that captures seasonality beats the naive baselines on the 44 held-out months; the strongest cut test RMSE by about 30 % versus a climatology baseline.
* Differences between the best models are within noise on 44 months (bootstrap intervals include zero), so rankings among them should not be over-read.

|Model (one-month-ahead, test: May 2012 – Dec 2015)|MAE|RMSE|
|-|-:|-:|
|Ridge|35.4|48.4|
|Random Forest|35.0|53.5|
|SARIMA(1,0,1)×(1,0,1)\[12]|42.2|60.8|
|Gradient Boosting|41.9|62.5|
|Climatology (baseline)|50.5|69.1|
|ARMA(1,1), non-seasonal|66.6|83.1|
|Persistence (baseline)|66.3|92.0|

!\[Test-period forecasts](reports/figures/04\_test\_forecast.png)

## Approach

1. **Explore and validate:** gap-free date index, quality report, monthly/annual aggregation (rainfall summed, other variables averaged), seasonality, flow-duration curve, lag structure.
2. **Statistics:** regression with HAC-robust errors and VIF; Mann-Kendall, Theil-Sen, Hamed-Rao and seasonal trend tests with Holm correction.
3. **Time series:** STL, ADF/KPSS, ACF/PACF, ARIMA/SARIMA grid, Ljung-Box diagnostics, one-step-ahead evaluation.
4. **ML forecasting:** leakage-safe lag features, Ridge / Random Forest / Gradient Boosting, `TimeSeriesSplit` tuning, chronological train/validation/test split, ablation and block bootstrap.

## Dashboard

```bash
streamlit run app/streamlit\\\\\\\_app.py
```

## Repository structure

```text
data/        Cleaned daily CSV and data dictionary
notebooks/   01 EDA, 02 statistics, 03 ARIMA/SARIMA, 04 ML forecasting
src/         Data loading, feature engineering, metrics
app/         Streamlit dashboard
tests/       Data-integrity and leakage tests
reports/     Figures and result tables
```

## How to reproduce

```bash
git clone <this-repo>
cd <this-repo>
python -m venv .venv
.venv\\\\\\\\Scripts\\\\\\\\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in order (`01` to `04`; `03` takes a few minutes), then the last cell of `04` to build the combined comparison. Run the tests with `pip install pytest` and `pytest`. Random Forest scores can differ slightly across machines and library versions.

## Limitations

* One catchment, 36 years; results may not transfer to other basins.
* Only 44 test months; close models cannot be separated reliably.
* Reservoir releases and upstream rainfall are not in the data and plausibly explain part of the unexplained variance.
* Large floods dominate RMSE; extreme-event (quantile / log-scale) models are a natural next step.
* Forecast horizon is one month; multi-horizon forecasting is future work.
* ## \## License

## Code: \[MIT](LICENSE). Data: supplied by the course instructor; reuse subject to the instructor's terms.

