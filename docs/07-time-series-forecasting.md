# 07 · Time Series & Forecasting

Predicting the future of a series. The domain where naive evaluation fools people
most often. Paired project:
[P2 — airline passengers](../projects/p2_time_series_forecasting).

## First, understand the series

Decompose into **trend + seasonality + residual**. If seasonal swings *grow with
the level*, the structure is **multiplicative** (use a log transform or a
multiplicative model) — the airline series in P2 is the textbook case. Check for:

- **Stationarity** — do mean/variance drift? Many classical models need a
  stationary series; differencing (subtracting the previous value) often achieves
  it. Test with ADF/KPSS, but the plot usually tells you.
- **Autocorrelation** — ACF/PACF plots reveal how the series relates to its own
  past and guide ARIMA orders.

## Model families

| Family | Reach for it when |
|--------|-------------------|
| **Seasonal-naive baseline** | Always — it's the bar every real model must clear |
| **Exponential smoothing (ETS / Holt-Winters)** | Clear trend + seasonality, one series, want a robust default. *Won P2.* |
| **ARIMA / SARIMA** | Autocorrelated series; you want interpretable structure and intervals |
| **Prophet** | Business series with holidays, multiple seasonalities, missing data, analyst-friendly knobs |
| **Gradient boosting on lag features** | Many related series, exogenous regressors, non-linear effects; competition-grade |
| **Deep models (N-BEATS, DeepAR, TFT)** | Many long series, rich covariates, and you have the data + engineering to justify them |

Reality check: on a *single* clean seasonal series, ETS/SARIMA usually beat fancy
ML. ML wins when you have **many** series and **exogenous features**.

## The two cardinal sins (and their fixes)

1. **Random train/test split** — leaks the future into the past. **Fix:** split
   chronologically; evaluate with a **rolling-origin backtest** (train on the
   past, forecast forward, roll the cutoff). P2 does this across three origins.
2. **Reporting only in-sample fit** — a model can fit history perfectly and
   forecast terribly. **Fix:** score out-of-sample, multi-origin, at the horizon
   you actually care about, **against the seasonal-naive baseline.**

## Metrics
- **MAPE** — intuitive (%), but explodes near zero and punishes over- vs.
  under-forecast asymmetrically. Used in P2 because the series is far from zero.
- **MAE / RMSE** — scale-dependent; fine for a single series.
- **MASE** — error relative to the naive baseline; comparable across series and
  the most honest single number. <1 means you beat naive.

## Feature engineering for ML forecasting
Lags (t-1, t-7, t-12…), rolling means/std, calendar features (day-of-week,
month, holiday flags), and exogenous drivers (price, weather, promotions). The
craft is in choosing lags that respect what's *knowable* at forecast time — a lag
shorter than your horizon is leakage.

---

## Python example — rolling-origin backtest with ETS vs. naive

```python
"""
Time series forecasting: rolling-origin evaluation on a synthetic
seasonal series. Demonstrates why random splits are wrong for time series.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

rng = np.random.default_rng(42)

# ── Synthetic annual-seasonal series (multiplicative) ─────────────────────────
n = 120  # 10 years of monthly data
t = np.arange(n)
trend = 100 + 0.5 * t
seasonal = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
noise = rng.normal(1, 0.05, n)
series = trend * seasonal * noise
idx = pd.date_range("2014-01", periods=n, freq="MS")
ts = pd.Series(series, index=idx)

# ── Rolling-origin backtest (3 origins, 12-step horizon) ─────────────────────
HORIZON = 12
origins = [84, 96, 108]  # months 84, 96, 108 → forecast 85-96, 97-108, 109-120

ets_errors, naive_errors = [], []

for origin in origins:
    train = ts.iloc[:origin]
    actual = ts.iloc[origin:origin + HORIZON].values

    # ETS (additive trend, multiplicative seasonality)
    model = ExponentialSmoothing(train, trend="add", seasonal="mul",
                                  seasonal_periods=12).fit(disp=False)
    ets_fc = model.forecast(HORIZON)

    # Seasonal naive: repeat the last full season
    naive_fc = train.iloc[-12:].values[:HORIZON]

    ets_errors.append(np.mean(np.abs(ets_fc - actual) / np.abs(actual)) * 100)
    naive_errors.append(np.mean(np.abs(naive_fc - actual) / np.abs(actual)) * 100)

print("Rolling-origin MAPE:")
for i, (e, n_) in enumerate(zip(ets_errors, naive_errors)):
    print(f"  Origin {origins[i]}: ETS={e:.1f}%  Naive={n_:.1f}%")
print(f"  Mean    : ETS={np.mean(ets_errors):.1f}%  Naive={np.mean(naive_errors):.1f}%")

# ── Plot last forecast origin ─────────────────────────────────────────────────
origin = origins[-1]
train = ts.iloc[:origin]
ets_fc = ExponentialSmoothing(train, trend="add", seasonal="mul",
                               seasonal_periods=12).fit(disp=False).forecast(HORIZON)
naive_fc = pd.Series(train.iloc[-12:].values, index=ts.iloc[origin:origin+HORIZON].index)

fig, ax = plt.subplots(figsize=(10, 4))
train.iloc[-24:].plot(ax=ax, label="Train (last 24 months)")
ts.iloc[origin:origin+HORIZON].plot(ax=ax, label="Actual", color="black")
ets_fc.plot(ax=ax, label="ETS forecast", style="--")
naive_fc.plot(ax=ax, label="Seasonal naive", style=":")
ax.set_title("Rolling-origin forecast — last origin")
ax.legend()
plt.tight_layout()
plt.savefig("forecast_comparison.png", dpi=120)
plt.close()
print("Plot saved. ETS beats naive because it adapts the trend.")
```

---

## Cross-references

- [P2](../projects/p2_time_series_forecasting) — full forecasting project (airline data)
- [36](36-energy.md) — energy load forecasting in practice
- [34](34-manufacturing.md) — sensor time series for predictive maintenance
