# 36 · Energy & Utilities

> [Forecasting](07-time-series-forecasting.md) demand, detecting grid anomalies, and optimising dispatch in one of
> the most consequential and data-rich physical systems on earth.

---

## Why data science here

Energy is a domain where decisions happen at millisecond timescales (frequency
regulation) and decade timescales (infrastructure investment) simultaneously. The
data is abundant — smart meters, SCADA systems, weather stations, satellite imagery
— but highly seasonal, non-stationary, and coupled to physics (supply must equal
demand at every instant or the grid fails).

The signature data science problem is **load forecasting**: predicting electricity
demand at various horizons (next hour, next day, next season) so that generation
can be dispatched efficiently. Forecast errors translate directly into either
wasted generation (too much supply = spinning reserve costs) or blackouts (too
little supply). A 1% MAPE improvement in national load forecasting is worth
millions of pounds in avoided balancing costs.

Renewables add complexity: solar and wind generation is intermittent and harder
to forecast than demand. Grid operators now need to forecast both sides of the
supply-demand balance.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Short-term load forecasting | Predict demand next 24–48h | SARIMA, [gradient boosting](05-supervised-learning.md) on calendar + weather |
| Long-term load forecasting | Predict demand next 1–5 years | Regression on economic indicators + demographics |
| Renewable generation forecasting | Predict solar/wind output | NWP-based models, GBM, [CNN](11-computer-vision.md) on NWP grids |
| Demand [anomaly detection](13-anomaly-detection.md) | Is this meter reading fraudulent/faulty? | [Isolation Forest](13-anomaly-detection.md), STL + control limits |
| Energy theft detection | Is this customer under-reporting? | Classification on consumption patterns |
| Price forecasting | What will the spot price be tomorrow? | Time series + market features |
| Grid fault detection | Is this [transformer](10-nlp-and-llms.md) about to fail? | Anomaly detection on SCADA |

---

## Key techniques

### 1. Short-term load forecasting (STLF)

Electricity demand has strong daily, weekly, and annual [seasonality](07-time-series-forecasting.md), plus
weather sensitivity (heating degree days, cooling degree days). The dominant
approach: gradient boosting with calendar features (hour, weekday, month,
holiday) and lagged demand (t-24h, t-48h, same day last week).

**Pitfall:** temperature is the strongest predictor but it is the *future*
temperature (forecast), not current. For operational forecasting, you need an
NWP (numerical weather prediction) input — not historical weather.

### 2. STL decomposition

Seasonal-Trend decomposition using LOESS separates a time series into trend,
seasonal, and residual components. Useful for: (1) visualising seasonality
structure, (2) detecting anomalies in the residual, (3) as a preprocessing
step for downstream models.

### 3. Prophet

Facebook's Prophet handles multiple seasonalities, holidays, and changepoints
with minimal tuning. Excellent for communicating forecasts to non-technical
stakeholders because it produces interpretable components. Often outperformed
by GBM on large datasets but fast to prototype.

### 4. Anomaly detection on meter data

Smart meter data contains zeros (outages), step changes (meter tampering),
and [drift](14-mlops-and-productionization.md) (meter ageing). STL residual + 3σ control limits is a simple first
pass. Isolation Forest handles multivariate meter clusters.

---

## Best practices & pitfalls

- **Evaluate with MAPE only where demand is never near zero.** MAPE is undefined
  at zero and misleading at small values. Use RMSE or sMAPE for intermittent loads.
- **Always compare to a naive seasonal baseline.** "Same as last week" or "same
  hour last year" is the correct benchmark for energy forecasting, not zero.
- **Public holidays break patterns.** Christmas Day looks like Sunday load profile.
  Bank holidays need explicit encoding.
- **Daylight saving time creates duplicate/missing hours.** Handle explicitly in
  your index construction.
- **Weather data lags.** Don't use actual future weather in a backtest — use the
  NWP forecast that would have been available at prediction time.
- **Long-term forecasts are wrong.** Embrace uncertainty quantification (prediction
  intervals) rather than pretending a 5-year forecast is a point estimate.

---

## Python example — electricity load forecasting

```python
"""
Short-term electricity load forecasting using gradient boosting.

Uses synthetic hourly load data with realistic seasonal structure,
temperature sensitivity, and weekly patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

rng = np.random.default_rng(42)
OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Generate synthetic hourly load data ─────────────────────────────────────
# 2 years of hourly data
n_hours = 24 * 365 * 2
idx = pd.date_range("2022-01-01", periods=n_hours, freq="h")

# Seasonal components
hour_effect = np.sin(2 * np.pi * (idx.hour - 6) / 24) * 500 + \
              np.sin(2 * np.pi * (idx.hour - 14) / 12) * 300
day_effect  = (idx.dayofweek >= 5).astype(float) * (-600)  # lower on weekends
year_effect = np.cos(2 * np.pi * idx.dayofyear / 365) * (-800)  # lower in summer

# Temperature proxy: cold in winter, warm in summer
temp = 15 - 12 * np.cos(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 3, n_hours)
heat_cool = np.where(temp < 8, (8 - temp) * 80,  # heating
             np.where(temp > 22, (temp - 22) * 60, 0))  # cooling

load = (5000 + hour_effect + day_effect + year_effect + heat_cool
        + rng.normal(0, 100, n_hours)).clip(1000)

ts = pd.DataFrame({"load": load, "temp": temp}, index=idx)

# ── Feature engineering ──────────────────────────────────────────────────────
ts["hour"]       = ts.index.hour
ts["dayofweek"]  = ts.index.dayofweek
ts["month"]      = ts.index.month
ts["is_weekend"] = (ts.index.dayofweek >= 5).astype(int)
ts["load_lag24"] = ts["load"].shift(24)  # same hour yesterday
ts["load_lag168"]= ts["load"].shift(168) # same hour last week

# Sin/cos encoding for cyclical features
for col, period in [("hour", 24), ("dayofweek", 7), ("month", 12)]:
    ts[f"{col}_sin"] = np.sin(2 * np.pi * ts[col] / period)
    ts[f"{col}_cos"] = np.cos(2 * np.pi * ts[col] / period)

ts = ts.dropna()

# ── Train / test split (temporal — last 8 weeks as test) ────────────────────
cutoff = ts.index.max() - pd.Timedelta(weeks=8)
train  = ts[ts.index <= cutoff]
test   = ts[ts.index > cutoff]

feat_cols = ["temp", "load_lag24", "load_lag168",
             "hour_sin","hour_cos","dayofweek_sin","dayofweek_cos",
             "month_sin","month_cos","is_weekend"]

X_tr, y_tr = train[feat_cols], train["load"]
X_te, y_te = test[feat_cols],  test["load"]

# Naive baseline: same hour last week
naive_pred = test["load_lag168"]

# GBM model
gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                 learning_rate=0.05, random_state=42)
gbm.fit(X_tr, y_tr)
gbm_pred = gbm.predict(X_te)

# ── Metrics ───────────────────────────────────────────────────────────────────
naive_mape = mean_absolute_percentage_error(y_te, naive_pred)
gbm_mape   = mean_absolute_percentage_error(y_te, gbm_pred)
naive_rmse = np.sqrt(mean_squared_error(y_te, naive_pred))
gbm_rmse   = np.sqrt(mean_squared_error(y_te, gbm_pred))

print(f"Naive (same hour last week): MAPE={naive_mape:.2%}  RMSE={naive_rmse:.0f} MW")
print(f"GBM:                         MAPE={gbm_mape:.2%}  RMSE={gbm_rmse:.0f} MW")

# ── Plot one week of forecasts ────────────────────────────────────────────────
sample = test.iloc[:168]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sample.index, y_te.iloc[:168], label="Actual", lw=1.5)
ax.plot(sample.index, gbm_pred[:168],  label=f"GBM (MAPE {gbm_mape:.1%})", alpha=0.8)
ax.plot(sample.index, naive_pred.iloc[:168], "--", label=f"Naive (MAPE {naive_mape:.1%})", alpha=0.6)
ax.set_ylabel("Load (MW)")
ax.set_title("Short-term load forecast — 1-week sample")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "load_forecast.png", dpi=120)
plt.close()

# ── Feature importance ────────────────────────────────────────────────────────
importance = pd.Series(gbm.feature_importances_, index=feat_cols).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
importance.plot.barh(ax=ax)
ax.set_title("Feature importance — load forecasting GBM")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=120)
plt.close()
print("Plots saved.")
```

---

## Cross-references

- [07](07-time-series-forecasting.md) — time series methods (SARIMA, Prophet, backtest)
- [13](13-anomaly-detection.md) — meter anomaly and fault detection
- [12](12-optimization.md) — energy dispatch optimisation
- [34](34-manufacturing.md) — predictive maintenance for plant equipment
