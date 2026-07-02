"""
Project 2 — Time Series Forecasting
===================================
Dataset : Monthly international airline passengers, 1949-1960 (144 points).
          Real open data, fetched from the jbrownlee/Datasets mirror on GitHub.
Goal    : Forecast the next 24 months and evaluate honestly.

Why this project matters for a portfolio: forecasting is where beginners most
often fool themselves. The two cardinal sins are (a) evaluating with a random
train/test split — which leaks the future into the past — and (b) reporting only
in-sample fit. This script does neither. It uses a *rolling-origin backtest*
(a.k.a. time-series cross-validation) and compares real models against a
seasonal-naive baseline, because a model that can't beat "same month last year"
is not worth deploying.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from dsmanual import seasonal_naive, mape   # tested, shared utilities

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
y = pd.read_csv(URL, parse_dates=["Month"], index_col="Month")["Passengers"].asfreq("MS")
print(f"Series: {len(y)} monthly points, {y.index.min():%Y-%m} to {y.index.max():%Y-%m}")

# ---------------------------------------------------------------- 1. decomposition
# Multiplicative because the seasonal swings GROW with the level (classic sign).
decomp = seasonal_decompose(y, model="multiplicative", period=12)
fig = decomp.plot(); fig.set_size_inches(9, 7); fig.suptitle("Multiplicative decomposition", y=1.01)
fig.tight_layout(); fig.savefig("decomposition.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 2. models
def fit_ets(train, h):
    m = ExponentialSmoothing(train, trend="add", seasonal="mul",
                             seasonal_periods=12).fit()
    return m.forecast(h).values

def fit_sarima(train, h):
    m = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return m.forecast(h).values

# ---------------------------------------------------------------- 3. rolling-origin backtest
# Expanding window: train on everything up to a cutoff, forecast H months, score,
# then move the cutoff forward. This mimics how the model is actually used.
H = 12
cutoffs = [len(y) - k for k in (36, 24, 12)]     # three evaluation origins
results = {"SeasonalNaive": [], "ETS": [], "SARIMA": []}
for c in cutoffs:
    train, actual = y.iloc[:c], y.iloc[c:c + H].values
    results["SeasonalNaive"].append(mape(actual, seasonal_naive(train, H)))
    results["ETS"].append(mape(actual, fit_ets(train, H)))
    results["SARIMA"].append(mape(actual, fit_sarima(train, H)))

bt = pd.DataFrame(results, index=[f"origin@{y.index[c-1]:%Y-%m}" for c in cutoffs])
bt.loc["MEAN"] = bt.mean()
print("\n=== Rolling-origin backtest — MAPE %% (lower is better) ===")
print(bt.round(2).to_string())
best = bt.loc["MEAN"].idxmin()
print(f"\nBest by mean MAPE: {best}")

# ---------------------------------------------------------------- 4. final forecast with the winner
fit_fn = {"ETS": fit_ets, "SARIMA": fit_sarima}.get(best, None)
future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=24, freq="MS")
if best == "SARIMA":
    m = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = m.get_forecast(24); mean, ci = fc.predicted_mean, fc.conf_int()
elif best == "ETS":
    m = ExponentialSmoothing(y, trend="add", seasonal="mul", seasonal_periods=12).fit()
    mean = pd.Series(m.forecast(24).values, index=future_idx); ci = None
else:
    mean = pd.Series(seasonal_naive(y, 24), index=future_idx); ci = None

fig, ax = plt.subplots(figsize=(10, 4.5))
y.plot(ax=ax, label="observed")
pd.Series(np.asarray(mean), index=future_idx).plot(ax=ax, label=f"{best} forecast", color="C1")
if ci is not None:
    ax.fill_between(future_idx, ci.iloc[:, 0], ci.iloc[:, 1], color="C1", alpha=.2,
                    label="95% interval")
ax.set(title=f"24-month forecast ({best})", ylabel="passengers (thousands)")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("forecast.png", dpi=120); plt.close(fig)

with open("metrics.md", "w") as f:
    f.write("# Project 2 results — rolling-origin backtest\n\n")
    f.write("MAPE (%) across three forecast origins, 12-month horizon each:\n\n")
    f.write(bt.round(2).to_markdown() + "\n\n")
    f.write(f"**Winner: {best}.** Note that both statistical models must be "
            "compared against the seasonal-naive baseline — reporting an absolute "
            "error without that reference is meaningless.\n")
print("\nSaved: decomposition.png, forecast.png, metrics.md")
