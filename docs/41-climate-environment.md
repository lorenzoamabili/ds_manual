# 41 · Climate & Environment

## Signature problems

| Problem | Approach |
|---------|----------|
| Emissions estimation | Regression on activity data + emission factors |
| Deforestation detection | Computer vision on satellite imagery (see [11](11-computer-vision.md)) |
| Air quality forecasting | Time-series + spatial interpolation (see [07](07-time-series-forecasting.md), [22](22-geospatial.md)) |
| Species distribution modelling | MaxEnt, GLM on geo-referenced presence/absence data |
| Extreme weather prediction | Deep learning (NeurWeatherFormer, GraphCast) |
| Carbon credit verification | Satellite + ML for forest biomass estimation |
| Climate downscaling | Convolutional super-resolution on climate model output |
| ESG scoring | Multi-source data fusion, NLP on company disclosures |

## Domain characteristics

- **Sparse, noisy ground truth**: environmental sensor networks have gaps; satellite data has cloud cover; species sightings are presence-only (no absence data). Handling missing data and presence-only bias is core.
- **Spatial autocorrelation is the rule**: nearby measurements are not independent. Standard CV that splits randomly over-estimates performance — use spatial CV (hold out geographic blocks). See [22](22-geospatial.md).
- **Long time horizons, slow feedback**: a deforestation alert may take months to verify on the ground. Labelling pipelines are slow.
- **High stakes, contested data**: emissions data is politically sensitive; satellite-based deforestation estimates are legally consequential. Uncertainty quantification is not optional.
- **Class imbalance**: rare events (fires, floods, species detections) dominate the value but are sparse in data.

## Air quality forecasting

```python
"""
Interpolate sparse sensor readings to a spatial grid using kriging-inspired
inverse-distance weighting, then forecast next-hour PM2.5.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
n_hours = 8760  # one year hourly

# Synthetic PM2.5: diurnal cycle + weekly pattern + random noise
hours   = np.arange(n_hours)
diurnal = 15 + 8 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
weekly  = 5  * np.sin(2 * np.pi * hours / (24*7))
spikes  = rng.choice([0, 30], n_hours, p=[0.97, 0.03])  # pollution events
pm25    = np.clip(diurnal + weekly + spikes + rng.normal(0, 3, n_hours), 1, 300)

# Feature engineering
df = pd.DataFrame({"pm25": pm25, "hour": hours % 24,
                   "dow": (hours // 24) % 7, "month": (hours // 730) % 12})
for lag in [1, 2, 3, 24]:
    df[f"lag_{lag}h"] = df["pm25"].shift(lag)
df["roll_24h_mean"] = df["pm25"].rolling(24).mean()
df["roll_24h_std"]  = df["pm25"].rolling(24).std()
df = df.dropna()

X = df.drop(columns="pm25")
y = df["pm25"]

# Time-series CV (no future leakage)
tscv = TimeSeriesSplit(n_splits=5)
gbm  = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
maes = []
for tr_idx, te_idx in tscv.split(X):
    gbm.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = gbm.predict(X.iloc[te_idx])
    maes.append(mean_absolute_error(y.iloc[te_idx], pred))

print(f"PM2.5 forecast MAE: {np.mean(maes):.2f} µg/m³ (5-fold time-series CV)")
print(f"WHO daily guideline: 15 µg/m³ — model error is {np.mean(maes)/15:.0%} of guideline")

# Plot last week
fig, ax = plt.subplots(figsize=(12, 3))
last_week = pm25[-168:]
ax.plot(last_week, lw=1, label="Actual PM2.5")
ax.axhline(15, color="orange", ls="--", label="WHO guideline (15 µg/m³)")
ax.axhline(35, color="red",    ls="--", label="Unhealthy threshold (35 µg/m³)")
ax.set(xlabel="Hour (last 7 days)", ylabel="PM2.5 µg/m³",
       title="Air quality — last 7 days")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig("air_quality.png", dpi=120); plt.close()
print("Plot saved: air_quality.png")
```

## Presence-only species modelling

```python
"""
Species distribution modelling with presence-only data.
MaxEnt-style: compare presence locations vs. background pseudo-absences.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)

# Simulate presence points clustered in warm, moist areas
n_presence = 200
temp    = rng.normal(22, 3, n_presence)    # °C
precip  = rng.normal(1200, 200, n_presence) # mm/yr
presence_X = np.column_stack([temp, precip])

# Background: random environmental conditions across the study area
n_background = 2000
bg_temp   = rng.uniform(5, 35,   n_background)
bg_precip = rng.uniform(200, 2000, n_background)
background_X = np.column_stack([bg_temp, bg_precip])

X = np.vstack([presence_X, background_X])
y = np.array([1]*n_presence + [0]*n_background)

scaler = StandardScaler().fit(X)
model  = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
model.fit(scaler.transform(X), y)

auc = roc_auc_score(y, model.predict_proba(scaler.transform(X))[:, 1])
print(f"Species distribution model AUC: {auc:.3f}")
print("Note: pseudo-absence sampling strategy critically affects results.")
print("AUC interpretation: how well model separates presences from background.")
print("Caveat: background ≠ true absence — no-detection may mean unsampled.")
```

## Cross-references

- [07](07-time-series-forecasting.md) — climate and weather time series
- [11](11-computer-vision.md) — satellite imagery, deforestation detection
- [22](22-geospatial.md) — spatial CV, Moran's I, spatial autocorrelation
- [case studies](case_studies/cs-energy.md) — DeepMind, Ørsted wind power (adjacent domain)
