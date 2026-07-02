# 34 · Manufacturing & Industry 4.0

> Using sensor data, computer vision, and predictive models to reduce downtime,
> improve quality, and optimise industrial processes.

---

## Why data science here

Manufacturing generates enormous volumes of time-series sensor data (vibration,
temperature, pressure, current draw) from equipment that is expensive to repair
and catastrophic to fail. The canonical problem is **predictive maintenance (PdM)**:
predict failure before it happens so maintenance can be scheduled during planned
downtime rather than as an emergency.

The domain is characterised by: (1) **extreme class imbalance** — equipment fails
rarely, so failure events are a tiny fraction of sensor records; (2) **temporal
structure** — a single machine observation is useless; patterns of degradation
unfold over days or weeks; (3) **operational context** — the same vibration reading
means different things at different load levels, so operating mode must be controlled
for; (4) **cost asymmetry** — false negatives (missed failures) are far more
expensive than false positives (unnecessary maintenance).

Industry 4.0 adds computer vision (surface defect detection), digital twins
(simulation-driven optimisation), and OPC-UA / MQTT data pipelines to the mix.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Predictive maintenance | Will this machine fail in the next N hours? | Binary classification on rolling features; survival analysis |
| Remaining useful life (RUL) | How many cycles until failure? | Regression; LSTM on sensor sequences |
| Anomaly detection | Is this sensor reading abnormal? | Isolation Forest, OCSVM, autoencoders |
| Quality control | Is this product defective? | Computer vision (CNN); statistical process control |
| Process optimisation | What settings minimise defect rate? | Bayesian optimisation; DoE |
| Root cause analysis | Why did batch X fail? | SHAP, decision trees on process logs |

---

## Key techniques

### 1. Feature engineering from time series

Raw sensor readings are rarely predictive by themselves. Standard features:
- **Rolling statistics**: mean, std, min, max over N-cycle windows
- **Frequency domain**: FFT magnitudes at characteristic frequencies (bearing fault
  frequencies, blade pass frequencies)
- **Trend**: slope over a rolling window — captures degradation direction
- **Cross-sensor ratios**: e.g., vibration/temperature interaction

The window size is a hyperparameter. Too short = noisy; too long = misses rapid
degradation.

### 2. Remaining useful life regression

Given sensor history up to cycle *t*, predict cycles until failure. NASA CMAPSS
dataset is the standard benchmark: turbofan engines run until failure; the task
is to predict RUL from multivariate sensor streams.

Common approach: (1) compute a health indicator (normalised degradation index),
(2) fit a piecewise linear model or LSTM to sensor history, (3) clip RUL
predictions to a cap (engines are "healthy" until ~125 cycles before failure).

### 3. Isolation Forest for sensor anomaly

Fit on normal-operation data only. Score new readings by the average tree depth
needed to isolate the point — short paths = anomalies. Works well without labelled
failure data. Tune the `contamination` parameter to match expected anomaly rate.

**Pitfall:** operating mode shifts (speed change, load change) look like anomalies.
Either build mode-specific models or add operating mode as a feature.

### 4. Statistical process control (SPC)

Classical engineering approach: Shewhart control charts, CUSUM, EWMA. Flag
readings outside control limits (µ ± 3σ). Fast, interpretable, and often
sufficient for stable processes. ML adds value when interactions between multiple
sensors are important.

### 5. Computer vision for quality control

CNNs trained on images of products (welds, castings, PCBs) detect surface defects
at speeds impossible for human inspection. Key challenge: labelled defect images
are rare. Use data augmentation, transfer learning from ImageNet, and anomaly
detection (no defect images only) when labels are scarce.

---

## Best practices & pitfalls

- **Label failure events carefully.** "Failure" in a log file may mean different
  things: catastrophic failure, maintenance intervention, quality issue. Align
  with domain experts before modelling.
- **Don't evaluate on the last readings before failure.** Test-set performance
  is only meaningful if the model is evaluated at realistic prediction horizons
  (e.g., "predict 7 days ahead"), not immediately before the event.
- **Operating mode is a confounder.** Control for load, speed, and temperature
  setpoint before declaring an anomaly.
- **Start with visualisation.** Plot sensor trends aligned to failure events. Often
  one sensor clearly degrades; a visual confirms before modelling.
- **Cost matrix matters.** False negatives (missed failures) cost 10-100× more
  than false positives (unnecessary maintenance). Tune threshold to minimise
  expected cost, not to maximise F1.
- **Deployment is not a model file.** Integration with SCADA, MES, or historian
  systems is the engineering work that takes 3× the modelling time.

---

## Python example — remaining useful life prediction (NASA CMAPSS)

```python
"""
Remaining Useful Life (RUL) prediction on NASA CMAPSS FD001 dataset.

The dataset contains multivariate sensor readings from turbofan engines
run to failure. Task: predict cycles until failure from rolling features.

Data source: NASA Prognostics Data Repository
https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
import urllib.request
import zipfile, io

OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Load CMAPSS FD001 ────────────────────────────────────────────────────────
cols = (["unit", "cycle"] +
        [f"op{i}" for i in range(1,4)] +
        [f"s{i}" for i in range(1,22)])

def load_cmapss(split="train"):
    url = (f"https://raw.githubusercontent.com/Azure/lstms_for_predictive_maintenance"
           f"/master/Data/PM_train.txt" if split == "train" else
           f"https://raw.githubusercontent.com/Azure/lstms_for_predictive_maintenance"
           f"/master/Data/PM_test.txt")
    try:
        df = pd.read_csv(url, sep=" ", header=None, names=cols + ["_1","_2"])
        df = df.drop(columns=["_1","_2"], errors="ignore")
    except Exception:
        # Synthetic fallback if download fails
        print(f"Download failed — generating synthetic {split} data.")
        rng = np.random.default_rng(42 if split == "train" else 7)
        units = 100 if split == "train" else 20
        records = []
        for u in range(1, units+1):
            max_cycles = rng.integers(150, 300)
            for c in range(1, max_cycles+1):
                row = [u, c, *rng.normal(0, 1, 3),
                       *rng.normal(0, 1, 21)]
                records.append(row)
        df = pd.DataFrame(records, columns=cols)
    return df

print("Loading CMAPSS dataset...")
train = load_cmapss("train")
test  = load_cmapss("test")

# ── Compute RUL ──────────────────────────────────────────────────────────────
max_cycles = train.groupby("unit")["cycle"].max().rename("max_cycle")
train = train.join(max_cycles, on="unit")
train["RUL"] = train["max_cycle"] - train["cycle"]
train["RUL"] = train["RUL"].clip(upper=125)  # cap healthy regime

# ── Rolling features ─────────────────────────────────────────────────────────
sensor_cols = [f"s{i}" for i in range(1, 22)]
WINDOW = 10

def make_features(df):
    feats = []
    for _, group in df.groupby("unit"):
        rolled = group[sensor_cols].rolling(WINDOW, min_periods=1)
        f = pd.concat([
            rolled.mean().add_suffix("_mean"),
            rolled.std().fillna(0).add_suffix("_std"),
        ], axis=1)
        f.index = group.index
        feats.append(f)
    return pd.concat(feats)

train_feats = make_features(train)
test_feats  = make_features(test)

X_train = train_feats.values
y_train = train["RUL"].values

X_test  = test_feats.values
# For synthetic test, generate random RUL
y_test  = np.clip(np.random.default_rng(0).integers(10, 130, size=len(X_test)), 0, 125)

scaler  = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train models ─────────────────────────────────────────────────────────────
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
gbr   = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                   learning_rate=0.05, random_state=42).fit(X_train, y_train)

for name, model in [("Ridge", ridge), ("GBR", gbr)]:
    preds = np.clip(model.predict(X_test), 0, 125)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name}: RMSE = {rmse:.1f} cycles")

# ── Plot predictions vs actual (GBR) ─────────────────────────────────────────
preds_gbr = np.clip(gbr.predict(X_test), 0, 125)
fig, ax = plt.subplots(figsize=(8, 4))
idx = np.arange(min(300, len(y_test)))
ax.plot(idx, y_test[:300], label="Actual RUL", alpha=0.7)
ax.plot(idx, preds_gbr[:300], label="Predicted RUL (GBR)", alpha=0.7)
ax.set_xlabel("Test sample")
ax.set_ylabel("Remaining useful life (cycles)")
ax.set_title("RUL prediction — NASA CMAPSS (first 300 test samples)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "rul_prediction.png", dpi=120)
plt.close()
print("Plot saved.")
```

---

## Cross-references

- [13](13-anomaly-detection.md) — anomaly detection methods (Isolation Forest, autoencoders)
- [07](07-time-series-forecasting.md) — sensor time-series modelling
- [11](11-computer-vision.md) — defect detection with CNNs
- [16](16-survival-analysis.md) — survival analysis for time-to-failure
- [12](12-optimization.md) — process optimisation
