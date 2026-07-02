# 13 · Anomaly Detection

Flagging the abnormal: fraud, intrusions, equipment failure, data-quality issues.
Defined by a hard constraint — anomalies are **rare and often unlabelled**, so you
usually can't treat it as ordinary supervised classification.

## Approaches by data situation
| Situation | Approach |
|-----------|----------|
| **No labels** (typical) | Unsupervised: model "normal", flag deviations |
| **A few labels** | Semi-supervised: train on *normal only*, flag what doesn't fit |
| **Enough labels of both** | Supervised classification with heavy imbalance handling ([05](05-supervised-learning.md)) |

## Methods
| Method | Idea | Good for |
|--------|------|----------|
| **Statistical (z-score, IQR, robust MAD)** | Points far from the centre | Univariate, well-understood distributions; the honest first pass |
| **Isolation Forest** | Anomalies are *easier to isolate* by random splits | Tabular, higher-dimensional; fast, few assumptions; strong default |
| **Local Outlier Factor (LOF)** | Compare a point's local density to its neighbours' | Clusters of varying density |
| **One-Class SVM** | Learn a boundary around normal data | Small/medium, well-behaved data |
| **Autoencoders** | Train to reconstruct normal data; high reconstruction error = anomaly | High-dimensional (images, sensor arrays), lots of normal data |
| **Time-series methods** | Forecast the series; large residual = anomaly; or STL residuals, change-point detection | Monitoring, telemetry, seasonal signals |

## The hard parts
- **Defining "normal" is the whole game**, and normal drifts. A model trained on
  last quarter flags this quarter's *legitimate* new behaviour. Plan to retrain.
- **Threshold setting** — anomaly scores are relative. Set the cut by the
  **alert budget** the team can actually investigate, not an arbitrary cutoff. Too
  sensitive → alert fatigue → ignored alerts → worse than nothing.
- **Evaluation with scarce labels** — when you have *some* labelled anomalies, use
  **precision@k** (of the top-k flagged, how many are real) and PR-AUC, because
  ROC-AUC looks flatteringly high when anomalies are <1%.
- **Context matters** — a value can be normal globally but anomalous *for this
  user / this hour / this machine*. Contextual and collective anomalies need
  features that encode the context, or per-entity models.

## A pragmatic recipe
Start with **robust statistical rules** (they're explainable and catch the obvious
cases), add **Isolation Forest** for multivariate structure, feed results to a
human review queue, and use the humans' verdicts to build a labelled set over time
— eventually enabling a supervised model for the well-understood anomaly types
while unsupervised methods keep watch for the novel ones.

---

## Python example — Isolation Forest vs. z-score on multivariate data

```python
"""
Anomaly detection: z-score univariate vs. Isolation Forest multivariate.
Shows that z-score misses anomalies in high-dimensional interaction space.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(42)

# Normal data: 2D Gaussian
n_normal   = 500
n_anomaly  = 20
X_normal   = rng.multivariate_normal([0, 0], [[1, 0.8],[0.8, 1]], n_normal)
# Anomalies: on the off-diagonal (normal marginals, anomalous joint)
X_anomaly  = rng.multivariate_normal([0, 0], [[1, -0.9],[-0.9, 1]], n_anomaly)
X          = np.vstack([X_normal, X_anomaly])
y          = np.array([0]*n_normal + [1]*n_anomaly)

# ── Z-score (per-feature, univariate) ────────────────────────────────────────
z_scores    = np.abs((X - X.mean(0)) / X.std(0)).max(axis=1)
pr_zscore   = average_precision_score(y, z_scores)

# ── Isolation Forest ──────────────────────────────────────────────────────────
iso         = IsolationForest(n_estimators=200, contamination=0.04, random_state=42)
iso.fit(X[y == 0])
iso_scores  = -iso.score_samples(X)
pr_iso      = average_precision_score(y, iso_scores)

print(f"Z-score  PR-AUC: {pr_zscore:.3f}")
print(f"Iso Forest PR-AUC: {pr_iso:.3f}")
print("z-score misses joint anomalies because each feature looks normal alone")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, scores, title in zip(axes,
    [z_scores, iso_scores],
    ["Z-score (univariate)", "Isolation Forest (multivariate)"]):
    sc = ax.scatter(X[:,0], X[:,1], c=scores, cmap="Reds", s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Anomaly score")
    ax.scatter(X_anomaly[:,0], X_anomaly[:,1], marker="x", s=80,
               c="blue", label="True anomalies")
    ax.set_title(title); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("anomaly_comparison.png", dpi=120)
plt.close()
```

---

## Cross-references

- [P7](../projects/p7_anomaly_detection) — full anomaly detection project
- [31](31-fintech.md) — fraud detection in financial services
- [34](34-manufacturing.md) — sensor anomaly in manufacturing
- [37](37-cybersecurity.md) — intrusion detection
