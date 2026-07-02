# 04 · Evaluation & Validation

A model is only as trustworthy as the scheme you used to evaluate it. This is the
most under-appreciated skill in the field — and the place where most published
results are wrong.

---

## Validation schemes — match the split to deployment

The golden rule: **your validation must mimic how the model will actually be used.**

| Scheme | When to use | Pitfall |
|--------|-------------|---------|
| **Hold-out split** | Large data, fast iteration | Noisy estimate on small sets; wasteful |
| **Stratified k-fold CV** | Tabular i.i.d. data, classification | Don't use for time series |
| **GroupKFold** | Rows cluster into entities (users, patients) | Must define groups carefully |
| **Rolling-origin CV** | Time series — train on past, test on future | Slow; pick horizon to match deployment |
| **Nested CV** | Hyperparameter tuning *and* performance estimation | Computationally expensive but the only unbiased scheme |

**Nested CV explained:** when you tune hyperparameters, the outer loop estimates
generalisation performance while the inner loop does model selection. Using the
same fold for both optimistically biases the reported score — sometimes by 3–5
percentage points on small datasets.

---

## Metrics — pick for the decision, not habit

### Regression

| Metric | Formula | Use when |
|--------|---------|----------|
| **RMSE** | √mean((y-ŷ)²) | Large errors disproportionately bad; same scale as y |
| **MAE** | mean(\|y-ŷ\|) | Robust, interpretable "typical error" |
| **MAPE** | mean(\|y-ŷ\|/\|y\|) × 100 | Relative errors matter; **fails if y near zero** |
| **MASE** | MAE / naive MAE | Scale-free; <1 means you beat the naive baseline (best for time series) |
| **R²** | 1 - SS_res/SS_tot | Communicating % variance explained; misleading outside training range |

### Classification

**Accuracy is almost always the wrong headline metric** on imbalanced data
(99% accuracy by predicting "no fraud" every time catches zero fraud).

| Metric | Meaning | Use when |
|--------|---------|----------|
| **Precision** | TP / (TP + FP) | False alarms are expensive |
| **Recall / sensitivity** | TP / (TP + FN) | Missing positives is expensive |
| **F1** | Harmonic mean precision × recall | Single-number threshold-dependent summary |
| **ROC-AUC** | Area under TPR vs FPR curve | Ranking quality; threshold-independent |
| **PR-AUC** | Area under precision-recall curve | Correct choice when positives are rare (<5%) |
| **Brier score** | mean((p - y)²) | Quality of probabilities, not just ranking |
| **Log-loss** | -mean(y log p + (1-y) log(1-p)) | Proper scoring rule for probabilistic output |

**Threshold choice is a business decision.** The default 0.5 is arbitrary.
Choose from the precision/recall curve based on the relative cost of false
positives vs. false negatives. A fraud model where FN costs 100× more than FP
should have a very different threshold than a spam filter.

---

## Calibration — the forgotten check

A model can rank perfectly (great AUC) yet output badly-scaled probabilities.
If the probability is used as a probability (risk scores, expected-value decisions),
**calibration must be checked**:

- Among all predictions of "20% probability", does ~20% actually occur?
- Plot **reliability diagram**: x = mean predicted probability in bins, y = actual
  fraction positive. Perfect calibration = diagonal line.
- Measure with **Brier score** (lower is better) or **Expected Calibration Error (ECE)**.
- Fix miscalibration: **Platt scaling** (logistic regression on scores) or
  **isotonic regression** on a held-out calibration set.

---

## Bias–variance trade-off

| Symptom | Cause | Fix |
|---------|-------|-----|
| Train score ≈ val score, both poor | High bias (underfitting) | More complex model, more features, less regularisation |
| Train score >> val score, big gap | High variance (overfitting) | More data, more regularisation, simpler model |
| Both near perfect but test is poor | Leakage → see [03](03-data-and-feature-engineering.md) | Hunt for the leak |

**Learning curves** (score vs. training-set size) diagnose which problem you have
and whether more data would help. If val score plateaus before 100% training data,
adding data helps. If train and val are both flat and poor, the model is underfitting.

---

## Python example — evaluation pipeline with all the right checks

```python
"""
Demonstrates the full evaluation workflow:
  - Stratified CV vs. hold-out
  - Multiple metrics (including PR-AUC and calibration)
  - Dumb baseline comparison
  - Threshold selection from cost matrix
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_recall_curve, PrecisionRecallDisplay
)

rng = np.random.default_rng(42)

# ── Dataset: imbalanced binary (10% positive) ─────────────────────────────────
X, y = make_classification(n_samples=3000, n_features=15, weights=[0.9, 0.1],
                            random_state=42)

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Majority class (baseline)": Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent"))
    ]),
    "Logistic Regression": Pipeline([
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scl", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Positive rate: {y.mean():.1%}\n")
results = {}
probas = {}

for name, pipe in models.items():
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    probas[name] = proba
    results[name] = {
        "Accuracy":  ((proba >= 0.5).astype(int) == y).mean(),
        "ROC-AUC":   roc_auc_score(y, proba),
        "PR-AUC":    average_precision_score(y, proba),
        "Brier":     brier_score_loss(y, proba),
    }

df = pd.DataFrame(results).T
print(df.round(3).to_string())
print("\nNote: baseline 'accuracy' looks fine (90%) — but PR-AUC is 0.1 (random).")

# ── Calibration plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot([0,1],[0,1],"k--", label="Perfect calibration")
for name, proba in list(probas.items())[1:]:
    frac, mean_pred = calibration_curve(y, proba, n_bins=10)
    ax.plot(mean_pred, frac, "o-", label=name)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Actual fraction positive")
ax.set_title("Calibration (reliability diagram)")
ax.legend(fontsize=8)

# ── PR curve + threshold selection ───────────────────────────────────────────
ax = axes[1]
best_name = max(results, key=lambda n: results[n]["PR-AUC"] if n != "Majority class (baseline)" else 0)
proba_best = probas[best_name]
PrecisionRecallDisplay.from_predictions(y, proba_best, ax=ax, name=best_name)

# Find threshold that maximises F1
precision_, recall_, thresholds_ = precision_recall_curve(y, proba_best)
f1s = 2 * precision_[:-1] * recall_[:-1] / (precision_[:-1] + recall_[:-1] + 1e-9)
best_thresh = thresholds_[np.argmax(f1s)]
ax.axvline(recall_[np.argmax(f1s)], color="red", linestyle="--", alpha=0.7,
           label=f"Best F1 threshold = {best_thresh:.2f}")
ax.set_title(f"Precision-Recall — {best_name}")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("evaluation_complete.png", dpi=120)
plt.close()

# ── Cost-based threshold ──────────────────────────────────────────────────────
# FN costs 10x more than FP (e.g. missed fraud vs. declined legitimate transaction)
FN_COST, FP_COST = 10, 1
costs = FN_COST * (1 - recall_[:-1]) + FP_COST * (1 - precision_[:-1])
cost_thresh = thresholds_[np.argmin(costs)]
print(f"\nMax-F1 threshold: {best_thresh:.3f}")
print(f"Min-cost threshold (FN=10×FP): {cost_thresh:.3f}")
print("These differ — choose threshold from business cost, not arbitrary 0.5")
```

---

## Sanity checks that catch real bugs

1. **Always beat the dumb baseline** (majority class, mean, last value). A model
   that can't beat it is not a result.
2. **Shuffle the labels** and retrain: performance should collapse to chance. If it
   doesn't, there is leakage ([03](03-data-and-feature-engineering.md)).
3. **Inspect the errors** — the systematic mistakes tell you where model and data
   are weak.
4. **Check calibration** if probabilities will be used as probabilities.
5. **Report the baseline** in every paper, report, and presentation. A model
   that improves AUC from 0.95 to 0.96 is very different from one that improves
   from 0.55 to 0.96.
