# 31 · FinTech

> Applying data science to financial services — fraud detection, credit risk,
> and algorithmic pricing — where model errors have direct monetary and regulatory cost.

---

## Why data science here

Finance was an early adopter of statistical modelling (actuarial science predates
ML by a century), so the domain has mature expectations: models must be
**explainable** (regulators demand it), **[calibrated](04-evaluation-and-validation.md)** (a predicted 2% default
rate must actually produce ~2% defaults), and **auditable** (model risk management
is a legal requirement in banking under SR 11-7 and BCBS 239).

The data is rich but dirty: transaction logs, credit bureau files, market tick data,
alternative data (mobile metadata, web browsing). [Class imbalance](05-supervised-learning.md) is extreme in
fraud (< 0.1% positive rate). [Temporal leakage](03-data-and-feature-engineering.md) is the most common modelling error —
using features that would not be available at prediction time.

Three things distinguish a finance-literate data scientist from a generic one:
(1) understanding of the **point-in-time problem** (features must reflect what was
known *at* decision time, not what we later learned), (2) appreciation for
**calibration** (not just discrimination), (3) familiarity with **regulatory
constraints** (GDPR, FCRA, EU AI Act all restrict what features you can use).

---

## Signature problems

| Problem | Formulation | Typical model |
|---------|-------------|---------------|
| Fraud detection | Binary classification; extreme imbalance | [XGBoost](05-supervised-learning.md) + threshold tuning; [Isolation Forest](13-anomaly-detection.md) |
| Credit scoring | PD (probability of default) estimation | [Logistic regression](05-supervised-learning.md) (scorecard), GBM |
| Credit limit / pricing | Regression or policy optimisation | GLM, causal uplift |
| Churn (banking) | Which customers will close accounts? | [Survival analysis](16-survival-analysis.md), classification |
| Market risk | VaR, expected shortfall | Historical simulation, GARCH |
| Algorithmic trading | Signal generation + execution | Time series + optimisation (NOT covered here) |
| AML / transaction monitoring | Flag suspicious sequences | Graph anomaly, sequence models |

---

## Key techniques

### 1. Credit scorecard (logistic regression + WoE)

The industry standard for retail credit is logistic regression with Weight of
Evidence (WoE) encoding. WoE transforms each predictor into `ln(P(event) /
P(non-event))` within bins, producing a score additive on the log-odds scale.
Scorecards survive regulatory scrutiny because every point is traceable to a
specific input value.

**Pitfall:** WoE binning must happen *inside* the CV fold. Binning on the full
dataset and then cross-validating is leakage — a common mistake in tutorial code.

### 2. [Gradient boosting](05-supervised-learning.md) for fraud

GBMs (XGBoost, [LightGBM](05-supervised-learning.md)) dominate fraud because they handle the non-linear
feature interactions that characterise fraud rings (amount × merchant category ×
time-of-day patterns). Use [PR-AUC](04-evaluation-and-validation.md) as the primary metric, not AUC-ROC — with
extreme imbalance, ROC is misleadingly optimistic.

**Pitfall:** transaction-level fraud data has user-level [clustering](06-unsupervised-learning.md). Split by
**user**, not by transaction, to avoid leakage where train and test share
transactions from the same fraudster.

### 3. Model calibration

A fraud model that outputs 0.9 for a transaction means "90% chance of fraud."
If the actual fraud rate among those transactions is 40%, the model is
miscalibrated. Use Platt scaling or isotonic regression post-hoc. Evaluate with
a calibration plot (predicted vs. actual probability in deciles).

### 4. Temporal [cross-validation](04-evaluation-and-validation.md)

Financial models degrade over time ([concept drift](14-mlops-and-productionization.md)). Always evaluate with a
**walk-forward** (expanding window) or **rolling-window** backtest. A single
random train/test split on historical data overstates real performance.

### 5. [SHAP](05-supervised-learning.md) for explainability

Regulators (and customers) can ask "why was my loan denied?" SHAP values give a
per-prediction, additive decomposition of the model output. For scorecards,
SHAP aligns with the WoE contribution. For GBMs, SHAP is the only defensible
explanation method.

---

## Best practices & pitfalls

- **Point-in-time features are mandatory.** Never join on a key that produces
  values from the future. Re-build features from raw event tables with cutoff dates.
- **Reject inference.** Your training data only includes applicants you approved.
  Declined applicants have unknown outcomes — this biases PD estimates. Acknowledge
  it; partially correct with reject inference methods if needed.
- **Regulatory constraints on features.** Protected attributes (race, gender,
  religion) are illegal in credit decisions in most jurisdictions. Proxies
  (zip code, surname) can also be illegal. Know the rules before you model.
- **Two-model approach for fraud.** Train a real-time model (simple, fast) for
  transaction-time blocking, and an offline model (slower, richer features) for
  post-hoc review. They serve different latency budgets.
- **PR-AUC, not accuracy.** With 0.1% fraud rate, a model that predicts "never
  fraud" gets 99.9% accuracy. Meaningless. Use [precision-recall](04-evaluation-and-validation.md).
- **Champion/challenger.** Production models should always have a logged baseline
  (champion) and a challenger getting a small traffic slice. Never deploy blind.

---

## Python example — credit scoring with logistic regression

```python
"""
Credit scoring: logistic regression with calibration check.

Uses the 'credit-g' (German Credit) dataset from OpenML — a standard
benchmark for credit scoring with 1000 applicants, 20 features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report
)

rng = np.random.default_rng(42)
OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading German Credit dataset from OpenML...")
credit = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
X, y = credit.data, (credit.target == "bad").astype(int)

cat_cols = X.select_dtypes("category").columns.tolist()
num_cols = X.select_dtypes(["int64", "float64"]).columns.tolist()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
])

lr  = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42))])
gbm = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Cross-validated probability estimates ────────────────────────────────────
proba_lr  = cross_val_predict(lr,  X, y, cv=cv, method="predict_proba")[:, 1]
proba_gbm = cross_val_predict(gbm, X, y, cv=cv, method="predict_proba")[:, 1]

results = {}
for name, proba in [("LogReg", proba_lr), ("GBM", proba_gbm)]:
    results[name] = {
        "ROC-AUC":  roc_auc_score(y, proba),
        "PR-AUC":   average_precision_score(y, proba),
        "Brier":    brier_score_loss(y, proba),
    }
    print(f"\n{name}: ROC-AUC={results[name]['ROC-AUC']:.3f} | "
          f"PR-AUC={results[name]['PR-AUC']:.3f} | "
          f"Brier={results[name]['Brier']:.3f}")

# ── Calibration plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
for name, proba in [("LogReg", proba_lr), ("GBM", proba_gbm)]:
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", label=name)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration curve — credit scoring")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "calibration.png", dpi=120)
plt.close()

print("\nCalibration plot saved.")
print("\nKey insight: PR-AUC is more informative than ROC-AUC when defaults are")
print("a minority class. Calibration matters when the predicted probability")
print("feeds a pricing or limit decision — not just a binary flag.")
```

---

## Cross-references

- [05](05-supervised-learning.md) — classification models, calibration, imbalance
- [13](13-anomaly-detection.md) — unsupervised fraud detection
- [04](04-evaluation-and-validation.md) — PR-AUC, calibration, temporal CV
- [09](09-causal-inference-and-experimentation.md) — causal pricing / uplift
- [19](19-responsible-ai-and-fairness.md) — fairness constraints in credit
