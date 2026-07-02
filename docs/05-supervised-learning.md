# 05 · Supervised Learning

Predicting a labelled outcome from features. The workhorse of applied DS. Paired
project: [P1 — breast-cancer classification](../projects/p1_supervised_learning).

## Choosing a model family

| Family | Reach for it when | Watch out for |
|--------|-------------------|---------------|
| **Linear / logistic regression** | You want a strong, interpretable baseline; coefficients matter; data is roughly linear | Needs scaling + explicit interactions; sensitive to outliers |
| **Regularised linear (Ridge/Lasso/ElasticNet)** | Many features, some irrelevant; want built-in selection (Lasso) | Choose λ by CV; Lasso arbitrary among correlated features |
| **Tree ensembles — Random Forest** | Tabular data, non-linear, minimal tuning, robust default | Larger models; less interpretable than one tree |
| **Gradient boosting (XGBoost / LightGBM / HistGB)** | You want top tabular accuracy | Needs tuning; can overfit; watch learning rate × n_estimators |
| **SVM** | Small/medium data, clear margin, kernels for non-linearity | Scales poorly to big n; needs scaling + kernel choice |
| **k-NN** | Simple, local structure, quick baseline | Slow at inference; curse of dimensionality; must scale |
| **Neural nets** | Unstructured data (images, text, audio) or huge tabular data | Overkill and often *worse* than boosting on typical tabular data |

**The empirical truth of tabular ML:** gradient-boosted trees win most tabular
competitions, but a well-regularised linear model is often within a whisker and
far more interpretable. In [P1](../projects/p1_supervised_learning), plain
logistic regression *beat* Random Forest and gradient boosting — start simple and
make complexity earn its place.

## Regularisation, in one paragraph

Regularisation trades a little bias for a lot less variance by penalising model
complexity. **L2 (Ridge)** shrinks coefficients smoothly; **L1 (Lasso)** drives
some to exactly zero (feature selection); **ElasticNet** blends both. In trees,
"regularisation" is `max_depth`, `min_samples_leaf`, `learning_rate`, and
subsampling. Tune the strength by cross-validation — never by test-set score.

## Class imbalance

Common in fraud, churn, disease. Options, roughly in order of preference:

1. **Do nothing to the data; fix the metric and threshold** (PR-AUC, cost-based
   threshold). Often sufficient.
2. **Class weights** (`class_weight="balanced"`) — cheap and effective.
3. **Resampling** (SMOTE oversampling, or undersampling) — can help, but SMOTE
   must be applied *inside* the CV fold, never before splitting (leakage).

## Interpretation you can defend

- **Coefficients** (linear) — direct partial effects, if specification is sound.
- **Permutation importance** — model-agnostic; how much does shuffling a feature
  hurt the score? (Used in [P1](../projects/p1_supervised_learning).) Prefer it to
  impurity-based importance, which is biased toward high-cardinality features.
- **SHAP values** — the current standard for per-prediction explanations; additive,
  consistent, and the plots communicate well to stakeholders.
- **Partial dependence / ICE** — show the shape of a feature's effect.

## The workflow, distilled

Baseline → Pipeline (no leakage) → cross-validated comparison of 2–3 families →
pick by the *business-relevant* metric → refit on all training data → **single**
held-out test → calibrate if probabilities matter → explain → document the
assumptions. That is exactly the arc of P1.

---

## Python example — full supervised workflow on breast cancer

```python
"""
Full supervised learning workflow:
  1. Baseline (majority-class)
  2. Logistic regression with regularisation
  3. Random Forest
  4. Gradient boosting (HistGradientBoosting)
  5. Comparison by ROC-AUC and PR-AUC (imbalanced-aware)
  6. Calibration check
  7. SHAP-style permutation importance
  8. Single held-out test (touched ONCE at the very end)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
feat_names = X.columns.tolist()

# ── Hold-out split — test set sealed until final step ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Model grid ────────────────────────────────────────────────────────────────
models = {
    "Baseline (majority)": Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent")),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED)),
    ]),
    "Hist Gradient Boosting": Pipeline([
        ("clf", HistGradientBoostingClassifier(max_iter=300, random_state=SEED)),
    ]),
}

# ── CV comparison ─────────────────────────────────────────────────────────────
rows = []
for name, pipe in models.items():
    auc  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    prauc = cross_val_score(pipe, X_train, y_train, cv=cv,
                            scoring="average_precision").mean()
    rows.append({"Model": name, "CV ROC-AUC": round(auc, 4),
                 "CV PR-AUC": round(prauc, 4)})
    print(f"{name:<30} ROC-AUC={auc:.4f}  PR-AUC={prauc:.4f}")

# ── Final evaluation on held-out test (ONCE) ─────────────────────────────────
best_pipe = models["Logistic Regression"]   # pick from CV, not test
best_pipe.fit(X_train, y_train)
y_prob = best_pipe.predict_proba(X_test)[:, 1]
test_auc   = roc_auc_score(y_test, y_prob)
test_prauc = average_precision_score(y_test, y_prob)
print(f"\nTest ROC-AUC: {test_auc:.4f}  |  Test PR-AUC: {test_prauc:.4f}")

# ── Calibration plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=8)
axes[0].plot([0, 1], [0, 1], "k--", label="Perfect")
axes[0].plot(mean_pred, frac_pos, "o-", label="Logistic Reg")
axes[0].set(xlabel="Mean predicted prob", ylabel="Fraction positive",
            title="Calibration curve")
axes[0].legend()

# ── Permutation importance ────────────────────────────────────────────────────
result = permutation_importance(best_pipe, X_test, y_test,
                                n_repeats=20, random_state=SEED,
                                scoring="roc_auc")
imp_df = pd.DataFrame({"feature": feat_names,
                        "importance": result.importances_mean}).sort_values(
    "importance", ascending=True).tail(12)
imp_df.plot.barh(x="feature", y="importance", ax=axes[1], legend=False)
axes[1].set(title="Permutation importance (top 12)", xlabel="ROC-AUC drop")
plt.tight_layout()
plt.savefig("supervised_workflow.png", dpi=120)
plt.close()
print("Plot saved: supervised_workflow.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\nCV comparison:")
print(pd.DataFrame(rows).to_string(index=False))
```

---

## Cross-references

- [P1](../projects/p1_supervised_learning) — end-to-end script for this workflow
- [03](03-data-and-feature-engineering.md) — Pipeline + leakage prevention
- [04](04-evaluation-and-validation.md) — ROC vs PR-AUC, calibration, cost threshold
- [19](19-responsible-ai-and-fairness.md) — fairness audit after you have a model
