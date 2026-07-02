"""
Project 1 — Supervised Learning End-to-End (Binary Classification)
=================================================================
Dataset : Wisconsin Breast Cancer (bundled with scikit-learn, no download).
Goal    : Predict malignant vs. benign tumours from 30 cell-nucleus features.

This script is a *template* for a real supervised workflow. It demonstrates the
practices that separate a portfolio-grade project from a notebook dump:

  1. A single sklearn Pipeline so preprocessing is fit ONLY on training folds
     (this is how you avoid data leakage — the #1 silent killer of DS projects).
  2. Stratified cross-validation for honest performance estimates.
  3. Multiple model families compared on the same footing.
  4. Threshold-independent metrics (ROC-AUC, PR-AUC) AND a calibration check,
     because a classifier that ranks well can still output badly-scaled scores.
  5. Permutation importance for model-agnostic explanation.

Run:  python train.py
Outputs: metrics.md, roc_pr.png, calibration.png, importance.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, brier_score_loss,
                             classification_report)
from sklearn.calibration import calibration_curve

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------- load & frame
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target            # target: 1 = benign, 0 = malignant
# Reframe so the POSITIVE class is the clinically important one (malignant).
y = 1 - y                                # now 1 = malignant (the event we care about)
print(f"Samples: {len(X)} | Features: {X.shape[1]} | Positive (malignant) rate: {y.mean():.3f}")

# Hold out a final test set the models never see during comparison/tuning.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- models
models = {
    "LogReg (L2)": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=1.0)),
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE)),
    ]),
    "HistGradientBoosting": Pipeline([
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]),
}

# ---------------------------------------------------------------- cross-validated comparison
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []
oof_scores = {}                          # out-of-fold predicted probabilities
for name, pipe in models.items():
    # cross_val_predict gives an honest OOF probability for every training row.
    proba = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    oof_scores[name] = proba
    rows.append({
        "model": name,
        "ROC_AUC": roc_auc_score(y_tr, proba),
        "PR_AUC":  average_precision_score(y_tr, proba),
        "Brier":   brier_score_loss(y_tr, proba),   # lower = better calibrated
    })
cv_table = pd.DataFrame(rows).set_index("model").round(4)
print("\n=== 5-fold cross-validated performance (training set) ===")
print(cv_table.to_string())

best_name = cv_table["PR_AUC"].idxmax()
best_pipe = models[best_name].fit(X_tr, y_tr)      # refit on ALL training data

# ---------------------------------------------------------------- final held-out test
test_proba = best_pipe.predict_proba(X_te)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)
test_row = {
    "ROC_AUC": roc_auc_score(y_te, test_proba),
    "PR_AUC":  average_precision_score(y_te, test_proba),
    "Brier":   brier_score_loss(y_te, test_proba),
}
print(f"\n=== Held-out TEST performance ({best_name}) ===")
print(pd.Series(test_row).round(4).to_string())
print("\n" + classification_report(y_te, test_pred, target_names=["benign", "malignant"]))

# ---------------------------------------------------------------- plots
# 1) ROC + PR curves (OOF, best model)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for name, proba in oof_scores.items():
    fpr, tpr, _ = roc_curve(y_tr, proba)
    ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_tr, proba):.3f})")
    prec, rec, _ = precision_recall_curve(y_tr, proba)
    ax[1].plot(rec, prec, label=f"{name} ({average_precision_score(y_tr, proba):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (out-of-fold)")
ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (out-of-fold)")
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("roc_pr.png", dpi=120); plt.close(fig)

# 2) Calibration curve — do predicted probabilities match observed frequencies?
fig, axc = plt.subplots(figsize=(5.2, 5))
for name, proba in oof_scores.items():
    frac_pos, mean_pred = calibration_curve(y_tr, proba, n_bins=10, strategy="quantile")
    axc.plot(mean_pred, frac_pos, "o-", label=name, ms=4)
axc.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
axc.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
        title="Calibration (out-of-fold)")
axc.legend(fontsize=8); axc.grid(alpha=.3)
fig.tight_layout(); fig.savefig("calibration.png", dpi=120); plt.close(fig)

# 3) Permutation importance on the held-out test set (model-agnostic)
perm = permutation_importance(best_pipe, X_te, y_te, n_repeats=20,
                              random_state=RANDOM_STATE, scoring="roc_auc")
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values()[-12:]
fig, axi = plt.subplots(figsize=(6.5, 5))
imp.plot.barh(ax=axi)
axi.set(xlabel="Drop in ROC-AUC when feature is shuffled",
        title=f"Permutation importance — {best_name}")
fig.tight_layout(); fig.savefig("importance.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- write report
with open("metrics.md", "w") as f:
    f.write(f"# Project 1 results — {best_name} selected\n\n")
    f.write("## 5-fold cross-validated (training set)\n\n")
    f.write(cv_table.to_markdown() + "\n\n")
    f.write("## Held-out test set\n\n")
    f.write(pd.Series(test_row).round(4).to_frame("value").to_markdown() + "\n\n")
    f.write("Interpretation: PR-AUC is the headline metric because the positive "
            "(malignant) class is the minority and the cost of a false negative is "
            "high. The calibration plot confirms whether the probabilities can be "
            "trusted as probabilities, not just as a ranking.\n")
print("\nSaved: metrics.md, roc_pr.png, calibration.png, importance.png")
