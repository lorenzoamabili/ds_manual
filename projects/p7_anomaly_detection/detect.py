"""
Project 7 - Anomaly Detection (Imbalanced Classification)
==========================================================
Dataset : Synthetic credit-card-style fraud data (seeded, reproducible).
Goal    : Detect rare anomalies (0.5% positive rate) without labelled
          failure data, and show why accuracy is a useless metric here.

Real lesson: On extreme class imbalance, accuracy is meaningless.
PR-AUC and F1 at the right threshold are what matter. Isolation Forest
beats naive classification when labels are scarce; supervised GBM wins
when you have enough labelled examples.

Run:  python detect.py
Outputs: metrics.md, anomaly_scores.png, pr_curve.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score,
    PrecisionRecallDisplay,
)

OUT = Path(__file__).parent
rng = np.random.default_rng(42)

# -- Simulate imbalanced fraud-like dataset -----------------------------------
# 99.5% normal, 0.5% fraud - typical of real card fraud
print("Generating synthetic fraud dataset (n=20 000, fraud_rate=0.5%)...")
X, y = make_classification(
    n_samples=20_000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    weights=[0.995, 0.005],
    flip_y=0.001,
    random_state=42,
)
print(f"  Positive rate: {y.mean():.2%}  ({y.sum()} fraud / {len(y)} total)")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
scaler  = StandardScaler().fit(X_tr)
X_tr_s  = scaler.transform(X_tr)
X_te_s  = scaler.transform(X_te)

# -- Baselines & models --------------------------------------------------------
# 1. Naive: predict-all-normal baseline
naive_pred  = np.zeros(len(y_te), dtype=int)

# 2. Isolation Forest (unsupervised - no labels used in training)
iso = IsolationForest(n_estimators=200, contamination=0.005, random_state=42)
iso.fit(X_tr_s)
iso_scores = -iso.score_samples(X_te_s)   # higher = more anomalous
iso_pred   = (iso_scores >= np.quantile(iso_scores, 0.995)).astype(int)

# 3. Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.005, novelty=True)
lof.fit(X_tr_s)
lof_scores = -lof.score_samples(X_te_s)
lof_pred   = (lof_scores >= np.quantile(lof_scores, 0.995)).astype(int)

# 4. Supervised GBM (uses labels - best case ceiling)
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
gbm.fit(X_tr_s, y_tr)
gbm_proba  = gbm.predict_proba(X_te_s)[:, 1]
gbm_pred   = (gbm_proba >= 0.5).astype(int)

# -- Evaluate ------------------------------------------------------------------
results = {}

def evaluate(name, y_true, y_pred, y_score):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "PR-AUC":    average_precision_score(y_true, y_score),
        "ROC-AUC":   roc_auc_score(y_true, y_score),
    }

results["Naive (all normal)"]  = evaluate("Naive",  y_te, naive_pred, naive_pred.astype(float))
results["Isolation Forest"]    = evaluate("IsoF",   y_te, iso_pred,   iso_scores)
results["Local Outlier Factor"]= evaluate("LOF",    y_te, lof_pred,   lof_scores)
results["GBM (supervised)"]    = evaluate("GBM",    y_te, gbm_pred,   gbm_proba)

df_results = pd.DataFrame(results).T
print("\n" + df_results.round(3).to_string())

# -- Write metrics.md ---------------------------------------------------------
with open(OUT / "metrics.md", "w") as f:
    f.write("# P7 · Anomaly Detection - Metrics\n\n")
    f.write(f"**Dataset:** Synthetic fraud (n=20 000, fraud_rate~0.5%)\n\n")
    f.write(df_results.round(3).to_markdown())
    f.write("\n\n## Key insight\n\n")
    f.write("The naive model (predict all normal) achieves **99.5% accuracy** - "
            "yet it catches zero fraud.\n")
    f.write("This is why accuracy is misleading on imbalanced data.\n")
    f.write("**PR-AUC** is the correct primary metric: it measures performance "
            "across all thresholds weighted toward the rare class.\n")

# -- Plot 1: anomaly score distributions --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, scores) in zip(axes, [("Isolation Forest", iso_scores),
                                      ("GBM probability", gbm_proba)]):
    ax.hist(scores[y_te == 0], bins=60, alpha=0.6, label="Normal", density=True)
    ax.hist(scores[y_te == 1], bins=60, alpha=0.8, label="Fraud",  density=True, color="red")
    ax.set_title(f"Score distribution - {name}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.legend()
plt.suptitle("Do fraud scores separate from normal scores?", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "anomaly_scores.png", dpi=120)
plt.close()

# -- Plot 2: Precision-Recall curves ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for label, scores in [("Isolation Forest", iso_scores),
                       ("LOF",             lof_scores),
                       ("GBM (supervised)",gbm_proba)]:
    PrecisionRecallDisplay.from_predictions(y_te, scores, name=label, ax=ax)
ax.set_title("Precision-Recall curve comparison\n(imbalanced data: baseline = 0.5%)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "pr_curve.png", dpi=120)
plt.close()

print("\nOutputs written: metrics.md, anomaly_scores.png, pr_curve.png")
print("\nLesson: naive accuracy = 99.5% but catches zero fraud.")
print(f"GBM PR-AUC = {results['GBM (supervised)']['PR-AUC']:.3f} - this is the real signal.")
