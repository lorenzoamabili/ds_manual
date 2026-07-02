# 37 · Cybersecurity Data Science

> Detecting intrusions, fraud, and malicious behaviour in high-dimensional, high-volume
> network and endpoint telemetry — where the adversary actively adapts to evade detection.

---

## Why data science here

Cybersecurity is an adversarial domain: the moment a detection model is deployed,
motivated adversaries probe it for blind spots and adapt. This distinguishes it
from most ML problems where the data-generating process is stable. A model that
achieves 99.9% accuracy in testing may be useless against a novel attack variant
that exploits its known decision boundary.

The data is high-dimensional and highly imbalanced: billions of network events per
day in a large enterprise, of which a vanishingly small fraction are malicious.
False positives are expensive (analyst time to investigate every alert), and false
negatives are catastrophic (missed breaches). The operational constraint is strict:
models must run at wire speed or near-wire speed.

Two broad paradigms: **signature-based detection** (rules matching known bad patterns
— fast, zero false positives on known threats, blind to novel ones) and **anomaly-
based detection** (ML on "normal" behaviour — catches novel threats, generates more
false positives). In practice, production systems layer both.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Intrusion detection | Is this network flow malicious? | Binary classification (rule + GBM) |
| Malware classification | Which malware family is this binary? | Multi-class classification on PE features |
| Insider threat | Is this employee behaving abnormally? | User/Entity Behaviour Analytics (UEBA) |
| Phishing detection | Is this email / URL malicious? | NLP + URL feature classification |
| Lateral movement | Is this authentication sequence anomalous? | Sequence anomaly, [graph analysis](21-graph-and-network-analysis.md) |
| Vulnerability prioritisation | Which CVEs should we patch first? | Regression/ranking on CVSS + exploit availability |
| Log [anomaly detection](13-anomaly-detection.md) | Are these log lines indicative of an attack? | [Isolation Forest](13-anomaly-detection.md), log parsing + anomaly |

---

## Key techniques

### 1. Isolation Forest for network anomaly

Fit on "normal" traffic. Score each connection by isolation depth. Works well
for point anomalies in individual flows. For contextual anomalies (normal activity
at an unusual time), add temporal features (hour, day) and per-entity baselines.

**Pitfall:** the contamination parameter must be set carefully. If 1 in 10,000
flows are malicious, setting contamination=0.01 will flag far too many legitimate
flows.

### 2. User and Entity Behaviour Analytics (UEBA)

Build a statistical baseline of each user's normal behaviour: typical login hours,
typical data volumes, typical remote hosts. Flag deviations. Rolling z-score or
Mahalanobis distance per user. More sophisticated: autoencoder per user trained on
their own history.

**Pitfall:** new employees have no baseline; accounts that are gradually
compromised look normal until the exfiltration event. Use peer-group baselines
(compare to similar users) as a fallback.

### 3. Graph-based detection

Attacks propagate through networks: a compromised host authenticates to a server,
which authenticates to a database. Graph features (PageRank, [betweenness centrality](21-graph-and-network-analysis.md),
community membership) on the authentication graph reveal lateral movement patterns
invisible in individual flow analysis.

### 4. NLP on logs and phishing

Log lines are semi-structured text. Drain or other log parsers extract templates;
anomalous templates (sequences or frequencies) are flagged. Phishing detection:
URL lexical features (length, entropy, special characters, TLD) + optional DOM
features fed to a classifier.

### 5. Threshold engineering

At sub-1% base rate, any threshold [calibration](04-evaluation-and-validation.md) error has enormous impact on alert
volume. Plot [precision-recall](04-evaluation-and-validation.md) at every threshold. Work with analysts to understand
how many alerts per day they can triage, then set the threshold to that volume —
this is the operational constraint, not an arbitrary 0.5.

---

## Best practices & pitfalls

- **Adversarial evaluation is mandatory.** Report performance on both known
  attack variants and held-out novel variants. AUC on the original test set is
  insufficient.
- **False positive budget is the binding constraint.** If analysts can triage 50
  alerts per day and your model fires 5,000, you have 4,950 wasted hours and a
  team that stops trusting the system. Set thresholds to operational budget, not
  to academic metrics.
- **Label quality is terrible.** Security logs are rarely labelled. Ground truth
  comes from incident reports, which are incomplete and delayed. Many threat
  hunts produce retroactive labels months later.
- **Distribution shift is the norm.** Network topology changes, new applications
  deploy, user behaviour shifts seasonally. Retrain frequently; monitor for [drift](14-mlops-and-productionization.md).
- **Explainability matters for triage.** An analyst can't act on a score of 0.87
  without context. [SHAP values](05-supervised-learning.md) or rule explanations that say "unusually large
  outbound transfer at 2AM from a host that has never connected to this IP"
  are actionable.

---

## Python example — network intrusion detection with Isolation Forest

```python
"""
Network intrusion detection on KDD Cup 1999 / NSL-KDD style data.

Uses sklearn's built-in make_classification to simulate network features
(if the real dataset is unavailable), then compares Isolation Forest vs
supervised GBM on a heavily imbalanced problem.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              PrecisionRecallDisplay, classification_report)

rng = np.random.default_rng(42)
OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Simulate network flow data ────────────────────────────────────────────────
# 99.5% normal, 0.5% attack (realistic enterprise imbalance)
X, y = make_classification(
    n_samples=20_000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    weights=[0.995, 0.005],
    flip_y=0.001,
    random_state=42,
)

print(f"Dataset: {len(y):,} samples | attack rate: {y.mean():.2%}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                            stratify=y, random_state=42)
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── Isolation Forest (unsupervised) ──────────────────────────────────────────
iso = IsolationForest(n_estimators=200, contamination=0.005, random_state=42)
iso.fit(X_tr_s)
iso_scores = -iso.score_samples(X_te_s)  # higher = more anomalous

# ── Supervised GBM ───────────────────────────────────────────────────────────
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                  learning_rate=0.05, random_state=42)
gbm.fit(X_tr_s, y_tr)
gbm_proba = gbm.predict_proba(X_te_s)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────
print(f"\nIsolation Forest — PR-AUC: {average_precision_score(y_te, iso_scores):.3f} | "
      f"ROC-AUC: {roc_auc_score(y_te, iso_scores):.3f}")
print(f"GBM              — PR-AUC: {average_precision_score(y_te, gbm_proba):.3f} | "
      f"ROC-AUC: {roc_auc_score(y_te, gbm_proba):.3f}")

# ── Precision-Recall curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
PrecisionRecallDisplay.from_predictions(y_te, iso_scores,
    name="Isolation Forest", ax=ax)
PrecisionRecallDisplay.from_predictions(y_te, gbm_proba,
    name="GBM (supervised)", ax=ax)
ax.set_title("Precision-Recall — network intrusion detection")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "pr_curve.png", dpi=120)
plt.close()

# ── Alert volume at fixed precision ──────────────────────────────────────────
print("\n--- Alert volume at different thresholds (GBM) ---")
for thresh in [0.3, 0.5, 0.7, 0.9]:
    preds = (gbm_proba >= thresh).astype(int)
    alerts = preds.sum()
    tp = ((preds == 1) & (y_te == 1)).sum()
    precision = tp / alerts if alerts > 0 else 0
    recall = tp / y_te.sum() if y_te.sum() > 0 else 0
    print(f"  thresh={thresh}: {alerts:3d} alerts | precision={precision:.0%} | recall={recall:.0%}")

print("\nKey insight: at 0.5% attack rate, even a 99% specific model fires")
print("~200 false positives per true positive. Threshold to your analyst capacity.")
print("Plots saved.")
```

---

## Cross-references

- [13](13-anomaly-detection.md) — anomaly detection methods in depth
- [05](05-supervised-learning.md) — classification with extreme imbalance
- [10](10-nlp-and-llms.md) — log parsing and phishing NLP
- [21](21-graph-and-network-analysis.md) — graph-based lateral movement detection
