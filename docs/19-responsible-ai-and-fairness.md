# 19 · Responsible AI & Fairness

Increasingly table-stakes, especially in the EU. Not a compliance afterthought — an
engineering and ethics discipline that, done well, is a differentiator. Paired
project: [P6 — fairness audit](../projects/p6_fairness_audit).

## Fairness is plural, and the definitions conflict

There is no single "fair." The main group-fairness criteria:

| Criterion | Requires equal… | Right when |
|-----------|-----------------|-----------|
| **Demographic parity** | selection rate across groups | Outcomes *should* be equal by policy (representation) |
| **Equal opportunity** | true-positive rate across groups | Missing a qualified person is the key harm |
| **Equalised odds** | TPR *and* FPR across groups | Both error types matter across groups |
| **[Calibration](04-evaluation-and-validation.md)** | predicted probs mean the same per group | Scores are used as probabilities |

**An impossibility result:** except in degenerate cases you cannot satisfy
calibration *and* equalised odds simultaneously when base rates differ. So fairness
requires a *choice* grounded in context and values — a policy question, not a
technical one. P6 makes this trade-off explicit.

## "We didn't use the protected attribute" is not a defence
Removing race/gender/age from the features does **not** prevent discrimination,
because other features proxy for them (postcode → race; certain measures →
gender). P6 demonstrates a model that never sees the protected attribute yet fails
the 80% rule, because a feature it *does* use was historically mismeasured for one
group. Fairness must be *measured on outcomes*, not assumed from feature lists.

## Where bias enters the pipeline
- **Historical bias** — the world (and thus the training labels) is already unequal;
  the model faithfully reproduces it.
- **Representation/sampling bias** — some groups under-represented in the data.
- **Measurement bias** — the label or a feature is a worse proxy for one group
  (the P6 mechanism).
- **Aggregation bias** — one model for groups that behave differently (Simpson-style
  effects).
- **Deployment/feedback bias** — the model shapes future data (over-policing a
  region generates more recorded incidents there, confirming the model).

## Mitigation, by stage
- **Pre-processing** — reweighting, resampling, or learning fair representations.
- **In-processing** — fairness constraints/regularisers in the objective (adversarial
  debiasing, constrained optimisation).
- **Post-processing** — group-specific thresholds to equalise a chosen metric (P6's
  approach; simple, model-agnostic, but note legal constraints on explicit
  group-based decisions in some jurisdictions).

## Beyond fairness: the wider responsible-AI surface
- **Transparency & explainability** — [SHAP](05-supervised-learning.md)/counterfactuals, and **model cards** +
  **datasheets for datasets** documenting intended use, training data, evaluation
  by subgroup, and limitations. (This repo ships cards for its models — see
  [`cards/`](../cards).)
- **Privacy** — data minimisation, anonymisation (and its limits — re-identification
  is easy), differential privacy for strong guarantees, federated learning to avoid
  centralising raw data.
- **Robustness & security** — distribution shift, adversarial examples, data
  poisoning, and prompt injection for [LLM](10-nlp-and-llms.md) systems.
- **Accountability** — human-in-the-loop for high-stakes decisions, audit trails,
  the right to contest an automated decision, and a named owner.

## The regulatory context (EU-centric)
- **GDPR** — lawful basis, purpose limitation, data-subject rights, and Article 22
  restrictions on solely-automated decisions with legal/significant effects (right
  to human review and to an explanation of the logic).
- **EU AI Act** — risk-tiered obligations: unacceptable-risk uses banned;
  **high-risk** systems (hiring, credit, essential services) face requirements on
  data governance, documentation, transparency, human oversight, and accuracy/
  robustness. Building the documentation and subgroup evaluation habits now is far
  cheaper than retrofitting them.

The pragmatic stance: measure disparate impact and subgroup performance as a
standard part of *every* consequential model's evaluation, document it in a model
card, and make the fairness/accuracy trade-off a visible decision rather than an
accident.

---

## Python example — fairness audit: disparate impact + equalised odds

```python
"""
Fairness audit pipeline:
  1. Train a classifier ignoring protected attribute
  2. Measure disparate impact (80% rule) and equalised odds
  3. Apply post-processing threshold adjustment to equalise TPR
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

rng = np.random.default_rng(42)
n = 2000

# Synthetic hiring dataset: group 0 historically disadvantaged
group = rng.binomial(1, 0.4, n)   # 0 = majority, 1 = minority
skill = rng.normal(0, 1, n) + 0.3 * group  # minority has same skills
# Historical label: biased (minority hired at 60% the rate, even at same skill)
p_hire = 1 / (1 + np.exp(-(0.8 * skill - 0.6 * (group == 0).astype(float))))
hired = rng.binomial(1, p_hire)

X = pd.DataFrame({"skill": skill, "years_exp": rng.uniform(0, 10, n)})
y = hired
g = group

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, y, g, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_tr)
clf = LogisticRegression(random_state=42).fit(scaler.transform(X_tr), y_tr)
proba = clf.predict_proba(scaler.transform(X_te))[:, 1]
preds = (proba >= 0.5).astype(int)

# ── Disparate impact (80% rule) ───────────────────────────────────────────────
rate_0 = preds[g_te == 0].mean()
rate_1 = preds[g_te == 1].mean()
di = rate_1 / rate_0   # ratio of minority / majority selection rate
print(f"Selection rate — majority: {rate_0:.2%}  minority: {rate_1:.2%}")
print(f"Disparate impact ratio: {di:.3f}  ({'PASS' if di >= 0.8 else 'FAIL'} 80% rule)")

# ── Equalised odds: TPR and FPR per group ─────────────────────────────────────
def group_metrics(y_true, y_pred, mask):
    tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask]).ravel()
    return tp/(tp+fn), fp/(fp+tn)   # TPR, FPR

tpr0, fpr0 = group_metrics(y_te, preds, g_te == 0)
tpr1, fpr1 = group_metrics(y_te, preds, g_te == 1)
print(f"\nEqualised odds check:")
print(f"  TPR — majority: {tpr0:.2%}  minority: {tpr1:.2%}  gap: {abs(tpr0-tpr1):.2%}")
print(f"  FPR — majority: {fpr0:.2%}  minority: {fpr1:.2%}  gap: {abs(fpr0-fpr1):.2%}")

# ── Post-processing: per-group threshold to equalise TPR ─────────────────────
def best_threshold_for_tpr(proba, y, target_tpr, n_thresh=200):
    """Find the threshold that hits closest to target_tpr."""
    thresholds = np.linspace(0.01, 0.99, n_thresh)
    tprs = [(((proba >= t) & (y == 1)).sum() / (y == 1).sum()) for t in thresholds]
    return thresholds[np.argmin(np.abs(np.array(tprs) - target_tpr))]

target_tpr = (tpr0 + tpr1) / 2   # equalise at midpoint
t0 = best_threshold_for_tpr(proba[g_te == 0], y_te[g_te == 0], target_tpr)
t1 = best_threshold_for_tpr(proba[g_te == 1], y_te[g_te == 1], target_tpr)

preds_adj = preds.copy()
preds_adj[g_te == 0] = (proba[g_te == 0] >= t0).astype(int)
preds_adj[g_te == 1] = (proba[g_te == 1] >= t1).astype(int)

tpr0_adj, _ = group_metrics(y_te, preds_adj, g_te == 0)
tpr1_adj, _ = group_metrics(y_te, preds_adj, g_te == 1)
di_adj = preds_adj[g_te == 1].mean() / preds_adj[g_te == 0].mean()

print(f"\nAfter threshold adjustment:")
print(f"  TPR — majority: {tpr0_adj:.2%}  minority: {tpr1_adj:.2%}")
print(f"  Disparate impact: {di_adj:.3f}  ({'PASS' if di_adj >= 0.8 else 'FAIL'} 80% rule)")
print(f"\nLesson: 'we didn't use the protected attribute' is not enough.")
print(f"Measure outcomes by group — every consequential model, every time.")
```

---

## Cross-references

- [P6](../projects/p6_fairness_audit) — full fairness audit project
- [38](38-hrtech.md) — fairness in hiring (legally critical)
- [31](31-fintech.md) — fairness in credit scoring (FCRA, EU AI Act)
- [cards/](../cards) — model cards for all projects
