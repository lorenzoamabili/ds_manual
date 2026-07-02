# 33 · HealthTech & Clinical Data Science

> Applying predictive modelling and causal inference to improve patient outcomes,
> accelerate drug discovery, and optimise care delivery.

---

## Why data science here

Healthcare data is simultaneously the richest and most constrained: EHRs contain
decades of longitudinal patient histories, but HIPAA, GDPR, and local data
governance make it expensive to access. The stakes are high — a mis-calibrated
sepsis model that fails silently kills people. Regulatory approval (FDA, CE mark)
is required for anything that influences clinical decisions.

What distinguishes clinical data science from generic ML: (1) **censoring** —
patients are lost to follow-up or the study ends before they have the outcome
of interest; (2) **confounding by indication** — sicker patients get more
treatment, so naive comparisons always make interventions look ineffective;
(3) **external validity** — a model trained at one hospital often fails at another
because clinical coding practices, population demographics, and care protocols
differ.

The dominant modelling need is not classification accuracy but **calibration** and
**decision support** — a 12% predicted mortality risk means something specific to a
clinician ordering treatment.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Readmission risk | Will patient be readmitted within 30 days? | Binary classification (logistic, GBM) |
| Mortality / deterioration | Risk score in the next N hours | Classification, survival analysis |
| Treatment effect estimation | Does drug X reduce complications? | RCT analysis, propensity scoring |
| Length-of-stay prediction | How many bed-days will this patient need? | Regression, count models |
| Disease progression | How fast will this patient's condition worsen? | Survival / longitudinal modelling |
| Clinical NLP | Extract diagnoses from free-text notes | NER, BERT-based classifiers |
| Drug discovery | Which molecules are active against target? | Graph NNs, molecular fingerprints |
| Genomics | Which variants associate with phenotype? | GWAS, polygenic risk scores |

---

## Key techniques

### 1. Survival analysis (time-to-event)

When the outcome is "how long until event X" (death, discharge, relapse) and some
patients don't have the event by study end (they are **censored**), standard
regression discards that information. Kaplan-Meier estimates the survival curve
non-parametrically; Cox proportional hazards models add covariates. See
[16](16-survival-analysis.md) for full treatment.

**Pitfall:** informative censoring — if patients who are improving are more likely
to be lost to follow-up, the survival estimate is biased. Cannot be fixed by Cox alone.

### 2. Propensity score methods

When treatment is not randomised, patients who receive it differ from those who
don't. Propensity score matching or inverse probability weighting (IPW) re-balances
covariates to approximate a randomised comparison. See [09](09-causal-inference-and-experimentation.md).

**Pitfall:** propensity methods adjust for *observed* confounders only. If an
important confounder (e.g., disease severity) is unmeasured, bias remains.

### 3. Risk stratification with calibrated classifiers

Predict P(outcome) for each patient. Report in risk deciles. The key demand:
**calibration** — the top-10% risk group should actually experience the outcome at
the rate the model predicts. Miscalibrated models mislead triage.

Evaluate: Hosmer-Lemeshow test, reliability diagrams, Brier score.

### 4. Clinical NLP

Free-text notes, discharge summaries, and radiology reports contain information
not encoded in structured fields. BERT-based models (BioBERT, ClinicalBERT) extract
diagnoses, medications, and outcomes. Regex + rule systems remain competitive for
well-defined entity types.

### 5. Temporal feature construction

EHR data is longitudinal. Features must reflect the state of the patient at a
specific point in time. A "last creatinine" feature must be the last measurement
*before* the prediction window, not the one taken during the event. Getting this
right is 80% of the implementation work.

---

## Best practices & pitfalls

- **Calibration > discrimination in clinical settings.** A model with AUC 0.82 and
  poor calibration is less useful than one with AUC 0.78 and perfect calibration,
  because clinicians reason from risk percentages.
- **External validation is mandatory.** A model trained at one hospital that isn't
  tested at another is not ready for deployment. Distribution shift is the norm.
- **Missing data is not random.** Clinicians order tests because they're suspicious.
  "No measurement" is often signal. Don't drop rows with missing labs — impute
  carefully or model the missingness as a feature.
- **Label leakage is easy in EHR data.** The diagnosis code you're predicting is
  often recorded before the outcome you're trying to predict (admission → ICD coding
  happens retrospectively). Use only data available at the time of prediction.
- **Ethics and fairness.** Models trained on historical care data encode historical
  disparities (race, sex, SES). Audit for disparate impact before deployment.
  See [19](19-responsible-ai-and-fairness.md).

---

## Python example — patient readmission risk prediction

```python
"""
30-day readmission prediction on the Diabetes 130-US hospitals dataset.

This UCI dataset contains 10 years of clinical care data for diabetic
patients. We predict 30-day readmission (binary) from structured features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

rng = np.random.default_rng(42)
OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading Diabetes 130-US hospitals dataset from OpenML...")
data = fetch_openml("diabetes130us", version=1, as_frame=True, parser="auto")
X_raw, y_raw = data.data, data.target

# Target: readmitted within 30 days (">30" and "NO" → 0, "<30" → 1)
y = (y_raw == "<30").astype(int)

# Drop high-cardinality or leaky columns
drop_cols = [c for c in ["encounter_id", "patient_nbr"] if c in X_raw.columns]
X = X_raw.drop(columns=drop_cols, errors="ignore")

# Replace "?" with NaN
X = X.replace("?", np.nan)

cat_cols = X.select_dtypes(["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(["int64", "float64"]).columns.tolist()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
])

lr  = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.1, max_iter=500, random_state=42))])
gbm = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(
    n_estimators=100, max_depth=3, random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Dataset: {X.shape[0]:,} patients | positive rate: {y.mean():.1%}")

# ── Cross-validated evaluation ───────────────────────────────────────────────
results = {}
for name, pipe in [("LogReg", lr), ("GBM", gbm)]:
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    results[name] = {
        "proba": proba,
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC":  average_precision_score(y, proba),
        "Brier":   brier_score_loss(y, proba),
    }
    print(f"{name}: ROC-AUC={results[name]['ROC-AUC']:.3f} | "
          f"PR-AUC={results[name]['PR-AUC']:.3f} | "
          f"Brier={results[name]['Brier']:.3f}")

# ── Calibration plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0,1], [0,1], "k--", label="Perfect")
for name, res in results.items():
    frac, mean_pred = calibration_curve(y, res["proba"], n_bins=10)
    ax.plot(mean_pred, frac, "o-", label=name)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration — 30-day readmission")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "calibration.png", dpi=120)
plt.close()

print("\nKey insight: in clinical settings, calibration is often more important")
print("than AUC. A risk of '15%' must actually mean 15% of patients readmitted.")
```

---

## Cross-references

- [16](16-survival-analysis.md) — time-to-event modelling (mortality, progression)
- [09](09-causal-inference-and-experimentation.md) — treatment effect estimation
- [05](05-supervised-learning.md) — classification and calibration
- [04](04-evaluation-and-validation.md) — calibration metrics
- [19](19-responsible-ai-and-fairness.md) — fairness in clinical AI
- [10](10-nlp-and-llms.md) — clinical NLP
