# 38 · HRTech & People Analytics

> Using data to understand the workforce — who will leave, who to hire, and how
> to structure teams for performance.

---

## Why data science here

People analytics has matured from "HR reporting" to predictive and causal
modelling, driven by the realisation that employee attrition costs 50-200% of an
annual salary to replace. The domain has unique ethical constraints: employees
have privacy rights, predictions about people have legal implications, and
the HR function is wary of models that feel like surveillance.

The data is internally generated: HRIS records (tenure, role, salary band,
performance ratings, promotions), engagement surveys, calendar metadata
(meeting load, after-hours work), and in some organisations, communication
metadata. External data (LinkedIn, Glassdoor ratings) is sometimes used but
legally and ethically constrained.

Three modelling problems dominate: (1) **attrition prediction** (who will leave in
the next 6-12 months?), (2) **hiring funnel optimisation** (which candidates will
succeed?), and (3) **workforce planning** (how many people with which skills do we
need in 3 years?). All three have significant causal complexity and ethical risk.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Voluntary attrition | Will this employee leave in next 12 months? | Binary classification |
| Time-to-leave | How long until this employee leaves? | [Survival analysis](16-survival-analysis.md) |
| Engagement risk | Is this team disengaged? | [Clustering](06-unsupervised-learning.md) + survey analysis |
| Hiring quality | Will this candidate succeed in 90 days? | Classification on structured interview + assessment data |
| Promotion prediction | Who is ready for promotion? | Classification (use carefully — bias risk) |
| Workforce demand | How many software engineers in 2027? | Time series + scenario modelling |
| Compensation equity | Are there unexplained pay gaps by gender/race? | Regression audit (Oaxaca-Blinder) |

---

## Key techniques

### 1. Attrition prediction

Standard binary classification on employee features. Key predictors from
literature: tenure (U-shaped risk — high in first 2 years and after 5 years),
time since last promotion, manager change, commute change, pay band vs. market,
engagement survey score.

**Pitfall:** HR data often has performance ratings that are [calibrated](04-evaluation-and-validation.md) per-manager,
not company-wide. A "3 out of 5" from one manager ≠ "3 out of 5" from another.
Normalise ratings within manager before using as a feature.

### 2. Survival analysis for time-to-attrition

When "will they leave?" becomes "when will they leave?", use survival analysis.
[Cox proportional hazards](16-survival-analysis.md) gives hazard ratios — interpretable HR-facing outputs
("employees without a promotion in 18 months have 2.3× the baseline leaving rate").
See [16](16-survival-analysis.md).

### 3. Segmentation for engagement

Cluster employees or teams on engagement survey dimensions (manager quality,
workload, career clarity, belonging). Produces actionable segments for HR
business partners. [K-means](06-unsupervised-learning.md) or GMM on 5-10 survey dimensions. Interpret cluster
centroids to label segments.

### 4. Compensation equity analysis

Regress salary on legitimate predictors (role, seniority, location, experience).
The residual captures unexplained variation. Decompose residuals by gender/race
to identify gaps not explained by legitimate factors (Oaxaca-Blinder decomposition).
This is both a fairness audit and a legal compliance tool in many jurisdictions.

### 5. Caution on predictive hiring

Predicting candidate success from structured data is technically feasible but
ethically high-risk: models often amplify historical hiring bias. In many
jurisdictions (EU AI Act, NYC Local Law 144), AI hiring tools require bias
audits and in some cases regulatory approval.

---

## Best practices & pitfalls

- **Attrition models surface symptoms, not causes.** A model says "this person has
  an 80% attrition probability." It doesn't say why. Always follow up with
  qualitative investigation before any HR intervention.
- **Avoid features that proxy for protected attributes.** Zip code, commute time,
  college name, and gap years can proxy for race, disability, or class. Audit
  feature importance for [disparate impact](19-responsible-ai-and-fairness.md).
- **Attrition labels are sparse and noisy.** Voluntary vs. involuntary attrition
  need separate models. Layoff periods corrupt the training set.
- **Privacy by design.** Aggregate to cohorts before surfacing to managers.
  An individual-level attrition score visible to a manager creates a self-fulfilling
  prophecy (manager treats the "high risk" employee differently, causing them to leave).
- **Models for intervention, not just prediction.** An attrition prediction without
  an actionable retention play is just expensive trivia. Tie every model to a
  decision or experiment.
- **See [19](19-responsible-ai-and-fairness.md).** Fairness in HR is legally and
  ethically critical. Run [demographic parity](19-responsible-ai-and-fairness.md), [equalised odds](19-responsible-ai-and-fairness.md), and individual fairness
  checks before deployment.

---

## Python example — employee attrition prediction (IBM HR dataset)

```python
"""
Employee attrition prediction on IBM HR Analytics dataset.

Public dataset with 1470 employees, 35 features, and a voluntary attrition label.
Available from: https://raw.githubusercontent.com/dsrscientist/dataset1/master/
                HR_Analytics.csv
Or from sklearn / kaggle (also available via seaborn-data).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              RocCurveDisplay, PrecisionRecallDisplay)

rng = np.random.default_rng(42)
OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Load IBM HR dataset ───────────────────────────────────────────────────────
url = ("https://raw.githubusercontent.com/IBM/employee-attrition-aif360"
       "/master/data/emp_attrition.csv")
print("Loading IBM HR Analytics dataset...")
try:
    df = pd.read_csv(url)
    # Standardise column names
    if "Attrition" not in df.columns and "attrition" in df.columns:
        df = df.rename(columns={"attrition": "Attrition"})
except Exception:
    print("Download failed — using synthetic HR data.")
    n = 1470
    df = pd.DataFrame({
        "Attrition": rng.choice(["Yes","No"], n, p=[0.16, 0.84]),
        "Age": rng.integers(22, 60, n),
        "MonthlyIncome": rng.integers(2000, 15000, n),
        "YearsAtCompany": rng.integers(0, 20, n),
        "YearsSinceLastPromotion": rng.integers(0, 10, n),
        "JobSatisfaction": rng.integers(1, 5, n),
        "WorkLifeBalance": rng.integers(1, 4, n),
        "OverTime": rng.choice(["Yes","No"], n, p=[0.28, 0.72]),
        "NumCompaniesWorked": rng.integers(0, 9, n),
        "DistanceFromHome": rng.integers(1, 30, n),
        "Department": rng.choice(["Sales","R&D","HR"], n),
        "JobRole": rng.choice(["Manager","Engineer","Analyst","Sales Rep"], n),
        "MaritalStatus": rng.choice(["Single","Married","Divorced"], n),
        "EnvironmentSatisfaction": rng.integers(1, 5, n),
        "JobInvolvement": rng.integers(1, 4, n),
    })

# ── Prepare ───────────────────────────────────────────────────────────────────
target = "Attrition"
drop = [target, "EmployeeNumber", "EmployeeCount", "Over18",
        "StandardHours"] if "EmployeeNumber" in df.columns else [target]
X = df.drop(columns=[c for c in drop if c in df.columns])
y = (df[target] == "Yes").astype(int)

cat_cols = X.select_dtypes(["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(["int64", "float64"]).columns.tolist()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
])
lr  = Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.1, max_iter=500,
                                                          class_weight="balanced",
                                                          random_state=42))])
gbm = Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(
    n_estimators=100, max_depth=3, random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Dataset: {len(y):,} employees | attrition rate: {y.mean():.1%}")

# ── Cross-validate ────────────────────────────────────────────────────────────
for name, pipe in [("LogReg (balanced)", lr), ("GBM", gbm)]:
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    print(f"{name}: ROC-AUC={roc_auc_score(y, proba):.3f} | "
          f"PR-AUC={average_precision_score(y, proba):.3f}")

# ── Feature importance (fit once for plot) ────────────────────────────────────
gbm.fit(X, y)
feat_names = (num_cols +
              gbm.named_steps["pre"].named_transformers_["cat"]
              .get_feature_names_out(cat_cols).tolist())
importance = pd.Series(gbm.named_steps["clf"].feature_importances_,
                        index=feat_names).nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(7, 5))
importance.plot.barh(ax=ax)
ax.set_title("Top 15 features — attrition GBM")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=120)
plt.close()
print("Plot saved.")
print("\nKey insight: overtime, distance from home, and job satisfaction typically")
print("dominate attrition models — but surface these to HR as hypotheses to")
print("investigate, not as deterministic predictions.")
```

---

## Cross-references

- [05](05-supervised-learning.md) — classification, [class imbalance](05-supervised-learning.md), [SHAP](05-supervised-learning.md)
- [16](16-survival-analysis.md) — time-to-attrition with Cox
- [06](06-unsupervised-learning.md) — team engagement segmentation
- [19](19-responsible-ai-and-fairness.md) — bias audits for HR models (legal critical)
- [09](09-causal-inference-and-experimentation.md) — causal effects of HR interventions
