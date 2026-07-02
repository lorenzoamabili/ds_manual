# 03 · Data, Features & the Leakage Problem

Models are commodities; features and clean data are where projects are won. And
**data leakage is the single most common reason a model that looked brilliant in
development fails in production.**

## EDA with intent

EDA is not "make lots of plots." It is a series of questions:

1. **Shape & types** — rows, columns, dtypes. Are IDs stored as numbers (and being
   treated as continuous)?
2. **Missingness** — how much, and *is it random?* Missingness is often
   informative (a blank "income" may signal something). Visualise the missingness
   *pattern*, don't just count.
3. **Distributions** — skew, outliers, zero-inflation. Decide transforms here.
4. **Target relationship** — how does each feature relate to the target? This is
   where hypotheses for features come from.
5. **Leakage smell test** — for any feature that looks *too* predictive, ask: would
   I actually know this value at prediction time? (This one question prevents most
   disasters.)

## Cleaning decisions and their trade-offs

- **Missing values.** Options: drop rows (only if MCAR and few), drop columns (if
  mostly empty), or impute. Simple imputation (median/mode) + a **"was-missing"
  indicator** is a strong, honest default. Model-based imputation (KNN, iterative)
  can leak if done before the train/test split.
- **Outliers.** Don't reflexively delete. Decide if they're errors (fix/drop) or
  real extremes (keep, and use robust methods or a transform). Winsorising caps
  without discarding.
- **Categoricals.** One-hot for low cardinality; **target/mean encoding** for high
  cardinality — but target encoding *leaks the target* unless done inside
  cross-validation folds. This is a classic trap.
- **Scaling.** Required for distance- and gradient-based methods (SVM, k-NN,
  k-means, neural nets, regularised linear). Irrelevant for tree ensembles.

## Feature engineering — the highest-leverage work

- **Domain features** beat clever models. A single ratio a domain expert suggests
  can outperform weeks of tuning.
- **Interactions & non-linearities** — for linear models, add them explicitly
  (products, splines, polynomials). Trees find them automatically.
- **Temporal features** — from a timestamp: hour, day-of-week, is-holiday, and
  crucially **lag and rolling-window** features for series (see
  [07](07-time-series-forecasting.md)).
- **Aggregations** — group-level stats (this user's average, this store's median)
  are often the strongest features in tabular problems.

## Leakage: the taxonomy

Leakage = information available at training time that will **not** be available at
prediction time (or that encodes the target). Types:

1. **Target leakage** — a feature computed from, or a proxy for, the outcome.
   Example: "number of late-payment reminders sent" predicting default — the
   reminders happen *because* of impending default.
2. **Train/test contamination** — fitting *any* transform (scaler, imputer, target
   encoder, feature selector) on the full dataset **before** splitting. The test
   set has bled into training. **Fix: fit every transform inside a Pipeline, on
   training folds only** — exactly what [Project 1](../projects/p1_supervised_learning)
   demonstrates.
3. **Temporal leakage** — using future information to predict the past. In time
   series you must split *chronologically*, never randomly
   ([Project 2](../projects/p2_time_series_forecasting)).
4. **Group leakage** — the same entity (patient, user) in both train and test, so
   the model memorises the entity rather than learning the pattern. Use
   `GroupKFold` to keep groups intact.

> The tell-tale sign of leakage is a validation score that is *suspiciously* good.
> When a result looks too good to be true, hunt for leakage before you celebrate.

---

## Python example — EDA, imputation, feature engineering, leakage guard

```python
"""
Full data preparation pipeline:
  - EDA with missingness map
  - Imputation inside Pipeline (no leakage)
  - Feature engineering: log transform, interaction, rolling aggregation
  - Leakage guard: shuffle-label test
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

rng = np.random.default_rng(42)
n   = 600

# ── Synthetic tabular dataset with missing values and skew ───────────────────
df = pd.DataFrame({
    "age":        rng.integers(18, 75, n).astype(float),
    "income":     rng.lognormal(10, 0.8, n),          # skewed
    "credit_util":rng.beta(2, 5, n),
    "months_cust":rng.integers(1, 120, n).astype(float),
    "missed_pay": rng.binomial(1, 0.2, n),
})
# Inject 12% missingness in income and age
for col in ["income", "age"]:
    missing_idx = rng.choice(n, size=int(0.12*n), replace=False)
    df.loc[missing_idx, col] = np.nan

y = (rng.binomial(1, 1/(1+np.exp(-(
    -2 + df["missed_pay"].fillna(0) * 2
    - df["credit_util"] * 1.5
    + df["income"].fillna(df["income"].median()).apply(np.log) * 0.3
))), 1)).astype(int)

# ── EDA: missingness ──────────────────────────────────────────────────────────
print("Missingness rate:")
print(df.isnull().mean().round(3).to_string())
print(f"\nTarget rate: {y.mean():.1%}")
print(f"\nDistribution skew:")
print(df.skew().round(2).to_string())

# ── Feature engineering ───────────────────────────────────────────────────────
df["log_income"]     = np.log1p(df["income"])         # tame skew
df["util_x_missed"]  = df["credit_util"] * df["missed_pay"]   # interaction
df["log_months"]     = np.log1p(df["months_cust"])

feat_cols = ["age", "log_income", "credit_util", "log_months",
             "missed_pay", "util_x_missed"]
X = df[feat_cols]

# ── Pipeline: impute THEN scale THEN fit — all inside CV ─────────────────────
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # fit on training fold only
    ("scale",  StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
])

cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
score = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
print(f"\nCV ROC-AUC (honest): {score:.4f}")

# ── Leakage guard: shuffle labels → must drop to chance ──────────────────────
y_shuffled = rng.permutation(y)
score_null = cross_val_score(pipe, X, y_shuffled, cv=cv, scoring="roc_auc").mean()
print(f"CV ROC-AUC (shuffled labels): {score_null:.4f}  (should be ≈ 0.50)")
assert score_null < 0.65, "Shuffled labels gave high AUC — investigate leakage!"
print("Leakage guard passed: shuffled-label score collapsed to chance.")

# ── Missingness visualisation ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
miss_map = df[["age","income","credit_util","months_cust","missed_pay"]].isnull()
ax1.imshow(miss_map.values.T, aspect="auto", cmap="Reds", interpolation="nearest")
ax1.set_yticks(range(len(miss_map.columns)))
ax1.set_yticklabels(miss_map.columns)
ax1.set_xlabel("Sample index"); ax1.set_title("Missingness map (red = missing)")

df["income"].dropna().apply(np.log).hist(ax=ax2, bins=30, color="#3498db", alpha=0.7)
ax2.set_xlabel("log(income)"); ax2.set_title("Income after log transform\n(tamed skew)")
plt.tight_layout()
plt.savefig("eda_leakage.png", dpi=120)
plt.close()
print("Plot saved.")
```

---

## Cross-references

- [P1](../projects/p1_supervised_learning) — canonical no-leakage Pipeline demo
- [04](04-evaluation-and-validation.md) — the shuffle-label leakage guard
- [18](18-sql-and-data-engineering.md) — SQL for data prep (fan-out, grain bugs)
