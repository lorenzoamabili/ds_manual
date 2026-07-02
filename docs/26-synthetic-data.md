# Synthetic Data Generation

Synthetic data addresses three problems in production DS: class imbalance,
privacy constraints on sensitive datasets, and insufficient training data.
Understanding when and how to generate synthetic data is a core skill.

---

## Why synthetic data?

| Problem | Synthetic data solution |
|---|---|
| Imbalanced classes | Oversample minority class (SMOTE) |
| Privacy-sensitive training data | Replace real records with synthetic records |
| Insufficient training data | Augment with plausible synthetic samples |
| Testing data pipelines | Generate schema-conformant test data |
| Counterfactual simulation | Simulate "what would have happened" |

---

## SMOTE — oversampling for imbalanced classes

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic minority
samples by interpolating between existing minority samples rather than
duplicating them. This reduces overfitting compared to random oversampling.

**How it works:** for each minority sample, find its K nearest minority
neighbours, pick one at random, create a new sample at a random point on
the line segment between them.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

# pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("imbalanced-learn not installed: pip install imbalanced-learn")

rng = np.random.default_rng(42)

# Imbalanced dataset: 2% positive rate
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=8, n_redundant=4,
    weights=[0.98, 0.02], flip_y=0, random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

print(f"Training set: {y_tr.sum()} positives out of {len(y_tr)} ({y_tr.mean():.1%})")
print(f"Test set:     {y_te.sum()} positives out of {len(y_te)} ({y_te.mean():.1%})")
print()

# Baseline: class_weight="balanced"
gbm_weighted = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_weighted.fit(X_tr, y_tr)
proba_w = gbm_weighted.predict_proba(X_te)[:, 1]

if HAS_IMBLEARN:
    # SMOTE + GBM
    smote_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("gbm",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])
    smote_pipe.fit(X_tr, y_tr)
    proba_s = smote_pipe.predict_proba(X_te)[:, 1]

    print("Results on test set:")
    print(f"{'Method':<25} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 48)
    for name, proba in [("GBM (class_weight)", proba_w), ("SMOTE + GBM", proba_s)]:
        print(f"{name:<25} {roc_auc_score(y_te, proba):>10.4f} "
              f"{average_precision_score(y_te, proba):>10.4f}")
else:
    print(f"GBM (class_weight) — ROC-AUC: {roc_auc_score(y_te, proba_w):.4f}  "
          f"PR-AUC: {average_precision_score(y_te, proba_w):.4f}")

print()
print("Note: SMOTE works best when the minority class is not noisy.")
print("SMOTE can hurt when minority samples cluster near the majority boundary.")
```

---

## Gaussian Copula — tabular synthetic data

For privacy-preserving synthetic data generation, the Gaussian Copula approach:
1. Learn marginal distributions for each column
2. Learn correlation structure via a Gaussian copula
3. Sample new rows that preserve both marginals and correlations

```python
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)

# Simulate a credit dataset with mixed column types
n = 500
real_df = pd.DataFrame({
    "age":          rng.integers(18, 75, n),
    "income":       rng.lognormal(10, 0.5, n).round(-2),   # skewed
    "credit_score": rng.normal(650, 80, n).clip(300, 850).round(),
    "defaulted":    rng.binomial(1, 0.08, n),
})
# Inject correlation: higher income -> higher credit score
real_df["credit_score"] += (real_df["income"] / 10_000 * 30).values.clip(0, 100)
real_df["credit_score"] = real_df["credit_score"].clip(300, 850).round()

print("Real data stats:")
print(real_df.describe().round(1))
print(f"\nCorrelation income-credit_score: {real_df['income'].corr(real_df['credit_score']):.3f}")

# Gaussian Copula: transform each column to normal via rank-based CDF
def to_normal(series: pd.Series) -> np.ndarray:
    ranks = series.rank(method="average")
    u     = ranks / (len(ranks) + 1)   # uniform via empirical CDF
    return stats.norm.ppf(u)

normal_cols = {col: to_normal(real_df[col]) for col in real_df.columns}
normal_mat  = np.column_stack(list(normal_cols.values()))
corr_mat    = np.corrcoef(normal_mat.T)

# Sample from multivariate normal with learned correlation
synth_normal = rng.multivariate_normal(np.zeros(4), corr_mat, size=n)

# Inverse transform back to original scale
def from_normal(synth_z: np.ndarray, orig_series: pd.Series) -> pd.Series:
    u = stats.norm.cdf(synth_z)
    return pd.Series(np.quantile(orig_series, u))

synth_df = pd.DataFrame({
    col: from_normal(synth_normal[:, i], real_df[col])
    for i, col in enumerate(real_df.columns)
})
synth_df["defaulted"] = (synth_df["defaulted"] > 0.5).astype(int)

print("\nSynthetic data stats:")
print(synth_df.describe().round(1))
print(f"\nCorrelation income-credit_score (synthetic): "
      f"{synth_df['income'].corr(synth_df['credit_score']):.3f}")
print("Correlation preserved ✓" if abs(
    real_df['income'].corr(real_df['credit_score']) -
    synth_df['income'].corr(synth_df['credit_score'])
) < 0.1 else "Correlation drift — check marginal transforms")
```

---

## Train-on-synthetic, test-on-real (TSTR)

The gold-standard evaluation: train a model on synthetic data, test on real data.
If performance matches train-on-real/test-on-real (TRTR), the synthetic data
is high-fidelity.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

def make_dataset(n, seed):
    rng_local = np.random.default_rng(seed)
    X = rng_local.normal(0, 1, (n, 8))
    # True relationship: positives cluster in top-right quadrant
    y = ((X[:, 0] + X[:, 1] > 1.0) & (X[:, 2] > 0)).astype(int)
    return X, y

# Real data
X_real, y_real = make_dataset(2000, seed=42)
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_real, y_real, random_state=42)

# Synthetic data (here: same DGP with different seed — simulates a good generator)
X_synth, y_synth = make_dataset(2000, seed=99)

def fit_eval(X_tr, y_tr, X_te, y_te):
    m = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

trtr = fit_eval(X_tr_r,  y_tr_r,  X_te_r, y_te_r)
tstr = fit_eval(X_synth, y_synth, X_te_r, y_te_r)

print(f"TRTR (train-real / test-real):      ROC-AUC = {trtr:.4f}  (upper bound)")
print(f"TSTR (train-synth / test-real):     ROC-AUC = {tstr:.4f}")
print(f"Fidelity gap: {abs(trtr - tstr):.4f}")
print()
print("Rule of thumb: gap < 0.05 → synthetic data is usable for training")
```

---

## Privacy evaluation — membership inference attack

Synthetic data that memorises training records is not private. The standard
privacy test: can an adversary distinguish real records from synthetic ones?

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)

# Real training data and synthetic data
X_real  = rng.normal(0, 1, (500, 10))
X_synth = rng.normal(0, 1, (500, 10))   # independent samples = perfect privacy

# Membership inference: can a classifier tell them apart?
X_combined = np.vstack([X_real, X_synth])
y_combined = np.array([1]*500 + [0]*500)   # 1=real, 0=synthetic

from sklearn.model_selection import cross_val_score
attacker = RandomForestClassifier(n_estimators=100, random_state=42)
auc = cross_val_score(attacker, X_combined, y_combined,
                      cv=5, scoring="roc_auc").mean()

print(f"Membership inference AUC: {auc:.4f}")
print("  0.50 = perfect privacy (indistinguishable)")
print("  1.00 = no privacy (synthetic = memorised real data)")
print()
print("Rule: AUC > 0.55 → investigate privacy leakage")
```

---

## When synthetic data backfires

- **Distribution shift**: synthetic data trained on historical data inherits
  historical biases. Synthetic samples do not introduce new signal.
- **SMOTE on noisy boundaries**: creates synthetic samples in ambiguous regions,
  worsening the classifier.
- **Overconfidence**: a model trained on synthetic data can appear better than
  TRTR performance because synthetic labels are cleaner than real-world noise.
- **Regulatory**: in healthcare and finance, regulators may not accept models
  trained on synthetic data as equivalent to real-data models. Check jurisdiction.

---

## SDV library for production synthetic data

```python
# pip install sdv
# SDV provides Gaussian Copula, CTGAN, TVAE with privacy metrics

# from sdv.single_table import GaussianCopulaSynthesizer
# synth = GaussianCopulaSynthesizer(metadata)
# synth.fit(real_df)
# synthetic_df = synth.sample(num_rows=1000)
# from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
# quality_report = evaluate_quality(real_df, synthetic_df, metadata)

print("SDV workflow (requires: pip install sdv):")
print("  1. Define metadata (column types, primary key)")
print("  2. GaussianCopulaSynthesizer(metadata).fit(real_df)")
print("  3. .sample(num_rows=N)  ->  synthetic_df")
print("  4. evaluate_quality(real_df, synthetic_df, metadata)")
print()
print("CTGAN: GAN-based, better on complex distributions but slower to train")
print("TVAE:  VAE-based, good on mixed types (numerical + categorical)")
```
