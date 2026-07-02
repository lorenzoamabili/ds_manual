# AutoML & Hyperparameter Tuning

Finding good hyperparameters by hand is expensive and unreliable. Systematic
search — Bayesian optimisation in particular — produces better results in fewer
evaluations than grid or random search.

---

## Why not grid search?

Grid search evaluates every combination. With 5 hyperparameters each with 4
values, that's 4⁵ = 1024 fits. Worse: it wastes most evaluations on regions
that are clearly suboptimal because it has no memory of previous results.

**Random search** (Bergstra & Bengio 2012) is a free win over grid search: it
samples hyperparameters independently, which means each sample is useful for
every dimension even if the others aren't interesting. For the same budget, random
search usually finds a better solution.

**Bayesian optimisation** goes further: it fits a surrogate model (Gaussian
Process or Tree Parzen Estimator) over the hyperparameter space and uses it
to select the next point to evaluate, trading off exploitation (near the best
known point) and exploration (high uncertainty regions).

---

## Optuna — modern hyperparameter optimisation

Optuna uses **TPE** (Tree-structured Parzen Estimator), an efficient Bayesian
method that works well with conditional hyperparameters (hyperparameters that
only exist when another hyperparameter takes a specific value).

### Python example — optimising a GBM on breast cancer

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
cv   = StratifiedKFold(5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbm",   GradientBoostingClassifier(**params, random_state=42)),
    ])
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"Best ROC-AUC: {study.best_value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Compare with default GBM
baseline = cross_val_score(
    Pipeline([("s", StandardScaler()), ("g", GradientBoostingClassifier(random_state=42))]),
    X, y, cv=cv, scoring="roc_auc"
).mean()
print(f"\nDefault GBM ROC-AUC: {baseline:.4f}")
print(f"Optimised GBM ROC-AUC: {study.best_value:.4f}")
print(f"Lift: +{study.best_value - baseline:.4f}")
```

---

## Learning curves — diagnosing bias vs variance

Before tuning, plot learning curves to diagnose whether the problem is
underfitting (high bias → more features/model capacity) or overfitting
(high variance → more data/regularisation).

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
train_sizes = np.linspace(0.1, 1.0, 10)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, pipe) in zip(axes, [
    ("Logistic Regression (low capacity)",
     Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])),
    ("GBM (high capacity)",
     Pipeline([("s", StandardScaler()), ("m", GradientBoostingClassifier(n_estimators=200))])),
]):
    tr_sizes, tr_scores, val_scores = learning_curve(
        pipe, X, y, cv=5, scoring="roc_auc",
        train_sizes=train_sizes, n_jobs=-1
    )
    ax.plot(tr_sizes, tr_scores.mean(1), "o-", label="Train")
    ax.fill_between(tr_sizes, tr_scores.mean(1)-tr_scores.std(1),
                               tr_scores.mean(1)+tr_scores.std(1), alpha=0.2)
    ax.plot(tr_sizes, val_scores.mean(1), "s-", label="CV")
    ax.fill_between(tr_sizes, val_scores.mean(1)-val_scores.std(1),
                               val_scores.mean(1)+val_scores.std(1), alpha=0.2)
    ax.set(xlabel="Training size", ylabel="ROC-AUC", title=name, ylim=(0.8, 1.01))
    ax.legend()

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=120)
print("Saved learning_curves.png")
print()
print("Diagnosis guide:")
print("  High train / low CV score  → overfitting → add regularisation or data")
print("  Both scores low            → underfitting → increase model capacity")
print("  Scores converge at high N  → need more data to improve")
```

---

## Feature selection

Too many features increase noise, training time, and risk of overfitting.
Three practical methods:

### Variance threshold — remove near-constant features

```python
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=1000, n_features=30, n_informative=10,
                            n_redundant=5, random_state=42)

# Add 5 near-constant features
rng = np.random.default_rng(42)
X = np.hstack([X, np.tile([0, 0, 0, 1, 0], (1000, 1))])

sel = VarianceThreshold(threshold=0.01)
X_sel = sel.fit_transform(X)
print(f"Features before: {X.shape[1]}, after variance filter: {X_sel.shape[1]}")
```

### Recursive feature elimination with CV (RFECV)

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

X, y = make_classification(n_samples=800, n_features=20, n_informative=8,
                            n_redundant=4, random_state=42)

rfecv = RFECV(
    GradientBoostingClassifier(n_estimators=50, random_state=42),
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc",
    min_features_to_select=3,
)
rfecv.fit(X, y)
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"CV ROC-AUC with all features:     {rfecv.cv_results_['mean_test_score'][-1]:.4f}")
print(f"CV ROC-AUC with optimal features: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_ - rfecv.min_features_to_select]:.4f}")
```

### Permutation importance — post-fit model-agnostic

```python
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=15, n_informative=6,
                            n_redundant=3, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

result = permutation_importance(model, X_te, y_te, n_repeats=15,
                                 scoring="roc_auc", random_state=42)
order = np.argsort(result.importances_mean)[::-1]

print("Permutation importance (ROC-AUC drop when feature shuffled):")
for rank, i in enumerate(order[:10]):
    print(f"  {rank+1:2d}. feature_{i:02d}: "
          f"{result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")
print("\nNegative = shuffling that feature improves score → likely noise feature")
```

---

## Early stopping — free regularisation

For iterative algorithms (GBM, neural nets), early stopping monitors validation
loss and halts when it stops improving. It is the simplest and most effective
overfitting prevention technique.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

# ❌ No early stopping: runs all 500 trees
gbm_full = GradientBoostingClassifier(n_estimators=500, random_state=42)
gbm_full.fit(X_tr, y_tr)

# ✅ With early stopping via staged_predict_proba
gbm_es = GradientBoostingClassifier(n_estimators=500, random_state=42,
                                     subsample=0.8)  # subsample enables OOB estimate
gbm_es.fit(X_tr, y_tr)

val_scores = [
    roc_auc_score(y_te, proba[:, 1])
    for proba in gbm_es.staged_predict_proba(X_te)
]
best_n = int(np.argmax(val_scores)) + 1
print(f"Best n_estimators via early stopping: {best_n}")
print(f"ROC-AUC at n={best_n}: {val_scores[best_n-1]:.4f}")
print(f"ROC-AUC at n=500:    {val_scores[-1]:.4f}")
print(f"Saved {500 - best_n} unnecessary trees")
```

---

## AutoML frameworks

| Framework | Strengths | Weakness |
|---|---|---|
| **Optuna** | Flexible, any objective, fast TPE | DIY: you write the trial function |
| **FLAML** | Zero-config, fast, production-safe | Less flexible for custom pipelines |
| **Auto-sklearn** | Full sklearn pipeline search | Slow, memory-heavy |
| **H2O AutoML** | Distributed, good UI | Java dependency |
| **TPOT** | Genetic algorithm on pipelines | Very slow |

For most DS workflows: Optuna for targeted tuning, FLAML for quick baselines.

---

## Best practices

1. **Define metric first** — tune the metric that matters, not accuracy.
2. **Cross-validate inside the objective** — never tune on the test set.
3. **Use log-scale for learning rates** — `suggest_float(..., log=True)`.
4. **Set a time budget, not a trial count** — `study.optimize(timeout=600)`.
5. **Plot optimisation history** — `optuna.visualization.plot_optimization_history`.
6. **Refit on all data after tuning** — best params found on CV splits, not full data.
7. **Save the study** — `optuna.create_study(storage="sqlite:///optuna.db")` for restart.
