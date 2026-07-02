# 01 · Lifecycle & Reproducibility

Modelling is maybe 10% of the job. The other 90% is framing, data, validation,
and making the whole thing reproducible. This is where credibility is won or lost.

## The lifecycle (CRISP-DM, de-romanticised)

1. **Business framing** — What decision changes based on this? If none, stop. Turn
   the vague ask ("understand churn") into a decision + a metric + a target
   population. *The most common failure is solving a well-specified version of the
   wrong problem.*
2. **Data understanding** — Where does each column come from, and *when* is it
   known? A feature that is only available *after* the outcome is [leakage](03-data-and-feature-engineering.md) waiting
   to happen (see [03](03-data-and-feature-engineering.md)).
3. **Data preparation** — Cleaning, joining, [feature engineering](03-data-and-feature-engineering.md). Usually the
   largest time sink. Everything here must be expressible as a transform that can
   be re-run on new data.
4. **Modelling** — Start with a baseline (mean, last value, [logistic regression](05-supervised-learning.md)).
   Complex models earn their place only by beating it.
5. **Evaluation** — Against the *baseline* and the *business metric*, on data the
   model has never seen, with a validation scheme that matches deployment
   ([04](04-evaluation-and-validation.md)).
6. **Deployment & monitoring** — The model meets reality; reality drifts
   ([14](14-mlops-and-productionization.md)).

It is a loop, not a line. Expect to revisit framing after seeing the data.

## Project structure that scales

A convention beats a clever one-off. A workable default:

```
project/
├── data/            # raw/ (read-only, never edited) and interim/, processed/
├── notebooks/       # exploration only — nothing important lives ONLY here
├── src/             # importable, testable functions (the real code)
│   ├── data.py      # loading + cleaning
│   ├── features.py  # feature transforms (usable at train AND inference)
│   └── model.py     # train / evaluate / predict
├── models/          # serialised artefacts (gitignored or tracked via DVC/registry)
├── reports/         # figures + written findings
├── tests/
├── requirements.txt / environment.yml
└── README.md
```

Rule of thumb: **notebooks are for thinking, `src/` is for truth.** Anything a
result depends on must live in version-controlled code, not in a cell you ran
once in an order you can't reconstruct.

## Reproducibility checklist

- [ ] **Seed everything** (`numpy`, framework RNGs, split seeds) and record it.
- [ ] **Pin dependencies** (`requirements.txt` with versions, or a lockfile).
- [ ] **Data is versioned**, or at minimum hashed/dated so "which data?" is answerable.
- [ ] **One command reproduces the result** from raw data to figure.
- [ ] **Config over hard-coding** — paths, hyperparameters, and thresholds in one place.
- [ ] **[Experiment tracking](14-mlops-and-productionization.md)** — params, metrics, and artefacts logged (MLflow,
      Weights & Biases). "Which run produced this number?" should never be a mystery.
- [ ] **Environment captured** — a container or `environment.yml`, so "works on my
      machine" becomes "works in the image."

If a colleague can't reproduce your headline number from a clean checkout, you
don't have a result — you have an anecdote.

---

## Python example — one-command reproducible pipeline

```python
"""
Demonstrates a minimal reproducible project structure:
  - Seeded randomness
  - Transformation inside Pipeline (no leakage)
  - Results written to file (not just printed)
  - Importable utility functions (not notebook-only logic)

This is the pattern every project in this repo follows.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ── Seed everything — first line of any script ────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)
OUT  = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Load data — from a known, versioned source ────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

# ── Pipeline — scaler fit ONLY on training folds ──────────────────────────────
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
])

# ── Cross-validation — stratified, seeded ─────────────────────────────────────
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

result = {"roc_auc_mean": scores.mean(), "roc_auc_std": scores.std(), "seed": SEED}
print(f"ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# ── Write results to file — not just print ────────────────────────────────────
results_df = pd.DataFrame([result])
results_df.to_csv(OUT / "results.csv", index=False)
print(f"Results written to results.csv")

# ── Checklist verification ────────────────────────────────────────────────────
print("\nReproducibility checklist:")
print("  [x] SEED set and recorded in output")
print("  [x] Preprocessing inside Pipeline (no fit-before-split)")
print("  [x] Results written to file (not ephemeral print)")
print("  [x] Data from versioned source (sklearn bundled dataset)")
print("  [x] No notebook-only magic — this script is importable and re-runnable")
```

---

## Cross-references

- [03](03-data-and-feature-engineering.md) — leakage (the reproducibility failure mode)
- [04](04-evaluation-and-validation.md) — honest evaluation schemes
- [14](14-mlops-and-productionization.md) — [MLOps](14-mlops-and-productionization.md): taking reproducibility to production
