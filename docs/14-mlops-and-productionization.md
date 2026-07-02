# 14 · MLOps & Productionisation

A model in a notebook creates zero value. MLOps is the engineering discipline that
turns a model into a reliable, monitored, reproducible service — and it's often
what separates a data scientist who ships from one who prototypes.

## The lifecycle beyond training
```
data → train → package → deploy → monitor → (detect drift) → retrain → …
```
It's a loop with the world in it. The model is the easy part; the loop is the job.

## Reproducibility & tracking (the foundation)
- **Experiment tracking** — log params, metrics, code version, and data version for
  every run (MLflow, W&B). "Which run made this number?" must be answerable months
  later.
- **Data & artefact versioning** — DVC or a feature store; a model is only
  reproducible if its *training data* is.
- **Environments as code** — a container image pins the exact runtime. "Works on my
  machine" is not a deployment strategy.
- **Model registry** — versioned, stage-tagged (staging/production) artefacts with
  lineage back to the run that produced them.

## Serving patterns
| Pattern | Use when |
|---------|----------|
| **Batch** | Predictions can be precomputed (nightly scores) — simplest, cheapest |
| **Real-time (REST/gRPC)** | Predictions needed on demand (fraud check at checkout) |
| **Streaming** | Continuous event flows (Kafka + online scoring) |
| **Edge / on-device** | Latency/privacy constraints (mobile, sensors) |

**Training/serving skew** is the classic production bug: the features are computed
one way in the training pipeline and another way at serving time, so the live
model sees inputs it was never trained on. A shared feature-transformation library
(or feature store) is the fix.

## Monitoring — the part everyone under-invests in
A deployed model degrades silently. Monitor:
- **Operational** — latency, throughput, error rates, uptime (like any service).
- **Data drift** — are incoming feature distributions shifting from training
  (PSI, KS test, population stability)? Often the *first* sign of trouble.
- **Concept drift** — has the relationship between features and target changed?
  (The pandemic broke countless demand models overnight.)
- **Prediction & performance monitoring** — track the output distribution now;
  track accuracy once labels arrive (which may be delayed). Set alerts.

## Deployment safety
- **Shadow deployment** — run the new model alongside the old on live traffic
  *without* acting on it; compare before switching.
- **Canary / gradual rollout** — send a small % of traffic to the new model first.
- **A/B test the model** — the ultimate check that the new model improves the
  *business* metric, not just offline accuracy
  ([09](09-causal-inference-and-experimentation.md)).
- **Rollback plan** — always be able to revert to the previous version instantly.

## Testing ML systems (more than unit tests)
- Data validation (schema, ranges, nulls) at ingestion — e.g. Great Expectations.
- Tests on the *transformation* code (the deterministic part).
- Behavioural tests on the model (invariances, directional expectations,
  minimum-functionality checks — see the CheckList methodology for NLP).
- A CI pipeline that retrains and re-evaluates on a fixed benchmark before any
  promotion to production.

## A sane maturity path
Don't build a platform on day one. **Level 0:** manual, scripted, tracked with
MLflow, containerised. **Level 1:** automated training pipeline + registry +
monitoring. **Level 2:** automated retraining triggered by drift, with CI/CD. Most
teams get enormous value at Level 0–1; reach for Level 2 when the cost of staleness
justifies it.

---

## Python example — MLflow experiment tracking + data drift detection

```python
"""
Demonstrates MLOps Level 0 patterns without needing a running MLflow server:
  - Structured experiment logging (to local directory)
  - Data drift detection (PSI + KS test)
  - Model serialisation and loading (joblib)
  - Behavioural tests on loaded model (invariance checks)

Run `mlflow ui` in the same directory to browse the tracked experiments.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Optional MLflow — graceful fallback if not installed
try:
    import mlflow
    MLFLOW = True
    mlflow.set_experiment("breast_cancer_demo")
except ImportError:
    MLFLOW = False
    print("mlflow not installed — skipping tracking (install with: pip install mlflow)")

SEED = 42
OUT  = Path("mlops_demo")
OUT.mkdir(exist_ok=True)

# ── Train ────────────────────────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

params = {"C": 1.0, "max_iter": 1000, "seed": SEED}
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=params["C"], max_iter=params["max_iter"],
                                  random_state=SEED)),
])
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

# ── Log to MLflow ─────────────────────────────────────────────────────────────
if MLFLOW:
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.sklearn.log_model(pipe, "model")
        print(f"MLflow run logged. ROC-AUC={roc_auc:.4f}")
else:
    print(f"ROC-AUC={roc_auc:.4f}")

# ── Serialise model ───────────────────────────────────────────────────────────
model_path = OUT / "model.pkl"
joblib.dump(pipe, model_path)
print(f"Model saved to {model_path}")

# ── Load + behavioural tests (run in CI) ─────────────────────────────────────
loaded = joblib.load(model_path)

# Directional invariance: high-risk sample should score higher than low-risk
hi_risk = X_test.iloc[y_test.values == 1].head(1)
lo_risk = X_test.iloc[y_test.values == 0].head(1)
p_hi = loaded.predict_proba(hi_risk)[0, 1]
p_lo = loaded.predict_proba(lo_risk)[0, 1]
print(f"\nBehavioural test — directional invariance:")
print(f"  high-risk sample score: {p_hi:.3f}")
print(f"  low-risk  sample score: {p_lo:.3f}")
assert p_hi > p_lo, "Directional invariance FAILED — investigate!"
print("  PASS")

# Score-on-noise should be near 0.5
noise = pd.DataFrame(np.random.default_rng(SEED).standard_normal((100, X.shape[1])),
                     columns=X.columns)
noise_scores = loaded.predict_proba(noise)[:, 1]
print(f"\nBehavioural test — noise inputs → scores near 0.5:")
print(f"  noise score mean={noise_scores.mean():.3f}, std={noise_scores.std():.3f}")

# ── Data drift simulation ─────────────────────────────────────────────────────
print("\n── Data drift detection ─────────────────────────────────────────────────")
rng = np.random.default_rng(SEED)

# Simulate production data with 20% covariate shift on top 2 features
X_prod = X_test.copy()
X_prod.iloc[:, 0] += rng.normal(loc=2.0, scale=0.5, size=len(X_prod))  # shift mean
X_prod.iloc[:, 1] *= rng.uniform(1.3, 1.5, size=len(X_prod))            # scale

# KS test per feature
drift_rows = []
for col in X_test.columns[:8]:   # check first 8 features for brevity
    ks_stat, p_val = stats.ks_2samp(X_train[col], X_prod[col])
    drift_rows.append({"feature": col, "KS_stat": round(ks_stat, 3),
                       "p_value": round(p_val, 4),
                       "drift_flag": "YES" if p_val < 0.05 else "no"})

drift_df = pd.DataFrame(drift_rows)
print(drift_df.to_string(index=False))
drifted = (drift_df["drift_flag"] == "YES").sum()
print(f"\n{drifted}/{len(drift_df)} features show significant drift (α=0.05)")
print("Action: inspect, re-validate model on new distribution, consider retraining.")
```

---

## Cross-references

- [01](01-lifecycle-and-reproducibility.md) — reproducibility foundations (seeds, Pipelines)
- [04](04-evaluation-and-validation.md) — monitoring metrics and evaluation offline
- [09](09-causal-inference-and-experimentation.md) — A/B test a deployed model properly
