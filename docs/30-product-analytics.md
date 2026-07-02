# 30 · Product Analytics

> Measuring and improving user behaviour in digital products — the dominant flavour
> of "data scientist" at tech and SaaS companies.

---

## Why data science here

Product analytics is what most big-tech DS roles *actually* mean. The business
question is always a variant of: **did this change make the product better, and for
whom?** The data is behavioural: clicks, sessions, feature activations, purchases,
support tickets. It is high-volume, dirty, and causally treacherous — most of what
looks like signal is selection bias.

The domain sits at the intersection of statistics (experimentation), forecasting
(growth modelling), and causal inference (what drove the change). Unlike FinTech or
HealthTech, the "label" is rarely external — you define metrics (DAU, retention D7,
LTV), and the definition of success is itself a modelling choice.

Three failure modes kill most product DS work: (1) metric selection that optimises
a proxy at the expense of the real goal, (2) launching an A/B test without adequate
power, (3) calling a "trend" causal when it is seasonal.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Retention modelling | P(user active at day N \| day-0 cohort) | Cohort survival curves, logistic regression |
| Feature impact | Did shipping X change retention? | A/B test, diff-in-diff, CUPED |
| Funnel drop-off | Where do users abandon the conversion flow? | Funnel analysis, chi-square per step |
| DAU / growth forecasting | How many users next quarter? | Decomposition + regression, Prophet |
| Churn prediction | Which users will stop using the product? | Binary classification (LightGBM, LR) |
| Engagement segmentation | Who are the power users vs. casuals? | Clustering (KMeans, GMM) on behaviour vectors |
| Metric sensitivity | Is D7 retention the right north-star? | Correlation analysis vs. LTV |

---

## Key techniques

### 1. Cohort retention analysis

Track users by their first-activity date (acquisition cohort) and measure the
fraction still active N days later. The output is a triangle heatmap. The lesson:
aggregate retention hides heterogeneity; product changes show up as kinks in the
cohort curves, not in the aggregate DAU line.

**Pitfall:** mixing acquisition cohorts of different sizes inflates variance for
recent cohorts. Weight by cohort size or restrict to mature cohorts for comparisons.

### 2. Funnel analysis

Model conversion as a sequence of Bernoulli steps. Chi-square or Fisher's exact
test per step to identify where the problem is. Segment by device, source, and user
vintage to find *who* drops off, not just *where*.

**Pitfall:** funnel order assumes a linear path. Most products have non-linear
flows. Session-replay data often reveals the real path.

### 3. A/B testing (online experiments)

Randomise users to control/treatment. Test the primary metric with a two-sample
t-test or Mann-Whitney (for non-normal outcomes like revenue). Report effect size
and CI, not just p-value. See [17](17-experimentation-advanced.md) for CUPED,
sequential testing, and multi-arm bandits.

**Pitfall:** peeking at p-values before the pre-specified sample size is reached
inflates Type I error. Pre-register stopping criteria.

### 4. Causal inference without an experiment

When you can't randomise — a competitor launched, a market crashed, a feature
shipped to all users at once — reach for quasi-experimental methods:
diff-in-diff, synthetic control, regression discontinuity. See
[09](09-causal-inference-and-experimentation.md).

**Pitfall:** parallel-trends assumption is not testable — validate with pre-period
placebo tests.

### 5. Growth accounting / decomposition

Decompose DAU change into: new users + resurrected users − churned users.
Forces precise language: "growth slowed" could mean acquisition fell, churn rose,
or resurrection dropped. Each has a different fix.

---

## Best practices & pitfalls

- **Define "active" before you model.** One session? One meaningful action? "Active
  = any login" is almost always wrong; it counts drive-by visits as retained users.
- **DAU is a lagging indicator.** By the time DAU moves, the cohort problem is
  weeks old. Track D1, D7, D30 retention per cohort in real time.
- **Power before you ship.** Run a power calculation (or use a simulation) before
  starting an experiment. Underpowered tests waste weeks and give inconclusive
  results that get interpreted as "no effect."
- **Novelty effect is real.** New features often spike engagement for 1–2 weeks
  regardless of real value. Run experiments long enough to let it wash out.
- **Metric traps.** Optimising for click-through inflates rage-clicks. Optimising
  for session length can trap users in confusion loops. Always pair a primary metric
  with a guardrail metric (e.g., retention AND NPS).
- **Simpson's paradox in funnels.** An overall conversion improvement can hide a
  decline in a key segment if that segment's share changed. Always segment.

---

## Python example — cohort retention analysis

```python
"""
Cohort retention analysis on synthetic product data.

Simulates a 6-month user base with realistic retention decay and
computes a D1/D7/D14/D30 retention triangle.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Simulate user event log ─────────────────────────────────────────────────
N_USERS = 5_000
N_DAYS  = 90

# Each user has a first-activity day (acquisition date)
acquisition_day = rng.integers(0, 60, size=N_USERS)

# Retention probability decays with days-since-acquisition
# p(active on day d | acquired on day 0) ≈ 0.6 * exp(-0.05 * d)
records = []
for uid in range(N_USERS):
    acq = acquisition_day[uid]
    for d in range(acq, N_DAYS):
        days_since = d - acq
        p = 0.60 * np.exp(-0.05 * days_since)
        if rng.random() < p:
            records.append({"user_id": uid, "day": d, "cohort_day": acq})

events = pd.DataFrame(records)

# ── Build retention matrix ───────────────────────────────────────────────────
cohort_sizes = events.groupby("cohort_day")["user_id"].nunique().rename("cohort_size")
events["days_since_acq"] = events["day"] - events["cohort_day"]

# Pivot: rows = cohort day, columns = days_since_acquisition
retention = (
    events.groupby(["cohort_day", "days_since_acq"])["user_id"]
    .nunique()
    .unstack(fill_value=0)
)
retention = retention.div(cohort_sizes, axis=0)

# Keep only d0, d1, d7, d14, d30
keep = [c for c in [0, 1, 7, 14, 30] if c in retention.columns]
retention = retention[keep]

print("\nRetention matrix (first 10 cohorts):")
print(retention.head(10).round(2).to_string())

# ── Plot heatmap ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(retention.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

ax.set_xticks(range(len(keep)))
ax.set_xticklabels([f"D{k}" for k in keep])
ax.set_yticks(range(0, len(retention), 5))
ax.set_yticklabels(retention.index[::5])
ax.set_xlabel("Days since acquisition")
ax.set_ylabel("Acquisition cohort (day)")
ax.set_title("Cohort retention heatmap")
plt.colorbar(im, ax=ax, label="Retention rate")
plt.tight_layout()
plt.savefig(OUT / "retention_heatmap.png", dpi=120)
plt.close()

# ── Summary stats ────────────────────────────────────────────────────────────
summary = retention.mean().rename("avg_retention")
print("\nAverage retention across cohorts:")
print(summary.round(3).to_string())
```

---

## Cross-references

- [09](09-causal-inference-and-experimentation.md) — A/B testing and causal methods
- [17](17-experimentation-advanced.md) — CUPED, sequential testing, bandits
- [07](07-time-series-forecasting.md) — growth and DAU forecasting
- [05](05-supervised-learning.md) — churn prediction models
- [06](06-unsupervised-learning.md) — user segmentation and clustering
