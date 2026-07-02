# 16 · Survival / Time-to-Event Analysis

The function for questions of the form **"how long until X, and what changes
that?"** — time-to-churn, time-to-failure, time-to-default, time-to-event in
trials. Paired project: [P5 — Cox on censored churn](../projects/p5_survival_analysis).

## Why it needs its own toolkit: censoring

The defining feature is **censoring**: at analysis time, many subjects *haven't had
the event yet*. You can't drop them (they're often the majority and the healthiest
cases) and you can't treat "no event" as the event not happening (it may happen
tomorrow). Survival methods use the partial information "survived at least this
long" correctly. Ordinary regression/classification cannot, which is why
"predict churn in 30 days" as a binary target quietly wastes information and bakes
in a fixed horizon.

- **Right-censoring** (most common) — the event hasn't happened by end of
  observation.
- **Left-truncation** — subjects enter observation only after some time (immortal
  time bias if ignored).

## The core objects
- **Survival function S(t)** — probability of surviving past time *t*.
- **Hazard h(t)** — instantaneous event rate at *t*, given survival so far. The
  "risk right now."
- **Hazard ratio (HR)** — the multiplicative effect of a covariate on the hazard.
  HR = 0.5 → half the instantaneous rate; HR > 1 → faster to the event.

## The three workhorse methods
| Method | Type | Use for |
|--------|------|---------|
| **Kaplan-Meier** | Non-parametric | Estimating & plotting S(t) for a group; the standard visual |
| **Log-rank test** | Non-parametric | Testing whether two+ survival curves differ |
| **Cox proportional hazards** | Semi-parametric | Quantifying covariate effects (HRs) without assuming a baseline hazard shape — the default regression model. [P5 recovers planted HRs.] |
| **Parametric (Weibull, AFT)** | Parametric | When you need to extrapolate beyond observed time or want a full time distribution |

## Assumptions & pitfalls
- **Proportional hazards** (Cox) — the HR is constant over time. Check it
  (Schoenfeld residuals; a KM crossing is a red flag). If violated, use
  time-varying covariates, stratification, or an AFT model.
- **Non-informative censoring** — censoring must be unrelated to the risk of the
  event. If sick patients drop out (informative), estimates are biased.
- **Competing risks** — when other events preclude the one you care about (a
  customer can't churn if their account is deleted). Standard KM overestimates;
  use cumulative-incidence functions / Fine-Gray models.

## Beyond the basics
- **Survival machine learning** — Random Survival Forests, gradient-boosted Cox,
  and DeepSurv handle non-linearities and interactions while respecting censoring.
- **Evaluation** — the concordance index (**C-index**, a survival analogue of AUC:
  does the model rank who fails sooner?) and time-dependent AUC / Brier scores.

## Practical stack
`lifelines` (the friendliest API: `KaplanMeierFitter`, `CoxPHFitter`),
`scikit-survival` (ML survival models, sklearn-compatible), or `statsmodels`
`duration` (used in P5, no extra dependency). In R, the `survival` package is the
reference implementation.

---

## Python example — Kaplan-Meier + Cox on customer churn

```python
"""
Survival analysis: Kaplan-Meier curves + Cox proportional hazards.
Synthetic churn data with a known hazard ratio for a pricing feature.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

rng = np.random.default_rng(42)
n = 500

# Simulate: premium plan has 0.5x the churn hazard (HR=0.5)
plan = rng.binomial(1, 0.4, n)           # 1 = premium
tenure_limit = 24                          # months observed
base_hazard = 0.06
true_duration = rng.exponential(1 / (base_hazard * (1 - 0.5*plan)))
censored = rng.uniform(6, tenure_limit, n)
observed_duration = np.minimum(true_duration, censored)
event_occurred = (true_duration <= censored).astype(int)

df = pd.DataFrame({"duration": observed_duration, "event": event_occurred,
                   "premium": plan, "age": rng.integers(20, 60, n)})

# ── Kaplan-Meier by plan type ─────────────────────────────────────────────────
kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7, 4))
for group, label in [(0, "Standard"), (1, "Premium")]:
    mask = df["premium"] == group
    kmf.fit(df.loc[mask,"duration"], df.loc[mask,"event"], label=label)
    kmf.plot_survival_function(ax=ax)
ax.set_xlabel("Months"); ax.set_ylabel("Survival probability")
ax.set_title("Kaplan-Meier: churn survival by plan type")
plt.tight_layout()
plt.savefig("kaplan_meier.png", dpi=120)
plt.close()

# ── Cox proportional hazards ──────────────────────────────────────────────────
cph = CoxPHFitter()
cph.fit(df[["duration","event","premium","age"]], duration_col="duration",
        event_col="event")
print(cph.print_summary())
print(f"\nRecovered HR for premium: {np.exp(cph.params_['premium']):.3f}")
print(f"True HR: 0.500  (premium = 50% lower hazard)")
```

---

## Cross-references

- [P5](../projects/p5_survival_analysis) — full survival project (planted HRs)
- [33](33-healthtech.md) — clinical survival analysis
- [38](38-hrtech.md) — time-to-attrition with Cox
