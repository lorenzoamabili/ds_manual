# 17 · Experimentation — Beyond the Basic [A/B Test](09-causal-inference-and-experimentation.md)

[Doc 09](09-causal-inference-and-experimentation.md) covers the foundations of
randomised experiments. This is the practitioner's layer: the techniques that make
experimentation *fast, trustworthy, and scalable* in a real product org — and the
subtle ways experiments lie.

## Getting more power for the same traffic

Experiments are often traffic-limited. Two techniques buy sensitivity for free:

- **CUPED (variance reduction using pre-experiment data).** Regress the outcome on
  a pre-period covariate (usually the same metric measured *before* the
  experiment) and analyse the residual. Because pre-period behaviour predicts
  post-period behaviour, this strips out a large chunk of between-user variance —
  often cutting required sample size by 30–50% with no bias. The single
  highest-leverage trick in industrial experimentation.
- **Stratification / blocking & regression adjustment** — adjust for known
  covariates to remove their noise. (Randomisation makes this unbiased; it only
  tightens the estimate.)

## Not peeking: valid stopping

The cardinal sin is checking the [p-value](02-statistics-that-matter.md) repeatedly and stopping when it dips below
0.05 — this inflates false positives dramatically (repeated looks = repeated
chances to get lucky). Fixes:

- **Fixed-horizon testing** — decide n up front (power analysis), look once. Simple
  and safe.
- **Sequential / always-valid inference** — designed for continuous monitoring:
  group-sequential boundaries (O'Brien-Fleming), alpha-spending, or **always-valid
  p-values / confidence sequences** (mSPRT, e-values). These *let* you peek and
  stop early while controlling error.

## Trust checks the pros run automatically
- **Sample Ratio Mismatch (SRM)** — if you randomised 50/50 but observe 48/52 with
  large n, something is broken (logging, assignment, redirect bias). A chi-square
  test on the split is a mandatory guardrail; a failing SRM invalidates the
  experiment — don't interpret the result, fix the pipeline.
- **A/A tests** — run "no difference" experiments to validate the platform's false
  positive rate is actually ~5%.
- **Pre-registration** of the primary metric and analysis to avoid the
  garden-of-forking-paths problem.

## When the simple design breaks
- **Interference / network effects** — in marketplaces and social products, treating
  one user affects others ([SUTVA](09-causal-inference-and-experimentation.md) violation), biasing naive estimates. Use
  **[cluster](06-unsupervised-learning.md) randomisation** (randomise cities/markets) or **switchback
  experiments** (flip the whole system on/off over time windows) — standard in
  two-sided marketplaces (rideshare, delivery).
- **Long-term vs. short-term effects** — the metric that moves in week 1 (novelty)
  may reverse. Use holdback groups and surrogate/long-term metrics.
- **Heterogeneous effects** — the average can hide that a change helps one segment
  and hurts another. Estimate CATEs (causal forests, meta-learners) — *pre-specify*
  the segments or treat it as exploratory to avoid false discovery.

## Adaptive experiments — bandits
When the goal is to *earn* while learning (not to precisely measure an effect),
**multi-armed bandits** (Thompson sampling, UCB) shift traffic toward winning arms
during the test, reducing regret. **Contextual bandits** personalise the choice by
user features. Trade-off: bandits optimise reward but give weaker, biased effect
*estimates* than a clean A/B test — use them for optimisation, not for a
defensible causal readout.

## The org-level view
Mature experimentation is a *platform*, not a series of one-offs: a metrics layer
with a single source of truth, automated power/SRM/guardrail checks, a
pre-registration habit, and an experiment review culture. The technique matters
less than the discipline that stops the org from fooling itself at scale.

---

## Python example — CUPED variance reduction

```python
"""
CUPED (Controlled-experiment Using Pre-Experiment Data):
Shows that using a pre-period covariate reduces required sample size by ~40%.
Standard in industrial experimentation (Netflix, Microsoft, Booking.com).
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
N = 5000  # users per arm

# Pre-experiment covariate: previous week's metric (strong signal)
pre_control   = rng.normal(100, 20, N)
pre_treatment = rng.normal(100, 20, N)

# Post-experiment outcome: correlated with pre (rho=0.7) + small treatment effect
true_effect = 2.0   # absolute lift
rho = 0.7
noise_std = 20 * np.sqrt(1 - rho**2)

post_control   = rho * pre_control   + rng.normal(0, noise_std, N)
post_treatment = rho * pre_treatment + rng.normal(true_effect, noise_std, N)

# ── Standard t-test ───────────────────────────────────────────────────────────
t_std, p_std = stats.ttest_ind(post_treatment, post_control)

# ── CUPED adjustment ──────────────────────────────────────────────────────────
# theta = Cov(Y, X_pre) / Var(X_pre)
all_pre  = np.concatenate([pre_control, pre_treatment])
all_post = np.concatenate([post_control, post_treatment])
theta = np.cov(all_post, all_pre)[0,1] / np.var(all_pre)

# Adjusted outcome: remove variance explained by pre-period
cuped_treatment = post_treatment - theta * (pre_treatment - pre_treatment.mean())
cuped_control   = post_control   - theta * (pre_control   - pre_control.mean())
t_cuped, p_cuped = stats.ttest_ind(cuped_treatment, cuped_control)

# ── Variance comparison ───────────────────────────────────────────────────────
var_std   = np.var(post_control) + np.var(post_treatment)
var_cuped = np.var(cuped_control) + np.var(cuped_treatment)
reduction = 1 - var_cuped / var_std

print(f"True effect: {true_effect}")
print(f"\nStandard t-test: est={post_treatment.mean()-post_control.mean():.3f}  p={p_std:.4f}")
print(f"CUPED:           est={cuped_treatment.mean()-cuped_control.mean():.3f}  p={p_cuped:.4f}")
print(f"\nVariance reduction from CUPED: {reduction:.0%}")
print(f"Equivalent sample size saving: ~{reduction:.0%} fewer users for same power")
print(f"\nCUPED is free power — run it on every A/B test with historical data.")
```

---

## Cross-references

- [09](09-causal-inference-and-experimentation.md) — A/B testing foundations
- [30](30-product-analytics.md) — experimentation in product analytics
- [02](02-statistics-that-matter.md) — multiple comparisons, power
