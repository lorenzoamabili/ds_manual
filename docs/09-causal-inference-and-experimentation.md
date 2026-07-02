# 09 · Causal Inference & Experimentation

The question that prediction *cannot* answer: **did X cause Y, and by how much?**
This is the highest-leverage, most-undersupplied skill in applied DS, because
almost every business decision is causal ("if we *do* this, what happens?"). Paired
project: [P3 — recovering a known ATE](../projects/p3_causal_inference).

## The core idea: potential outcomes

Each unit has two potential outcomes: Y(1) if treated, Y(0) if not. The causal
effect is Y(1) − Y(0). The **fundamental problem**: you only ever observe *one* of
them. Causal inference is the science of estimating the missing counterfactual.

- **ATE** — average treatment effect over the whole population.
- **ATT** — average effect on the treated (what P3's matching estimates).
- **CATE / HTE** — conditional/heterogeneous effects: for *whom* does it work?

## The gold standard: randomised experiments (A/B tests)

Randomisation makes treated and control groups **exchangeable** — equal in
expectation on everything, observed and unobserved. That's why a simple
difference in means is unbiased. Getting A/B tests right in practice:

- **Power & sample size up front.** Fix the minimum detectable effect, α, and
  power (usually 80%); compute n *before* launching. Peeking early and stopping on
  significance inflates false positives massively.
- **Randomisation unit = analysis unit.** Randomise by user, analyse by user. If
  you randomise by user but measure per-session, your SEs are wrong (clustered).
- **One metric to rule it (OEC).** Pre-register the primary metric; treat the rest
  as guardrails/diagnostics to avoid the multiple-comparisons trap
  ([02](02-statistics-that-matter.md)).
- **Watch for interference (SUTVA violations).** Marketplaces and social networks
  break the "no spillover" assumption — treatment of one unit affects others.
  Cluster-randomise or use switchback designs.
- **Novelty & primacy effects** — early behaviour ≠ steady state; run long enough.

## When you can't randomise: observational causal inference

Often you only have observational data (ethics, cost, legacy logs). Then you must
*assume* your way to identification, and defend the assumptions. The key one is
**conditional ignorability / no unmeasured confounding**: given the covariates
you've measured, treatment is as-good-as-random. This is untestable — argue it
with domain knowledge, and probe it with sensitivity analysis.

Estimators (all demonstrated in P3, which plants a **true ATE of 3.0** and shows
the naive estimate is **−1.5** — wrong sign — while the proper methods recover ~3):

| Method | Idea | Robust if… |
|--------|------|-----------|
| **Regression adjustment** | Model Y from covariates + treatment; average the modelled gap | …the **outcome model** is correct |
| **IPW** | Weight each unit by 1/P(treatment received) to rebuild a pseudo-population where treatment is independent of covariates | …the **propensity model** is correct |
| **Matching (PSM / caliper)** | Pair each treated unit with a control of similar propensity | …matching achieves balance & common support |
| **Doubly-robust (AIPW / TMLE)** | Combine outcome model + propensity weighting | …**either** model is correct (two shots on goal) |

**Always check covariate balance.** Report standardised mean differences (SMD)
before and after adjustment; you want |SMD| < 0.1. P3 shows IPW pulling imbalance
from ~0.4 to <0.03. Balance diagnostics are exactly what reviewers scrutinise.

## Quasi-experimental designs (when even ignorability is a stretch)

- **Difference-in-Differences (DiD)** — compare the *change* in a treated group to
  the change in a control group; differences out fixed group traits. Assumes
  **parallel trends** absent treatment.
- **Regression Discontinuity (RD)** — when treatment is assigned by a threshold
  (score ≥ cutoff), compare units just above vs. just below; near the cutoff it's
  as-good-as random.
- **Instrumental Variables (IV)** — an instrument that affects treatment but not
  the outcome except *through* treatment; recovers effects despite unmeasured
  confounding. Strong assumptions, weak-instrument dangers.

## Practical discipline
- **Draw the DAG.** A causal graph forces you to state what you believe causes
  what, and reveals which variables to adjust for — and which to *not* (never
  condition on a collider or a mediator you don't intend to block).
- **Estimate uncertainty** — bootstrap or robust/clustered SEs; a causal estimate
  without an interval is not a result.
- **Do a sensitivity analysis** — how strong would an unmeasured confounder have to
  be to overturn your conclusion? (E-values, Rosenbaum bounds.)
- **Tools:** `DoWhy` (encodes the identify→estimate→refute workflow), `EconML`
  (heterogeneous effects, ML-based), `causalml`, `statsmodels`.

---

## Python example — A/B test: power calculation + two-sample test

```python
"""
A/B testing workflow:
  1. Power calculation before launch
  2. Simulate the experiment
  3. Test with t-test + report effect size
  4. Show why peeking inflates false-positive rate
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# ── 1. Power calculation (before the experiment) ─────────────────────────────
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
# Want to detect a 5% relative lift on conversion rate=10%, α=0.05, power=0.80
baseline, lift = 0.10, 0.005   # absolute effect = 0.5 percentage points
effect_size = lift / np.sqrt(baseline * (1 - baseline))  # Cohen's h approx
n_required = int(analysis.solve_power(effect_size=effect_size,
                                       alpha=0.05, power=0.80)) + 1
print(f"Required n per arm: {n_required:,}  (to detect +{lift:.1%} lift at 80% power)")

# ── 2. Simulate the experiment ────────────────────────────────────────────────
n = n_required
control   = rng.binomial(1, 0.10, n).astype(float)
treatment = rng.binomial(1, 0.105, n).astype(float)  # true effect = +0.5pp

# ── 3. Analyse once (correct approach) ───────────────────────────────────────
t, p = stats.ttest_ind(treatment, control)
diff = treatment.mean() - control.mean()
se   = np.sqrt(treatment.var(ddof=1)/n + control.var(ddof=1)/n)
ci_lo, ci_hi = diff - 1.96*se, diff + 1.96*se

print(f"\nCorrect analysis (look once at n={n:,}):")
print(f"  Effect: {diff:+.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  p = {p:.3f}  {'Significant' if p < 0.05 else 'Not significant'}")

# ── 4. Peeking simulation: false-positive inflation ───────────────────────────
# Null experiment (no true effect), peek every 10 observations
N_SIM, PEEK_EVERY = 2000, 50
fp_peeker, fp_once = 0, 0

for _ in range(N_SIM):
    null_c = rng.binomial(1, 0.10, n_required).astype(float)
    null_t = rng.binomial(1, 0.10, n_required).astype(float)  # no true effect

    # Peeking: stop as soon as p < 0.05
    peeked_sig = False
    for cutoff in range(PEEK_EVERY, n_required + 1, PEEK_EVERY):
        _, p_peek = stats.ttest_ind(null_t[:cutoff], null_c[:cutoff])
        if p_peek < 0.05:
            peeked_sig = True
            break
    fp_peeker += peeked_sig

    # Look once at the end
    _, p_once = stats.ttest_ind(null_t, null_c)
    fp_once += (p_once < 0.05)

print(f"\nNull experiment (no true effect), {N_SIM} simulations:")
print(f"  Look-once FPR:  {fp_once/N_SIM:.1%}  (expected ≈ 5%)")
print(f"  Peeking FPR:    {fp_peeker/N_SIM:.1%}  (inflated!)")
print(f"\nConclusion: peeking nearly doubles the false-positive rate.")
```

---

## Cross-references

- [P3](../projects/p3_causal_inference) — doubly-robust ATE estimation with planted truth
- [17](17-experimentation-advanced.md) — CUPED, sequential testing, bandits
- [30](30-product-analytics.md) — experimentation in product analytics
- [35](35-martech.md) — uplift modelling (who responds to treatment?)
