# 20 · Bayesian & Probabilistic Modelling

A modelling *philosophy* as much as a toolkit: represent everything you don't know
as a probability distribution, start from a prior, and update it with data to get a
posterior. Powerful precisely where classical ML is weakest — small data, nested
structure, and honest uncertainty.

## When to reach for it
- **Small or expensive data** — priors regularise and stop overfitting where a
  neural net would flail.
- **Hierarchical / grouped data** — the killer app (see below).
- **You need real uncertainty** — a full posterior, not a point estimate, so you can
  say "70% probability B beats A" — language stakeholders actually understand.
- **You have genuine prior knowledge** — physical constraints, past studies, expert
  bounds — that it would be wasteful to ignore.

## The one idea worth internalising: partial pooling
With many small groups (stores, users, regions, experiments), you face a dilemma:

- **No pooling** — a separate estimate per group; noisy, overfits tiny groups.
- **Complete pooling** — one global estimate; ignores real group differences.
- **Partial pooling (hierarchical/multilevel models)** — the Bayesian answer.
  Groups share a common prior, so small groups are *shrunk* toward the global mean
  while large groups keep their own signal. You get the best of both automatically.

This is the right model for "estimate a rate for each of 500 stores, some with 5
sales and some with 50,000," conversion by segment, per-user effects, and
meta-analysis. It is quietly one of the most useful tools in applied statistics.

## How inference actually happens
The posterior is rarely available in closed form, so:
- **MCMC** (Hamiltonian Monte Carlo / NUTS) — gold-standard, asymptotically exact
  samples from the posterior. What PyMC and Stan use by default. Slower.
- **Variational inference (ADVI)** — approximates the posterior with optimisation;
  much faster, scales to big data, at the cost of some accuracy.
- **Conjugate/analytic** — for simple models (Beta-Binomial, Normal-Normal) the
  posterior is exact and instant — the basis of Bayesian [A/B testing](09-causal-inference-and-experimentation.md).

Always run **diagnostics**: R-hat ≈ 1, sufficient effective sample size, no
divergences, and **posterior predictive checks** (does data simulated from the
fitted model resemble the real data?). A fit you didn't diagnose is not a fit.

## Common applications
- **Bayesian A/B testing** — direct probability that a variant is best, and expected
  loss; more intuitive than p-values ([09](09-causal-inference-and-experimentation.md)).
- **Marketing Mix Modelling (MMM)** — Bayesian regression with adstock/saturation
  priors to attribute sales to channels; a resurgent MarTech workhorse in the
  post-cookie world.
- **Time series** — structural/state-space models (Prophet is Bayesian-ish under the
  hood; `PyMC`/`orbit` for full control).
- **Hierarchical GLMs** everywhere grouped data appears.

## Tools
`PyMC` (Pythonic, NUTS, great docs) and `Stan`/`cmdstanpy` (the reference,
cross-language); `NumPyro`/`Pyro` (JAX/PyTorch-backed, fast, scalable);
`ArviZ` for diagnostics and posterior visualisation; `bambi` for a formula
interface to PyMC.

## The trade-off, honestly
Bayesian models are more work to specify, slower to fit, and demand more statistical
care than `model.fit()`. The payoff is [calibrated](04-evaluation-and-validation.md) uncertainty, graceful handling of
small/structured data, and the ability to encode knowledge. Use it where those
matter; reach for boosting when you just need a fast, accurate point prediction on
lots of tabular data.

---

## Python example — Bayesian A/B test with conjugate Beta-Binomial

```python
"""
Bayesian A/B testing using the Beta-Binomial conjugate model.
No MCMC needed — the posterior is analytically exact for proportions.
Demonstrates: posterior updating, probability-of-superiority, expected loss.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# ── Simulate A/B experiment (binomial conversions) ────────────────────────────
n_a, conv_a = 2000, 180   # control:  9.0% conversion
n_b, conv_b = 2000, 216   # variant: 10.8% conversion (true lift = +1.8pp)

# ── Prior: Beta(1,1) = uniform (no prior knowledge) ──────────────────────────
alpha_prior, beta_prior = 1, 1

# ── Posterior: Beta(alpha + successes, beta + failures) ──────────────────────
post_a = stats.beta(alpha_prior + conv_a, beta_prior + n_a - conv_a)
post_b = stats.beta(alpha_prior + conv_b, beta_prior + n_b - conv_b)

# ── Monte Carlo: P(B > A) and expected loss ────────────────────────────────────
N_MC = 100_000
samples_a = post_a.rvs(N_MC, random_state=42)
samples_b = post_b.rvs(N_MC, random_state=43)

prob_b_better = (samples_b > samples_a).mean()
expected_loss  = np.maximum(samples_a - samples_b, 0).mean()

print(f"Conversion A: {conv_a}/{n_a} = {conv_a/n_a:.2%}")
print(f"Conversion B: {conv_b}/{n_b} = {conv_b/n_b:.2%}")
print(f"\nP(B > A) = {prob_b_better:.1%}")
print(f"Expected loss if we choose B: {expected_loss:.4f} ({expected_loss*100:.2f} pp)")
print(f"\nFrequentist equivalent:")
from statsmodels.stats.proportion import proportions_ztest
z, p = proportions_ztest([conv_b, conv_a], [n_b, n_a])
print(f"  z = {z:.2f}, p = {p:.4f}")
print(f"\nBayesian gives: 'We're {prob_b_better:.0%} confident B is better'")
print(f"Frequentist gives: 'If null were true, we'd see this data {p:.1%} of the time'")
print(f"The Bayesian statement is what stakeholders actually want to know.")

# ── Plot posterior distributions ───────────────────────────────────────────────
x = np.linspace(0.06, 0.16, 500)
fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(x, post_a.pdf(x), alpha=0.4, label=f"Control A (conv={conv_a/n_a:.1%})")
ax.fill_between(x, post_b.pdf(x), alpha=0.4, label=f"Variant B (conv={conv_b/n_b:.1%})")
ax.axvline(conv_a/n_a, color="C0", linestyle="--", lw=1)
ax.axvline(conv_b/n_b, color="C1", linestyle="--", lw=1)
ax.set_xlabel("Conversion rate")
ax.set_ylabel("Posterior density")
ax.set_title(f"Bayesian A/B test — P(B>A) = {prob_b_better:.1%}")
ax.legend()
plt.tight_layout()
plt.savefig("bayesian_ab_test.png", dpi=120)
plt.close()
print("\nPlot saved: posterior distributions show the uncertainty around each rate.")
```

---

## Cross-references

- [09](09-causal-inference-and-experimentation.md) — A/B testing foundations
- [17](17-experimentation-advanced.md) — [sequential testing](17-experimentation-advanced.md) and bandits
- [35](35-martech.md) — Bayesian MMM for marketing mix
