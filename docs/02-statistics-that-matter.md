# 02 · The Statistics That Actually Matter

You don't need to re-derive the CLT. You need a working command of a small set of
ideas that come up constantly and that people get subtly wrong.

---

## Estimation & uncertainty

- **A point estimate without an interval is half a result.** Always attach a
  confidence/[credible interval](20-bayesian-and-probabilistic.md) or a standard error.
- **The bootstrap is your universal tool.** Resample the data with replacement,
  recompute the statistic, repeat 1 000×; the spread of those values *is* your
  sampling distribution. Works when a closed-form SE is hard or the statistic is
  weird (median, AUC, a ratio).
- **Standard error ≠ standard deviation.** SD describes spread in the data; SE
  describes uncertainty in an *estimate* and shrinks with √n.

---

## Hypothesis testing — and its traps

- A **p-value** is P(data this extreme | null true). It is *not* P(null true), not
  effect size, and not importance. A tiny p-value on a trivial effect is common
  with large n.
- **Report the effect size and its interval**, not just significance. "Statistically
  significant" answers "is it real?"; the interval answers "is it big enough to care?".
- **Multiple comparisons inflate false positives.** Test 20 things at α=0.05 and
  you expect one spurious hit. Correct with Bonferroni (conservative) or
  Benjamini–Hochberg FDR (more power).
- **Power** is P(detect a real effect of a given size). Underpowered studies both
  miss real effects *and* exaggerate the ones they do find (winner's curse).
  Run a power calculation *before* the experiment ([09](09-causal-inference-and-experimentation.md)).

---

## Distributions to recognise on sight

| Distribution | Arises from | Diagnostic | Transform |
|---|---|---|---|
| Normal | Sums of many small effects | QQ-plot, Shapiro-Wilk | — |
| Log-normal | Products, exponential growth | Heavy right tail; log → Normal | log(x) |
| Poisson | Count of rare events in fixed interval | Mean ≈ variance | — |
| Negative binomial | Overdispersed counts (variance > mean) | Very common in real data | — |
| Power-law / Pareto | Wealth, city sizes, social networks | Straight line on log-log plot | log-log |
| Beta | Rates/proportions in [0,1] | — | logit(x) |

**When the mean lies:** log-normal and power-law distributions have means that are
dominated by extreme values. The median (or geometric mean after log-transform) is
the right summary. Income, social-media followers, revenue per customer — all
log-normal in practice.

---

## Regression: the engine underneath everything

OLS deserves deep understanding because it underlies so much:

- Coefficients are **partial effects** ("holding all other variables fixed") — only
  meaningful if the model is correctly specified.
- **Residual diagnostics are mandatory.** Plot residuals vs. fitted: patterns mean
  your model is missing structure. QQ-plot of residuals: heavy tails mean your SEs
  are wrong.
- **Heteroskedasticity** (variance of residuals changes with fitted value) makes
  coefficients unbiased but SEs wrong. Fix: HC3 robust standard errors.
- **Multicollinearity** inflates SEs and makes individual coefficients unstable.
  Detect with VIF (>10 is a problem). Fix: ridge, drop one of correlated pair.
- **[Clustered](06-unsupervised-learning.md) data** (repeat measurements per patient, store, user) produces
  correlated residuals. Fix: clustered standard errors or mixed-effects model.

---

## Correlation: what it is and isn't

- **Pearson r** measures *linear* association. r=0 does not mean independent
  (Anscombe's quartet, datasaurus dozen).
- **Spearman ρ** is Pearson on ranks — handles monotone non-linear and is robust
  to outliers.
- **Partial correlation** controls for a third variable — use when you want "X–Y
  correlation holding Z fixed."
- Correlation in the presence of a [confounder](09-causal-inference-and-experimentation.md) is almost always wrong (see
  [09](09-causal-inference-and-experimentation.md)).

---

## Python example — bootstrap CI and multiple comparisons

```python
"""
Demonstrates bootstrap confidence intervals, t-test, effect size,
and Benjamini-Hochberg correction for multiple tests.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ── 1. Bootstrap CI for a median ─────────────────────────────────────────────
# Revenue per customer: log-normal (mean lies, median is better summary)
revenue = rng.lognormal(mean=4.0, sigma=1.2, size=500)

def bootstrap_ci(data, stat_fn, n_boot=2000, alpha=0.05, seed=42):
    rng_ = np.random.default_rng(seed)
    boot_stats = [stat_fn(rng_.choice(data, size=len(data), replace=True))
                  for _ in range(n_boot)]
    lo, hi = np.quantile(boot_stats, [alpha/2, 1 - alpha/2])
    return stat_fn(data), lo, hi

med, lo, hi = bootstrap_ci(revenue, np.median)
print(f"Median revenue: {med:.1f}  95% CI [{lo:.1f}, {hi:.1f}]")
print(f"Mean revenue:   {revenue.mean():.1f}  (inflated by extreme values)")

# ── 2. Effect size (Cohen's d) ────────────────────────────────────────────────
def cohens_d(a, b):
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / pooled_std

control = rng.normal(100, 15, 200)
treatment = rng.normal(103, 15, 200)  # tiny real effect
t_stat, p_val = stats.ttest_ind(treatment, control)
d = cohens_d(treatment, control)
print(f"\nA/B test: t={t_stat:.2f}, p={p_val:.3f}, Cohen's d={d:.3f}")
print(f"p<0.05: {p_val < 0.05} — but d={d:.2f} is a tiny effect. Is it meaningful?")

# ── 3. Multiple comparisons correction ──────────────────────────────────────
# Simulate 20 A/B tests; 1 is truly effective, 19 are noise
n_tests = 20
p_values = [stats.ttest_ind(rng.normal(0,1,200), rng.normal(0,1,200)).pvalue
            for _ in range(n_tests - 1)]
p_values.append(stats.ttest_ind(rng.normal(0,1,200), rng.normal(0.5,1,200)).pvalue)
p_values = np.array(p_values)

reject_uncorrected = p_values < 0.05
reject_bh, _, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

print(f"\nMultiple testing ({n_tests} tests, 1 truly effective):")
print(f"  Uncorrected: {reject_uncorrected.sum()} 'significant' (expected ~1 false positive)")
print(f"  BH-corrected: {reject_bh.sum()} 'significant'")

# ── 4. Residual diagnostics ───────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
X = rng.normal(0, 1, (200, 1))
y = 3 * X.flatten() + rng.normal(0, 1, 200)

model = LinearRegression().fit(X, y)
residuals = y - model.predict(X)
fitted = model.predict(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(fitted, residuals, alpha=0.5, s=20)
ax1.axhline(0, color="red", lw=1)
ax1.set_xlabel("Fitted values"); ax1.set_ylabel("Residuals")
ax1.set_title("Residuals vs. Fitted\n(should be random noise around 0)")

stats.probplot(residuals, plot=ax2)
ax2.set_title("QQ-plot\n(points on line = normally distributed residuals)")

plt.tight_layout()
plt.savefig("residual_diagnostics.png", dpi=120)
plt.close()

print("\nKey rules:")
print("  - Report effect size + CI, not just p-value")
print("  - Correct for multiple comparisons")
print("  - Always plot residuals before trusting a regression")
```

---

## [Bayesian](20-bayesian-and-probabilistic.md) vs. frequentist, pragmatically

Don't pick a religion. Use the frame that fits:

- **Frequentist** for well-defined experiments with clear repeated-sampling
  interpretation ([A/B tests](09-causal-inference-and-experimentation.md), regulated clinical trials).
- **Bayesian** when you have real prior information, small data, or want a direct
  probability statement ("70% chance B beats A") stakeholders actually understand.
  Hierarchical Bayesian models excel at "many small groups" (per-store, per-user).

See [20](20-bayesian-and-probabilistic.md) for the full Bayesian treatment.

---

## The one formula to memorise

**Bayes' theorem in English:** [posterior](20-bayesian-and-probabilistic.md) ∝ likelihood × prior. After seeing data,
your belief is your prior belief updated by how probable the data is under each
hypothesis. This is literally what good scientific reasoning should always do.
