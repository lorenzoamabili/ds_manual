"""
Project 3 — Causal Inference from Observational Data
====================================================
Setup   : Semi-synthetic data with a KNOWN true effect. We simulate a confounded
          "treatment" (e.g. enrolled in a training programme) whose assignment
          depends on covariates that ALSO drive the outcome (income). Because we
          built the data, we know the ground-truth Average Treatment Effect (ATE)
          and can check which estimators recover it.

Why simulate? On real observational data you can never see the true effect, so
you can't tell a good estimator from a biased one. Recovering a planted effect is
the standard way to validate a causal pipeline before trusting it on real data.

Estimators demonstrated:
  0. Naive difference in means         -> biased by confounding
  1. Regression adjustment (outcome)   -> unbiased IF outcome model correct
  2. IPW (propensity model)            -> unbiased IF propensity model correct
  3. Propensity-score matching         -> nearest-neighbour on the score
  4. Doubly-robust (AIPW)              -> unbiased if EITHER model is correct
Plus covariate-balance diagnostics (standardised mean differences) before/after
weighting, which are the assumption checks reviewers actually look for.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from dsmanual import standardised_mean_difference, clip_propensity  # tested utilities

rng = np.random.default_rng(7)
N = 4000
TRUE_ATE = 3.0                      # the effect we plant, in $1000s of income

# ---------------------------------------------------------------- 1. simulate
age   = rng.normal(40, 10, N)
educ  = rng.normal(13, 3, N)
prior = rng.normal(30, 8, N)       # prior income, a strong confounder
# Propensity: older, less-educated, lower-prior-income people enrol MORE.
logit = -0.04*age - 0.15*educ - 0.06*prior + 6.5
p = 1/(1+np.exp(-logit))
T = rng.binomial(1, p)             # treatment assignment (confounded)
# Outcome depends on covariates AND treatment (constant effect = TRUE_ATE).
noise = rng.normal(0, 4, N)
Y = 0.10*age + 0.8*educ + 0.9*prior + TRUE_ATE*T + noise
df = pd.DataFrame(dict(age=age, educ=educ, prior=prior, treat=T, Y=Y))
X = df[["age", "educ", "prior"]].values
print(f"N={N} | treated={T.mean():.2%} | true ATE = {TRUE_ATE}")

# ---------------------------------------------------------------- 2. naive
naive = df.loc[df["treat"] == 1, "Y"].mean() - df.loc[df["treat"] == 0, "Y"].mean()

# ---------------------------------------------------------------- 3. regression adjustment
# Fit one outcome model, predict everyone's Y under T=1 and under T=0, average the gap.
Xt = np.column_stack([X, df["treat"].values])
out = LinearRegression().fit(Xt, df.Y)
mu1 = out.predict(np.column_stack([X, np.ones(N)]))
mu0 = out.predict(np.column_stack([X, np.zeros(N)]))
reg_ate = (mu1 - mu0).mean()

# ---------------------------------------------------------------- 4. propensity model + IPW
ps_model = LogisticRegression(max_iter=1000).fit(X, df["treat"])
ps = ps_model.predict_proba(X)[:, 1]
ps = clip_propensity(ps, eps=0.01)                # trim to avoid exploding weights
w = np.where(df["treat"] == 1, 1/ps, 1/(1-ps))          # inverse-probability weights
ipw_ate = (np.sum(w*df["treat"]*df.Y)/np.sum(w*df["treat"])
           - np.sum(w*(1-df["treat"])*df.Y)/np.sum(w*(1-df["treat"])))

# ---------------------------------------------------------------- 5. propensity-score matching
treated_idx = np.where(df["treat"] == 1)[0]
control_idx = np.where(df["treat"] == 0)[0]
nn = NearestNeighbors(n_neighbors=1).fit(ps[control_idx].reshape(-1, 1))
_, m = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
matched_control = control_idx[m.ravel()]
psm_ate = (df.Y.values[treated_idx] - df.Y.values[matched_control]).mean()  # ATT

# ---------------------------------------------------------------- 6. doubly-robust (AIPW)
dr = (mu1 - mu0
      + df["treat"].values*(df.Y.values - mu1)/ps
      - (1-df["treat"].values)*(df.Y.values - mu0)/(1-ps)).mean()

# ---------------------------------------------------------------- 7. covariate balance (SMD)
def smd(col, weights=None):
    """Thin adapter over the tested dsmanual.standardised_mean_difference."""
    t, c = (df["treat"] == 1).values, (df["treat"] == 0).values
    wt = weights[t] if weights is not None else None
    wc = weights[c] if weights is not None else None
    return standardised_mean_difference(df.loc[t, col], df.loc[c, col], wt, wc)

bal = pd.DataFrame({
    "before": {c: abs(smd(c)) for c in ["age", "educ", "prior"]},
    "after_IPW": {c: abs(smd(c, w)) for c in ["age", "educ", "prior"]},
})
print("\n=== Covariate balance |SMD| (want < 0.1 after weighting) ===")
print(bal.round(3).to_string())

# ---------------------------------------------------------------- results
est = pd.Series({
    "TRUE ATE": TRUE_ATE,
    "Naive diff-in-means": naive,
    "Regression adjustment": reg_ate,
    "IPW": ipw_ate,
    "PSM (ATT)": psm_ate,
    "Doubly-robust (AIPW)": dr,
}).round(3)
print("\n=== Estimated treatment effect ===")
print(est.to_string())

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
colors = ["#444"] + ["#c0392b" if abs(v-TRUE_ATE) > 0.5 else "#27ae60"
                     for v in est.values[1:]]
ax[0].barh(est.index[::-1], est.values[::-1], color=colors[::-1])
ax[0].axvline(TRUE_ATE, ls="--", color="k", lw=1)
ax[0].set(title="Estimated vs. true ATE (dashed = truth)", xlabel="effect")
bal.plot.barh(ax=ax[1]); ax[1].axvline(0.1, ls="--", color="k", lw=1)
ax[1].set(title="Covariate imbalance |SMD|", xlabel="|standardised mean diff|")
fig.tight_layout(); fig.savefig("results.png", dpi=120); plt.close(fig)

with open("metrics.md", "w") as f:
    f.write(f"# Project 3 results — true ATE = {TRUE_ATE}\n\n")
    f.write(est.to_frame("estimate").to_markdown() + "\n\n")
    f.write("The naive comparison is badly biased (confounders push treated units' "
            "baseline outcomes down). Regression, IPW, matching and AIPW all recover "
            "the planted effect. AIPW is *doubly robust*: consistent if EITHER the "
            "outcome model OR the propensity model is correctly specified.\n\n")
    f.write("## Covariate balance |SMD|\n\n")
    f.write(bal.round(3).to_markdown() + "\n")
print("\nSaved: results.png, metrics.md")
