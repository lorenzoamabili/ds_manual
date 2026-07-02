"""
Project 5 — Survival / Time-to-Event Analysis
=============================================
Setup : Simulated time-to-churn data with right-censoring and KNOWN hazard ratios,
        so we can check that the Cox model recovers the truth. Survival analysis is
        the right tool whenever you care about *when* an event happens and some
        subjects haven't had it yet (censoring) — churn timing, equipment failure,
        clinical time-to-event, loan default timing.

Why not ordinary regression? Because you cannot just drop or ignore the customers
who haven't churned yet — that "censored" information ("survived at least this
long") is real signal, and standard regression has no way to use it correctly.

Demonstrates:
  - Kaplan-Meier survival curves by group (the non-parametric summary).
  - The log-rank test for whether two survival curves differ.
  - Cox proportional-hazards regression to quantify covariate effects (hazard ratios).
Uses statsmodels (no external survival library needed).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

RNG = np.random.default_rng(11)
N = 3000
# True log-hazard ratios we plant, then try to recover:
TRUE = {"premium_plan": -0.7,   # premium customers churn slower (HR<1, protective)
        "high_charges":  0.5,   # expensive plans churn faster (HR>1)
        "n_tickets":     0.3}   # each support ticket raises the hazard

# ---------------------------------------------------------------- simulate
premium   = RNG.binomial(1, 0.4, N)
high_chg  = RNG.binomial(1, 0.5, N)
tickets   = RNG.poisson(1.0, N)
X = pd.DataFrame({"premium_plan": premium, "high_charges": high_chg, "n_tickets": tickets})

lin = (TRUE["premium_plan"]*premium + TRUE["high_charges"]*high_chg
       + TRUE["n_tickets"]*tickets)
baseline_rate = 0.04
event_time = RNG.exponential(1 / (baseline_rate*np.exp(lin)))   # exponential survival
censor_time = RNG.uniform(0, 40, N)                              # administrative censoring
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(int)                 # 1 = churned, 0 = censored
print(f"N={N} | events (churned): {event.mean():.1%} | censored: {(1-event).mean():.1%}")

# ---------------------------------------------------------------- Kaplan-Meier by group
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for val, lab, col in [(1, "premium", "C0"), (0, "standard", "C1")]:
    m = premium == val
    sf = SurvfuncRight(time[m], event[m])
    ax[0].step(sf.surv_times, sf.surv_prob, where="post", label=lab, color=col)
ax[0].set(xlabel="months", ylabel="survival probability (still a customer)",
          title="Kaplan-Meier: premium customers survive longer")
ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_ylim(0, 1)

# log-rank test between the two groups
chisq, pval = survdiff(time, event, premium)
print(f"\nLog-rank test (premium vs standard): chi2={chisq:.1f}, p={pval:.2e}")

# ---------------------------------------------------------------- Cox PH regression
model = PHReg(time, X, status=event).fit()
res = pd.DataFrame({
    "true_log_HR": [TRUE[c] for c in X.columns],
    "est_log_HR":  np.asarray(model.params),
    "hazard_ratio": np.exp(np.asarray(model.params)),
    "p_value":     np.asarray(model.pvalues),
}, index=X.columns).round(3)
print("\n=== Cox proportional-hazards estimates ===")
print(res.to_string())

# forest-style plot of hazard ratios with recovery of the truth
ax[1].errorbar(np.exp(np.asarray(model.params)), range(len(X.columns)),
               xerr=None, fmt="o", color="C0", label="estimated HR")
ax[1].scatter(np.exp([TRUE[c] for c in X.columns]), range(len(X.columns)),
              marker="x", color="r", s=80, label="true HR")
ax[1].axvline(1, ls="--", color="k", lw=1)
ax[1].set_yticks(range(len(X.columns))); ax[1].set_yticklabels(X.columns)
ax[1].set(xlabel="hazard ratio (>1 = churns faster)", title="Cox recovers the planted hazard ratios")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("survival.png", dpi=120); plt.close(fig)

with open("metrics.md", "w") as f:
    f.write(f"# Project 5 — survival analysis ({event.mean():.0%} events, rest censored)\n\n")
    f.write("## Cox proportional-hazards: estimated vs. planted effects\n\n")
    f.write(res.to_markdown() + "\n\n")
    f.write(f"Log-rank (premium vs standard): chi2={chisq:.1f}, p={pval:.1e}.\n\n")
    f.write("A hazard ratio of 0.5 means half the instantaneous churn rate. The Cox "
            "model recovers the planted log-hazard-ratios despite ~half the data being "
            "censored — which is exactly the information ordinary regression throws away.\n")
print("\nSaved: survival.png, metrics.md")
