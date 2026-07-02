"""
Project 6 - Algorithmic Fairness Audit
======================================
Setup : A hiring/lending-style screening model. A protected attribute `group` does
        NOT enter the model, yet the model still discriminates - because a feature it
        DOES use was measured with historical bias against one group. This is the
        realistic failure mode: "we didn't use the protected attribute" is not a
        fairness guarantee.

Demonstrates:
  - Why omitting the protected attribute does not prevent disparate impact.
  - Group fairness metrics: selection rate, disparate-impact ratio (the 80% rule),
    true-positive rate (equal opportunity), false-positive rate.
  - A mitigation: group-specific decision thresholds that equalise opportunity,
    and the accuracy trade-off it costs.

Fairness is not one number - these metrics can conflict, and which one matters is a
context and policy question, not a purely technical one (see docs/19).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(5)
N = 8000

# ---------------------------------------------------------------- simulate
group = RNG.binomial(1, 0.5, N)                       # protected attribute (0 / 1)
skill = RNG.normal(0, 1, N)                           # true latent ability, EQUAL by design
# True qualification depends only on skill -> equal true base rates across groups.
qualified = (skill + RNG.normal(0, 0.3, N) > 0.4).astype(int)
# Observed score is measured with BIAS against group 1 (historical under-measurement).
observed_score = skill - 0.6*group + RNG.normal(0, 0.5, N)
df = pd.DataFrame({"group": group, "observed_score": observed_score, "qualified": qualified})
print(f"True qualification rate - group0: {qualified[group==0].mean():.1%}, "
      f"group1: {qualified[group==1].mean():.1%}  (equal by construction)")

tr, te = train_test_split(df, test_size=0.4, random_state=0, stratify=df.qualified)
# The model uses ONLY the (biased) score, NOT the protected attribute.
clf = LogisticRegression().fit(tr[["observed_score"]], tr.qualified)
te = te.assign(p=clf.predict_proba(te[["observed_score"]])[:, 1])

# ---------------------------------------------------------------- fairness metrics
def audit(te, thresholds):
    """thresholds: dict group->cutoff. Returns per-group rates."""
    rows = {}
    for g in (0, 1):
        s = te[te.group == g]
        pred = (s.p >= thresholds[g]).astype(int)
        pos = pred == 1
        tp = ((pred == 1) & (s.qualified == 1)).sum()
        fp = ((pred == 1) & (s.qualified == 0)).sum()
        rows[g] = {
            "selection_rate": pos.mean(),
            "TPR": tp / max((s.qualified == 1).sum(), 1),   # equal-opportunity metric
            "FPR": fp / max((s.qualified == 0).sum(), 1),
            "accuracy": (pred == s.qualified).mean(),
        }
    return pd.DataFrame(rows).T

single = 0.5
before = audit(te, {0: single, 1: single})
di_before = before.loc[1, "selection_rate"] / before.loc[0, "selection_rate"]
print("\n=== Single threshold (0.5) - the 'unaware' model ===")
print(before.round(3).to_string())
print(f"Disparate-impact ratio (group1/group0 selection): {di_before:.2f} "
      f"({'FAILS' if di_before < 0.8 else 'passes'} the 80% rule)")

# ---------------------------------------------------------------- mitigation: equalise opportunity
# Lower group 1's threshold until its TPR matches group 0's at threshold 0.5.
target_tpr = before.loc[0, "TPR"]
g1 = te[te.group == 1].sort_values("p")
thr1 = single
for t in np.linspace(0.5, 0.1, 41):
    pred = (g1.p >= t).astype(int)
    tpr = ((pred == 1) & (g1.qualified == 1)).sum() / max((g1.qualified == 1).sum(), 1)
    if tpr >= target_tpr:
        thr1 = t; break
after = audit(te, {0: single, 1: thr1})
di_after = after.loc[1, "selection_rate"] / after.loc[0, "selection_rate"]
print(f"\n=== Group-specific thresholds (group0={single}, group1={thr1:.2f}) ===")
print(after.round(3).to_string())
print(f"Disparate-impact ratio: {di_after:.2f}  |  "
      f"overall accuracy {before['accuracy'].mean():.3f} -> {after['accuracy'].mean():.3f}")

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
x = np.arange(2)
for i, (lab, tbl) in enumerate([("before (single threshold)", before),
                                ("after (equalised opportunity)", after)]):
    ax[0].bar(x + i*0.35, tbl["selection_rate"], 0.35, label=lab)
ax[0].set_xticks(x + 0.17); ax[0].set_xticklabels(["group 0", "group 1"])
ax[0].axhline(0, color="k", lw=.6)
ax[0].set(ylabel="selection rate", title="Selection rate by group")
ax[0].legend(fontsize=8)
for i, (lab, tbl) in enumerate([("before", before), ("after", after)]):
    ax[1].bar(x + i*0.35, tbl["TPR"], 0.35, label=lab)
ax[1].set_xticks(x + 0.17); ax[1].set_xticklabels(["group 0", "group 1"])
ax[1].set(ylabel="true positive rate", title="Equal opportunity (TPR) by group")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("fairness.png", dpi=120); plt.close(fig)

with open("metrics.md", "w") as f:
    f.write("# Project 6 - fairness audit\n\n")
    f.write("## 'Unaware' model, single threshold\n\n")
    f.write(before.round(3).to_markdown() + "\n\n")
    f.write(f"Disparate-impact ratio = **{di_before:.2f}** - "
            f"{'fails' if di_before<0.8 else 'passes'} the 80% rule despite the model "
            "never seeing the protected attribute.\n\n")
    f.write("## After equalising opportunity (group-specific thresholds)\n\n")
    f.write(after.round(3).to_markdown() + "\n\n")
    f.write(f"Disparate-impact ratio = **{di_after:.2f}**; overall accuracy moves "
            f"{before['accuracy'].mean():.3f} -> {after['accuracy'].mean():.3f}. "
            "The gap closes at a small accuracy cost - the fairness/accuracy trade-off "
            "made explicit rather than hidden.\n")
print("\nSaved: fairness.png, metrics.md")
