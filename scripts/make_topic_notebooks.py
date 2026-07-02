"""
Generate high-quality topic notebooks with rich inline visualisations.
Run: python scripts/make_topic_notebooks.py
"""
import json, textwrap
from pathlib import Path

ROOT   = Path(__file__).parent.parent
NB_DIR = ROOT / "notebooks"

SETUP = """\
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "font.family":       "sans-serif",
    "lines.linewidth":   2.2,
    "patch.edgecolor":   "none",
})
C = ["#2563EB","#DC2626","#16A34A","#D97706","#7C3AED","#0891B2","#DB2777"]
print("Setup complete")
"""


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(src).strip()}

def code(src):
    return {"cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": textwrap.dedent(src).strip()}

def save(name, notebook):
    path = NB_DIR / name
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"  wrote notebooks/{name}")


# =============================================================================
# 01 · Statistics fundamentals
# =============================================================================
def nb_statistics():
    return nb([
        md("""# Statistics That Matter for Data Science

Practical statistical concepts that appear in every DS project:
**hypothesis testing**, the **bootstrap**, **power analysis**, and
**multiple comparisons correction**.
"""),
        md("## Setup"), code(SETUP),

        # ── 1. Sampling distributions ────────────────────────────────────────
        md("""## 1. Sampling distributions and the Central Limit Theorem

The mean of $n$ independent draws converges to a Normal distribution
regardless of the population distribution — this underpins almost every
standard test.
"""),
        code("""\
rng = np.random.default_rng(42)
n_samples, n_obs = 5000, [1, 5, 30, 200]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, n in zip(axes, n_obs):
    means = [rng.exponential(scale=1, size=n).mean() for _ in range(n_samples)]
    ax.hist(means, bins=50, color=C[0], alpha=0.85, density=True, edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(means), color=C[1], lw=2, linestyle="--", label=f"mean={np.mean(means):.2f}")
    ax.set(title=f"n = {n}", xlabel="Sample mean", ylabel="Density" if n==1 else "")
    ax.legend(fontsize=9)
fig.suptitle("CLT: sample means of Exponential(1) converge to Normal as n grows",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
"""),

        # ── 2. Bootstrap CI ──────────────────────────────────────────────────
        md("""## 2. Bootstrap confidence intervals

Model-free uncertainty estimation — no distributional assumptions,
works on any statistic (median, ratio, AUC).
"""),
        code("""\
rng   = np.random.default_rng(42)
data  = rng.lognormal(mean=4, sigma=0.8, size=250)   # skewed revenue data
n_boot = 10_000

boot_means   = [rng.choice(data, size=len(data), replace=True).mean()  for _ in range(n_boot)]
boot_medians = [np.median(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, vals, stat, col in zip(axes,
        [boot_means, boot_medians], ["Mean", "Median"], [C[0], C[2]]):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    ax.hist(vals, bins=60, color=col, alpha=0.85, density=True,
            edgecolor="white", linewidth=0.3)
    ax.axvline(lo, color="black", lw=1.8, linestyle="--", label=f"95% CI [{lo:.0f}, {hi:.0f}]")
    ax.axvline(hi, color="black", lw=1.8, linestyle="--")
    ax.axvline(np.mean(vals), color=C[1], lw=2, label=f"Point estimate: {np.mean(vals):.0f}")
    ax.set(title=f"Bootstrap {stat} (n=250, log-normal revenue)",
           xlabel=stat, ylabel="Density")
    ax.legend()
plt.tight_layout()
plt.show()
print(f"Mean    95% CI: [{np.percentile(boot_means,2.5):.1f}, {np.percentile(boot_means,97.5):.1f}]")
print(f"Median  95% CI: [{np.percentile(boot_medians,2.5):.1f}, {np.percentile(boot_medians,97.5):.1f}]")
print("Median CI is narrower -> median is more efficient for log-normal data")
"""),

        # ── 3. Power analysis ────────────────────────────────────────────────
        md("""## 3. Power analysis — how many samples?

Underpowered studies produce inflated estimates *when* they happen to
reach significance (winner's curse). Run power analysis *before* the
experiment.
"""),
        code("""\
from statsmodels.stats.power import TTestIndPower

analysis  = TTestIndPower()
baseline  = 0.10
lifts_pp  = np.arange(0.5, 5.1, 0.5)   # effect sizes in percentage points
alphas    = [0.01, 0.05, 0.10]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: required n vs effect size
for alpha, col in zip(alphas, [C[1], C[0], C[2]]):
    ns = []
    for lift in lifts_pp:
        p_treat   = baseline + lift / 100
        sigma     = ((baseline*(1-baseline) + p_treat*(1-p_treat)) / 2)**0.5
        effect_d  = (lift / 100) / sigma
        n         = analysis.solve_power(effect_size=effect_d, alpha=alpha, power=0.80)
        ns.append(n)
    ax1.plot(lifts_pp, ns, "o-", color=col, label=f"alpha={alpha}")
ax1.set(xlabel="True lift (percentage points)", ylabel="n per arm (power=80%)",
        title="Sample size vs lift size\n(baseline conversion=10%)", yscale="log")
ax1.legend()
ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: f"{int(x):,}"))

# Panel 2: power vs n for a 2pp lift
lift_pp = 2.0
p_treat = baseline + lift_pp / 100
sigma   = ((baseline*(1-baseline) + p_treat*(1-p_treat)) / 2)**0.5
eff_d   = (lift_pp / 100) / sigma
ns2     = np.linspace(500, 20_000, 200)
for alpha, col in zip(alphas, [C[1], C[0], C[2]]):
    powers = [analysis.solve_power(effect_size=eff_d, nobs1=n, alpha=alpha, power=None)
              for n in ns2]
    ax2.plot(ns2, powers, color=col, label=f"alpha={alpha}")
ax2.axhline(0.80, color="grey", lw=1.5, linestyle="--", label="Power=80%")
ax2.axhline(0.90, color="grey", lw=1.0, linestyle=":")
ax2.set(xlabel="n per arm", ylabel="Statistical power",
        title=f"Power vs n for +{lift_pp}pp lift\n(baseline={baseline:.0%})")
ax2.legend()
plt.tight_layout()
plt.show()
"""),

        # ── 4. Multiple comparisons ──────────────────────────────────────────
        md("""## 4. Multiple comparisons — Benjamini-Hochberg correction

Running 20 tests at alpha=5% gives ~1 false positive in expectation.
The BH correction controls the **False Discovery Rate** (FDR) rather
than the per-test alpha — more powerful than Bonferroni.
"""),
        code("""\
from statsmodels.stats.multitest import multipletests

rng = np.random.default_rng(42)
n_tests   = 30
n_true    = 5   # truly significant hypotheses

p_true   = rng.uniform(0, 0.005, n_true)    # small p (true effects)
p_noise  = rng.uniform(0.05, 1.0, n_tests - n_true)  # null hypotheses
p_values = np.concatenate([p_true, p_noise])
is_true  = np.array([True]*n_true + [False]*(n_tests-n_true))
rng.shuffle(p_values)   # mix them up (shuffle doesn't track truth labels — demo only)

_, reject_bonf, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")
_, reject_bh,   _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: p-value distribution
ax = axes[0]
ax.scatter(range(n_tests), sorted(p_values), color=C[0], s=50, zorder=3, label="p-value")
ax.axhline(0.05, color=C[1], lw=1.8, linestyle="--", label="Raw alpha=0.05")
bonf_thresh = 0.05 / n_tests
ax.axhline(bonf_thresh, color=C[3], lw=1.8, linestyle="-.", label=f"Bonferroni={bonf_thresh:.4f}")
# BH threshold line
sorted_p = np.sort(p_values)
bh_thresholds = 0.05 * (np.arange(1, n_tests+1)) / n_tests
bh_line_idx = np.where(sorted_p <= bh_thresholds)[0]
if len(bh_line_idx):
    ax.axhline(sorted_p[bh_line_idx[-1]], color=C[2], lw=1.8, linestyle=":", label=f"BH adaptive threshold")
ax.set(xlabel="Test rank", ylabel="p-value",
       title="p-values: raw vs corrected thresholds")
ax.legend(fontsize=9)

# Right: rejection comparison
ax = axes[1]
methods = ["Raw (p<0.05)", "Bonferroni", "Benjamini-Hochberg"]
n_reject = [(p_values < 0.05).sum(), reject_bonf.sum(), reject_bh.sum()]
bars = ax.bar(methods, n_reject, color=[C[1], C[3], C[2]], width=0.5)
ax.bar_label(bars, padding=3, fontsize=11, fontweight="bold")
ax.set(ylabel="Rejections", title="Number of rejected hypotheses\n(5 are truly significant)",
       ylim=(0, max(n_reject)+3))
ax.axhline(n_true, color="black", lw=1.5, linestyle="--", label=f"True positives = {n_true}")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Raw:    {(p_values < 0.05).sum()} rejections (inflation risk)")
print(f"Bonf:   {reject_bonf.sum()} rejections (too conservative)")
print(f"BH:     {reject_bh.sum()} rejections (controls FDR at 5%)")
"""),
    ])


# =============================================================================
# 02 · Model evaluation
# =============================================================================
def nb_evaluation():
    return nb([
        md("""# Model Evaluation & Validation

Choosing the right metric, avoiding leakage in evaluation,
calibrating probabilities, and setting decision thresholds from business costs.
"""),
        md("## Setup"), code(SETUP),

        md("""## 1. ROC-AUC vs PR-AUC — why metric choice matters for imbalance

ROC-AUC looks great even for a useless classifier on imbalanced data
because it counts true negatives. PR-AUC is brutally honest.
"""),
        code("""\
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve,
                              roc_auc_score, average_precision_score)

rng = np.random.default_rng(42)
X, y = make_classification(n_samples=12_000, n_features=12, n_informative=7,
                            weights=[0.97, 0.03], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

models = {
    "Logistic Reg": LogisticRegression(class_weight="balanced", max_iter=500),
    "Random Forest": RandomForestClassifier(100, class_weight="balanced", random_state=42),
    "GBM":          GradientBoostingClassifier(100, random_state=42),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
naive_ap = y_te.mean()

for (name, model), col in zip(models.items(), C):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:,1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    pre, rec, _ = precision_recall_curve(y_te, proba)
    roc = roc_auc_score(y_te, proba)
    ap  = average_precision_score(y_te, proba)
    axes[0].plot(fpr, tpr, color=col, label=f"{name} (AUC={roc:.3f})")
    axes[1].plot(rec, pre, color=col, label=f"{name} (AP={ap:.3f})")

axes[0].plot([0,1],[0,1],"--", color="grey", lw=1.2, label="Random")
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC curve  (3% positive rate)\nAll models look decent here")
axes[0].legend(fontsize=9)

axes[1].axhline(naive_ap, color="grey", lw=1.5, linestyle="--",
                label=f"No-skill baseline ({naive_ap:.3f})")
axes[1].set(xlabel="Recall", ylabel="Precision",
            title="PR curve  (same data, same models)\nNow quality differences are visible")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.show()
print(f"Positive rate: {y_te.mean():.1%}  ->  use PR-AUC for imbalanced tasks")
"""),

        md("""## 2. Calibration curves

A model predicting P=0.7 should be right ~70% of the time.
GBMs are typically overconfident (sigmoid-shaped calibration curve);
Logistic Regression is usually well-calibrated.
"""),
        code("""\
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=8000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

gbm     = GradientBoostingClassifier(200, random_state=42).fit(X_tr, y_tr)
gbm_cal = CalibratedClassifierCV(
    GradientBoostingClassifier(200, random_state=42), method="isotonic", cv=5
).fit(X_tr, y_tr)
lr      = LogisticRegression(max_iter=500).fit(X_tr, y_tr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for model, name, col, ls in [
    (gbm,     "GBM (raw)",        C[1], "--"),
    (gbm_cal, "GBM (calibrated)", C[0], "-"),
    (lr,      "Logistic Reg",     C[2], "-"),
]:
    frac_pos, mean_pred = calibration_curve(y_te, model.predict_proba(X_te)[:,1], n_bins=12)
    ax1.plot(mean_pred, frac_pos, "o-", color=col, ls=ls, lw=2, label=name)

ax1.plot([0,1],[0,1], "--", color="grey", lw=1.2, label="Perfect calibration")
ax1.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
        title="Calibration curves\n(closer to diagonal = better calibrated)")
ax1.legend()

# Histogram of predicted probabilities
for model, name, col in [
    (gbm,     "GBM (raw)",   C[1]),
    (gbm_cal, "GBM (calib)", C[0]),
    (lr,      "LR",          C[2]),
]:
    proba = model.predict_proba(X_te)[:,1]
    ax2.hist(proba, bins=40, alpha=0.55, color=col, label=name, density=True)

ax2.set(xlabel="Predicted probability", ylabel="Density",
        title="Predicted probability distributions\n(GBM raw is overconfident near 0 and 1)")
ax2.legend()
plt.tight_layout()
plt.show()
"""),

        md("""## 3. Cost-sensitive thresholds

The default threshold (0.5) is optimal only when FP and FN have equal costs.
In cancer screening (FN = missed cancer >> FP = unnecessary biopsy), lower thresholds are better.
"""),
        code("""\
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])
pipe.fit(X_tr, y_tr)
proba = pipe.predict_proba(X_te)[:,1]

cost_fp, cost_fn = 200, 8_000   # GBP: biopsy vs missed cancer
thresholds = np.arange(0.05, 0.96, 0.02)
costs = []
precs, recs = [], []
for t in thresholds:
    pred = (proba >= t).astype(int)
    fp = ((pred==1)&(y_te==0)).sum()
    fn = ((pred==0)&(y_te==1)).sum()
    costs.append(fp*cost_fp + fn*cost_fn)
    precs.append(precision_score(y_te, pred, zero_division=0))
    recs.append(recall_score(y_te, pred))

best_t = thresholds[np.argmin(costs)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(thresholds, costs, color=C[0], lw=2)
ax.axvline(0.5,   color="grey", lw=1.5, linestyle="--", label="Default t=0.50")
ax.axvline(best_t, color=C[1], lw=2,   linestyle="--",
           label=f"Optimal t={best_t:.2f} (min cost)")
ax.set(xlabel="Decision threshold", ylabel="Total expected cost (GBP)",
       title="Cost vs threshold\n(FP cost=£200, FN cost=£8,000)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: f"£{int(x):,}"))

ax = axes[1]
ax.plot(thresholds, precs, color=C[2], lw=2, label="Precision")
ax.plot(thresholds, recs,  color=C[3], lw=2, label="Recall")
ax.axvline(best_t, color=C[1], lw=2, linestyle="--",
           label=f"Optimal t={best_t:.2f}")
ax.set(xlabel="Threshold", ylabel="Score", title="Precision vs Recall\nvs threshold")
ax.legend(fontsize=9)

ax = axes[2]
ax.scatter(recs, precs, c=thresholds, cmap="viridis_r", s=40)
sc = ax.scatter(
    [recs[np.argmin(costs)]], [precs[np.argmin(costs)]],
    marker="*", color=C[1], s=250, zorder=5, label=f"Optimal t={best_t:.2f}"
)
ax.set(xlabel="Recall", ylabel="Precision", title="PR curve (color = threshold)")
ax.legend(fontsize=9)
plt.colorbar(
    plt.cm.ScalarMappable(cmap="viridis_r",
                          norm=mpl.colors.Normalize(thresholds.min(), thresholds.max())),
    ax=ax, label="Threshold"
)
plt.tight_layout()
plt.show()
print(f"Default threshold (0.50) cost: £{costs[np.argmin(np.abs(thresholds-0.5))]:,}")
print(f"Optimal threshold ({best_t:.2f}) cost: £{int(min(costs)):,}")
print(f"Savings: £{costs[np.argmin(np.abs(thresholds-0.5))] - min(costs):,.0f}")
"""),
    ])


# =============================================================================
# 03 · A/B testing & causal inference
# =============================================================================
def nb_ab():
    return nb([
        md("""# A/B Testing & Causal Inference

Running valid experiments, avoiding common traps (peeking, SUTVA violations),
and measuring causal effects rather than correlations.
"""),
        md("## Setup"), code(SETUP),

        md("""## 1. The peeking problem

Checking results before your pre-specified sample size and stopping at
first significance inflates the false-positive rate from 5% to ~30%.
"""),
        code("""\
from scipy import stats

rng    = np.random.default_rng(42)
n_days, n_per_day, n_sims, alpha = 30, 100, 3000, 0.05

fp_fixed, fp_peeking = 0, 0
peeking_stop_days = []

for _ in range(n_sims):
    ctrl, trt = [], []
    found = False
    for day in range(n_days):
        ctrl += rng.binomial(1, 0.10, n_per_day).tolist()
        trt  += rng.binomial(1, 0.10, n_per_day).tolist()
        if len(ctrl) >= 40:
            _, p = stats.ttest_ind(ctrl, trt)
            if p < alpha and not found:
                fp_peeking += 1
                peeking_stop_days.append(day + 1)
                found = True
    _, p = stats.ttest_ind(ctrl, trt)
    if p < alpha:
        fp_fixed += 1

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
strategies = ["Fixed horizon\n(alpha=5%)", "Peeking\n(check daily)"]
fp_rates   = [fp_fixed/n_sims, fp_peeking/n_sims]
bars = ax.bar(strategies, [r*100 for r in fp_rates], color=[C[2], C[1]], width=0.4)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=12, fontweight="bold")
ax.axhline(5, color="grey", lw=1.5, linestyle="--", label="Nominal alpha=5%")
ax.set(ylabel="False positive rate (%)", ylim=(0, 45),
       title=f"FPR in A/A tests\n({n_sims:,} simulations, 30 days, 100 obs/day)")
ax.legend()

ax = axes[1]
if peeking_stop_days:
    ax.hist(peeking_stop_days, bins=range(1,32), color=C[1], alpha=0.85,
            edgecolor="white", linewidth=0.4)
ax.set(xlabel="Day of early stop", ylabel="Count",
       title="When does peeking falsely 'find' significance?\n(mostly in early days when variance is high)")
plt.tight_layout()
plt.show()
print(f"Fixed-horizon FPR: {fp_fixed/n_sims:.1%}  (nominal = 5%)")
print(f"Peeking FPR:       {fp_peeking/n_sims:.1%}  (inflated by {fp_peeking/fp_fixed:.1f}x)")
"""),

        md("""## 2. CUPED — variance reduction with pre-experiment data

Regress out a pre-experiment covariate correlated with the post-experiment
metric. Equivalent sample size reduction shown below.
"""),
        code("""\
from scipy import stats

rng = np.random.default_rng(42)
n   = 2000

pre   = rng.normal(100, 20, n)
treat = rng.binomial(1, 0.5, n)
post  = 0.8*pre + treat*3 + rng.normal(0, 12, n)

theta      = np.cov(post, pre)[0,1] / np.var(pre)
post_cuped = post - theta*(pre - pre.mean())

var_reduction = 1 - np.var(post_cuped) / np.var(post)

# Standard vs CUPED estimate
t_std, p_std = stats.ttest_ind(post[treat==1], post[treat==0])
t_cup, p_cup = stats.ttest_ind(post_cuped[treat==1], post_cuped[treat==0])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scatter: pre vs post (correlation)
ax = axes[0]
for t, col, label in [(0, C[0], "Control"), (1, C[1], "Treatment")]:
    ax.scatter(pre[treat==t], post[treat==t], alpha=0.2, s=10, color=col, label=label)
ax.set(xlabel="Pre-experiment metric", ylabel="Post-experiment metric",
       title=f"Pre-post correlation\n(r = {np.corrcoef(pre, post)[0,1]:.2f})")
ax.legend(markerscale=2, fontsize=9)

# Distributions before and after CUPED
ax = axes[1]
for data, name, col, lw in [(post, "Raw", C[0], 1.5), (post_cuped, "CUPED", C[2], 2.5)]:
    ctrl_vals = data[treat==0]; trt_vals = data[treat==1]
    ax.hist(ctrl_vals - ctrl_vals.mean(), bins=40, alpha=0.35, color=col, density=True)
    ax.hist(trt_vals  - trt_vals.mean(),  bins=40, alpha=0.35, color=col, density=True)
ax.set(xlabel="Demeaned metric", ylabel="Density",
       title=f"Variance reduction: {var_reduction:.0%}\n(CUPED = narrower = more power)")

# p-value comparison
ax = axes[2]
metrics   = ["Standard t-test", "CUPED t-test"]
p_values  = [p_std, p_cup]
bar_colors = [C[0] if p>0.05 else C[2] for p in p_values]
bars = ax.bar(metrics, [-np.log10(p) for p in p_values], color=bar_colors, width=0.4)
ax.axhline(-np.log10(0.05), color=C[1], lw=1.8, linestyle="--",
           label="p=0.05 threshold")
ax.bar_label(bars, labels=[f"p={p:.4f}" for p in p_values], padding=4, fontsize=10)
ax.set(ylabel="-log10(p-value)  (higher = more significant)",
       title="CUPED achieves same significance\nwith fewer samples")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Variance reduction:  {var_reduction:.1%}")
print(f"Equivalent to:       {1/(1-var_reduction):.1f}x more data")
"""),

        md("""## 3. Heterogeneous treatment effects (T-learner)

Not everyone responds equally to a treatment. The T-learner estimates
per-user CATE by training separate response surface models and subtracting.
"""),
        code("""\
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(42)
n   = 4000

age   = rng.uniform(20, 70, n)
spend = rng.lognormal(4, 0.8, n)
p_t   = 1 / (1 + np.exp(-(age - 45) / 10))
treat = rng.binomial(1, p_t, n)
true_cate = 4 + 0.08*(60 - age) + 0.0008*spend
outcome   = 15 + 0.05*age + 0.003*spend + true_cate*treat + rng.normal(0, 4, n)

X = np.column_stack([age, spend])
gbm1 = GradientBoostingRegressor(100, random_state=42).fit(X[treat==1], outcome[treat==1])
gbm0 = GradientBoostingRegressor(100, random_state=42).fit(X[treat==0], outcome[treat==0])
cate_hat = gbm1.predict(X) - gbm0.predict(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# True vs estimated CATE
ax = axes[0]
ax.scatter(true_cate, cate_hat, alpha=0.15, s=8, color=C[0])
lo, hi = min(true_cate.min(), cate_hat.min()), max(true_cate.max(), cate_hat.max())
ax.plot([lo,hi],[lo,hi], "--", color=C[1], lw=2, label=f"r={np.corrcoef(true_cate,cate_hat)[0,1]:.2f}")
ax.set(xlabel="True CATE", ylabel="Estimated CATE",
       title="T-learner: true vs estimated CATE\n(Pearson r shown)")
ax.legend()

# CATE vs age
ax = axes[1]
order = np.argsort(age)
ax.scatter(age, true_cate, alpha=0.15, s=8, color=C[0], label="True CATE")
ax.scatter(age, cate_hat,  alpha=0.15, s=8, color=C[1], label="Estimated CATE")
ax.set(xlabel="Age", ylabel="CATE", title="CATE heterogeneity by age\n(younger users respond more)")
ax.legend(markerscale=2, fontsize=9)

# Cumulative gain: target high-CATE users
ax = axes[2]
order = np.argsort(-cate_hat)
cumulative_true = np.cumsum(true_cate[order]) / true_cate.sum()
cumulative_rand = np.linspace(0, 1, n)
ax.plot(np.linspace(0,100,n), cumulative_true*100, color=C[0], lw=2, label="CATE targeting")
ax.plot([0,100],[0,100], "--", color="grey", lw=1.5, label="Random targeting")
ax.fill_between(np.linspace(0,100,n), cumulative_true*100, np.linspace(0,100,n),
                alpha=0.15, color=C[0])
ax.set(xlabel="% customers targeted", ylabel="% cumulative uplift captured",
       title="Cumulative gain curve\n(CATE targeting outperforms random)")
ax.legend()
plt.tight_layout()
plt.show()
top20_gain = cumulative_true[int(0.2*n)] * 100
print(f"Targeting top 20% by CATE captures {top20_gain:.0f}% of total uplift")
print(f"vs {20:.0f}% if targeting randomly")
"""),
    ])


# =============================================================================
# 04 · NLP text classification
# =============================================================================
def nb_nlp():
    return nb([
        md("""# NLP — Text Classification

Progressive workflow: bag-of-words -> TF-IDF + linear models -> embeddings.
Always establish a strong linear baseline before reaching for transformers.
"""),
        md("## Setup"), code(SETUP),

        md("""## 1. TF-IDF features — what does the model actually learn?

TF-IDF reweights raw counts by inverse document frequency — common words
(``the``, ``is``) get down-weighted; discriminative words get amplified.
"""),
        code("""\
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

# Overlapping computer topics -> harder -> LR/SVC beat NB
CATS = ["comp.graphics", "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"]
train = fetch_20newsgroups(subset="train", categories=CATS,
                            remove=("headers","footers","quotes"))
test  = fetch_20newsgroups(subset="test",  categories=CATS,
                            remove=("headers","footers","quotes"))

models = {
    "Naive Bayes": Pipeline([("v", TfidfVectorizer(max_features=30_000)),
                              ("m", MultinomialNB())]),
    "Logistic Reg": Pipeline([("v", TfidfVectorizer(max_features=30_000, ngram_range=(1,2))),
                               ("m", LogisticRegression(C=5, max_iter=1000, random_state=42))]),
    "LinearSVC":   Pipeline([("v", TfidfVectorizer(max_features=30_000, ngram_range=(1,2))),
                              ("m", LinearSVC(C=1.0, random_state=42))]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(train.data, train.target)
    preds = pipe.predict(test.data)
    from sklearn.metrics import f1_score, accuracy_score
    results[name] = {
        "f1":  f1_score(test.target, preds, average="macro"),
        "acc": accuracy_score(test.target, preds),
        "pipe": pipe, "preds": preds,
    }

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
names = list(results.keys())
f1s   = [results[n]["f1"]  for n in names]
accs  = [results[n]["acc"] for n in names]
x = np.arange(len(names))
w = 0.35
b1 = ax.bar(x - w/2, [f*100 for f in f1s],  w, color=C[0], label="Macro F1")
b2 = ax.bar(x + w/2, [a*100 for a in accs], w, color=C[2], label="Accuracy")
ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
ax.set(xticks=x, xticklabels=names, ylabel="Score (%)", ylim=(50,100),
       title="Model comparison on overlapping comp.* categories\n(LinearSVC beats NB on ambiguous text)")
ax.legend()

# Confusion matrix for best model
best_name = max(results, key=lambda n: results[n]["f1"])
cm = confusion_matrix(test.target, results[best_name]["preds"])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
im = axes[1].imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
axes[1].set(xticks=range(4), yticks=range(4),
            xticklabels=[c.split(".")[-1] for c in CATS],
            yticklabels=[c.split(".")[-1] for c in CATS],
            xlabel="Predicted", ylabel="True",
            title=f"Confusion matrix - {best_name}\n(% of true class, diagonal = correct)")
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f"{cm_pct[i,j]:.0f}%", ha="center", va="center",
                     color="white" if cm_pct[i,j]>50 else "black", fontsize=9)
plt.colorbar(im, ax=axes[1], label="% of true class")
plt.tight_layout()
plt.show()
"""),

        md("""## 2. Top discriminative features per class

TF-IDF + logistic regression is interpretable — the coefficients directly
show which words the model associates with each class.
"""),
        code("""\
lr_pipe = models["LinearSVC"]
# Use LogisticRegression for coefficient access
lr_coef_pipe = Pipeline([("v", TfidfVectorizer(max_features=30_000, ngram_range=(1,2))),
                          ("m", LogisticRegression(C=5, max_iter=1000, random_state=42))])
lr_coef_pipe.fit(train.data, train.target)

vocab = lr_coef_pipe.named_steps["v"].get_feature_names_out()
coef  = lr_coef_pipe.named_steps["m"].coef_
short_cats = [c.split(".")[-1] for c in CATS]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
n_top = 12
for ax, cls_idx, cat in zip(axes, range(4), short_cats):
    top_idx = np.argsort(coef[cls_idx])[-n_top:][::-1]
    top_w   = coef[cls_idx][top_idx]
    colors  = [C[0] if w > 0 else C[1] for w in top_w]
    ax.barh(range(n_top), top_w[::-1], color=colors[::-1])
    ax.set(yticks=range(n_top), yticklabels=vocab[top_idx[::-1]],
           title=f"{cat}", xlabel="Coefficient")
    ax.invert_yaxis()
fig.suptitle("Top discriminative words per class (Logistic Regression)", fontsize=13)
plt.tight_layout()
plt.show()
"""),

        md("""## 3. Semantic similarity with TF-IDF cosine

A lightweight semantic search without any embeddings model — useful
as a baseline before adding dense retrieval.
"""),
        code("""\
from sklearn.metrics.pairwise import cosine_similarity

# Small illustrative corpus
corpus = [
    "machine learning fraud detection credit card anomaly",
    "neural network deep learning classification banking",
    "natural language processing text classification sentiment",
    "transformer BERT fine-tuning NLP tasks",
    "survival analysis customer churn time-to-event censoring",
    "A/B testing causal inference experiment randomisation",
    "recommender system collaborative filtering matrix factorisation",
    "gradient boosting decision tree ensemble feature importance",
]
queries = [
    "detecting fraud with ML in finance",
    "NLP models for text tasks",
    "customer retention analysis",
]

vec      = TfidfVectorizer(ngram_range=(1,2)).fit(corpus)
doc_vecs = vec.transform(corpus)
q_vecs   = vec.transform(queries)
sims     = cosine_similarity(q_vecs, doc_vecs)

fig, axes = plt.subplots(1, len(queries), figsize=(15, 4))
short_docs = [d[:35]+"..." for d in corpus]
for ax, query, sim_row in zip(axes, queries, sims):
    order = np.argsort(sim_row)[::-1]
    bars  = ax.barh(range(len(corpus)), sim_row[order], color=C[0], alpha=0.8)
    ax.set(yticks=range(len(corpus)),
           yticklabels=[short_docs[i] for i in order],
           xlabel="Cosine similarity",
           title=f'Query: "{query[:30]}..."')
    ax.invert_yaxis()
    # Highlight top-1
    ax.get_children()[0].set_color(C[1])
fig.suptitle("TF-IDF cosine similarity: corpus retrieval results", fontsize=12)
plt.tight_layout()
plt.show()
"""),
    ])


# =============================================================================
# 05 · Feature engineering
# =============================================================================
def nb_features():
    return nb([
        md("""# Feature Engineering & Data Preparation

The highest-leverage work in any ML project.
Covers imputation, encoding, transforms, leakage prevention, and diagnostics.
"""),
        md("## Setup"), code(SETUP),

        md("""## 1. Leakage — the most common and most damaging failure

Fitting preprocessing on the full dataset before train/test splitting
leaks test statistics into training. The accuracy looks better but the
model has peeked at the test set.
"""),
        code("""\
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

X, y = load_breast_cancer(return_X_y=True)
cv   = StratifiedKFold(5, shuffle=True, random_state=42)

# 1. WRONG: scaler fit on full dataset
from sklearn.preprocessing import StandardScaler as SS
scores_wrong = []
for train_idx, val_idx in cv.split(X, y):
    X_sc = SS().fit_transform(X)   # fit on ALL data including val fold
    m    = LogisticRegression(max_iter=1000).fit(X_sc[train_idx], y[train_idx])
    scores_wrong.append(m.score(X_sc[val_idx], y[val_idx]))

# 2. CORRECT: scaler inside Pipeline
pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])
scores_correct = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

# 3. Shuffle-label leakage guard
scores_shuffle = cross_val_score(
    pipe, X, np.random.default_rng(42).permutation(y),
    cv=cv, scoring="roc_auc"
)

fig, ax = plt.subplots(figsize=(10, 5))
configs = [
    ("WRONG\n(scaler on full data)",  scores_wrong,   C[1]),
    ("CORRECT\n(scaler in Pipeline)", scores_correct, C[2]),
    ("Shuffled labels\n(leakage guard)", scores_shuffle, C[0]),
]
positions = np.arange(len(configs))
for pos, (label, scores, col) in zip(positions, configs):
    ax.scatter([pos]*len(scores), scores, color=col, alpha=0.7, s=60, zorder=3)
    ax.plot([pos-0.25, pos+0.25], [np.mean(scores)]*2, color=col, lw=3)
ax.set(xticks=positions, xticklabels=[c[0] for c in configs],
       ylabel="ROC-AUC (5-fold CV)",
       title="Leakage inflates CV score — and shuffled labels should collapse to ~0.5")
ax.axhline(0.5, color="grey", lw=1.2, linestyle="--", label="Chance (no signal)")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Leaked scaler:  {np.mean(scores_wrong):.4f} +/- {np.std(scores_wrong):.4f}")
print(f"Correct:        {np.mean(scores_correct):.4f} +/- {np.std(scores_correct):.4f}")
print(f"Shuffle guard:  {np.mean(scores_shuffle):.4f} (should be ~0.50)")
"""),

        md("""## 2. Missing data patterns and imputation strategies

Always visualise missingness before imputing. Random vs. structured missingness
demands different approaches.
"""),
        code("""\
rng = np.random.default_rng(42)
n   = 500
df  = pd.DataFrame({
    "age":    rng.normal(45, 12, n),
    "income": rng.lognormal(10, 0.5, n),
    "score":  rng.normal(650, 80, n),
    "tenure": rng.exponential(3, n),
    "target": rng.binomial(1, 0.15, n),
})
# Introduce structured missingness: income missing more for young users
miss_prob = 0.05 + 0.3*(df["age"] < 30)
df.loc[rng.random(n) < miss_prob,      "income"] = np.nan
df.loc[rng.random(n) < 0.12,           "score"]  = np.nan
df.loc[rng.random(n) < 0.08,           "tenure"] = np.nan

from sklearn.impute import SimpleImputer

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Missing rate bar chart
ax = axes[0]
miss_rates = (df.isna().mean() * 100).sort_values(ascending=False)
miss_rates = miss_rates[miss_rates > 0]
bars = ax.bar(miss_rates.index, miss_rates.values, color=C[1], alpha=0.85)
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set(ylabel="Missing rate (%)", title="Missingness by feature")

# Missingness heatmap
ax = axes[1]
miss_matrix = df.isna().astype(int)
im = ax.imshow(miss_matrix.T, aspect="auto", cmap="Reds", interpolation="none")
ax.set(yticks=range(len(df.columns)), yticklabels=df.columns,
       xlabel="Row index", title="Missingness pattern\n(red = missing)")
plt.colorbar(im, ax=ax)

# Imputation comparison: income distribution
ax = axes[2]
raw   = df["income"].dropna()
mean_imp  = df["income"].fillna(df["income"].mean())
median_imp = df["income"].fillna(df["income"].median())
ax.hist(raw.values,       bins=40, alpha=0.55, density=True, color=C[0], label=f"Observed (n={len(raw)})")
ax.hist(mean_imp.values,  bins=40, alpha=0.45, density=True, color=C[1], label="Mean imputed")
ax.hist(median_imp.values,bins=40, alpha=0.45, density=True, color=C[2], label="Median imputed")
ax.set(xlabel="Income", title="Imputation strategies for income\n(mean imputation biased for log-normal)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
print("Rule: mean imputation is biased for skewed distributions -> use median or model-based")
"""),

        md("""## 3. Target encoding inside cross-validation

Target encoding must be fit only on training folds to avoid leakage.
Leave-one-out or k-fold target encoding are standard safe implementations.
"""),
        code("""\
rng = np.random.default_rng(42)
n   = 3000
cats = [f"cat_{i}" for i in range(40)]
df  = pd.DataFrame({
    "category": rng.choice(cats, n),
    "feature1": rng.normal(0, 1, n),
})
# Planted true effect: cat_0 has 3x higher conversion
df["y"] = (rng.uniform(0,1,n) < 0.05 + 0.10*(df["category"]=="cat_0")).astype(int)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
df["te_safe"]   = np.nan
df["te_leaked"] = df.groupby("category")["y"].transform("mean")   # WRONG: uses all data

for tr_idx, val_idx in kf.split(df):
    train_fold = df.iloc[tr_idx]
    mean_map   = train_fold.groupby("category")["y"].mean()
    global_mean = train_fold["y"].mean()
    df.loc[df.index[val_idx], "te_safe"] = (
        df.iloc[val_idx]["category"].map(mean_map).fillna(global_mean)
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, col, title in [
    (axes[0], "te_leaked", "Leaked target encoding\n(fit on all data - WRONG)"),
    (axes[1], "te_safe",   "Safe target encoding\n(fit per CV fold - CORRECT)"),
]:
    top = df.groupby("category")[col].mean().sort_values(ascending=False).head(10)
    color_list = [C[1] if idx=="cat_0" else C[0] for idx in top.index]
    bars = ax.bar(range(len(top)), top.values, color=color_list)
    ax.set(xticks=range(len(top)), xticklabels=top.index, ylabel="Target encoding value",
           title=title)
    ax.tick_params(axis='x', rotation=45)

fig.suptitle("cat_0 (red bar) should have highest encoding - does leakage exaggerate it?",
             fontsize=11)
plt.tight_layout()
plt.show()
leaked_cat0 = df.groupby("category")["te_leaked"].mean()["cat_0"]
safe_cat0   = df.groupby("category")["te_safe"].mean()["cat_0"]
print(f"cat_0 leaked encoding: {leaked_cat0:.4f}")
print(f"cat_0 safe encoding:   {safe_cat0:.4f}")
print(f"True conversion rate:  {df[df['category']=='cat_0']['y'].mean():.4f}")
"""),
    ])


# =============================================================================
# 06 · Time-series forecasting
# =============================================================================
def nb_timeseries():
    return nb([
        md("""# Time-Series Forecasting

Rolling-origin backtesting, decomposition, and forecast evaluation.
Key rule: **never use a standard train/test split for time series** — it leaks
future information into training.
"""),
        md("## Setup"), code(SETUP),

        md("""## 1. Decomposition — trend, seasonality, noise

Before modelling, decompose the series to understand its components.
Additive decomposition: `y = trend + seasonality + residual`.
"""),
        code("""\
from statsmodels.tsa.seasonal import seasonal_decompose

rng = np.random.default_rng(42)
t   = np.arange(144)

# Synthetic series: trend + monthly seasonality + noise
trend      = 100 + 0.8*t
seasonality = 20 * np.sin(2*np.pi*t/12)
noise       = rng.normal(0, 5, 144)
y           = trend + seasonality + noise
dates       = pd.date_range("2012-01", periods=144, freq="MS")
series      = pd.Series(y, index=dates)

result = seasonal_decompose(series, model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax, data, title, col in zip(axes,
    [series, result.trend, result.seasonal, result.resid],
    ["Original", "Trend", "Seasonal (period=12)", "Residual"],
    [C[0], C[2], C[3], C[1]]
):
    ax.plot(data.index, data.values, color=col, lw=1.8)
    ax.fill_between(data.index, data.values, alpha=0.15, color=col)
    ax.set_ylabel(title, fontsize=10)
    ax.grid(True, alpha=0.3)
axes[0].set_title("Additive decomposition: trend + seasonality + residual", fontsize=12)
plt.tight_layout()
plt.show()
print(f"Trend range: {result.trend.dropna().min():.0f} to {result.trend.dropna().max():.0f}")
print(f"Seasonal amplitude: +/-{result.seasonal.max():.1f}")
print(f"Residual std: {result.resid.dropna().std():.2f}  (noise level)")
"""),

        md("""## 2. Rolling-origin backtest

Split data into an expanding training window, forecast horizon steps ahead,
collect errors. This mirrors real deployment: you always forecast into the future.
"""),
        code("""\
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def seasonal_naive(train, h, period=12):
    last_season = train.values[-period:]
    reps = (h // period) + 1
    return pd.Series(
        np.tile(last_season, reps)[:h],
        index=pd.date_range(train.index[-1], periods=h+1, freq="MS")[1:]
    )

def mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

# Rolling-origin evaluation
origin_step = 6
h           = 6
origins     = range(72, len(series) - h, origin_step)
results     = {"Seasonal Naive": [], "ETS": [], "ARIMA(1,1,1)x12": []}

for origin in origins:
    train  = series.iloc[:origin]
    actual = series.iloc[origin:origin+h]

    # Naive
    fc_naive = seasonal_naive(train, h)
    results["Seasonal Naive"].append(mape(actual.values, fc_naive.values))

    # ETS
    try:
        fc_ets = ExponentialSmoothing(train, trend="add", seasonal="add",
                                       seasonal_periods=12).fit(optimized=True).forecast(h)
        results["ETS"].append(mape(actual.values, fc_ets.values))
    except Exception:
        results["ETS"].append(np.nan)

    # ARIMA
    try:
        fc_arima = ARIMA(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit().forecast(h)
        results["ARIMA(1,1,1)x12"].append(mape(actual.values, fc_arima.values))
    except Exception:
        results["ARIMA(1,1,1)x12"].append(np.nan)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Boxplot of MAPE distributions
ax = axes[0]
data_box = [np.array(v)[~np.isnan(v)] for v in results.values()]
bp = ax.boxplot(data_box, labels=list(results.keys()), patch_artist=True,
                medianprops=dict(color="white", lw=2))
for patch, col in zip(bp["boxes"], C):
    patch.set_facecolor(col)
    patch.set_alpha(0.75)
ax.set(ylabel="MAPE (%)", title="Rolling-origin backtest\nMAPE distribution by model")

# Mean MAPE per origin
ax = axes[1]
origin_idx = list(range(len(list(results.values())[0])))
for (name, vals), col in zip(results.items(), C):
    clean = np.array(vals, dtype=float)
    ax.plot(origin_idx, clean, "o-", color=col, alpha=0.8, ms=4, label=name)
ax.set(xlabel="Backtest origin index", ylabel="MAPE (%)",
       title="MAPE across rolling origins\n(lower = better, stability matters too)")
ax.legend()

plt.tight_layout()
plt.show()
for name, vals in results.items():
    clean = [v for v in vals if not np.isnan(v)]
    print(f"{name:<22}  mean MAPE = {np.mean(clean):.1f}%  (n={len(clean)} origins)")
"""),

        md("""## 3. Forecast uncertainty — prediction intervals

A point forecast without intervals is incomplete. Prediction intervals
quantify how much the forecast could plausibly be wrong.
"""),
        code("""\
from statsmodels.tsa.holtwinters import ExponentialSmoothing

train = series.iloc[:120]
test  = series.iloc[120:]
h     = len(test)

fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                            seasonal_periods=12).fit(optimized=True)
fc  = fit.forecast(h)

# Bootstrap prediction intervals
n_boot = 500
boot_forecasts = np.zeros((n_boot, h))
resid = fit.resid.values
for b in range(n_boot):
    boot_resid = np.random.choice(resid, size=len(train), replace=True)
    boot_series = pd.Series(train.values + boot_resid, index=train.index)
    try:
        boot_fit = ExponentialSmoothing(boot_series, trend="add", seasonal="add",
                                         seasonal_periods=12).fit(optimized=True)
        boot_forecasts[b] = boot_fit.forecast(h).values
    except Exception:
        boot_forecasts[b] = fc.values

pi_80_lo = np.percentile(boot_forecasts, 10, axis=0)
pi_80_hi = np.percentile(boot_forecasts, 90, axis=0)
pi_95_lo = np.percentile(boot_forecasts, 2.5, axis=0)
pi_95_hi = np.percentile(boot_forecasts, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(train.index[-36:], train.values[-36:], color=C[0], lw=2, label="Training (last 3 yrs)")
ax.plot(test.index, test.values, color="black", lw=2, label="Actual")
ax.plot(fc.index,   fc.values,   color=C[1], lw=2, linestyle="--", label="ETS forecast")
ax.fill_between(fc.index, pi_80_lo, pi_80_hi, color=C[1], alpha=0.25, label="80% PI")
ax.fill_between(fc.index, pi_95_lo, pi_95_hi, color=C[1], alpha=0.12, label="95% PI")
ax.axvline(test.index[0], color="grey", lw=1.5, linestyle=":", label="Forecast origin")
ax.set(xlabel="Date", ylabel="Passengers",
       title="ETS forecast with bootstrap prediction intervals\n(wider = more uncertain)")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
plt.tight_layout()
plt.show()

coverage_80 = np.mean((test.values >= pi_80_lo) & (test.values <= pi_80_hi))
coverage_95 = np.mean((test.values >= pi_95_lo) & (test.values <= pi_95_hi))
print(f"80% PI coverage: {coverage_80:.0%}  (should be ~80%)")
print(f"95% PI coverage: {coverage_95:.0%}  (should be ~95%)")
print(f"Forecast MAPE:   {mape(test.values, fc.values):.1f}%")
"""),
    ])


# =============================================================================
# 07 · Clustering & dimensionality reduction
# =============================================================================
def nb_clustering():
    return nb([
        md("""# Clustering & Dimensionality Reduction

Unsupervised learning: finding structure without labels.
Key challenge: evaluating quality without ground truth.
"""),
        md("## Setup"), code(SETUP),

        md("""## 1. KMeans — choosing k with the silhouette score

The elbow method is subjective. The silhouette score is objective:
measures how similar each point is to its own cluster vs. the nearest other cluster.
Range: -1 (wrong cluster) to +1 (tight, well-separated cluster).
"""),
        code("""\
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

rng = np.random.default_rng(42)
X_raw, y_true = make_blobs(n_samples=600, centers=4, cluster_std=1.1,
                            random_state=42)
X = StandardScaler().fit_transform(X_raw)

# Sweep k
k_range = range(2, 10)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(k, random_state=42, n_init=10).fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))

best_k = k_range[np.argmax(sil_scores)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Elbow plot
ax = axes[0]
ax.plot(list(k_range), inertias, "o-", color=C[0], lw=2)
ax.set(xlabel="k", ylabel="Inertia (within-cluster SS)",
       title="Elbow plot\n(elbow is subjective - hard to read)")

# Silhouette plot
ax = axes[1]
ax.plot(list(k_range), sil_scores, "o-", color=C[2], lw=2)
ax.axvline(best_k, color=C[1], lw=2, linestyle="--", label=f"Best k={best_k}")
ax.set(xlabel="k", ylabel="Silhouette score",
       title="Silhouette score vs k\n(peak = optimal k, objective)")
ax.legend()

# Cluster scatter at best k
ax = axes[2]
km_best = KMeans(best_k, random_state=42, n_init=10).fit(X)
for c in range(best_k):
    mask = km_best.labels_ == c
    ax.scatter(X[mask,0], X[mask,1], s=20, alpha=0.7, color=C[c], label=f"Cluster {c}")
centers = km_best.cluster_centers_
ax.scatter(centers[:,0], centers[:,1], marker="X", s=180, color="black", zorder=5, label="Centroids")
ax.set(xlabel="PC1", ylabel="PC2", title=f"KMeans k={best_k}\n(silhouette={sil_scores[best_k-2]:.3f})")
ax.legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
plt.show()
print(f"Best k by silhouette: {best_k}  (true k=4)")
print(f"Best silhouette:      {max(sil_scores):.3f}")
"""),

        md("""## 2. PCA — variance explained and biplots

PCA finds the directions of maximum variance. The scree plot shows
how many components capture "most" of the variance.
"""),
        code("""\
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

X_wine, y_wine = load_wine(return_X_y=True)
feature_names   = load_wine().feature_names
X_sc            = StandardScaler().fit_transform(X_wine)
pca             = PCA().fit(X_sc)
X_pca           = pca.transform(X_sc)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scree plot
ax = axes[0]
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
n_comp = np.argmax(cumvar >= 80) + 1
ax.bar(range(1, len(cumvar)+1), pca.explained_variance_ratio_*100,
       color=C[0], alpha=0.8, label="Individual")
ax.plot(range(1, len(cumvar)+1), cumvar, "o-", color=C[1], lw=2, label="Cumulative")
ax.axhline(80, color="grey", lw=1.5, linestyle="--", label="80% threshold")
ax.axvline(n_comp, color=C[2], lw=2, linestyle="--")
ax.set(xlabel="Principal component", ylabel="Variance explained (%)",
       title=f"Scree plot\n({n_comp} PCs explain 80% of variance)")
ax.legend(fontsize=9)

# Scatter PC1 vs PC2
ax = axes[1]
wine_classes = load_wine().target_names
for cls, col in zip([0,1,2], C):
    mask = y_wine == cls
    ax.scatter(X_pca[mask,0], X_pca[mask,1], s=30, alpha=0.8, color=col,
               label=wine_classes[cls])
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
       ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
       title="PC1 vs PC2\n(wine varieties well-separated)")
ax.legend()

# Biplot (loadings)
ax = axes[2]
loadings = pca.components_[:2].T
scale    = 3
ax.scatter(X_pca[:,0], X_pca[:,1], s=10, alpha=0.2, color="grey")
for i, (lx, ly) in enumerate(loadings):
    ax.annotate("", xy=(lx*scale, ly*scale), xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color=C[1], lw=1.5))
    ax.text(lx*scale*1.1, ly*scale*1.1, feature_names[i],
            fontsize=7, ha="center", color=C[1])
ax.set(xlabel="PC1", ylabel="PC2",
       title="Biplot: feature loadings\n(arrow direction = feature contribution)")
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
plt.tight_layout()
plt.show()
print(f"Top 2 features driving PC1: {[feature_names[i] for i in np.argsort(np.abs(pca.components_[0]))[-2:][::-1]]}")
"""),

        md("""## 3. DBSCAN — density-based clustering (handles noise)

KMeans requires specifying k and assumes spherical clusters.
DBSCAN discovers arbitrary-shaped clusters and labels outliers as noise.
"""),
        code("""\
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import silhouette_score

datasets = [
    ("Two moons",   *make_moons(500,  noise=0.08, random_state=42)),
    ("Two circles", *make_circles(500, noise=0.05, factor=0.5, random_state=42)),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, (name, X_d, y_d) in enumerate(datasets):
    # True labels
    for cls in np.unique(y_d):
        axes[row,0].scatter(X_d[y_d==cls,0], X_d[y_d==cls,1], s=12, alpha=0.7, color=C[cls])
    axes[row,0].set(title=f"{name}\nGround truth", xticks=[], yticks=[])

    # KMeans
    km_labels = KMeans(2, random_state=42, n_init=10).fit_predict(X_d)
    for cls in np.unique(km_labels):
        axes[row,1].scatter(X_d[km_labels==cls,0], X_d[km_labels==cls,1],
                            s=12, alpha=0.7, color=C[cls])
    sil_km = silhouette_score(X_d, km_labels)
    axes[row,1].set(title=f"KMeans k=2\nsil={sil_km:.3f} (fails on non-convex)", xticks=[], yticks=[])

    # DBSCAN
    db = DBSCAN(eps=0.15, min_samples=5).fit(X_d)
    db_labels = db.labels_
    noise_mask = db_labels == -1
    for cls in np.unique(db_labels[~noise_mask]):
        axes[row,2].scatter(X_d[db_labels==cls,0], X_d[db_labels==cls,1],
                            s=12, alpha=0.7, color=C[cls])
    axes[row,2].scatter(X_d[noise_mask,0], X_d[noise_mask,1],
                        s=12, alpha=0.5, color="grey", marker="x", label="Noise")
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise    = noise_mask.sum()
    axes[row,2].set(title=f"DBSCAN\n{n_clusters} clusters, {n_noise} noise pts", xticks=[], yticks=[])

cols = ["Ground truth", "KMeans k=2", "DBSCAN"]
for ax, col in zip(axes[0], cols):
    ax.set_title(f"{col}\n{ax.get_title().split(chr(10))[-1]}", fontsize=10)

fig.suptitle("KMeans fails on non-convex clusters; DBSCAN handles them", fontsize=12)
plt.tight_layout()
plt.show()
"""),
    ])


# =============================================================================
# 08 · Survival analysis
# =============================================================================
def nb_survival():
    return nb([
        md("""# Survival Analysis

Time-to-event modelling with censoring. Used for churn, medical outcomes,
equipment failure. Key insight: **censored observations are not missing data**
— they carry information ("survived at least this long").
"""),
        md("## Setup"), code(SETUP + "\nfrom lifelines import KaplanMeierFitter, CoxPHFitter\nfrom lifelines.statistics import logrank_test"),

        md("""## 1. Kaplan-Meier curves

Non-parametric survival function estimator. Handles censoring correctly.
Compare curves between groups with the log-rank test.
"""),
        code("""\
rng = np.random.default_rng(42)
n   = 600

# Simulate customer churn data
monthly_plan = rng.binomial(1, 0.55, n).astype(bool)
base_rate    = np.where(monthly_plan, 0.08, 0.03)   # monthly churn hazard
tenure       = np.zeros(n)
event        = np.zeros(n, dtype=bool)

for i in range(n):
    for month in range(1, 37):
        if rng.random() < base_rate[i]:
            tenure[i] = month
            event[i]  = True
            break
    if not event[i]:
        tenure[i] = 36   # censored at study end

df = pd.DataFrame({
    "tenure": tenure, "event": event,
    "plan":   np.where(monthly_plan, "Monthly", "Annual"),
    "charges": rng.lognormal(5, 0.4, n),
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# KM curves by plan type
ax = axes[0]
colors_plan = {"Monthly": C[1], "Annual": C[2]}
kmf = KaplanMeierFitter()
for plan, col in colors_plan.items():
    mask = df["plan"] == plan
    kmf.fit(df.loc[mask, "tenure"], df.loc[mask, "event"], label=plan)
    kmf.plot_survival_function(ax=ax, color=col, ci_show=True, ci_alpha=0.15)

# Log-rank test
m_mask = df["plan"] == "Monthly"
lr = logrank_test(df.loc[m_mask, "tenure"],  df.loc[~m_mask, "tenure"],
                  df.loc[m_mask, "event"],   df.loc[~m_mask, "event"])
ax.set(xlabel="Months", ylabel="Survival probability",
       title=f"Kaplan-Meier curves by plan type\nLog-rank p={lr.p_value:.4f}")
ax.legend(title="Plan type")

# At-risk table proxy (survival at key months)
ax = axes[1]
months = [6, 12, 18, 24, 30, 36]
for plan, col in colors_plan.items():
    mask = df["plan"] == plan
    kmf.fit(df.loc[mask, "tenure"], df.loc[mask, "event"])
    surv_at_months = [kmf.predict(m) for m in months]
    ax.plot(months, surv_at_months, "o-", color=col, lw=2, label=plan)
    for m, s in zip(months, surv_at_months):
        ax.annotate(f"{s:.0%}", xy=(m, s), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=col)

ax.set(xlabel="Month", ylabel="Survival probability",
       title="Survival probability at key months\n(Annual plan retains much better)")
ax.legend(title="Plan type")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.show()
print(f"Log-rank test p-value: {lr.p_value:.6f}")
print(f"Monthly 12-month survival:  {kmf.predict(12):.1%}")
"""),

        md("""## 2. Cox Proportional Hazards — covariate effects

Semi-parametric model: estimates **hazard ratios** (HR) for each covariate.
HR > 1 = increases hazard (faster churn); HR < 1 = protective.
"""),
        code("""\
from lifelines import CoxPHFitter

df_cox = df.copy()
df_cox["monthly_plan"] = (df_cox["plan"] == "Monthly").astype(int)
df_cox = df_cox.drop(columns=["plan"])

cph = CoxPHFitter()
cph.fit(df_cox, duration_col="tenure", event_col="event")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Coefficient forest plot
ax = axes[0]
summary = cph.summary
coefs   = summary["coef"]
se      = summary["se(coef)"]
hr      = np.exp(coefs)
hr_lo   = np.exp(coefs - 1.96*se)
hr_hi   = np.exp(coefs + 1.96*se)

y_pos = range(len(coefs))
ax.errorbar(hr, y_pos, xerr=[hr-hr_lo, hr_hi-hr], fmt="o",
            color=C[0], ms=8, capsize=5, lw=2)
ax.axvline(1.0, color="grey", lw=1.5, linestyle="--", label="HR=1 (no effect)")
ax.set(yticks=list(y_pos), yticklabels=coefs.index,
       xlabel="Hazard Ratio (95% CI)",
       title="Cox PH: Hazard Ratios\n(HR>1 = faster churn, HR<1 = slower)")
ax.legend()
for i, (h, lo, hi, name) in enumerate(zip(hr, hr_lo, hr_hi, coefs.index)):
    color = C[1] if h > 1 else C[2]
    ax.scatter(h, i, color=color, s=80, zorder=5)

# Predicted survival for representative customers
ax = axes[1]
profiles = pd.DataFrame({
    "monthly_plan": [1, 0, 1, 0],
    "charges":      [df["charges"].quantile(0.25)] * 2 + [df["charges"].quantile(0.75)] * 2,
})
labels = ["Monthly, low charges", "Annual, low charges",
          "Monthly, high charges", "Annual, high charges"]
colors_prof = [C[1], C[2], C[3], C[0]]
for profile, label, col in zip(profiles.itertuples(index=False), labels, colors_prof):
    row = pd.DataFrame([profile._asdict()])
    sf  = cph.predict_survival_function(row, times=range(1, 37))
    ax.plot(sf.index, sf.values.flatten(), color=col, lw=2, label=label)

ax.set(xlabel="Months", ylabel="Survival probability",
       title="Predicted survival by customer profile\n(Cox PH)")
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.show()
print("\nHazard ratios (exponentiated coefficients):")
print(np.exp(cph.params_).round(3))
"""),
    ])


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    save("01-statistics-fundamentals.ipynb", nb_statistics())
    save("02-model-evaluation.ipynb",        nb_evaluation())
    save("03-ab-testing.ipynb",              nb_ab())
    save("04-nlp-text-classification.ipynb", nb_nlp())
    save("05-feature-engineering.ipynb",     nb_features())
    save("06-time-series-forecasting.ipynb", nb_timeseries())
    save("07-clustering.ipynb",              nb_clustering())
    save("08-survival-analysis.ipynb",       nb_survival())
    print("Done.")
