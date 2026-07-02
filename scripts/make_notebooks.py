"""
Generate Jupyter notebooks from project scripts and doc examples.

For each project: reads the .py file, splits on section-comment boundaries
(lines starting with '# ──'), wraps each section in a code cell, and
inserts markdown cells with the lesson narrative.

Run: python scripts/make_notebooks.py
"""
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ── nbformat helpers ──────────────────────────────────────────────────────────

def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(source).strip()}


SETUP_CELL = """\
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

# High-quality plot defaults
mpl.rcParams.update({
    "figure.dpi":        120,
    "figure.figsize":    (10, 5),
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
    "lines.linewidth":   2.0,
    "patch.edgecolor":   "none",
})
PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]
print("Setup complete - plots will render inline with high-quality defaults")
"""


def notebook_prep(source: str) -> str:
    """Strip headless backend; replace savefig with show for inline display."""
    lines = []
    for line in source.splitlines():
        # Remove Agg backend (headless - suppresses inline display)
        if re.search(r'matplotlib\.use\(["\']Agg["\']\)', line):
            continue
        # Replace savefig(...) with plt.show() so plots appear inline
        if re.search(r'plt\.savefig\(', line):
            indent = len(line) - len(line.lstrip())
            lines.append(" " * indent + "plt.tight_layout()")
            lines.append(" " * indent + "plt.show()")
            continue
        lines.append(line)
    return "\n".join(lines)


def code(source: str, prep: bool = False) -> dict:
    src = textwrap.dedent(source).strip()
    if prep:
        src = notebook_prep(src)
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def save(path: Path, notebook: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def split_sections(script_text: str) -> list[tuple[str, str]]:
    """
    Split a script into (header, code_body) pairs.
    Splits on lines starting with '# ──' or '# =='.
    """
    section_re = re.compile(r'^# [─═]{2,}.*$', re.MULTILINE)
    positions = [m.start() for m in section_re.finditer(script_text)]
    positions.append(len(script_text))

    # Preamble before first section marker
    preamble = script_text[:positions[0]] if positions else script_text

    sections = []
    for i, pos in enumerate(positions[:-1]):
        block = script_text[pos:positions[i + 1]]
        lines = block.splitlines()
        header = lines[0].lstrip('# ─═').strip() if lines else ""
        body = "\n".join(lines[1:]).strip()
        if body:
            sections.append((header, body))

    return preamble.strip(), sections


# ── Per-project notebook metadata ────────────────────────────────────────────

PROJECTS = {
    "p1_supervised_learning": {
        "title": "P1 · Supervised Classification — Breast Cancer",
        "lesson": """
## Real lesson: no-leakage pipelines; simple models often win

This notebook walks through a complete supervised learning workflow:
fit preprocessing **inside** a `Pipeline` so transforms are learned only on
training folds, compare a logistic regression baseline against Random Forest and
gradient boosting, evaluate on a single sealed test set, and explain predictions
with permutation importance.

**The punchline:** plain logistic regression achieves ROC-AUC ≈ 0.99 on this
dataset — matching the complex ensemble. Always start simple.
""",
        "script": "train.py",
    },
    "p2_time_series_forecasting": {
        "title": "P2 · Time-Series Forecasting — Airline Passengers",
        "lesson": """
## Real lesson: rolling-origin backtest; always beat a naïve baseline

Standard train/test splits are wrong for time series (they leak future into past).
This notebook uses a **rolling-origin backtest** — iteratively training up to time T
and evaluating at T+h — which matches how the model will actually be deployed.

**The punchline:** a model that doesn't beat a seasonal naïve baseline is
actively harmful to deploy. Always compare against the simplest forecast.
""",
        "script": "forecast.py",
    },
    "p3_causal_inference": {
        "title": "P3 · Causal Inference — Doubly-Robust ATE Estimation",
        "lesson": """
## Real lesson: naïve comparison can flip the sign; use doubly-robust estimators

When treatment is not randomly assigned, simple mean-difference estimates are
confounded. This notebook plants a known ATE (= 3.0), then shows how naïve
regression recovers the wrong sign, while a doubly-robust AIPW estimator
recovers the truth.

**The punchline:** the estimator that's wrong in the most plausible way is the
naïve one. Propensity scores + outcome models together are robust to one
mis-specification.
""",
        "script": "causal.py",
    },
    "p4_unsupervised_learning": {
        "title": "P4 · Unsupervised Learning — Wine Clustering with KMeans + PCA",
        "lesson": """
## Real lesson: scale before distance; pick k with the silhouette score

KMeans uses Euclidean distance, so unscaled features dominate. This notebook
shows the StandardScaler → KMeans pipeline, uses the silhouette score to choose k
(not the elbow, which is subjective), and validates against known wine labels
using ARI.

**The punchline:** on wine data the true k is 3 and the silhouette peak is at 3.
Unsupervised methods can recover ground truth when the signal is strong.
""",
        "script": "cluster.py",
    },
    "p5_survival_analysis": {
        "title": "P5 · Survival Analysis — Cox PH on Customer Churn",
        "lesson": """
## Real lesson: censoring is signal, not missing data

Customers who haven't churned yet are **censored observations** — they tell us
the customer survived *at least* this long. Dropping them would bias every estimate.
This notebook uses Kaplan-Meier curves and a Cox Proportional Hazards model to
recover planted hazard ratios from semi-synthetic churn data.

**The punchline:** the Cox model recovers the planted HR (month-to-month ≈ 2.5×
annual) within the 95% CI. Censoring handled correctly → unbiased estimates.
""",
        "script": "survival.py",
    },
    "p6_fairness_audit": {
        "title": "P6 · Fairness Audit — Credit Scoring",
        "lesson": """
## Real lesson: dropping the protected attribute ≠ fair

A model trained without a protected group column can still produce disparate
outcomes if correlated proxies remain in the feature set. This notebook trains
a GBM credit scorer, audits it for disparate impact (80% rule) and equalised
odds, then applies a threshold-adjustment post-processing fix.

**The punchline:** fairness-unaware models routinely fail the 80% rule.
Post-processing thresholds can repair equalised odds without retraining.
""",
        "script": "fairness.py",
    },
    "p7_anomaly_detection": {
        "title": "P7 · Anomaly Detection — Fraud on Imbalanced Data",
        "lesson": """
## Real lesson: accuracy is 99.5% and catches zero fraud — use PR-AUC

With 0.5% fraud rate, a model that predicts "normal" for every transaction
achieves 99.5% accuracy. Accuracy is meaningless here. This notebook compares
Isolation Forest (unsupervised) vs. GBM (supervised) on a synthetic fraud dataset
and evaluates with Precision-Recall curves.

**The punchline:** PR-AUC exposes what accuracy hides. The supervised GBM
substantially outperforms the unsupervised baseline when labels are available.
""",
        "script": "detect.py",
    },
    "p8_recommender": {
        "title": "P8 · Recommender System — MovieLens 100K",
        "lesson": """
## Real lesson: RMSE hides popularity bias; evaluate coverage + Precision@K

A recommender optimised for RMSE tends to recommend only popular items, failing
the long tail. This notebook compares a popularity baseline vs. user-based
collaborative filtering vs. SVD matrix factorisation, and evaluates all three
on RMSE, coverage, and Precision@10.

**The punchline:** popularity has the best RMSE but worst coverage. SVD balances
accuracy and coverage. Always evaluate recommenders beyond RMSE.
""",
        "script": "recommend.py",
    },
    "p9_nlp_classification": {
        "title": "P9 · NLP Classification — 20 Newsgroups",
        "lesson": """
## Real lesson: TF-IDF + logistic regression is a strong baseline before transformers

Before reaching for BERT, fit a TF-IDF + linear model. It is fast, interpretable,
and often within a few points of a fine-tuned transformer on short-text
classification. This notebook compares Naïve Bayes vs. Logistic Regression vs.
LinearSVC, and extracts top features per class for interpretability.

**The punchline:** LinearSVC with TF-IDF outperforms Naïve Bayes on the
ambiguous comp.* categories where word-count independence breaks down.
""",
        "script": "classify.py",
    },
    "p10_optimization": {
        "title": "P10 · Optimisation — Budget Allocation, Knapsack, Staffing",
        "lesson": """
## Real lesson: predictions → actions requires an optimisation layer

A model that predicts revenue-per-pound by channel creates value only when
combined with a constrained allocation solver. This notebook demonstrates three
classic OR problems: LP budget allocation (scipy), 0/1 campaign selection (PuLP),
and staff-to-shift assignment (Hungarian algorithm).

**The punchline:** the LP optimal allocation yields 28.7% more revenue than equal
splitting — pure prediction without optimisation leaves money on the table.
""",
        "script": "optimize.py",
    },
    "p11_bandit": {
        "title": "P11 · Multi-Armed Bandits — Epsilon-Greedy vs UCB1 vs Thompson Sampling",
        "lesson": """
## Real lesson: Thompson sampling dominates; bandits are the right tool for online decisions

When you must allocate a budget across uncertain options and learn as you go,
standard A/B testing is wasteful — it allocates equally to all arms regardless
of what's being observed. Bandits adaptively shift budget to better-performing
arms.

This notebook compares three strategies on 5 marketing message variants and
extends to contextual bandits (LinUCB), where arm selection is conditioned on
user features.

**The punchline:** Thompson sampling achieves the lowest regret because its
Bayesian Beta-Binomial model naturally balances exploration (uncertain arms)
and exploitation (known good arms).
""",
        "script": "bandit.py",
    },
    "p12_rag": {
        "title": "P12 · Retrieval-Augmented Generation (RAG) — Building a Knowledge Retriever",
        "lesson": """
## Real lesson: retrieval quality is the ceiling — evaluate it first

A RAG system is only as good as its retriever. If the retriever doesn't surface
the relevant context, the LLM cannot generate a correct answer regardless of its
capability. This notebook builds a complete retrieval pipeline, evaluates it with
Recall@K and MRR, and demonstrates why hybrid (sparse + dense) retrieval
outperforms either alone.

**The punchline:** Recall@3 = 100% with TF-IDF on this well-structured knowledge
base. In production with noisier documents, dense embedding retrieval with
cross-encoder reranking closes the gap.
""",
        "script": "rag.py",
    },
}

CASE_STUDY = {
    "title": "Case Study · Churn Prediction with Uplift Modelling",
    "lesson": """
## The central lesson: target uplift, not risk

The intuitive approach — identify the highest-risk churners and offer them a
retention discount — is wrong. Some high-risk customers will churn *regardless*
of the offer (lost causes); others won't churn anyway (sure things). Only the
**persuadables** — those who churn without the offer but stay with it — generate
positive ROI.

This notebook takes the problem from business framing → data → churn model →
causal uplift model → budget recommendation. The simulated pilot RCT quantifies
the difference:

| Strategy | Net value |
|---|---|
| Target by predicted uplift | **+£55.7k** |
| Target by churn risk (naïve) | +£6.6k |
| Treat everyone | –£34k |

Risk ≠ uplift. Modelling causal effect is the whole job.
""",
}


def project_notebook(proj_dir: Path, meta: dict) -> dict:
    script_path = proj_dir / meta["script"]
    if not script_path.exists():
        print(f"  SKIP {script_path} (not found)")
        return None

    src = script_path.read_text(encoding="utf-8")
    preamble, sections = split_sections(src)

    cells = []

    # Title + lesson
    cells.append(md(f"# {meta['title']}\n{meta['lesson']}"))

    # Notebook setup cell: inline plots + style defaults
    cells.append(md("## Setup"))
    cells.append(code(SETUP_CELL))

    # Preamble (docstring + imports) — strip module docstring, keep imports
    preamble_code = re.sub(r'^"""[\s\S]*?"""', "", preamble).strip()
    if preamble_code:
        cells.append(md("### Imports and constants"))
        cells.append(code(preamble_code, prep=True))

    # Each section
    for header, body in sections:
        if header:
            title = header.title().replace("_", " ")
            cells.append(md(f"## {title}"))
        cells.append(code(body, prep=True))

    return nb(cells)


def case_study_notebook(cs_dir: Path, meta: dict) -> dict:
    script_path = cs_dir / "run.py"
    if not script_path.exists():
        print(f"  SKIP {script_path} (not found)")
        return None

    src = script_path.read_text(encoding="utf-8")
    preamble, sections = split_sections(src)

    cells = [md(f"# {meta['title']}\n{meta['lesson']}")]
    cells.append(md("## Setup"))
    cells.append(code(SETUP_CELL))

    preamble_code = re.sub(r'^"""[\s\S]*?"""', "", preamble).strip()
    if preamble_code:
        cells.append(md("### Imports and constants"))
        cells.append(code(preamble_code, prep=True))

    for header, body in sections:
        if header:
            cells.append(md(f"## {header.title()}"))
        cells.append(code(body, prep=True))

    return nb(cells)


def make_topic_notebooks() -> None:
    """
    Create standalone topic notebooks in notebooks/ that demonstrate
    key concepts from the docs with runnable examples.
    """
    nb_dir = ROOT / "notebooks"

    # ── Topic 1: Statistics fundamentals ─────────────────────────────────────
    save(nb_dir / "01-statistics-fundamentals.ipynb", nb([
        md("# Statistics That Matter for Data Science\n\n"
           "Practical statistical concepts that appear in every DS project:\n"
           "hypothesis testing, the bootstrap, power analysis, multiple comparisons."),
        md("## Hypothesis testing and p-values"),
        code("""\
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# Simulate A/B test: does the treatment group have higher conversion?
control   = rng.binomial(1, 0.10, 1000)   # 10% conversion
treatment = rng.binomial(1, 0.12, 1000)   # 12% conversion (true lift = +2pp)

t_stat, p_val = stats.ttest_ind(control, treatment)
print(f"Control mean:   {control.mean():.3f}")
print(f"Treatment mean: {treatment.mean():.3f}")
print(f"Observed diff:  {treatment.mean() - control.mean():+.3f}")
print(f"t-statistic:    {t_stat:.3f}")
print(f"p-value:        {p_val:.4f}  {'(significant at α=0.05)' if p_val < 0.05 else '(not significant)'}")"""),
        md("## Bootstrap confidence intervals\n\n"
           "Model-free uncertainty estimate — no distributional assumptions required."),
        code("""\
import numpy as np

rng = np.random.default_rng(42)
data = rng.lognormal(mean=3, sigma=0.5, size=200)  # skewed revenue data

# Bootstrap 95% CI for the mean
n_boot = 5000
boot_means = [rng.choice(data, size=len(data), replace=True).mean()
              for _ in range(n_boot)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"Sample mean:        {data.mean():.2f}")
print(f"Bootstrap 95% CI:   [{ci_low:.2f}, {ci_high:.2f}]")
print(f"Parametric 95% CI:  [{data.mean() - 1.96*data.std()/len(data)**0.5:.2f}, "
      f"{data.mean() + 1.96*data.std()/len(data)**0.5:.2f}]")
print("Bootstrap CI is wider — correctly accounts for skew")"""),
        md("## Power analysis — how many samples do I need?"),
        code("""\
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()

# Scenario: detect a 2pp lift in conversion rate (baseline 10%)
# Baseline std ≈ sqrt(0.10 * 0.90)
baseline_std = (0.10 * 0.90) ** 0.5
effect_size = 0.02 / baseline_std   # Cohen's d

n = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
print(f"Effect size (Cohen's d): {effect_size:.3f}")
print(f"Required n per arm:      {int(n) + 1}")
print(f"Total sample required:   {2*(int(n)+1)}")
print()
# Sensitivity: how does n change with power requirement?
for power in [0.70, 0.80, 0.90, 0.95]:
    n = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=power)
    print(f"  Power={power:.0%} → n per arm = {int(n)+1}")"""),
        md("## Multiple comparisons — Benjamini-Hochberg correction"),
        code("""\
import numpy as np
from statsmodels.stats.multitest import multipletests

rng = np.random.default_rng(42)

# Simulate 20 hypothesis tests; only 3 truly significant
p_values = np.concatenate([
    rng.uniform(0, 0.01, 3),   # truly significant
    rng.uniform(0.05, 1.0, 17) # noise
])
rng.shuffle(p_values)

reject_bonf, _, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
reject_bh,   _, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print(f"Raw p < 0.05:      {(p_values < 0.05).sum()} rejections (likely includes false positives)")
print(f"Bonferroni:        {reject_bonf.sum()} rejections (very conservative)")
print(f"Benjamini-Hochberg:{reject_bh.sum()} rejections (controls FDR at 5%)")
print()
print("BH is the standard for DS/ML multiple-testing scenarios")"""),
    ]))

    # ── Topic 2: Model evaluation ─────────────────────────────────────────────
    save(nb_dir / "02-model-evaluation.ipynb", nb([
        md("# Model Evaluation & Validation\n\n"
           "Choosing the right metric, avoiding leakage in evaluation, "
           "calibrating probabilities, and setting decision thresholds from costs."),
        md("## ROC-AUC vs PR-AUC — when each matters"),
        code("""\
import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove this line in Jupyter
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

# Imbalanced dataset: 1% positive rate
X, y = make_classification(n_samples=10_000, n_features=10, n_informative=6,
                            weights=[0.99, 0.01], random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=42)

model = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
model.fit(X_tr, y_tr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax1)
ax1.set_title("ROC curve — looks great (AUC≈0.97)\\nbut positive rate = 1%")

PrecisionRecallDisplay.from_estimator(model, X_te, y_te, ax=ax2)
ax2.axhline(y_te.mean(), ls="--", color="grey", label=f"No-skill baseline ({y_te.mean():.2f})")
ax2.set_title("PR curve — honest picture\\nfor imbalanced data")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()  # in script: plt.savefig(...)
print("Rule: use PR-AUC when positive rate < 10%")"""),
        md("## Calibration — do probabilities mean probabilities?"),
        code("""\
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

gbm = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
gbm_cal = CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=100,
    random_state=42), method="isotonic", cv=5).fit(X_tr, y_tr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0,1],[0,1],"k--", label="Perfect")
for model, label in [(gbm, "GBM (raw)"), (gbm_cal, "GBM (calibrated)")]:
    fp, mp = calibration_curve(y_te, model.predict_proba(X_te)[:,1], n_bins=10)
    ax.plot(mp, fp, "o-", label=label)
ax.set(xlabel="Mean predicted prob", ylabel="Fraction positive",
       title="Calibration: does P(y=1|score=0.7) ≈ 0.70?")
ax.legend()
plt.tight_layout()
plt.show()
print("Calibration matters whenever probabilities drive decisions (risk scores, thresholds)")"""),
        md("## Cost-based decision threshold"),
        code("""\
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
pipe_lr = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(X_tr), y_tr)
proba = pipe_lr.predict_proba(StandardScaler().fit(X_tr).transform(X_te))[:,1]

# Cost of FP (unnecessary biopsy) vs FN (missed cancer)
cost_fp, cost_fn = 500, 10_000   # £

print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<10} {'Expected cost':<15}")
print("-" * 50)
for thresh in np.arange(0.2, 0.8, 0.1):
    pred = (proba >= thresh).astype(int)
    fp = ((pred==1) & (y_te==0)).sum()
    fn = ((pred==0) & (y_te==1)).sum()
    total_cost = fp * cost_fp + fn * cost_fn
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred)
    print(f"{thresh:<12.1f} {prec:<12.3f} {rec:<10.3f} £{total_cost:,}")
print("\\nOptimal threshold = min expected cost, not 0.5")"""),
    ]))

    # ── Topic 3: A/B testing ──────────────────────────────────────────────────
    save(nb_dir / "03-ab-testing.ipynb", nb([
        md("# A/B Testing & Causal Inference\n\n"
           "Running valid experiments, avoiding common traps (peeking, SUTVA violations), "
           "and measuring causal effects rather than correlations."),
        md("## The peeking problem — why you can't check results early"),
        code("""\
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# Simulate 1000 A/A tests (no true effect) where analyst peeks daily
n_days     = 30
n_per_day  = 50
n_sims     = 1000
alpha      = 0.05

fp_fixed_horizon = 0  # check only at end
fp_peeking       = 0  # check every day, stop at first significance

for _ in range(n_sims):
    control   = []
    treatment = []
    found_early = False
    for day in range(n_days):
        control   += rng.binomial(1, 0.10, n_per_day).tolist()
        treatment += rng.binomial(1, 0.10, n_per_day).tolist()
        # Peeking: test every day
        if len(control) >= 20:
            _, p = stats.ttest_ind(control, treatment)
            if p < alpha and not found_early:
                fp_peeking += 1
                found_early = True
    # Fixed horizon: test only at end
    _, p = stats.ttest_ind(control, treatment)
    if p < alpha:
        fp_fixed_horizon += 1

print(f"False positive rate (fixed horizon, α=0.05): {fp_fixed_horizon/n_sims:.1%}  (expected ≈5%)")
print(f"False positive rate (peek every day):         {fp_peeking/n_sims:.1%}  (inflated!)")
print()
print("Fix: commit to sample size before starting, or use sequential testing (CUPED/mSPRT)")"""),
        md("## CUPED — variance reduction with pre-experiment data"),
        code("""\
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n   = 1000

# Pre-experiment metric (highly correlated with post-experiment)
pre  = rng.normal(100, 20, n)
# Post-experiment: treatment lifts mean by 3 units
treat = rng.binomial(1, 0.5, n)
post  = pre * 0.8 + treat * 3 + rng.normal(0, 15, n)

# Standard t-test
t_std, p_std = stats.ttest_ind(post[treat==1], post[treat==0])

# CUPED adjustment: regress pre out
theta = np.cov(post, pre)[0,1] / np.var(pre)
post_cuped = post - theta * (pre - pre.mean())
t_cup, p_cup = stats.ttest_ind(post_cuped[treat==1], post_cuped[treat==0])

var_reduction = 1 - np.var(post_cuped) / np.var(post)
print(f"True ATE:            3.0")
print(f"Naive estimate:      {post[treat==1].mean() - post[treat==0].mean():.2f}")
print(f"CUPED estimate:      {post_cuped[treat==1].mean() - post_cuped[treat==0].mean():.2f}")
print()
print(f"Standard p-value:    {p_std:.4f}")
print(f"CUPED p-value:       {p_cup:.4f}")
print(f"Variance reduction:  {var_reduction:.1%}")
print("CUPED needs fewer samples to achieve the same power")"""),
        md("## Causal forest — heterogeneous treatment effects"),
        code("""\
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

rng = np.random.default_rng(42)
n   = 3000

# Covariates: age, spend level
age   = rng.uniform(20, 70, n)
spend = rng.lognormal(4, 1, n)

# Treatment assignment (not random — older users more likely treated)
p_treat = 1 / (1 + np.exp(-(age - 45) / 10))
treat   = rng.binomial(1, p_treat, n)

# Heterogeneous effect: young high-spenders respond more
true_cate = 5 + 0.1 * (60 - age) + 0.001 * spend
outcome   = 20 + 0.05 * age + 0.005 * spend + true_cate * treat + rng.normal(0, 5, n)

X = np.column_stack([age, spend])

# Two-model (T-learner) CATE estimate
gbm1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm0 = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm1.fit(X[treat==1], outcome[treat==1])
gbm0.fit(X[treat==0], outcome[treat==0])
cate_hat = gbm1.predict(X) - gbm0.predict(X)

corr = np.corrcoef(true_cate, cate_hat)[0,1]
print(f"Correlation of true CATE vs estimated CATE: {corr:.3f}")
print(f"True mean CATE:      {true_cate.mean():.2f}")
print(f"Estimated mean CATE: {cate_hat.mean():.2f}")
print()
print("High-CATE users (top quartile):")
top = cate_hat > np.percentile(cate_hat, 75)
print(f"  Mean estimated effect: {cate_hat[top].mean():.2f}")
print(f"  Mean true effect:      {true_cate[top].mean():.2f}")
print("Target the high-CATE segment for maximum ROI")"""),
    ]))

    # ── Topic 4: NLP basics ───────────────────────────────────────────────────
    save(nb_dir / "04-nlp-text-classification.ipynb", nb([
        md("# NLP — Text Classification from Scratch to Fine-tuning\n\n"
           "Progressive workflow: bag-of-words → TF-IDF + linear → embeddings.\n"
           "Always establish a strong linear baseline before reaching for transformers."),
        md("## TF-IDF + Logistic Regression baseline"),
        code("""\
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 4 well-separated categories
CATS = ["sci.med", "sci.space", "rec.sport.baseball", "talk.politics.guns"]
train = fetch_20newsgroups(subset="train", categories=CATS,
                            remove=("headers","footers","quotes"))
test  = fetch_20newsgroups(subset="test",  categories=CATS,
                            remove=("headers","footers","quotes"))

pipe = Pipeline([
    ("vec", TfidfVectorizer(max_features=30_000, ngram_range=(1,2))),
    ("clf", LogisticRegression(C=5.0, max_iter=1000, random_state=42)),
])
pipe.fit(train.data, train.target)
preds = pipe.predict(test.data)
print(classification_report(test.target, preds, target_names=CATS))"""),
        md("## Top features per class — model interpretability"),
        code("""\
import numpy as np

vocab = pipe.named_steps["vec"].get_feature_names_out()
coef  = pipe.named_steps["clf"].coef_

print("Top 10 features per class:\\n")
for i, cat in enumerate(CATS):
    top10 = np.argsort(coef[i])[-10:][::-1]
    print(f"{cat}:")
    print("  " + ", ".join(vocab[top10]))
    print()"""),
        md("## Sentence embeddings for semantic similarity"),
        code("""\
# Demonstrates the concept with TF-IDF cosine similarity
# In production: replace with sentence-transformers embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "machine learning for fraud detection in banking",
    "credit card fraud prevention with neural networks",
    "natural language processing for sentiment analysis",
    "deep learning for customer review classification",
    "random forest for churn prediction",
]
query = "detecting financial fraud with ML"

vec = TfidfVectorizer().fit(docs + [query])
doc_vecs   = vec.transform(docs)
query_vec  = vec.transform([query])
sims = cosine_similarity(query_vec, doc_vecs)[0]

print(f"Query: '{query}'\\n")
print("Similarity scores:")
for doc, sim in sorted(zip(docs, sims), key=lambda x: -x[1]):
    print(f"  {sim:.3f}  {doc}")
print()
print("Note: in production use sentence-transformers for semantic similarity")
print("      (pip install sentence-transformers)")"""),
    ]))

    # ── Topic 5: Feature engineering ─────────────────────────────────────────
    save(nb_dir / "05-feature-engineering.ipynb", nb([
        md("# Feature Engineering & Data Preparation\n\n"
           "The highest-leverage work in any ML project. "
           "Covers imputation, encoding, transforms, leakage prevention, and the shuffle-label test."),
        md("## Leakage — the most common failure mode\n\n"
           "Fitting a scaler on the full dataset before splitting **leaks** test statistics into training.\n"
           "Always fit transforms inside a Pipeline."),
        code("""\
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# ❌ WRONG: fit scaler before CV (leaks test set stats into training)
from sklearn.preprocessing import StandardScaler as SS
X_scaled_WRONG = SS().fit_transform(X)   # full dataset used!
wrong_score = cross_val_score(
    LogisticRegression(max_iter=1000, random_state=42),
    X_scaled_WRONG, y, cv=cv, scoring="roc_auc"
).mean()

# ✅ CORRECT: scaler inside Pipeline (fit only on training folds)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])
correct_score = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

print(f"WRONG  (scaler fit on full data): ROC-AUC = {wrong_score:.4f}  <- optimistically biased")
print(f"CORRECT (scaler inside Pipeline): ROC-AUC = {correct_score:.4f}  <- honest estimate")
print()
print("The difference looks small here but grows with small datasets and many features")"""),
        md("## Shuffle-label leakage guard"),
        code("""\
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
rng  = np.random.default_rng(42)
cv   = StratifiedKFold(5, shuffle=True, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

real_score      = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
shuffled_score  = cross_val_score(pipe, X, rng.permutation(y),
                                   cv=cv, scoring="roc_auc").mean()

print(f"Real labels:     ROC-AUC = {real_score:.4f}")
print(f"Shuffled labels: ROC-AUC = {shuffled_score:.4f}  (should collapse to ~0.50)")
print()
if shuffled_score > 0.65:
    print("WARNING: shuffled labels give high AUC → leakage suspected!")
else:
    print("PASS: shuffled labels collapse to chance — no obvious leakage")"""),
        md("## High-cardinality categoricals — target encoding inside CV"),
        code("""\
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

rng = np.random.default_rng(42)
n   = 2000

# Simulated e-commerce data: 50 product categories, conversion as target
category = rng.choice([f"cat_{i}" for i in range(50)], n)
y = (rng.uniform(0,1,n) < 0.05 + 0.1 * (category == "cat_0")).astype(int)

df = pd.DataFrame({"category": category, "y": y})

# Target encoding INSIDE CV (to avoid leakage)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
df["target_enc"] = np.nan

for tr_idx, val_idx in kf.split(df):
    train_fold = df.iloc[tr_idx]
    mean_map   = train_fold.groupby("category")["y"].mean()
    global_mean = train_fold["y"].mean()
    df.loc[df.index[val_idx], "target_enc"] = (
        df.iloc[val_idx]["category"].map(mean_map).fillna(global_mean)
    )

print("Target encoding (fold means, no leakage):")
print(df.groupby("category")["target_enc"].mean().sort_values(ascending=False).head(10))
print()
print("cat_0 should have highest encoding (0.15) — injected true effect")"""),
    ]))

    print(f"  wrote notebooks/01-statistics-fundamentals.ipynb")
    print(f"  wrote notebooks/02-model-evaluation.ipynb")
    print(f"  wrote notebooks/03-ab-testing.ipynb")
    print(f"  wrote notebooks/04-nlp-text-classification.ipynb")
    print(f"  wrote notebooks/05-feature-engineering.ipynb")


def main():
    print("Generating project notebooks...")
    for name, meta in PROJECTS.items():
        proj_dir = ROOT / "projects" / name
        notebook = project_notebook(proj_dir, meta)
        if notebook:
            save(proj_dir / "notebook.ipynb", notebook)

    print("\nGenerating case study notebook...")
    cs_dir = ROOT / "case_study_churn_uplift"
    nb_cs = case_study_notebook(cs_dir, CASE_STUDY)
    if nb_cs:
        save(cs_dir / "notebook.ipynb", nb_cs)

    print("\nGenerating topic notebooks...")
    import subprocess, sys
    subprocess.run([sys.executable,
                    str(ROOT / "scripts" / "make_topic_notebooks.py")], check=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
