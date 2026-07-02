# 15 · Communication & Visualisation

The most technically correct analysis that no one understands or acts on has
created zero value. This is not a soft skill bolted on at the end — it is where the
value is realised, and it is a genuine differentiator.

## Analysis serves a decision, not a curiosity
Before presenting anything, answer: *what decision does this change, and for
whom?* Structure the communication around that, not around the order you did the
work. Lead with the answer (the "BLUF" — bottom line up front), then support it.
Executives want the recommendation and its confidence; peers want the method;
almost no one wants your chronological journey.

## Visualisation principles that hold up
- **Choose the chart for the comparison**, not for novelty:
  - *trend over time* → line;
  - *comparison across categories* → bar (start the axis at zero);
  - *distribution* → histogram/box/violin;
  - *relationship between two variables* → scatter;
  - *part-to-whole* → stacked bar (rarely a pie; never a 3-D pie).
- **Maximise the data-ink ratio** — remove chartjunk, gridlines, and decoration
  that don't carry information (Tufte's core lesson).
- **Encode with position and length** (the perceptually accurate channels) before
  colour and area (which humans judge poorly).
- **Colour with intent** — sequential for ordered magnitude, diverging for a
  meaningful midpoint, categorical (few, distinct) for groups. Check
  colour-blind safety; never rely on red/green alone.
- **Annotate the point.** The best charts tell the reader what to notice — a title
  that states the finding ("Churn doubled after the price change"), a highlighted
  series, a reference line.

## Honesty in charts
- Don't truncate axes to exaggerate (bar charts start at zero; line charts can be
  framed but label it).
- Show uncertainty — error bars, intervals, or "n=…". A point estimate presented as
  a hard fact is a communication *and* an ethical failure.
- Don't imply causation from a correlational plot
  ([09](09-causal-inference-and-experimentation.md)).

## Communicating uncertainty to non-technical audiences
- Translate probabilities into frequencies ("about 1 in 20") — people reason better
  with counts than percentages.
- Give ranges, not false precision ("between 8% and 12%", not "10.3%").
- Be explicit about what you *don't* know and what would change your conclusion.
  [Calibrated](04-evaluation-and-validation.md) humility builds far more credibility than confident certainty that
  later breaks.

## Tools
`matplotlib`/`seaborn` for analysis and publication; `plotly`/`Altair` for
interactive; `D3.js`/`Observable` for bespoke, explanatory, web-native
visualisation; dashboards (Streamlit, Dash, Tableau, Power BI) for
self-serve monitoring — but a dashboard is not an analysis; it answers questions
someone already knew to ask, whereas an analysis finds the question.

## The [reproducible](01-lifecycle-and-reproducibility.md) narrative
For a portfolio especially: a result is more persuasive when the reader can see the
path from raw data to conclusion. A clean notebook or report that interleaves the
reasoning, the code, and the figures — and that *runs* — demonstrates rigour in a
way a polished slide never can. That is the entire design philosophy of this
manual's [projects](../projects).

---

## Python example — publication-quality figures following Tufte principles

```python
"""
Demonstrates chart best practices:
  1. Bar chart with honest zero baseline + uncertainty
  2. Calibration curve with uncertainty band
  3. Distribution comparison (KDE + rug)
  4. Annotated trend line (finding in title, key event marked)

All using matplotlib with a minimal, high data-ink-ratio style.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ── Minimal style — strip chartjunk ──────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# ── 1. Bar chart: model comparison with CI ────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X_s = StandardScaler().fit_transform(X)
cv  = StratifiedKFold(5, shuffle=True, random_state=42)

model_names  = ["Baseline", "Logistic Reg", "Rand. Forest"]
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
clfs = [DummyClassifier("most_frequent"),
        LogisticRegression(max_iter=1000, random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42)]
means, errs = [], []
for clf in clfs:
    s = cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc")
    means.append(s.mean()); errs.append(s.std())

ax = axes[0, 0]
bars = ax.bar(model_names, means, yerr=errs, capsize=4,
              color=["#aaaaaa", "#3498db", "#2ecc71"], alpha=0.85)
ax.set_ylim(0, 1.05)                    # axis starts at zero — honest
ax.axhline(0.5, color="red", lw=0.8, ls="--", label="Chance (0.50)")
ax.set(ylabel="ROC-AUC", title="Logistic regression matches Random Forest\non breast cancer (mean ± 1 SD, 5-fold CV)")
ax.legend(fontsize=8)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, m + 0.01, f"{m:.3f}",
            ha="center", va="bottom", fontsize=8)

# ── 2. Calibration curve ──────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.25,
                                            stratify=y, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
prob = lr.predict_proba(X_te)[:, 1]
fp, mp = calibration_curve(y_te, prob, n_bins=8)

ax = axes[0, 1]
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
ax.plot(mp, fp, "o-", color="#e74c3c", label="Logistic Reg")
ax.fill_between(mp, fp - 0.04, fp + 0.04, alpha=0.15, color="#e74c3c",
                label="±0.04 uncertainty band")
ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
       title="Calibration: model probabilities are reliable\n(close to diagonal)")
ax.legend(fontsize=8)

# ── 3. Distribution comparison ────────────────────────────────────────────────
ax = axes[1, 0]
mal = X[:, 0][y == 1]   # mean radius, malignant
ben = X[:, 0][y == 0]   # mean radius, benign
for vals, label, colour in [(mal, "Malignant", "#e74c3c"),
                              (ben, "Benign",    "#3498db")]:
    kde = gaussian_kde(vals)
    xs  = np.linspace(vals.min(), vals.max(), 200)
    ax.plot(xs, kde(xs), lw=2, color=colour, label=label)
    ax.fill_between(xs, kde(xs), alpha=0.15, color=colour)
    ax.plot(vals, np.full_like(vals, -0.003), "|", color=colour, alpha=0.3)
ax.set(xlabel="Mean radius", ylabel="Density",
       title="Mean radius strongly separates classes\n(use position/area encoding, not 3-D pie)")
ax.legend()

# ── 4. Annotated trend line ───────────────────────────────────────────────────
ax = axes[1, 1]
months = pd.date_range("2023-01", periods=24, freq="MS")
churn  = 8 + np.cumsum(rng.normal(0, 0.3, 24))
churn[12:] += 3.5    # price increase effect

ax.plot(months, churn, lw=2, color="#9b59b6")
ax.fill_between(months, churn - 0.8, churn + 0.8, alpha=0.15, color="#9b59b6")
ax.axvline(months[12], color="black", lw=1, ls="--")
ax.text(months[12], churn.max() - 0.3, " Price\n increase", fontsize=8)
ax.set(ylabel="Monthly churn rate (%)",
       title="Churn rose 3.5 pp after the price increase in Jan 2024")
ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("viz_best_practices.png", dpi=130)
plt.close()
print("Chart saved: viz_best_practices.png")
print("\nPrinciples demonstrated:")
print("  1. Bar chart: zero-baseline, error bars, data labels")
print("  2. Calibration: uncertainty band, finding-first title")
print("  3. Distribution: KDE + rug, position/area encoding")
print("  4. Trend: annotated event, uncertainty band, BLUF title")
```

---

## Cross-references

- [04](04-evaluation-and-validation.md) — calibration curves (plot from communication angle)
- [09](09-causal-inference-and-experimentation.md) — communicating [A/B test](09-causal-inference-and-experimentation.md) results honestly
- [19](19-responsible-ai-and-fairness.md) — charts for subgroup fairness reporting
