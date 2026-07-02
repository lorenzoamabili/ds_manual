# 35 · MarTech (Marketing Analytics)

> Using data to acquire, engage, and retain customers — from segmentation and
> targeting to attribution and campaign lift measurement.

---

## Why data science here

Marketing was one of the first business functions to industrialise data science,
driven by the need to personalise at scale (millions of customers, thousands of
messages). The domain's signature challenge is **causal attribution**: customers
interact with many touchpoints before converting, and crediting one channel with
the sale is inherently a [causal inference](09-causal-inference-and-experimentation.md) problem, not a correlation one.

MarTech sits at the intersection of: (1) **segmentation** (who are my customers?),
(2) **prediction** (who will churn, convert, or respond?), (3) **causal inference**
(did this campaign *cause* more purchases?), and (4) **optimisation** (how do I
allocate a finite budget across channels?). All four are needed; most marketing
teams have only the first two.

The churn-uplift case study in this repo ([`case_study_churn_uplift/`](../case_study_churn_uplift))
is a concrete MarTech problem: targeting persuadables (high uplift) beats targeting
churners (high risk), and the difference is £55k on a simulated pilot.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Customer segmentation | Group customers by behaviour | RFM + [KMeans](06-unsupervised-learning.md), [UMAP](06-unsupervised-learning.md) + DBSCAN |
| Churn prediction | Who will cancel in the next 30 days? | Binary classification |
| Campaign uplift | Did the campaign *cause* more sales? | Uplift modelling, [A/B test](09-causal-inference-and-experimentation.md) |
| Attribution modelling | Which touchpoints drove conversion? | Shapley-value attribution, Markov chains |
| LTV prediction | How much will this customer spend lifetime? | Pareto/NBD, regression, survival |
| Next best offer | What product should we show this customer? | Recommendation + contextual bandit |
| Send-time optimisation | When should we send this email? | Classification / historical pattern |

---

## Key techniques

### 1. RFM segmentation

Score each customer on Recency, Frequency, Monetary value. Quantile-score each
dimension (1–5), combine, and [cluster](06-unsupervised-learning.md). Produces named segments: Champions,
Loyal, At-Risk, Hibernating, Lost. Each segment maps to a different CRM action.

**Pitfall:** RFM is backward-looking. It tells you who was valuable, not who will
be valuable. Augment with forward-looking predictive churn/LTV scores.

### 2. Uplift modelling

Separate "will they buy?" from "does the campaign make them more likely to buy?"
There are four customer types: persuadables (buy only if treated), sure things
(buy regardless), lost causes (won't buy regardless), and sleeping dogs (buy
*less* if treated). Standard response models can't distinguish these. Uplift
models estimate the individual treatment effect.

Two-model approach: `uplift = P(buy | treat) − P(buy | control)`. Train one
response model on treated customers and one on controls, then score the difference.
S-learner (one model with treatment as a feature) is simpler. See
[09](09-causal-inference-and-experimentation.md) and the
[case study](../case_study_churn_uplift).

**Pitfall:** uplift requires a randomised holdout (A/B) to train on. Without
random assignment, the "control" group is not comparable to the treatment group.

### 3. Multi-touch attribution

A customer sees 5 ads, 3 emails, and a paid search result before converting. Who
gets credit? Last-touch attribution (100% to the last click) is wrong but common.
Data-driven attribution models each touchpoint's marginal contribution using
Shapley values (cooperative game theory) or Markov chain transition matrices.

**Pitfall:** attribution is observational. It tells you correlation of touchpoints
with conversion, not causation. Budget decisions based on attribution require
corroborating experiments.

### 4. Customer lifetime value (CLV)

**Contractual setting** (subscription): CLV = margin × (1 / churn rate) in steady
state. Model churn with [survival analysis](16-survival-analysis.md).

**Non-contractual setting** (retail): use Pareto/NBD model (purchase count) +
Gamma-Gamma model (average order value). Or fit ML on historical transaction data
with appropriate feature construction (no [leakage](03-data-and-feature-engineering.md) on future aggregates).

### 5. A/B testing for campaigns

Random holdout is the gold standard. Key decisions: unit of randomisation (user,
cookie, geo), primary metric, sample size (power calculation), duration (avoid
novelty effects). See [09](09-causal-inference-and-experimentation.md) and
[17](17-experimentation-advanced.md).

---

## Best practices & pitfalls

- **Targeting uplift, not risk.** Sending a retention offer to all high-risk
  churners treats "sure things" (who would stay anyway) and "sleeping dogs"
  (who leave because of the offer). The case study quantifies this: risk targeting
  nets £6.6k; uplift targeting nets £55.7k on the same budget.
- **Holdout groups are sacred.** Never use the holdout for anything other than
  measurement. Contaminating it (letting holdout customers receive messages) makes
  the measurement meaningless.
- **Look-alike modelling.** Using ML to find "customers like your best customers"
  in a prospect database amplifies whatever biases are in your historical data.
  Validate on fresh cohorts.
- **Email metrics are proxy metrics.** Open rate is gameable (bot opens). Click
  rate is better but measures intent, not outcome. Always tie to revenue or
  retention when possible.
- **[Seasonality](07-time-series-forecasting.md) confounds.** Campaigns run in December will look great because
  everyone buys in December. Control for seasonality in uplift measurement.

---

## Python example — RFM segmentation on UCI Online Retail

```python
"""
RFM customer segmentation on UCI Online Retail dataset.

Computes Recency, Frequency, Monetary value per customer,
then applies KMeans to produce actionable customer segments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)
OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Load data ────────────────────────────────────────────────────────────────
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "00352/Online%20Retail.xlsx")
print("Loading UCI Online Retail dataset...")
try:
    df = pd.read_excel(url, engine="openpyxl")
    print(f"Loaded {len(df):,} rows.")
except Exception:
    print("Download failed — using synthetic RFM data.")
    n = 2000
    df = pd.DataFrame({
        "InvoiceNo": [f"INV{i:05d}" for i in rng.integers(1, 800, n)],
        "Description": rng.choice([f"PROD_{i}" for i in range(50)], n),
        "Quantity": rng.integers(1, 10, n),
        "InvoiceDate": pd.date_range("2010-01-01", periods=n, freq="1h"),
        "UnitPrice": rng.uniform(1, 50, n),
        "CustomerID": rng.integers(10000, 12000, n),
        "Country": "United Kingdom",
    })

# ── Clean ────────────────────────────────────────────────────────────────────
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
if "Country" in df.columns:
    df = df[df["Country"] == "United Kingdom"]
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# ── Compute RFM ───────────────────────────────────────────────────────────────
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency  =("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency=("InvoiceNo",   "nunique"),
    Monetary =("Revenue",     "sum"),
).reset_index()

print(f"\nCustomers: {len(rfm):,}")
print(rfm[["Recency","Frequency","Monetary"]].describe().round(1).to_string())

# ── Cluster ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(np.log1p(rfm[["Recency","Frequency","Monetary"]]))

# Elbow method
inertias = []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(ks), inertias, "o-")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow method")

# Final model with k=4
km4 = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
rfm["Segment"] = km4.labels_

# PCA for 2D visualisation
pca = PCA(n_components=2, random_state=42)
X2  = pca.fit_transform(X)
colors = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
for seg in range(4):
    mask = rfm["Segment"] == seg
    ax2.scatter(X2[mask, 0], X2[mask, 1],
                label=f"Segment {seg} (n={mask.sum()})",
                color=colors[seg], alpha=0.6, s=20)
ax2.set_title("RFM segments (PCA projection)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "rfm_segments.png", dpi=120)
plt.close()

# ── Segment profiles ──────────────────────────────────────────────────────────
profile = rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1)
print("\nSegment profiles (raw means):")
print(profile.to_string())
print("\nInterpretation guide:")
print("  Low recency + high freq + high monetary = Champions")
print("  High recency + low freq + low monetary  = Hibernating / Lost")
```

---

## Cross-references

- [06](06-unsupervised-learning.md) — clustering for segmentation
- [09](09-causal-inference-and-experimentation.md) — A/B testing, uplift, causal inference
- [17](17-experimentation-advanced.md) — [CUPED](17-experimentation-advanced.md), [sequential testing](17-experimentation-advanced.md)
- [08](08-recommendation-systems.md) — personalisation and next-best-offer
- [32](32-retail-ecommerce.md) — overlap with retail analytics
- [`case_study_churn_uplift/`](../case_study_churn_uplift) — full uplift case study
