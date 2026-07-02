# 32 · Retail & E-commerce

> Using data to sell the right product, at the right price, to the right customer,
> at the right time — across supply chains that span continents.

---

## Why data science here

Retail generates some of the richest transactional data outside of finance: every
scan, click, basket, return, and delivery event is logged. The business problems
map cleanly onto core DS functions — demand forecasting, recommenders, pricing
optimisation, and clustering — which is why retail has been an ML testbed since
the Walmart / beer-and-diapers era.

The hard part is not the models — it is the **operational coupling**. A demand
forecast feeds replenishment which constrains promotions which affects pricing.
Errors compound. A 5% forecast error on a fast-moving SKU means stockouts on
weekends; a 5% price error means margin erosion at scale. Data scientists in
retail spend as much time on data pipelines and stakeholder alignment as on models.

The domain is also highly seasonal, promotional, and interrupted by external events
(weather, competitor moves, macroeconomic shocks). Standard i.i.d. assumptions
almost never hold.

---

## Signature problems

| Problem | Formulation | Typical approach |
|---------|-------------|------------------|
| Demand forecasting | Predict SKU-level sales N weeks ahead | SARIMA, Prophet, LightGBM on lag features |
| Price optimisation | Set price to maximise revenue or margin | Elasticity modelling, constrained optimisation |
| Market basket analysis | Which products are bought together? | Association rules (Apriori/FP-Growth) |
| Customer segmentation | Group customers by value/behaviour | RFM + KMeans |
| Recommender | Which product to show next? | Collaborative filtering, content-based |
| Markdown optimisation | Clear end-of-season inventory profitably | Dynamic programming, regression |
| Returns prediction | Will this order be returned? | Binary classification |
| Supply chain risk | Will this supplier be late? | Classification, survival analysis |

---

## Key techniques

### 1. Market basket analysis (association rules)

Find item sets that co-occur in baskets above a support threshold, then rank by
lift (= observed co-occurrence / expected under independence). The classic output:
*"customers who buy X also buy Y with 3.2× the expected frequency."*

Key metrics:
- **Support:** P(X ∩ Y) — how often does the pair appear?
- **Confidence:** P(Y|X) — given X, how often is Y also bought?
- **Lift:** confidence / P(Y) — is the association stronger than chance?

**Pitfall:** high-support rules are dominated by the most popular items (milk,
bread). Filter by lift > 1 and minimum support to find genuinely informative rules.

### 2. Demand forecasting

Hierarchical time series: forecasts at national level disagree with the sum of
store-level forecasts. Reconcile with **bottom-up** (sum store forecasts) or
**top-down** (share national forecast) methods. LightGBM on lag features + calendar
features often beats ARIMA on large SKU count because it handles cross-SKU
information and avoids unit-root issues.

**Pitfall:** promotional periods violate stationarity. Either model promotions as
external regressors or exclude promoted weeks from baseline training.

### 3. Price elasticity

Estimate demand elasticity ε = ∂log(Q)/∂log(P). Can use OLS on log-log model
if prices vary exogenously. More often, instrument with cost shocks or use
a quasi-experimental design (geo-split pricing tests). Then set price to
maximise revenue: P* = MC / (1 + 1/ε).

**Pitfall:** observed price variation is often endogenous (promotions correlate
with demand spikes). Naive OLS gives biased elasticity.

### 4. RFM customer segmentation

Score customers on Recency (days since last purchase), Frequency (number of orders),
and Monetary value (total spend). KMeans or decile-based segmentation groups
customers into actionable tiers (champions, at-risk, hibernating, lost). Feed
segments into CRM for targeted campaigns.

---

## Best practices & pitfalls

- **Forecast at the level you act.** A national forecast is useless if replenishment
  is decided by store and SKU. Forecast at the action level.
- **Naive baseline first.** "Last year's sales × trend factor" is the industry
  baseline. Beat it before celebrating a fancy model.
- **Calendar features matter more than model complexity.** Day-of-week, week-of-year,
  holiday flags, promotion flags — these explain most retail variance.
- **Intermittent demand is different.** Slow-moving SKUs have many zero-sales days.
  Use Croston's method or zero-inflated models, not standard ARIMA.
- **Association rules produce noise at scale.** With 10k SKUs, you get millions of
  rules. Pair with a business filter (margin contribution, inventory level).
- **Segmentation without activation is useless.** An RFM segment is only valuable
  if CRM can act on it differently.

---

## Python example — market basket analysis with mlxtend

```python
"""
Market basket analysis on UCI Online Retail dataset.

Finds product association rules using FP-Growth (efficient implementation
of Apriori). Filters by lift > 1 and support threshold.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent if "__file__" in dir() else Path(".")

# ── Load UCI Online Retail dataset ───────────────────────────────────────────
# Public dataset: UCI Machine Learning Repository
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00352/Online%20Retail.xlsx"
)
print("Loading UCI Online Retail dataset (may take ~20s on first run)...")

try:
    df = pd.read_excel(url, engine="openpyxl")
except Exception:
    # Fallback: synthetic basket data if download fails
    print("Download failed — using synthetic basket data.")
    rng = np.random.default_rng(42)
    products = [f"PROD_{i:03d}" for i in range(50)]
    records = []
    for inv in range(2000):
        n = rng.integers(2, 8)
        items = rng.choice(products, size=n, replace=False)
        for item in items:
            records.append({"InvoiceNo": inv, "Description": item,
                            "Quantity": 1, "CustomerID": rng.integers(1, 500)})
    df = pd.DataFrame(records)

# ── Clean ────────────────────────────────────────────────────────────────────
df = df[df["Quantity"] > 0].dropna(subset=["InvoiceNo", "Description"])
if "CustomerID" in df.columns:
    df = df.dropna(subset=["CustomerID"])
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
df["Description"] = df["Description"].str.strip()

# UK only (if Country column exists)
if "Country" in df.columns:
    df = df[df["Country"] == "United Kingdom"]

print(f"Transactions: {df['InvoiceNo'].nunique():,} | Products: {df['Description'].nunique():,}")

# ── Build basket matrix ──────────────────────────────────────────────────────
basket = (
    df.groupby(["InvoiceNo", "Description"])["Quantity"]
    .sum()
    .unstack(fill_value=0)
    .clip(upper=1)  # binarise: bought or not
)

print(f"Basket matrix: {basket.shape[0]} invoices × {basket.shape[1]} products")

# ── Association rules with mlxtend ───────────────────────────────────────────
try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules

    # Use top-100 products for speed
    top_products = (basket.sum().nlargest(100)).index
    basket_top = basket[top_products]

    freq_items = fpgrowth(basket_top, min_support=0.02, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
    rules = rules.sort_values("lift", ascending=False)

    print(f"\nTop 10 rules by lift:")
    cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    print(rules[cols].head(10).to_string(index=False))

    # ── Plot top rules ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    top_rules = rules.head(20)
    ax.scatter(top_rules["support"], top_rules["confidence"],
               c=top_rules["lift"], cmap="YlOrRd", s=60, edgecolors="k", linewidths=0.5)
    plt.colorbar(ax.collections[0], label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Association rules (top 20 by lift)")
    plt.tight_layout()
    plt.savefig(OUT / "association_rules.png", dpi=120)
    plt.close()
    print("Plot saved.")

except ImportError:
    print("mlxtend not installed. Run: pip install mlxtend")
    print("Skipping rule mining — run the install and re-execute.")

# ── RFM segmentation ─────────────────────────────────────────────────────────
if "InvoiceDate" in df.columns and "CustomerID" in df.columns:
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot = df["InvoiceDate"].max()
    if "UnitPrice" in df.columns:
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]
        rfm = df.groupby("CustomerID").agg(
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        rfm = rfm[rfm["Monetary"] > 0]

        # Rank into 1-5 quintiles (5 = best)
        for col, ascending in [("Recency", True), ("Frequency", False), ("Monetary", False)]:
            rfm[f"{col[0]}_score"] = pd.qcut(rfm[col], 5, labels=[5,4,3,2,1] if ascending else [1,2,3,4,5])

        rfm["RFM_score"] = (rfm["R_score"].astype(int)
                           + rfm["F_score"].astype(int)
                           + rfm["M_score"].astype(int))

        print(f"\nRFM summary (n={len(rfm):,} customers):")
        print(rfm[["Recency","Frequency","Monetary","RFM_score"]].describe().round(1).to_string())
```

---

## Cross-references

- [07](07-time-series-forecasting.md) — demand forecasting methods
- [08](08-recommendation-systems.md) — product recommendation
- [06](06-unsupervised-learning.md) — customer segmentation
- [35](35-martech.md) — MarTech overlap (CRM, campaign optimisation)
- [12](12-optimization.md) — pricing and inventory optimisation
