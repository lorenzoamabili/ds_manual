# 18 · SQL & Data Engineering Literacy

The unglamorous truth: most of a data scientist's real-world hours are spent
getting data into a usable shape, and most of that happens in **SQL** against a
warehouse — not in pandas. Fluency here is what companies actually screen for, and
it's where the modeling-heavy curricula leave the biggest gap.

## SQL beyond SELECT

The dividing line between a beginner and a professional is comfort with:

- **Window functions** — the single most important intermediate SQL skill.
  `ROW_NUMBER()`, `RANK()`, `LAG()/LEAD()`, and running aggregates
  (`SUM() OVER (PARTITION BY ... ORDER BY ...)`). They express "per-group
  running/ranked/previous" logic that would otherwise need ugly self-joins —
  deduplication, sessionisation, period-over-period change, first/last event.
- **CTEs (`WITH`)** — decompose a gnarly query into named, readable steps. Prefer
  chained CTEs over nested subqueries; they read top-to-bottom like a pipeline.
- **The join zoo** — inner/left/anti/semi joins, and knowing that a careless join
  against a non-unique key **fans out** rows and silently corrupts every
  downstream aggregate. Always know the grain (one row per _what_?) on both sides.
- **Aggregation subtleties** — `GROUP BY` grain, `HAVING` vs `WHERE`, `COUNT(*)` vs
  `COUNT(col)` vs `COUNT(DISTINCT col)`, and how NULLs behave in each.

## Data modelling
- **Dimensional modelling (star schema)** — fact tables (events/measures, e.g.
  orders) surrounded by dimension tables (descriptive context, e.g. customer,
  product, date). The backbone of analytics warehouses; understand grain, facts,
  and slowly-changing dimensions (SCDs).
- **Normalisation vs. denormalisation** — normalise to avoid update anomalies in
  transactional systems (OLTP); denormalise for fast reads in analytics (OLAP).
  Know which world you're in.
- **The grain of a table** is the most important thing to state and preserve. Most
  data bugs are grain bugs.

## The modern data stack (ELT)
The industry moved from ETL to **ELT**: load raw data into the warehouse first,
then transform in-warehouse with SQL.

- **Warehouses / engines** — Snowflake, BigQuery, Redshift, Databricks; columnar,
  massively parallel. Understand partitioning and clustering (they make or break
  query cost/speed).
- **dbt** — transformations as version-controlled, tested, documented SQL models
  with dependency graphs. It brought software engineering (tests, CI, lineage,
  modularity) to the analytics layer; increasingly table-stakes.
- **Orchestration** — Airflow, Dagster, Prefect: schedule and monitor pipelines as
  DAGs, with retries and alerting.
- **Ingestion** — Fivetran/Airbyte (managed connectors), Kafka/streaming for
  real-time.
- **Data quality & contracts** — tests on freshness, uniqueness, nullity, accepted
  values (dbt tests, Great Expectations); data contracts to stop upstream schema
  changes from silently breaking models.

## Performance intuition
- Filter early, project only needed columns (columnar stores charge by column read).
- Beware `SELECT *` and accidental cross joins on big tables.
- Push work into the warehouse (set-based SQL) rather than pulling millions of rows
  into pandas to loop over them.
- Read the query plan when something's slow; understand what a partition prune or a
  broadcast vs. shuffle join is doing.

## What "good enough" looks like for a DS
You don't need to be a data engineer, but you should be able to: write correct
windowed, multi-CTE SQL against a star schema without fanning out; model a clean
table at a stated grain; contribute a tested dbt model; and reason about why a
query is slow or expensive. That level of literacy removes your dependence on
others for the 90% of the job that is data wrangling.

---

## Python / SQL example — window functions and the fan-out trap

```python
"""
Demonstrates the window-function patterns every DS needs, and the
fan-out grain bug that silently corrupts aggregates.
Uses pandas to simulate the SQL patterns (no DB required).
"""
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# ── Simulated orders table (grain: 1 row per order) ─────────────────────────
orders = pd.DataFrame({
    "order_id":    range(1000),
    "customer_id": rng.integers(1, 201, 1000),
    "order_date":  pd.date_range("2023-01-01", periods=1000, freq="8h"),
    "revenue":     rng.exponential(50, 1000).round(2),
})

# ── 1. Window function: running total per customer ────────────────────────────
# SQL equivalent:
#   SUM(revenue) OVER (PARTITION BY customer_id ORDER BY order_date)
orders = orders.sort_values(["customer_id","order_date"])
orders["running_total"] = orders.groupby("customer_id")["revenue"].cumsum()

# ── 2. Window function: days since previous order per customer ────────────────
# SQL equivalent:
#   DATEDIFF(order_date, LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date))
orders["prev_order_date"] = orders.groupby("customer_id")["order_date"].shift(1)
orders["days_since_prev"] = (orders["order_date"] - orders["prev_order_date"]).dt.days

# ── 3. Window function: customer rank by total revenue ────────────────────────
cust_total = orders.groupby("customer_id")["revenue"].sum().reset_index()
cust_total["rank"] = cust_total["revenue"].rank(ascending=False, method="dense").astype(int)
print("Top 5 customers by revenue:")
print(cust_total.nsmallest(5, "rank").to_string(index=False))

# ── 4. The fan-out bug ────────────────────────────────────────────────────────
# Suppose we have a promotions table: 1 order can have MULTIPLE promotions
promos = pd.DataFrame({
    "order_id":    rng.integers(0, 1000, 400),  # non-unique key
    "promo_code":  rng.choice(["SAVE10","SAVE20","FREE_SHIP"], 400),
    "discount":    rng.uniform(1, 20, 400).round(2),
})

# WRONG: naive join fans out orders for rows with multiple promos
wrong_join = orders.merge(promos, on="order_id", how="left")
wrong_revenue = wrong_join["revenue"].sum()
true_revenue  = orders["revenue"].sum()
print(f"\nFan-out bug:")
print(f"  True total revenue:  £{true_revenue:,.0f}")
print(f"  After naive join:    £{wrong_revenue:,.0f}  ← inflated by {wrong_revenue/true_revenue:.1f}x")
print(f"  Orders fanned out to {len(wrong_join)} rows from {len(orders)}")

# CORRECT: aggregate promos first (one row per order), then join
promo_agg = promos.groupby("order_id").agg(
    n_promos=("promo_code","count"),
    total_discount=("discount","sum"),
).reset_index()
correct_join = orders.merge(promo_agg, on="order_id", how="left")
print(f"  After correct join:  £{correct_join['revenue'].sum():,.0f}  ← correct grain preserved")
print(f"\nRule: aggregate to the target grain BEFORE joining.")
```

---

## Cross-references

- [03](03-data-and-feature-engineering.md) — feature engineering (including lag features from SQL)
- [01](01-lifecycle-and-reproducibility.md) — data versioning and pipeline reproducibility
- [14](14-mlops-and-productionization.md) — feature stores and training/serving skew
