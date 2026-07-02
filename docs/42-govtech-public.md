# 42 · GovTech & Public Sector

## Signature problems

| Problem | Approach |
|---------|----------|
| Benefit fraud detection | Classification, anomaly detection (see [13](13-anomaly-detection.md)) |
| Tax gap estimation | Regression, stratified sampling |
| Social care risk scoring | Survival analysis, classification — high ethical stakes |
| Policy impact evaluation | Causal inference: DiD, RDD, IV (see [09](09-causal-inference-and-experimentation.md)) |
| Public health surveillance | Anomaly detection on syndromic data, spatial clustering |
| Smart city traffic | Spatial-temporal forecasting (see [07](07-time-series-forecasting.md), [22](22-geospatial.md)) |
| Document processing (benefits, planning) | NLP + document classification (see [10](10-nlp-and-llms.md)) |
| Electoral boundary analysis | Geospatial clustering (see [22](22-geospatial.md)) |

## Domain characteristics

- **Highest-stakes fairness requirements**: government decisions affect citizens' rights (benefits, housing, custody). Disparate impact is a legal issue, not just an ethical one. See [19](19-responsible-ai-and-fairness.md).
- **Administrative data quality**: government data is often collected for administrative, not analytical, purposes. Missing not at random is endemic (people who don't engage with services have sparse records).
- **Right to explanation**: GDPR Article 22 and UK AI Act require explainability for automated decisions affecting individuals. Black-box models face regulatory barriers.
- **Procurement and transparency constraints**: public sector ML must often be auditable by third parties. Open-source, documented, reproducible pipelines matter more than marginal accuracy gains.
- **Long policy cycles**: interventions take years to evaluate. Short-term proxy metrics (application rates, processing times) are necessary but risk Goodhart effects.

## Causal policy evaluation: Regression Discontinuity Design

```python
"""
Regression Discontinuity Design (RDD):
Estimate the causal effect of a policy applied at a sharp threshold.
Example: job training programme for workers with earnings < £20,000.
Workers just below/above the threshold are otherwise comparable.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
n = 2000

# Running variable: annual earnings (centred at threshold)
earnings = rng.uniform(-10000, 10000, n)  # deviation from £20k threshold
treatment = (earnings < 0).astype(int)     # below threshold → treated

# Outcome: re-employment within 12 months
# True ATE = +0.15 (15 pp lift from programme)
re_employed = (
    0.5                               # base rate
    + 0.003 * earnings / 1000        # smooth trend in earnings
    + 0.15  * treatment               # true causal effect
    + rng.normal(0, 0.1, n)
).clip(0, 1)
# Convert to binary outcome
y = (re_employed > rng.uniform(0, 1, n)).astype(int)

df = pd.DataFrame({"earnings": earnings, "treatment": treatment, "y": y})

# RDD estimate: fit separate linear trend on each side of cutoff
bandwidth = 3000   # use only observations within £3k of cutoff
near = df[np.abs(df["earnings"]) < bandwidth].copy()

def rdd_fit(data, side):
    mask = (data["earnings"] < 0) if side == "left" else (data["earnings"] >= 0)
    d = data[mask]
    m = LinearRegression().fit(d[["earnings"]], d["y"])
    return m.predict([[0]])[0]

y_left  = rdd_fit(near, "left")   # predicted outcome just below cutoff (treated)
y_right = rdd_fit(near, "right")  # predicted outcome just above cutoff (control)
rdd_estimate = y_left - y_right

print(f"RDD estimate of programme effect: {rdd_estimate:+.3f}")
print(f"True planted effect:              +0.150")
print(f"Bandwidth: ±£{bandwidth:,.0f} from threshold")

# Visualisation
fig, ax = plt.subplots(figsize=(9, 4))
bins = np.linspace(-10000, 10000, 40)
for sign, color, label in [(-1,"#3498db","Treated (< £20k)"),
                             (1,"#e74c3c","Control (≥ £20k)")]:
    mask = near["earnings"].apply(np.sign) == sign
    x_b = pd.cut(near.loc[mask, "earnings"], bins=20).apply(lambda i: i.mid)
    y_b = near.loc[mask, "y"].groupby(x_b).mean()
    ax.scatter(y_b.index, y_b.values, color=color, s=40, label=label, alpha=0.8)
ax.axvline(0, color="black", ls="--", lw=1, label="Policy threshold (£20k)")
ax.set(xlabel="Earnings deviation from threshold (£)", ylabel="Re-employment rate",
       title=f"RDD: estimated programme effect = {rdd_estimate:+.3f}")
ax.legend(); plt.tight_layout()
plt.savefig("rdd_policy.png", dpi=120); plt.close()
print("Plot saved: rdd_policy.png")
```

## Fairness audit on public sector scoring

```python
"""
Mandatory fairness check for any government-facing model.
Disparate impact ratio and equalised-odds gap by demographic group.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
n = 3000

# Synthetic benefit eligibility dataset
age        = rng.integers(18, 65, n)
income_k   = rng.lognormal(3.0, 0.5, n)  # £k/yr
deprivation= rng.integers(1, 6, n)        # 1=least deprived, 5=most
protected  = rng.binomial(1, 0.3, n)      # protected group (e.g. ethnicity)

# Ground truth eligibility (no group effect in true model)
logit = -2 + 0.05*(65-age) + 0.3*deprivation - 0.02*income_k
y_true = (logit + rng.logistic(0, 1, n) > 0).astype(int)

X = np.column_stack([age, income_k, deprivation])
scaler = StandardScaler().fit(X)
model  = LogisticRegression(random_state=42).fit(scaler.transform(X), y_true)
y_pred = model.predict(scaler.transform(X))

# Disparate Impact Ratio (should be > 0.80 per 80% rule)
approval_A = y_pred[protected == 0].mean()
approval_B = y_pred[protected == 1].mean()
dir_ratio  = min(approval_A, approval_B) / max(approval_A, approval_B)

# Equalised odds gap (TPR difference between groups)
tpr_A = ((y_pred[protected==0]==1) & (y_true[protected==0]==1)).sum() / (y_true[protected==0]==1).sum()
tpr_B = ((y_pred[protected==1]==1) & (y_true[protected==1]==1)).sum() / (y_true[protected==1]==1).sum()

print(f"Approval rate — Group A: {approval_A:.2%}  Group B: {approval_B:.2%}")
print(f"Disparate Impact Ratio: {dir_ratio:.3f}  {'PASS (≥0.80)' if dir_ratio >= 0.8 else 'FAIL (<0.80)'}")
print(f"TPR gap (equalised odds): {abs(tpr_A - tpr_B):.3f}  (target < 0.05)")
print("\nGovernment ML must pass both checks before deployment.")
print("See doc 19 for post-processing threshold adjustment if tests fail.")
```

## Cross-references

- [09](09-causal-inference-and-experimentation.md) — DiD, RDD, IV for policy evaluation
- [19](19-responsible-ai-and-fairness.md) — fairness, GDPR, EU AI Act requirements
- [13](13-anomaly-detection.md) — fraud detection in public sector
- [case studies](case_studies/cs-hrtech.md) — IBM attrition (Goodhart lessons apply equally here)
