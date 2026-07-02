"""
CASE STUDY - Retention offers: who should we actually target?
=============================================================
A worked, end-to-end study that goes framing -> data -> model -> CAUSAL uplift ->
a decision, on a reproducible simulated business scenario.

THE SCENARIO (framing)
----------------------
A subscription business loses customers each month. Marketing can send a
retention offer (a discount) but it costs money and can't be sent to everyone.
Leadership asks: "Use the data to reduce churn." The naive reading - "predict who
will churn and target them" - is WRONG, and this study shows why with numbers.

The right question is causal: for whom does the offer *change* the outcome? That
is the customer's UPLIFT (churn probability without the offer minus with it). A
high-risk customer who will churn regardless (bad-service driven) is a wasted
offer; a price-sensitive "persuadable" is where the money is. Some loyal
customers are "sleeping dogs" - the offer reminds them to reconsider and slightly
*raises* churn. Targeting by risk cannot tell these apart; targeting by uplift can.

We ran a pilot RCT: a random 50% of customers received the offer. Because
assignment was randomised, we can estimate uplift honestly and evaluate any
targeting policy without confounding (see docs/09).

ECONOMICS
---------
  offer cost      = £8 per offer
  saved customer  = £150 (retained customer lifetime value)
  -> treat a customer only if uplift * 150 > 8, i.e. uplift > 0.053.

Run: python run.py   ->   metrics.md, business_case.md, uplift_analysis.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

RNG = np.random.default_rng(2024)
OFFER_COST, SAVED_VALUE = 8.0, 150.0
TREAT_THRESHOLD = OFFER_COST / SAVED_VALUE          # uplift break-even ~0.053

# ---------------------------------------------------------------- 1. DATA (simulated RCT)
def sigmoid(z): return 1 / (1 + np.exp(-z))

def simulate(n=24000):
    tenure   = RNG.integers(1, 72, n)               # months as customer
    charges  = RNG.normal(70, 25, n).clip(20, 140)  # monthly £
    tickets  = RNG.poisson(1.2, n)                  # support tickets (dissatisfaction)
    m2m      = RNG.binomial(1, 0.55, n)             # month-to-month contract
    # Baseline churn (no offer): driven by dissatisfaction, price, short tenure, m2m.
    base_logit = (-1.0 + 0.55*tickets + 0.014*(charges-70) - 0.02*tenure + 0.8*m2m)
    p_control = sigmoid(base_logit)
    # UPLIFT (churn reduction if treated): large for price-sensitive persuadables
    # (high charges + m2m), ~0 when churn is service-driven (tickets), NEGATIVE for
    # loyal long-tenure customers (sleeping dogs).
    persuadable = m2m * (charges > 75)
    uplift = (0.32*persuadable                            # persuadable price-sensitive
              - 0.015*tickets                             # service issues: offer won't fix
              - 0.06*(tenure > 48))                       # sleeping dogs (offer backfires)
    p_treat = (p_control - uplift).clip(0.01, 0.99)
    treat = RNG.binomial(1, 0.5, n)                       # randomised pilot
    p_actual = np.where(treat == 1, p_treat, p_control)
    churn = RNG.binomial(1, p_actual)
    return pd.DataFrame(dict(tenure=tenure, charges=charges, tickets=tickets,
                             m2m=m2m, treat=treat, churn=churn,
                             true_uplift=p_control - p_treat))    # kept only for validation

df = simulate()
FEATS = ["tenure", "charges", "tickets", "m2m"]
print(f"Customers: {len(df)} | overall churn: {df.churn.mean():.1%} | "
      f"treated: {df.treat.mean():.0%}")
print(f"Pilot ATE (randomised): control churn {df.loc[df.treat==0,'churn'].mean():.1%} "
      f"vs treated {df.loc[df.treat==1,'churn'].mean():.1%}")

tr, te = train_test_split(df, test_size=0.4, random_state=0, stratify=df.churn)

# ---------------------------------------------------------------- 2. RISK model (the naive approach)
risk_model = HistGradientBoostingClassifier(random_state=0).fit(tr[FEATS], tr.churn)
te = te.assign(risk=risk_model.predict_proba(te[FEATS])[:, 1])

# ---------------------------------------------------------------- 3. UPLIFT model (T-learner)
# Two models: churn among treated, churn among control. Uplift = P(churn|control) - P(churn|treat).
m_treat = HistGradientBoostingClassifier(random_state=0).fit(
    tr.loc[tr.treat == 1, FEATS], tr.loc[tr.treat == 1, "churn"])
m_ctrl = HistGradientBoostingClassifier(random_state=0).fit(
    tr.loc[tr.treat == 0, FEATS], tr.loc[tr.treat == 0, "churn"])
te = te.assign(pred_uplift=(m_ctrl.predict_proba(te[FEATS])[:, 1]
                            - m_treat.predict_proba(te[FEATS])[:, 1]))

# ---------------------------------------------------------------- 4. POLICY EVALUATION
# Because treatment was randomised, the incremental effect of "treat this set" is
# unbiased: within the set, (control churn rate) - (treated churn rate).
def realised_uplift(mask):
    s = te[mask]
    t, c = s[s.treat == 1], s[s.treat == 0]
    if len(t) == 0 or len(c) == 0:
        return 0.0
    return c.churn.mean() - t.churn.mean()

def policy_value(score_col, budget_frac):
    """Net £ value of treating the top `budget_frac` customers ranked by score_col."""
    k = int(len(te) * budget_frac)
    targeted = te.nlargest(k, score_col).index
    mask = te.index.isin(targeted)
    up = realised_uplift(mask)                 # avg churn reduction among targeted
    saved = up * k * SAVED_VALUE               # customers saved * value
    cost = k * OFFER_COST
    return saved - cost

budgets = np.linspace(0.05, 1.0, 20)
val_uplift = [policy_value("pred_uplift", b) for b in budgets]
val_risk   = [policy_value("risk", b) for b in budgets]
val_random = [realised_uplift(te.index.isin(te.sample(frac=b, random_state=1).index))
              * int(len(te)*b) * SAVED_VALUE - int(len(te)*b)*OFFER_COST for b in budgets]

best_i = int(np.argmax(val_uplift))
best_budget, best_value = budgets[best_i], val_uplift[best_i]
value_treat_all = policy_value("pred_uplift", 1.0)

print("\n=== Net value (£) by targeting policy and budget ===")
comp = pd.DataFrame({"budget_%": (budgets*100).round(0).astype(int),
                     "target_by_uplift": val_uplift,
                     "target_by_risk": val_risk, "random": val_random}).round(0)
print(comp.iloc[[0, 4, 9, 14, 19]].to_string(index=False))
print(f"\nOptimal uplift policy: treat top {best_budget:.0%} -> net £{best_value:,.0f}")
print(f"Treat everyone: net £{value_treat_all:,.0f}  |  "
      f"Best risk-targeting: net £{max(val_risk):,.0f}")

# ---------------------------------------------------------------- 5. does model rank uplift correctly?
# Validation only possible because this is simulated: correlation of predicted vs true uplift.
rank_corr = np.corrcoef(te.pred_uplift, te.true_uplift)[0, 1]
print(f"\nUplift model ranking quality (corr with true uplift): {rank_corr:.3f}")

# ---------------------------------------------------------------- 6. plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(budgets*100, np.array(val_uplift)/1000, "o-", label="target by UPLIFT")
ax[0].plot(budgets*100, np.array(val_risk)/1000, "s-", label="target by RISK (naive)")
ax[0].plot(budgets*100, np.array(val_random)/1000, "--", color="grey", label="random")
ax[0].axhline(0, color="k", lw=.8); ax[0].axvline(best_budget*100, color="C0", ls=":", lw=1)
ax[0].set(xlabel="% of customers offered", ylabel="net value (£000s)",
          title="Value curve: uplift-targeting dominates risk-targeting")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].scatter(te.risk, te.pred_uplift, s=6, alpha=.3)
ax[1].axhline(TREAT_THRESHOLD, color="r", ls="--", lw=1, label=f"break-even uplift ({TREAT_THRESHOLD:.3f})")
ax[1].set(xlabel="predicted churn RISK", ylabel="predicted UPLIFT",
          title="Risk ≠ Uplift: high-risk customers are often unpersuadable")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("uplift_analysis.png", dpi=120); plt.close(fig)

# ---------------------------------------------------------------- 7. write up
with open("metrics.md", "w") as f:
    f.write("# Case study results\n\n")
    f.write(comp.iloc[[0,4,9,14,19]].round(0).to_markdown(index=False) + "\n\n")
    f.write(f"- Optimal policy: **treat top {best_budget:.0%} by predicted uplift** "
            f"-> net **£{best_value:,.0f}** on this test cohort.\n")
    f.write(f"- Treat-everyone: £{value_treat_all:,.0f}. "
            f"Best risk-targeting: £{max(val_risk):,.0f}.\n")
    f.write(f"- Uplift-model ranking corr with ground truth: {rank_corr:.3f}.\n")

with open("business_case.md", "w") as f:
    f.write(f"""# Retention offers - recommendation

## Bottom line
Do **not** target the customers most likely to churn. Target the customers whose
behaviour the offer actually *changes*. On the pilot cohort, ranking by predicted
**uplift** and offering to the top **{best_budget:.0%}** yields **£{best_value:,.0f}**
net value - versus **£{max(val_risk):,.0f}** for the intuitive "target the
high-risk" policy and **£{value_treat_all:,.0f}** for offering to everyone.

## Why the naive approach loses money
Churn risk and persuadability are different things. Many high-risk customers are
leaving because of unresolved service problems - a discount doesn't fix that, so
the offer is wasted. A minority of loyal customers are "sleeping dogs": the offer
reminds them to reconsider and slightly *increases* churn. The right-hand panel of
`uplift_analysis.png` shows risk and uplift are only weakly related.

## The decision rule
Offer to a customer when `predicted_uplift * £{SAVED_VALUE:.0f} > £{OFFER_COST:.0f}`,
i.e. predicted uplift above **{TREAT_THRESHOLD:.3f}**. This maximises expected net
value per customer and generalises to any budget.

## Caveats (what would change this)
- The uplift estimates rest on the pilot being a valid RCT (random assignment, no
  interference). Re-validate if the pilot was compromised.
- Effects can drift; re-estimate periodically and consider a holdout to keep
  measuring true incremental impact in production (docs/14).
- Simulated data here; on real data, add confidence intervals (bootstrap) around
  each policy's value before committing budget.
""")
print("\nSaved: metrics.md, business_case.md, uplift_analysis.png")
