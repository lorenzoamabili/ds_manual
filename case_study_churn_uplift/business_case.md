# Retention offers - recommendation

## Bottom line
Do **not** target the customers most likely to churn. Target the customers whose
behaviour the offer actually *changes*. On the pilot cohort, ranking by predicted
**uplift** and offering to the top **30%** yields **£55,664**
net value - versus **£6,602** for the intuitive "target the
high-risk" policy and **£-33,936** for offering to everyone.

## Why the naive approach loses money
Churn risk and persuadability are different things. Many high-risk customers are
leaving because of unresolved service problems - a discount doesn't fix that, so
the offer is wasted. A minority of loyal customers are "sleeping dogs": the offer
reminds them to reconsider and slightly *increases* churn. The right-hand panel of
`uplift_analysis.png` shows risk and uplift are only weakly related.

## The decision rule
Offer to a customer when `predicted_uplift * £150 > £8`,
i.e. predicted uplift above **0.053**. This maximises expected net
value per customer and generalises to any budget.

## Caveats (what would change this)
- The uplift estimates rest on the pilot being a valid RCT (random assignment, no
  interference). Re-validate if the pilot was compromised.
- Effects can drift; re-estimate periodically and consider a holdout to keep
  measuring true incremental impact in production (docs/14).
- Simulated data here; on real data, add confidence intervals (bootstrap) around
  each policy's value before committing budget.
