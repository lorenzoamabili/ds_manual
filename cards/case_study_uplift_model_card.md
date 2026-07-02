# Model Card — Retention uplift model (Case study)

## Overview
- **Purpose:** Estimate each customer's *uplift* (churn reduction if given a
  retention offer) to target offers by persuadability, not by risk.
- **Out-of-scope:** Not a churn-risk predictor for other uses; the uplift estimates
  are only valid for the specific offer and population of the pilot.
- **Version / date:** 0.1, 2026 (simulated demonstration data).

## Model details
- **Type:** T-learner — two HistGradientBoosting classifiers (churn|treated,
  churn|control); uplift = P(churn|control) − P(churn|treated).
- **Inputs:** tenure, monthly charges, support tickets, contract type.
- **Output:** predicted per-customer uplift; a treat/don't-treat decision via the
  break-even rule uplift × value > cost.

## Evaluation
- **Scheme:** 60/40 train/test; policies scored on the test set using the pilot's
  randomised assignment (unbiased incremental effect).
- **Headline:** Targeting top-30% by uplift → +£55.7k net vs +£6.6k for risk-based
  targeting and −£33.9k for treating everyone (on the test cohort).
- **Validity check:** predicted-vs-true uplift rank correlation ≈ 0.65 (possible
  only because data is simulated).

## Limitations & ethical considerations
- Rests on the pilot being a valid RCT (random assignment, no interference).
- Effects drift; production use needs a permanent holdout to keep measuring true
  incremental impact ([docs/14](../docs/14-mlops-and-productionization.md)).
- Simulated data — real deployment needs bootstrap CIs on each policy's value.

## Maintenance
- **Monitoring:** track realised uplift on the holdout; watch for offer fatigue.
- **Retraining trigger:** quarterly, or on a material shift in churn base rate.
