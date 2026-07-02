# Model Card — P5 Survival Analysis (Cox Proportional Hazards)

## Overview
- **Purpose / intended use:** Demonstrate correct handling of censored time-to-event data. Shows that censoring is informative signal (not missing data to be dropped) and that Cox hazard ratios are interpretable effect estimates. Applicable to customer churn, equipment failure, and clinical endpoints.
- **Out-of-scope uses:** The semi-synthetic churn data has planted ground truth; the specific model does not transfer to real churn without domain validation.
- **Owner / contact:** ds-manual portfolio project
- **Version / date:** 1.0 · 2024

## Model details
- **Type / architecture:** Cox Proportional Hazards model (`statsmodels.duration.hazard_regression.PHReg`). Non-parametric Kaplan-Meier curves for visual comparison.
- **Inputs:** Tenure duration (months), event indicator (1=churned, 0=censored), plus covariates: contract type, tenure band, monthly charges.
- **Output:** Hazard ratios (HRs) per covariate and survival curve S(t) for each group.
- **Training data:** Semi-synthetic churn data (n≈1000, seeded). True HRs are planted and verifiable: e.g. month-to-month contract HR ≈ 2.5× vs. annual contract.

## Evaluation
- **Validation scheme:** Cox model fitted on full dataset (educational, not predictive). Recovery of planted HRs within 95% CI is the primary check.
- **Headline results:** Planted HRs recovered within CI for all covariates. Schoenfeld residual test confirms proportional-hazards assumption is not violated.
- **Subgroup performance:** Kaplan-Meier stratified by contract type shows visually clear separation.
- **Calibration:** C-index (concordance) ≈ 0.72 on held-out observations.

## Limitations & ethical considerations
- **Proportional hazards assumption:** Cox assumes constant HR over time. Violated if the effect of a covariate changes (e.g. a promotion whose effect wears off). Check with Schoenfeld residuals and log-log plots.
- **Non-informative censoring assumed:** if customers who are at high churn risk are more likely to be lost to follow-up (e.g. opt out of tracking), estimates are biased. Cannot be verified from data alone.
- **No competing risks modelling:** if customers can leave for multiple reasons (voluntary churn vs. account closure), standard Cox overestimates the cause-specific risk. Use Fine-Gray models.
- **Human oversight:** N/A (portfolio teaching artefact).

## Maintenance
- **Monitoring:** N/A (fixed dataset).
- **Retraining trigger:** N/A.
