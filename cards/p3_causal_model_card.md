# Model Card — P3 Causal Inference (Doubly-Robust ATE Estimator)

## Overview
- **Purpose / intended use:** Estimate the average treatment effect (ATE) of a binary intervention from observational data using propensity weighting, matching, and doubly-robust (AIPW) estimation. Demonstrates that naive comparison gives wrong-sign estimates when confounding is present.
- **Out-of-scope uses:** Not for production causal decision-making without domain validation. The planted ground truth (ATE=3.0) is only possible in semi-synthetic data.
- **Owner / contact:** ds-manual portfolio project
- **Version / date:** 1.0 · 2024

## Model details
- **Type / architecture:** Three estimators compared — (1) Naive difference in means, (2) IPW (logistic propensity model), (3) Doubly-Robust AIPW (outcome model + propensity model). Propensity model: logistic regression. Outcome model: linear regression.
- **Inputs:** 5 continuous covariates (X1-X5), binary treatment indicator.
- **Output:** Estimated ATE with bootstrap 95% CI and standardised mean difference (SMD) for covariate balance.
- **Training data:** Semi-synthetic (seeded, known ATE=3.0). Confounding is strong enough to flip the naive estimate sign (naive ≈ −1.5).

## Evaluation
- **Validation scheme:** Ground truth ATE is planted; evaluate by how close each estimator gets. SMD before/after IPW weighting demonstrates covariate balance.
- **Headline results:** Naive estimate ≈ −1.5 (wrong sign); IPW ≈ 3.0; DR ≈ 3.16 ± 0.3 (95% CI includes truth). SMD reduced from ~0.4 to <0.03 after weighting.
- **Subgroup performance:** ATT (average treatment effect on the treated) reported separately.
- **Calibration:** Not applicable (point estimate + interval).

## Limitations & ethical considerations
- **Ignorability assumption:** all estimators require no unmeasured confounders. This is guaranteed by construction in P3 but must be argued from domain knowledge in real applications.
- **Semi-synthetic data:** the lesson holds under this data-generating process; real data may require additional diagnostics (E-values, sensitivity analysis).
- **Overlap / common support:** if propensity scores are near 0 or 1, IPW weights blow up. P3 clips them (see `dsmanual.clip_propensity`).
- **Human oversight:** causal estimates should always be reviewed by domain experts before informing decisions.

## Maintenance
- **Monitoring:** N/A (historical fixed dataset).
- **Retraining trigger:** N/A.
