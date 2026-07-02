# Model Card — Breast-cancer malignancy classifier (Project 1)

## Overview
- **Purpose / intended use:** Educational reference implementation of a leak-free
  supervised classification workflow. Ranks tumours by malignancy probability.
- **Out-of-scope uses:** NOT for clinical decision-making. Trained on one historical
  research dataset; no clinical validation, regulatory clearance, or prospective
  testing.
- **Owner / contact:** Lorenzo Amabili
- **Version / date:** 0.1, 2026

## Model details
- **Type:** L2-regularised logistic regression inside a `StandardScaler` → classifier
  `Pipeline` (selected over Random Forest and HistGradientBoosting by CV PR-AUC).
- **Inputs:** 30 cell-nucleus morphology features (radius, texture, concavity, …).
- **Output:** Probability of malignancy (positive class = malignant).
- **Training data:** [p1_breast_cancer_data_card.md](p1_breast_cancer_data_card.md).

## Evaluation
- **Validation scheme:** Stratified 5-fold CV on 75% training split; single held-out
  25% test set touched once.
- **Headline metrics:** Test ROC-AUC ≈ 0.996, PR-AUC ≈ 0.994, Brier ≈ 0.022. Baseline
  (predict majority) PR-AUC ≈ base rate 0.37.
- **Subgroup performance:** Dataset carries no demographic attributes, so subgroup
  fairness cannot be assessed — itself a documented limitation.
- **Calibration:** Reliability curve inspected (out-of-fold); probabilities usable
  as probabilities.

## Limitations & ethical considerations
- Single-source, historical data; real deployment would face scanner/site shift.
- Small n (569); intervals are wide despite high point metrics.
- High-stakes domain: a false negative is far costlier than a false positive, so any
  real use would set the threshold from clinical cost, not 0.5.

## Maintenance
- **Monitoring:** N/A (teaching artefact).
- **Retraining trigger:** N/A.
