# Model Card — Screening classifier with fairness audit (Project 6)

## Overview
- **Purpose:** Demonstrate a fairness audit — how a model that omits the protected
  attribute can still cause disparate impact, and how to mitigate it.
- **Out-of-scope:** Illustrative only; not a deployable screening tool.

## Model details
- **Type:** Logistic regression on a single (deliberately bias-affected) score.
  Protected attribute is intentionally excluded from features.
- **Output:** probability of "qualified"; decision via threshold(s).

## Evaluation
- **Scheme:** 60/40 split; metrics computed per protected group.
- **Fairness assessment (the point of the project):**
  - Single-threshold model: selection-rate ratio 0.58 → **fails the 80% rule**;
    TPR 0.58 (group 1) vs 0.84 (group 0).
  - After group-specific thresholds (equal opportunity): ratio ≈ 1.03, TPRs
    matched, overall accuracy 0.825 → 0.833.

## Limitations & ethical considerations
- Group-specific thresholds are legally constrained in some jurisdictions; the
  choice of fairness criterion is a policy decision, not a technical default
  ([docs/19](../docs/19-responsible-ai-and-fairness.md)).
- Simulated data with equal true base rates by construction.

## Maintenance
- Re-audit disparate impact and subgroup TPR/FPR on every retrain.
