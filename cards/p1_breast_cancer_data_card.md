# Data Card — Wisconsin Breast Cancer (Diagnostic)

## Overview
- **What it is:** 569 fine-needle-aspirate images of breast masses, reduced to 30
  numeric nucleus-morphology features, labelled benign/malignant.
- **Source & licence:** UCI ML Repository / bundled with scikit-learn; public,
  research use.
- **Collection:** University of Wisconsin, early 1990s.

## Composition
- **Rows / grain:** one row per tumour sample.
- **Features:** 30 continuous (mean/SE/worst of 10 morphology measures).
- **Target:** diagnosis (malignant = positive in Project 1). Known at labelling
  time; no temporal leakage risk.
- **Size & splits:** 569 rows; ~37% malignant; stratified 75/25 train/test with
  5-fold CV inside training.

## Quality & caveats
- **Missingness:** none.
- **Representation bias:** single institution, single era; not representative of
  modern imaging or diverse populations.
- **Sensitive attributes:** none present (no demographics) → fairness un-auditable.

## Appropriate use
- **Suitable for:** demonstrating supervised-ML methodology, pipelines, evaluation.
- **Not suitable for:** any clinical or diagnostic use.
