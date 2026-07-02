# Model Card — P7 Anomaly Detector (Isolation Forest)

## Overview
- **Purpose / intended use:** Detect rare anomalies (fraud, equipment faults) in tabular data where labelled examples are scarce or absent.
- **Out-of-scope uses:** Not a substitute for supervised classification when abundant labels exist; not calibrated for probability output.
- **Owner / contact:** ds-manual portfolio project
- **Version / date:** 1.0 · 2024

## Model details
- **Type / architecture:** Isolation Forest (200 trees) + Local Outlier Factor comparison; GBM as supervised ceiling.
- **Inputs:** 20 continuous features from a synthetic fraud-like dataset (seeded, reproducible).
- **Output:** Anomaly score (higher = more anomalous); binary flag at contamination-tuned threshold.
- **Training data:** Synthetic `make_classification` dataset (n=20 000, 0.5% positive rate, seed=42). See [data card](#data) below.

## Evaluation
- **Validation scheme:** Train/test split (70/30, stratified). Unsupervised models fit on train only.
- **Headline metrics:** See `metrics.md` for current run. GBM PR-AUC ≈ 0.8+; Isolation Forest ≈ 0.4–0.6 depending on contamination tuning.
- **Subgroup performance:** N/A — synthetic data, no protected attributes.
- **Calibration:** Isolation Forest scores are not calibrated probabilities; use for ranking only.

## Limitations & ethical considerations
- **Synthetic data:** Results may not transfer to real fraud patterns. Validate on domain data before deployment.
- **Threshold is operational:** Contamination parameter must be tuned to analyst alert budget, not a fixed value.
- **Distribution shift:** Anomaly detectors degrade silently as "normal" drifts. Retrain on a rolling window in production.
- **Fairness assessment:** Not applicable to synthetic data; in production, audit false-positive rates by demographic group.
- **Human oversight:** Anomaly scores should surface to human analysts for investigation, not trigger automated blocking.

## Maintenance
- **Monitoring:** Track fraction of flagged records over time; a spike indicates either real anomalies or concept drift.
- **Retraining trigger:** When flagged-rate baseline shifts by >50% or when a known anomaly is missed in retrospect.

---

## Data card (inline)

- **Dataset:** Synthetic via `sklearn.datasets.make_classification` (seed=42)
- **Size:** 20 000 samples, 20 features, 0.5% positive rate
- **Purpose:** Simulate extreme class imbalance to demonstrate PR-AUC over accuracy
- **Limitations:** No real-world distribution; all features are abstract numeric
