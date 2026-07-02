# Model Card — P4 Unsupervised Learning (KMeans Clustering)

## Overview
- **Purpose / intended use:** Demonstrate correct unsupervised clustering workflow — scale before clustering, choose k via silhouette, validate against known structure. Educational reference for customer segmentation and exploratory analysis.
- **Out-of-scope uses:** The wine cultivar classification is a pedagogical fixture. The techniques transfer; the wine model does not apply to other domains directly.
- **Owner / contact:** ds-manual portfolio project
- **Version / date:** 1.0 · 2024

## Model details
- **Type / architecture:** KMeans (k=3, n_init=10, random_state=42) on StandardScaler-transformed features. k selected by silhouette score over k=2..7.
- **Inputs:** 13 chemical measurements from wine samples (alcohol, malic acid, ash, alcalinity, magnesium, phenols, flavanoids, nonflavanoid phenols, proanthocyanins, colour intensity, hue, OD280/OD315, proline).
- **Output:** Cluster assignment (0, 1, 2) per wine sample. No direct business interpretation without domain profiling.
- **Training data:** UCI Wine dataset (178 samples, 3 cultivars, bundled with scikit-learn). True labels withheld during clustering, used only for ARI validation.

## Evaluation
- **Validation scheme:** Silhouette score for k selection (k=3 is best). Adjusted Rand Index (ARI) vs. true cultivar labels for recovery validation.
- **Headline results:** Silhouette ≈ 0.28 at k=3; ARI ≈ 0.90 vs. true cultivars. Without scaling, ARI drops to ~0.4 (proline dominates all distances).
- **Subgroup performance:** N/A — unsupervised; no protected attributes.
- **Calibration:** Not applicable.

## Limitations & ethical considerations
- **Clusters are not facts:** they are a compression of the data. The 3-cluster solution reflects chemical structure, not guaranteed cultivar correspondence.
- **Sensitive to initialisation:** n_init=10 mitigates but doesn't eliminate dependence on random seed. Always run multiple initialisations.
- **Euclidean distance assumes isotropic structure:** DBSCAN or GMM may be better for non-spherical clusters.
- **Human oversight:** Cluster labels must be interpreted by domain experts before any business action is taken.

## Maintenance
- **Monitoring:** N/A (fixed dataset).
- **Retraining trigger:** N/A.
