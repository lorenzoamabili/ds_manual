# Project 4 — clustering wine (unlabelled)

- Chosen k (max silhouette): **3**
- Adjusted Rand Index vs. true cultivars: **0.897** (1.0 = perfect, 0 = random)

Scaling first is non-negotiable: proline ranges in the hundreds while some features are < 1, so unscaled Euclidean distance would be dominated by a single feature. Silhouette correctly recovers k=3.
