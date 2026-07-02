# 06 · Unsupervised Learning

Finding structure without labels: clustering, dimensionality reduction, density
estimation. Paired project: [P4 — clustering wine](../projects/p4_unsupervised_learning).

---

## Clustering

| Algorithm | Idea | Use when | Weakness |
|-----------|------|----------|----------|
| **k-means** | Minimise within-cluster variance around k centroids | Roughly spherical, similar-size clusters; large data | Must pick k; assumes convex clusters; sensitive to scale & outliers |
| **Hierarchical (agglomerative)** | Merge nearest points/clusters into a dendrogram | Want nested structure or no fixed k; small–medium data | O(n²) memory; doesn't scale |
| **DBSCAN / HDBSCAN** | Group dense regions; label sparse points as noise | Arbitrary shapes, unknown k, outliers present | Sensitive to density parameters; HDBSCAN is the better default |
| **Gaussian Mixture (GMM)** | Soft, probabilistic clusters (elliptical) | Overlapping clusters; want membership probabilities | Assumes Gaussian components; pick k via BIC |

**Non-negotiable: scale first.** Distance-based methods are dominated by
whichever feature has the largest range. In [P4](../projects/p4_unsupervised_learning),
one feature (proline) is in the hundreds while others are below 1 — without
scaling, clustering just sorts on that column.

### Choosing k

| Method | How | Notes |
|--------|-----|-------|
| **Silhouette score** | (b-a)/max(a,b) per point, average | Higher = better-separated. P4 uses this and correctly recovers k=3 |
| **Elbow (inertia)** | Plot WCSS vs. k, find the bend | Subjective; rarely unambiguous |
| **Gap statistic** | Compare WCSS to a null distribution | More principled but slow |
| **BIC (for GMM)** | Penalised log-likelihood | Principled model selection |
| **Domain knowledge** | You know there are N customer tiers | Often the best constraint |

### Validating clusters

When ground-truth labels exist (even if unused in training), compute **Adjusted
Rand Index (ARI)** — P4 achieves ARI ≈ 0.90 against true wine cultivars. Without
labels, profile each cluster on business-meaningful dimensions to check that
segments are interpretable and actionable.

---

## Dimensionality reduction

| Method | Good for | Critical caveat |
|--------|----------|-----------------|
| **PCA** | Preprocessing, denoising, 2-D map, decorrelation | Linear only; interprets variance, not structure |
| **t-SNE** | 2-D visualisation of high-dim data | Distances between clusters are NOT meaningful; don't feed to models |
| **UMAP** | Visualisation + preserves more global structure | Better speed/quality than t-SNE; still for visualisation |
| **Autoencoders** | Non-linear compression of images/text | Needs data + GPU; overkill for tabular |
| **SVD / Truncated SVD** | Sparse matrices (text, recommenders) | Same idea as PCA without centring |

**The warning about t-SNE/UMAP:** they are for *seeing*, not measuring. The
apparent size and spacing of blobs are artefacts of algorithm parameters, not the
data. Never conclude "these two clusters are far apart" from a t-SNE plot.

**PCA variance explained** tells you how many components you need but not which
components are meaningful. Component 1 explaining 60% variance is a summary, not
an interpretation.

---

## Topic modelling (unsupervised NLP)

For text data, **Latent Dirichlet Allocation (LDA)** finds latent topics as
distributions over words. Each document is a mixture of topics; each topic is a
mixture of words. Output: top words per topic. Interpretation requires domain
knowledge. BERTopic (transformer embeddings + clustering) is the modern default.

---

## Where unsupervised learning shows up

- **Customer segmentation** — cluster on behaviour, profile each segment, run
  targeted campaigns ([35](35-martech.md), [32](32-retail-ecommerce.md))
- **Anomaly detection** — density/distance methods flag outliers ([13](13-anomaly-detection.md))
- **Feature compression** — PCA as a preprocessing step reduces noise before supervised modelling
- **Exploration** — a UMAP map is often the fastest way to *see* structure before investing in modelling
- **Recommenders** — SVD factorises the user-item matrix ([08](08-recommendation-systems.md), [P8](../projects/p8_recommender))

---

## Python example — full unsupervised pipeline

```python
"""
Unsupervised learning pipeline: PCA + KMeans + UMAP on the wine dataset.
Demonstrates: scale-first rule, silhouette for k, ARI validation, UMAP visualisation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline

data = load_wine()
X, y_true = data.data, data.target
feature_names = data.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── PCA: how many components capture 90% variance? ───────────────────────────
pca_full = PCA().fit(X_scaled)
cumvar = pca_full.explained_variance_ratio_.cumsum()
n90 = np.searchsorted(cumvar, 0.90) + 1
print(f"Components for 90% variance: {n90} (out of {X.shape[1]})")

# ── Choose k via silhouette ──────────────────────────────────────────────────
sil_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

best_k = max(sil_scores, key=sil_scores.get)
print(f"\nSilhouette scores: {sil_scores}")
print(f"Best k by silhouette: {best_k} (true k=3)")

# ── Cluster with best k ───────────────────────────────────────────────────────
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km_best.fit_predict(X_scaled)
ari = adjusted_rand_score(y_true, labels)
print(f"ARI vs. true cultivars: {ari:.3f}  (1.0 = perfect recovery)")

# ── 2D PCA projection ─────────────────────────────────────────────────────────
pca2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Silhouette elbow
axes[0].plot(list(sil_scores.keys()), list(sil_scores.values()), "o-")
axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Silhouette score")
axes[0].set_title("Choosing k via silhouette"); axes[0].legend()

# Clusters in PCA space
colors = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
for cl in range(best_k):
    mask = labels == cl
    axes[1].scatter(pca2[mask,0], pca2[mask,1], c=colors[cl],
                    label=f"Cluster {cl}", alpha=0.7, s=40)
axes[1].set_title(f"KMeans k={best_k} — PCA projection")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend()

# True labels for comparison
for cl in range(3):
    mask = y_true == cl
    axes[2].scatter(pca2[mask,0], pca2[mask,1], c=colors[cl],
                    label=data.target_names[cl], alpha=0.7, s=40, marker="^")
axes[2].set_title("True cultivars (for validation)")
axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2"); axes[2].legend()

plt.suptitle("Wine clustering: scale → k selection → cluster → validate", fontsize=11)
plt.tight_layout()
plt.savefig("clustering_pipeline.png", dpi=120)
plt.close()

# ── Feature importance via PCA loadings ─────────────────────────────────────
pca2_fit = PCA(n_components=2, random_state=42).fit(X_scaled)
loadings = pd.DataFrame(pca2_fit.components_.T, index=feature_names,
                         columns=["PC1","PC2"]) if True else None
try:
    import pandas as pd
    loadings = pd.DataFrame(pca2_fit.components_.T, index=feature_names,
                             columns=["PC1","PC2"])
    print("\nTop PC1 loadings (most separating dimension):")
    print(loadings["PC1"].abs().nlargest(5).to_string())
except ImportError:
    pass

print("\nLesson: scale first, choose k with silhouette, validate with ARI.")
print(f"Without scaling, proline (range ~300-1600) dominates every distance.")
```

---

## Common pitfalls

- **Clustering without scaling** — single high-range feature dominates. Always `StandardScaler` first.
- **Interpreting WCSS/inertia alone** — always monotonically decreasing; the "elbow" is usually vague.
- **Treating clusters as facts** — they are a compression of the data, not discovered natural kinds.
- **Not validating against business meaning** — a cluster that's statistically tight but uninterpretable is useless.
- **t-SNE parameters** — perplexity, learning rate, and n_iter all change the picture dramatically. Report them and don't over-interpret.
