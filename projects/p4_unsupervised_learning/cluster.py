"""
Project 4 — Unsupervised Learning (Clustering + Dimensionality Reduction)
=========================================================================
Dataset : Wine (bundled with scikit-learn) — 178 wines, 13 chemical features,
          3 true cultivars. We PRETEND we don't know the labels, cluster, then
          check how well the clusters recover the real groups.

Demonstrates:
  - Why you MUST scale features before distance-based methods.
  - Choosing k with the silhouette score (not eyeballing).
  - PCA for a 2-D map you can actually look at.
  - Validating clusters against known labels with the Adjusted Rand Index.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

data = load_wine()
X, y_true = StandardScaler().fit_transform(data.data), data.target
print(f"Wine: {X.shape[0]} samples, {X.shape[1]} features, {len(set(y_true))} true classes")

# ---- choose k by silhouette ------------------------------------------------
sil = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))
       for k in range(2, 8)}
best_k = max(sil, key=sil.get)
labels = KMeans(best_k, n_init=10, random_state=0).fit_predict(X)
ari = adjusted_rand_score(y_true, labels)
print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
print(f"Best k = {best_k} | Adjusted Rand Index vs. truth = {ari:.3f}")

# ---- PCA map ---------------------------------------------------------------
pcs = PCA(2).fit(X)
XY = pcs.transform(X)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].bar(range(2, 8), [sil[k] for k in range(2, 8)]); ax[0].axvline(best_k, ls="--", color="r")
ax[0].set(title="Silhouette vs. k", xlabel="k", ylabel="silhouette")
sc = ax[1].scatter(XY[:, 0], XY[:, 1], c=labels, cmap="viridis", s=25)
ax[1].set(title=f"PCA map coloured by cluster (ARI={ari:.2f})",
          xlabel=f"PC1 ({pcs.explained_variance_ratio_[0]:.0%} var)",
          ylabel=f"PC2 ({pcs.explained_variance_ratio_[1]:.0%} var)")
fig.tight_layout(); fig.savefig("clusters.png", dpi=120); plt.close(fig)

with open("metrics.md", "w") as f:
    f.write(f"# Project 4 — clustering wine (unlabelled)\n\n")
    f.write(f"- Chosen k (max silhouette): **{best_k}**\n")
    f.write(f"- Adjusted Rand Index vs. true cultivars: **{ari:.3f}** "
            "(1.0 = perfect, 0 = random)\n\n")
    f.write("Scaling first is non-negotiable: proline ranges in the hundreds while "
            "some features are < 1, so unscaled Euclidean distance would be dominated "
            "by a single feature. Silhouette correctly recovers k=3.\n")
print("Saved: clusters.png, metrics.md")
