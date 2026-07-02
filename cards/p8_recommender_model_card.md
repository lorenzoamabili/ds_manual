# Model Card — P8 Recommender System (SVD Matrix Factorisation)

## Overview
- **Purpose / intended use:** Personalised item recommendation from implicit/explicit user-item interactions. Demonstrates that RMSE alone is insufficient; coverage and precision@K must be tracked.
- **Out-of-scope uses:** Not suitable for cold-start users (zero history); not for real-time serving without additional engineering.
- **Owner / contact:** ds-manual portfolio project
- **Version / date:** 1.0 · 2024

## Model details
- **Type / architecture:** Three models compared — (1) Popularity baseline, (2) User-based CF (cosine similarity, k=20 neighbours), (3) SVD matrix factorisation (k=50 latent factors, mean-centred).
- **Inputs:** User-item rating matrix from MovieLens 100K (or synthetic fallback if download unavailable).
- **Output:** Top-K ranked item list per user.
- **Training data:** MovieLens 100K — 100 000 ratings, ~943 users, ~1 682 movies. Standard `ua.base` / `ua.test` split.

## Evaluation
- **Validation scheme:** Pre-defined MovieLens train/test split (temporal-ordered per user).
- **Headline metrics (see metrics.md for current run):**
  - RMSE: SVD < User-CF < Popularity
  - Precision@10: SVD > User-CF > Popularity
  - Catalog coverage: SVD >> User-CF >> Popularity (key point: popularity is near-zero diversity)
- **Subgroup performance:** Not evaluated (no demographic data in MovieLens 100K).
- **Calibration:** Not applicable — ranking model, not probability model.

## Limitations & ethical considerations
- **Popularity bias:** The popularity baseline concentrates recommendations on blockbusters. SVD improves but still favours frequently-rated items. Long-tail items remain under-recommended.
- **Filter bubbles:** Collaborative filtering reinforces historical preferences. Users rarely discover genres they've never rated.
- **Cold-start:** All three models fail for users/items with zero historical ratings. Hybrid approaches (content features + CF) are needed in production.
- **Offline metrics ≠ online performance:** Precision@K on held-out ratings is a proxy. A/B test is required to validate business impact.
- **Fairness assessment:** Recommendation systems can amplify popularity bias along demographic lines. Audit whether certain user groups receive lower-quality recommendations.
- **Human oversight:** Recommendations should include diversity injection and editorial guardrails in production.

## Maintenance
- **Monitoring:** Track precision@K and coverage on a holdout weekly. Watch for popularity-bias creep as the user base grows.
- **Retraining trigger:** When new items exceed 10% of the catalog (cold-start coverage drop) or when precision@K degrades >5%.
