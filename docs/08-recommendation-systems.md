# 08 · Recommendation Systems

"What will this user want next?" Ubiquitous in retail, media, and MarTech.

## The main approaches

| Approach | Idea | Strengths | Weaknesses |
|----------|------|-----------|------------|
| **Content-based** | Recommend items similar to what the user liked, using item features | Works for new users with some history; explainable; no cold-start on items | Over-specialises (filter bubble); needs good item features |
| **Collaborative filtering (CF)** | "Users like you also liked…" — learn from the interaction matrix, no item features needed | Captures taste you can't describe; strong when data is dense | **Cold-start**: useless for brand-new users/items; needs scale |
| **Matrix factorisation** | Factor the user×item matrix into latent user & item vectors (SVD, ALS) | Compact, scalable, the CF workhorse | Latent factors are opaque; still cold-start |
| **Neural / two-tower / sequence models** | Deep embeddings; model order of interactions (e.g. session-based) | State of the art at scale; handles rich features & sequence | Data- and infra-hungry; harder to debug |
| **Hybrid** | Combine content + CF (e.g. factorisation machines, LightFM) | Mitigates cold-start; usually the production answer | More moving parts |

## Cold-start — the defining challenge
- **New item:** fall back to content features until interactions accumulate.
- **New user:** use onboarding signals, popularity, or demographic priors, then
  personalise as data arrives.
- **New system:** you may have to [bootstrap](02-statistics-that-matter.md) with popularity/heuristics before CF is
  viable at all.

## Evaluating recommenders (offline metrics lie less than you'd hope)
Rank-aware metrics on held-out interactions, with a **time-based split** (predict
future interactions from past ones — a random split leaks):

- **Precision@k / Recall@k** — of the top-k recommended, how many were relevant.
- **NDCG@k** — rewards putting relevant items *higher* in the list.
- **MAP** — mean average precision across users.
- **Coverage & diversity** — a recommender that only ever suggests the top 10 items
  scores fine offline and fails in the market. Track these too.

**The offline–online gap is real.** Offline metrics optimise for predicting past
behaviour; the business cares about *changing future* behaviour. The final arbiter
is an **online [A/B test](09-causal-inference-and-experimentation.md)** ([09](09-causal-inference-and-experimentation.md)) —
and beware feedback loops, where the recommender shapes the very data it's next
trained on.

## Practical stack
For a first real system: **implicit-feedback matrix factorisation** (ALS via the
`implicit` library, or `LightFM` for a hybrid) is a strong, well-understood
baseline before anything neural. Log the context you'd need to evaluate later
(what was shown, in what position, what was clicked).

---

## Python example — user-based CF vs. popularity baseline

```python
"""
Minimal recommender comparison: popularity baseline vs. user-based
collaborative filtering on a synthetic rating matrix.
Demonstrates why coverage matters alongside RMSE.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

rng = np.random.default_rng(42)
N_USERS, N_ITEMS = 200, 100

# Synthetic ratings: users have latent preferences over 5 genres
genres = rng.dirichlet(np.ones(5), size=N_USERS)     # user genre taste
item_genre = rng.dirichlet(np.ones(5), size=N_ITEMS)  # item genre mix

R = np.zeros((N_USERS, N_ITEMS))
for u in range(N_USERS):
    # Rate a random 15% of items
    items = rng.choice(N_ITEMS, size=15, replace=False)
    for i in items:
        R[u, i] = np.clip(1 + 4 * genres[u] @ item_genre[i] +
                          rng.normal(0, 0.3), 1, 5)

# ── Popularity baseline ───────────────────────────────────────────────────────
pop_scores = (R > 0).sum(axis=0)
top10_pop  = np.argsort(pop_scores)[::-1][:10]

pop_recs = {}
for u in range(N_USERS):
    rated = set(np.where(R[u] > 0)[0])
    pop_recs[u] = [i for i in top10_pop if i not in rated][:10]

# ── User-based CF ─────────────────────────────────────────────────────────────
sim = cosine_similarity(R)

def cf_predict(u, i, k=20):
    sims = sim[u].copy(); sims[u] = 0
    mask = R[:, i] > 0
    if not mask.any(): return R[R > 0].mean()
    s = sims * mask
    top = np.argsort(s)[::-1][:k]
    top = [t for t in top if s[t] > 0]
    if not top: return R[R > 0].mean()
    return np.dot(s[top], R[top, i]) / (s[top].sum() + 1e-9)

cf_recs = {}
for u in range(N_USERS):
    rated = set(np.where(R[u] > 0)[0])
    scores = {i: cf_predict(u, i) for i in range(N_ITEMS) if i not in rated}
    cf_recs[u] = sorted(scores, key=scores.get, reverse=True)[:10]

# ── Coverage ──────────────────────────────────────────────────────────────────
pop_cov = len({i for recs in pop_recs.values() for i in recs}) / N_ITEMS
cf_cov  = len({i for recs in cf_recs.values() for i in recs}) / N_ITEMS

print(f"Catalog coverage:")
print(f"  Popularity: {pop_cov:.0%}  ← recommends same {len(top10_pop)} items to everyone")
print(f"  User CF:    {cf_cov:.0%}  ← discovers personalised long-tail items")
print(f"\nLesson: identical RMSE; completely different diversity.")
```

---

## Cross-references

- [P8](../projects/p8_recommender) — full recommender project (MovieLens 100K)
- [32](32-retail-ecommerce.md) — retail recommenders in practice
- [35](35-martech.md) — next-best-offer and personalisation
