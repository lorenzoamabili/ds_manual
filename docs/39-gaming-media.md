# 39 · Gaming & Media

## Signature problems

| Problem | Approach |
|---------|----------|
| Player churn prediction | [Survival analysis](16-survival-analysis.md), classification (see [16](16-survival-analysis.md), [05](05-supervised-learning.md)) |
| Matchmaking (skill-based) | TrueSkill ([Bayesian](20-bayesian-and-probabilistic.md) rating), bandit exploration |
| Monetisation / LTV prediction | Regression, [quantile regression](07-time-series-forecasting.md), survival models |
| Content recommendation | [Collaborative filtering](08-recommendation-systems.md), contextual bandits (see [08](08-recommendation-systems.md)) |
| Toxicity / content moderation | NLP classification, human-in-the-loop review (see [10](10-nlp-and-llms.md)) |
| Dynamic difficulty adjustment | Reinforcement learning, contextual bandits |
| Ad inventory yield optimisation | Constrained optimisation, auction theory (see [12](12-optimization.md)) |
| Streaming quality adaptation | Real-time anomaly + contextual model (see [13](13-anomaly-detection.md)) |

## Domain characteristics

- **Extremely high event frequency**: user actions arrive at sub-second cadence (clicks, movements, in-game events). Data engineering often dominates modelling time.
- **Sparse monetisation**: in F2P (free-to-play) games, typically 1-5% of players generate >80% of revenue. Extreme [class imbalance](05-supervised-learning.md) in LTV prediction.
- **Network effects in matchmaking**: a player's experience depends on who they're matched with — standard i.i.d. ML assumptions break.
- **Short content cycles**: a game's meta changes with each patch; a model trained before a patch may be obsolete the next day. Rapid retraining pipelines are essential.
- **Engagement vs. wellbeing tension**: maximising session time can conflict with player wellbeing (addictive design). Responsible ML here requires explicit wellbeing metrics as guardrails.

## Key metrics

- **DAU/MAU ratio** (stickiness): daily active / monthly active users. Target > 0.3 for healthy retention.
- **D1 / D7 / D30 retention**: fraction of new players returning after 1, 7, 30 days. D1 < 40% typically signals onboarding friction.
- **ARPU / ARPPU**: average revenue per (paying) user.
- **Session length distribution**: bimodal distribution (quick sessions + long sessions) signals different player segments.

## Player segmentation with RFM

Gaming adapts Retail's RFM (see [32](32-retail-ecommerce.md)):
- **Recency** → days since last login
- **Frequency** → sessions per week
- **Monetary** → lifetime spend OR in-game currency earned (as a proxy for engagement)

Clusters typically emerge: whales (high M, regular F), lapsed engaged players (low R, high historical F), casual churners.

## Matchmaking: TrueSkill sketch

```python
"""
TrueSkill-inspired Bayesian skill rating.
Each player's skill is modelled as a Gaussian N(μ, σ²).
Match outcome updates the posterior via Gaussian belief propagation.

pip install trueskill
"""
try:
    import trueskill
    env = trueskill.TrueSkill(draw_probability=0.02)

    # Two players start with default prior μ=25, σ=25/3
    alice = env.create_rating()
    bob   = env.create_rating()
    print(f"Before: Alice={alice}, Bob={bob}")

    # Alice wins
    alice, bob = env.rate_1vs1(alice, bob)
    print(f"After Alice wins: Alice={alice:.2f}, Bob={bob:.2f}")

    # Match quality: probability game is fair
    quality = env.quality_1vs1(alice, bob)
    print(f"Match quality (fairness): {quality:.2%}")
except ImportError:
    print("pip install trueskill to run this example")
```

## Toxicity detection pipeline

```python
"""
NLP-based toxicity classifier (TF-IDF + LR baseline).
In production: fine-tuned transformer + human review queue for edge cases.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Synthetic chat messages (replace with real moderation corpus)
rng = np.random.default_rng(42)
messages = [
    "gg ez lol noob", "nice game well played", "report this cheater",
    "good match everyone", "you are trash get out", "great teamwork",
    "go back to bronze where you belong", "that was a tough fight",
    "i will find you", "gg nice mechanics",
] * 50
labels = [1,0,1,0,1,0,1,0,1,0] * 50  # 1=toxic

pipe = Pipeline([
    ("vec", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", random_state=42)),
])
scores = cross_val_score(pipe, messages, labels, cv=5, scoring="f1")
print(f"Toxicity classifier F1: {scores.mean():.3f} ± {scores.std():.3f}")
print("Production note: route low-confidence predictions to human moderation queue")
```

## Cross-references

- [08](08-recommendation-systems.md) — content and player recommendations
- [09](09-causal-inference-and-experimentation.md) — [A/B testing](09-causal-inference-and-experimentation.md) game features
- [12](12-optimization.md) — ad yield and resource allocation
- [case studies](case_studies/cs-product-analytics.md) — Duolingo engagement loop (same domain patterns)
