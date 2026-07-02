# Product Analytics Case Studies

## Netflix — from engagement to retention as North Star

**Problem:** Netflix's early North Star metric was streaming hours. Engineers optimised for it. The result: auto-play, increasingly passive, easy-to-watch content surfaced over challenging content users actually valued — engagement went up, but long-term retention showed signs of weakening for high-intent subscribers.

**Approach:** Netflix shifted to a *retention-centric* metric. Key insight from their data science team: **the marginal streaming hour has negative value past a saturation point** — users who feel they "watched too much Netflix" churn at higher rates. The actual North Star became probability a subscriber will cancel in the next 28 days, and the product was optimised against that.

**Key technical decisions:**
- **Survival modelling** on subscriber cohorts (time-to-cancel) rather than aggregated engagement scores. See [doc 16](../16-survival-analysis.md).
- **Personalisation at scale**: Netflix runs 250+ [A/B tests](../09-causal-inference-and-experimentation.md) simultaneously; the recommendation system (a hybrid of [matrix factorisation](../08-recommendation-systems.md) + bandit exploration) is evaluated not on RMSE but on whether viewing a recommendation *leads to completion*, which is a stronger retention signal.
- **Counterfactual evaluation**: because they can't A/B test every algorithm change, they use logged-data counterfactual estimators (IPS, DR) to evaluate recommendation changes offline before shipping. See [doc 09](../09-causal-inference-and-experimentation.md).

**What failed first:** RMSE-optimised recommenders surfaced popular titles the user had already seen or explicitly skipped. Coverage collapse — the long tail of content that actually retains niche subscribers — was invisible in RMSE. They added *diversity* and *novelty* as explicit recommender objectives. See [doc 08](../08-recommendation-systems.md).

**Transferable lesson:** Your North Star metric must causally connect to the business outcome you care about, not just correlate. Optimising a proxy metric (engagement hours) can actively harm the true outcome (retention). Validate the causal link with experimentation before committing.

---

## Spotify — the discover weekly flywheel

**Problem:** Recommendation diversity: serving the same popular artists (exploitation) vs. surfacing unknown artists users will love (exploration). Cold-start: new songs have no listening history.

**Approach:** Spotify uses a two-stage architecture: (1) **candidate generation** via [collaborative filtering](../08-recommendation-systems.md) on implicit feedback (play, skip, save, share — not just listens); (2) **ranking** with a gradient-boosted model incorporating audio features ([CNN](../11-computer-vision.md) on raw audio spectrograms), NLP on playlist co-occurrence ("songs that appear in playlists called 'chill morning' [cluster](../06-unsupervised-learning.md) differently from 'workout'"), and user context (time of day, device, recent history).

**Key technical decisions:**
- **Implicit feedback** treated correctly: a skip after 5 seconds is a strong negative signal; partial listen (50-80%) is a weak positive; save is a strong positive. Most recommenders ignore the distinction. See [doc 08](../08-recommendation-systems.md).
- **Audio embeddings** as content features bypass the cold-start problem for new songs — even without listening history, a song can be placed in the embedding space by its acoustic features.
- **Contextual bandits** for playlist ordering: the first song in a session is ranked by an exploration-sensitive policy; subsequent songs exploit observed session signals.

**What failed first:** Pure collaborative filtering produced "bubble" recommendations — users stayed in a genre forever. They added an explicit *diversity constraint* to the ranking objective: the top-N list must span at least K distinct audio clusters.

**Transferable lesson:** RMSE and accuracy hide coverage collapse. Always evaluate recommendation systems with diversity, novelty, and coverage@K alongside accuracy. The business goal is often user discovery, not prediction accuracy.

---

## Duolingo — the push notification optimisation problem

**Problem:** Duolingo's core retention mechanism is the daily streak. Push notifications remind users to practise. Sending too many → users disable notifications. Too few → they forget and break streaks. The marginal notification can have negative expected value.

**Approach:** A **contextual bandit** model determines for each user at each time: send notification Y/N, and which variant of message copy. Reward = session started within 2 hours. The policy learns per-user send-time preferences and copy sensitivity.

**Key technical decisions:**
- **[Feature engineering](../03-data-and-feature-engineering.md)**: last-active hour, day-of-week pattern, streak length, language being learned, time-zone. Streak length is a *moderating variable* — users with streaks > 30 days respond less to reminders (intrinsically motivated) so the bandit learns to suppress for them.
- **Holdout evaluation**: a permanent 5% holdout with no optimisation serves as a control arm; this catches Goodhart's law problems where the bandit optimises notification-open rate rather than actual learning retention.
- **Long-term reward**: a short horizon (open rate) is easy to optimise but may cannibalise long-term retention. Duolingo shifted to 7-day retention as the reward signal, which requires delayed reward attribution — harder but more honest.

**What failed first:** The first bandit optimised open rate (user tapped the notification). Open rate went up; 7-day retention went *down*. The model learned to send sensational/alarming copy ("Your streak is about to end!") that users opened but resented, and eventually muted.

**Transferable lesson:** Optimise for the metric you actually care about, not the metric that's easy to measure. Proxy metrics (open rate) can be *anti-correlated* with the true metric (retention) once the model learns to exploit the proxy.

---

## Cross-cutting lessons

1. **North Star ≠ engagement** in subscription businesses. Retention is the lagging indicator; measure its leading signals (completion rate, voluntary return rate).
2. **Evaluation gap**: offline metrics (RMSE, precision@K) systematically miss diversity, novelty, and long-horizon effects. Run online tests; use logged counterfactual estimation where live tests are expensive.
3. **Goodhart's law everywhere**: when a proxy measure becomes a target, it stops being a good measure. Build permanent holdouts and delayed reward pipelines.
