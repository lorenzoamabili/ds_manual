# MarTech Case Studies

## Airbnb — experimentation at scale and the [novelty effect](17-experimentation-advanced.md)

**Problem:** Airbnb runs thousands of [A/B tests](09-causal-inference-and-experimentation.md) per year on pricing, search ranking, checkout flow, and messaging. The challenge: **novelty effects** inflate short-term metrics for any new UI change, masking true long-term impact. A feature that looks great in a 2-week test may have zero or negative long-term effect.

**Approach:** Airbnb's data science team published their **[CUPED](17-experimentation-advanced.md) (Controlled-experiment Using Pre-Experiment Data)** methodology as variance reduction technique, now widely used across industry. They also use **holdout groups** — a permanent 1% of users in a long-term holdout who never see any experimental features, allowing measurement of the compound effect of all shipped features versus the counterfactual of shipping nothing.

**Key technical decisions:**
- **CUPED**: uses pre-experiment metric values as a covariate to reduce variance in the treatment effect estimator. Because the pre-experiment value is highly correlated with the post-experiment value, this can reduce variance by 50-80%, allowing smaller samples or shorter tests. See [doc 17](../17-experimentation-advanced.md).
- **Metric taxonomy**: Airbnb separates *guardrail metrics* (must not degrade: booking rate, host satisfaction) from *success metrics* (must improve: revenue per search). A test that improves revenue but degrades host satisfaction fails, regardless of the primary metric.
- **Network effects**: Airbnb's marketplace has interference — treating one user changes supply availability for another. Standard A/B testing assumes independence ([SUTVA](09-causal-inference-and-experimentation.md)). For supply-side experiments, they use **switchback designs** (time-based random assignment in geographic clusters) instead of user-based random assignment.

**What failed first:** A search ranking experiment showed +3% booking rate in a 2-week test. Shipped. Long-term tracking showed the effect decayed to ~0.5% at 8 weeks — the bulk of the gain was novelty (users clicking on reordered results out of curiosity). Airbnb now requires a "novelty decay check": if the effect in week 2 is materially larger than week 4, flag for longer observation before shipping.

**Transferable lesson:** Short-term A/B tests systematically overestimate the long-term value of UI changes due to novelty effects. Run tests long enough to see effect stabilisation, or apply novelty decay corrections. See [doc 09](../09-causal-inference-and-experimentation.md).

---

## Booking.com — 1000 experiments running simultaneously

**Problem:** Booking.com runs one of the highest-density experimentation programmes in the world — reportedly 1,000+ concurrent A/B tests across their website. At this scale, the standard NHST framework breaks down: with enough tests, some will be significant by chance (multiple testing problem). Their 2019 blog post/paper documented the **[p-value](02-statistics-that-matter.md) peeking problem** specifically.

**Approach:** [Sequential testing](17-experimentation-advanced.md) with **always-valid p-values** (mSPRT — mixture Sequential Probability Ratio Test). Unlike fixed-horizon t-tests, mSPRT allows tests to be stopped early when evidence is strong *without* inflating Type I error — because the test is designed for sequential data.

**Key technical decisions:**
- **Multiple testing correction at scale**: even with valid individual tests, running 1,000 simultaneous tests at α=0.05 means ~50 false positives. Booking applies a variant of BH correction across the test portfolio. See [doc 02](../02-statistics-that-matter.md).
- **Heterogeneous treatment effects**: a feature that improves conversion for mobile users may harm desktop users. Booking reports treatment effects separately for pre-defined subgroups (device, market, user tenure), and applies a **minimum detectable effect** threshold that accounts for multiple comparisons within subgroups.
- **Automation**: at this scale, experiment analysis is fully automated. A shipping decision is made by an algorithm, not a human, for the vast majority of tests. Human review is reserved for guardrail violations and large-effect experiments.

**What failed first:** Fixed-horizon tests were being peeked at (analysts checking results daily and stopping early when significant). This caused a measured 30-40% false positive inflation in their experiment pipeline. They moved to mSPRT specifically to allow peeking without inflating error rates.

**Transferable lesson:** Peeking at interim results in fixed-horizon A/B tests inflates false positives. Either commit to a fixed horizon (no peeking) or switch to a sequential test designed for continuous monitoring. See [doc 09](../09-causal-inference-and-experimentation.md).

---

## LinkedIn — B2B attribution and multi-touch models

**Problem:** A LinkedIn advertiser runs campaigns (Sponsored Content, InMail, Display). A user clicks a Sponsored Content ad on Monday, ignores InMail on Wednesday, and converts (downloads a whitepaper) on Friday after seeing Display. Which campaign gets credit? Last-click attribution gives it all to Display; first-click gives it all to Sponsored Content. Both are wrong.

**Approach:** LinkedIn developed a **data-driven attribution (DDA)** model: a Shapley value decomposition of the conversion credit across all touchpoints in the buyer's journey, using the observed conversion rate of each touchpoint combination vs. the counterfactual of each touchpoint being absent.

**Key technical decisions:**
- **Shapley values** from cooperative game theory: the credit for touchpoint T is the average marginal contribution of T across all possible subsets of touchpoints. This is the same framework used for ML feature attribution ([SHAP](05-supervised-learning.md)). See [doc 05](../05-supervised-learning.md).
- **Counterfactual estimation problem**: we never observe the same buyer journey with and without an ad. LinkedIn estimates counterfactuals using a survival model on time-to-conversion as a function of touchpoint sequence, identifying the marginal lift of each touchpoint from observational data. This is a [causal inference](09-causal-inference-and-experimentation.md) problem. See [doc 09](../09-causal-inference-and-experimentation.md).
- **Latent journey modelling**: B2B buying involves multiple stakeholders (the individual who clicks ≠ the individual who signs the contract). LinkedIn uses LinkedIn member graph data to link contacts at the same company, attributing at the *account* level, not the individual level.

**What failed first:** The first DDA model was slow to run (Shapley computation is exponential in the number of touchpoints). They approximated it with a Monte Carlo permutation estimator (kernel SHAP) that converges in O(n log n) time. This reduced the computation from days to minutes.

**Transferable lesson:** Last-click and first-click attribution are heuristics that systematically misallocate credit to bottom-funnel channels. Data-driven attribution using causal or Shapley approaches is more accurate but requires solving a counterfactual estimation problem — which is a causal inference problem in disguise.

---

## Cross-cutting lessons

1. **Novelty effects** inflate short-term experiment results for UI changes. Require effect stabilisation over time before shipping.
2. **Peeking at p-values** in fixed-horizon tests inflates false positive rates by 30-40%. Use sequential testing or commit to a fixed horizon.
3. **Last-click attribution** is wrong and known to be wrong. Data-driven attribution is better; the underlying problem is causal.
4. **Network effects** (marketplace interference) violate SUTVA. Use [cluster](06-unsupervised-learning.md)-randomised or switchback designs when treatment spills over between users.
