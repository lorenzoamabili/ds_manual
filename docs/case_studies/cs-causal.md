# [Causal Inference](../09-causal-inference-and-experimentation.md) at Scale

## Uber — surge pricing and the elasticity problem

**Problem:** Uber's surge pricing multiplier is designed to balance supply and demand in real time. Setting the multiplier too high clears the queue but destroys rider demand. Too low and drivers can't find rides — supply collapse. The optimal multiplier depends on *price elasticity of demand* — how much does demand drop for each 1% price increase? This is a causal question: it cannot be answered from observational data because prices correlate with demand conditions (prices are high when demand is high — reverse causality).

**Approach:** **[Instrumental variable](../09-causal-inference-and-experimentation.md) estimation** using exogenous variation in the surge multiplier caused by Uber's own algorithm: the algorithm rounds surge to discrete levels (1.0x, 1.2x, 1.5x, 1.8x, 2.0x...) and the rounding creates quasi-random variation in price independent of underlying demand. This rounding discontinuity is used as an instrument.

**Key technical decisions:**
- **Geographic × time × multiplier-bucket** experimental cells: rather than a platform-wide [A/B test](../09-causal-inference-and-experimentation.md) (impossible — you can't run two prices in the same area simultaneously), Uber uses a **[switchback design](../17-experimentation-advanced.md)** — alternating high/low surge multiplier windows in matched geographic cells and comparing demand response.
- **Heterogeneous elasticity**: price elasticity varies by city, time of day, trip distance, and competitive landscape (cities with transit alternatives have higher elasticity). Uber uses **[doubly-robust](../09-causal-inference-and-experimentation.md) CATE estimators** (causal forests) to estimate heterogeneous treatment effects. See [doc 09](../09-causal-inference-and-experimentation.md).
- **Supply-side response**: higher prices attract more drivers (supply increases). If you estimate only the demand elasticity, you miss the supply response. The equilibrium model estimates both simultaneously.

**Transferable lesson:** Price elasticity from observational data is always [confounded](../09-causal-inference-and-experimentation.md) (prices are high when demand is high). Causal identification requires either randomised price experiments or credible instruments. The switchback design is the standard solution when user-level randomisation is infeasible due to market interference.

---

## Microsoft — ExP platform and heterogeneous treatment effects

**Problem:** Microsoft runs A/B tests at massive scale across Office, Azure, Bing, Xbox, and Teams. A product feature that increases revenue for enterprise users may decrease it for personal users. Standard A/B tests report average treatment effects (ATE) and miss this heterogeneity.

**Approach:** Microsoft's ExP (Experimentation Platform) team published extensively on **HTE (Heterogeneous Treatment Effects)** estimation. Their approach:
1. Run standard A/B test for ATE and guardrail metrics.
2. Post-hoc, apply causal forest (grf package) or **CATE estimation** to identify subgroups where effect differs from ATE.
3. Report the distribution of individual treatment effects (ITE) as a "treatment effect waterfall."

**Key technical decisions:**
- **Pre-specified subgroups vs. data-driven**: pre-specified subgroup tests (mobile vs. desktop, enterprise vs. personal) control for multiple testing. Data-driven HTE discovery (causal forest) is exploratory and requires additional testing before acting on findings.
- **Surrogate metrics**: for long-horizon outcomes (e.g., 12-month retention), short-term surrogate metrics (7-day activation) are used. Microsoft validated the surrogate relationship using historical data — testing that the surrogate is causally downstream of the treatment and upstream of the outcome. This is surrogate index methodology.
- **[CUPED](../17-experimentation-advanced.md) at Microsoft scale**: CUPED (using pre-experiment metric values as covariates) is standard across all Microsoft experiments, reducing variance by 20-60% and allowing 30-40% shorter experiment durations without loss of power. See [doc 17](../17-experimentation-advanced.md).

**Transferable lesson:** ATE from an A/B test is a population average that can hide sub-populations where the treatment is actively harmful. HTE analysis should be standard post-hoc analysis for any shipped feature, with pre-specified subgroups protected by multiple-testing correction.

---

## Spotify — podcast recommendation and causal uplift

**Problem:** Spotify wants to increase podcast listening. The naive approach: identify users who are "at risk of not listening to podcasts" and send them a recommendation. This is the classic risk-vs-uplift confusion (same as the churn case study in this repo). A user who doesn't listen to podcasts because they genuinely don't like them will not respond to a recommendation — and you've wasted a notification slot and potentially annoyed them.

**Approach:** **Uplift modelling** — exactly as in the case study. Train a causal model that estimates the *incremental probability* that a notification causes a user to listen to a podcast, conditional on receiving vs. not receiving the notification. Target only users with high uplift (persuadables), not users with low [propensity](../09-causal-inference-and-experimentation.md) who won't respond (lost causes) or users who would listen anyway (sure things).

**Key technical decisions:**
- **Two-model uplift approach**: train P(Y=1|T=1, X) and P(Y=1|T=0, X) separately, use the difference as the uplift score. See [case study](../../case_study_churn_uplift/run.py).
- **Experiment design for uplift**: to train and validate an uplift model, you need a randomised holdout where some users with high propensity score are NOT notified. This is the **experimental uplift** setup — harder to justify commercially (you're withholding a (potentially) beneficial notification from some users) but necessary for valid uplift estimation.
- **Long-tail content uplift**: Spotify found that recommendation notifications are more effective for *niche* podcasts (where the user wouldn't have discovered it organically) than for *popular* podcasts (which appear in algorithmic recommendations anyway). Targeting high-uplift users in the content long-tail became a product strategy.

**Transferable lesson:** Risk ≠ uplift. The case study in this repo demonstrates this on a simulated example; Spotify demonstrates it at scale in production. The optimal intervention target is the persuadable segment, which is orthogonal to the high-risk segment. Always model causal effect, not just predicted outcome.

---

## Cross-cutting lessons

1. **Observational data cannot identify causal effects without assumptions.** Instrumental variables, [difference-in-differences](../09-causal-inference-and-experimentation.md), and RCTs are the tools; the choice depends on what variation is available.
2. **ATE hides heterogeneity.** HTE analysis is standard good practice post-A/B, with pre-specified subgroups protected by multiple testing correction.
3. **Risk ≠ uplift** in every domain: who is most at risk is not who will most benefit from intervention. Model the causal effect, target the persuadable.
4. **Interference** (network effects, market equilibrium) invalidates standard A/B testing in platform and marketplace settings. Use switchback or [cluster](../06-unsupervised-learning.md)-randomised designs.
