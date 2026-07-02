# Retail & E-commerce Case Studies

## Amazon — demand forecasting at item × FC × day grain

**Problem:** Amazon forecasts demand for millions of SKUs, at thousands of fulfilment centres (FCs), at daily granularity — a hierarchy with tens of billions of cells. Getting it wrong causes either stockouts (lost sale, customer dissatisfaction) or overstock (carrying cost, markdown). The asymmetry in costs depends on the product category.

**Approach:** Amazon uses a **hierarchical forecasting** approach: aggregate forecasts (category-level, regional) constrain disaggregated item-level forecasts, ensuring coherence. The core model family is **DeepAR** (Amazon's own, open-sourced), a probabilistic recurrent neural network trained across all time series simultaneously, learning shared patterns (seasonality, price elasticity, promotional lift) while conditioning on item-level covariates.

**Key technical decisions:**
- **Probabilistic forecasts** over point forecasts: DeepAR outputs a distribution (typically negative binomial for count data). This is critical because inventory decisions are asymmetric: the *cost-optimal* safety stock depends on the full distribution, not just the mean.
- **External covariates**: promotions, price changes, weather, and competitor promotions are fed as known-future covariates. Ignoring promotions is the single biggest source of forecast error in retail.
- **Cold-start**: new products have no history. Amazon uses **cross-series transfer** — item embeddings trained across all products let new items borrow patterns from similar established products.

**What failed first:** Early ensemble models (per-SKU time series models like ARIMA/ETS) failed at the long tail — millions of slow-moving SKUs have too few observations to fit reliable models. DeepAR's cross-series learning solved the long-tail problem by pooling data across all SKUs. See [doc 07](../07-time-series-forecasting.md).

**Transferable lesson:** Point forecasting optimises MAE/RMSE. Inventory decisions need the *distribution* of demand. Quantile or probabilistic forecasts let you set service levels (fill rate targets) explicitly rather than guessing safety stock multipliers.

---

## Walmart — demand forecasting with weather

**Problem:** Walmart's data science team published a landmark result: weather explains more variance in short-term demand for certain categories (umbrellas, ice cream, snow shovels) than any other single covariate. The challenge is integrating weather forecasts (which themselves carry uncertainty) into demand models.

**Approach:** A **feature-augmented gradient boosting** model where weather forecast data (temperature, precipitation, storm indicators) from NOAA is joined to item × store × day training data. The model learns interaction effects: the demand elasticity for beer and charcoal is multiplicatively higher when temperature > 25°C AND it's a weekend.

**Key technical decisions:**
- **Hierarchical aggregation**: forecasts at item level are noisy; store-category level is more stable. Walmart uses **middle-out** forecasting: forecast at store-category, then disaggregate to items using historical share ratios.
- **The "hurricane effect"**: demand for emergency supplies spikes 3-5 days before a predicted hurricane, then collapses at landfall. This is a short-duration, high-amplitude pattern that standard time series models miss without weather covariates.
- **Feature lag alignment**: weather forecasts at T+1, T+2, T+3 days are joined correctly to demand at the corresponding future date — a common engineering mistake is joining on calendar date without accounting for the forecast horizon.

**What failed first:** Models trained without promotional lift flags dramatically underforecast during promotional periods, leading to systemic stockouts on promoted items. A promoted item can see 5-10× normal demand — no time-series pattern is large enough to extrapolate to that. Promotions must be explicit features, not expected to be learned from history alone.

**Transferable lesson:** Feature engineering beats model sophistication in retail forecasting. A gradient boosting model with good features (promotions, weather, holidays, price) routinely outperforms complex deep learning on tabular retail data.

---

## Instacart — last-mile ETA and substitution

**Problem:** When a shopper is in-store fulfilling an order, an item may be out of stock. Instacart must instantly suggest a substitution the *customer* will accept — not just any available alternative. Customer acceptance rate of substitutions directly affects NPS and repeat order rate.

**Approach:** A **two-tower neural network** (query tower: customer preferences, order context; item tower: product attributes, price, nutritional content) trained on historical acceptance/rejection of substitutions. At inference, the shopper's app retrieves candidate substitutes from the same category and ranks them by the model score.

**Key technical decisions:**
- **Implicit preference learning**: Instacart doesn't ask customers to rate products. Preferences are inferred from reorder history, search queries, and item interaction time. This is the same implicit feedback problem as Spotify — a long dwell time in the product detail page is a weak positive signal.
- **Contextual features**: the *reason* for substitution matters. "Out of stock" vs "shopper can't find it" leads to different optimal substitutes (the latter might mean the customer's preferred item is actually available).
- **Delivery ETA prediction**: a separate gradient boosting model estimates shopper travel time from store to customer using store-zone, distance, time-of-day, and real-time traffic. The ETA is surfaced in the app; errors in ETA predictions damage customer trust more than small delays, so calibration matters. See [doc 04](../04-evaluation-and-validation.md).

**What failed first:** The first substitution model optimised for nutritional similarity (same calories, protein, fat). Customers rejected substitutions that were nutritionally equivalent but different *brands* — brand loyalty is stronger than nutritional equivalence. The model had to learn the brand preference from reorder data explicitly.

**Transferable lesson:** "Similar" is domain-specific. In grocery, brand loyalty and price sensitivity dominate nutritional equivalence. Let the customer's *behaviour* (acceptance, rejection, reorder) define similarity, not your prior about what matters.

---

## Cross-cutting lessons

1. **Probabilistic over point**: retail decisions (inventory, staffing, procurement) are inherently risk-based. Output distributions, not means.
2. **Promotions are mandatory features** in any retail demand model. Missing them causes the largest systematic errors.
3. **Cross-series learning** (DeepAR-style) is necessary for long-tail SKUs with sparse history.
4. **Substitution and recommendation** quality is measured by acceptance behaviour, not feature similarity — let behaviour define ground truth.
