# Forecasting at Scale Case Studies

## Uber — ride demand forecasting and the spatial-temporal problem

**Problem:** Uber needs to forecast demand (ride requests) at the level of hexagonal spatial cells (H3 grid) × 15-minute time windows, up to 60 minutes ahead. This enables proactive driver repositioning ("your next trip is likely to be from the airport in 30 minutes — head there now"). The forecast has ~1M cells per city; a single model must generalise across all of them.

**Approach:** Uber's **DeepETA** and demand forecasting systems use a **hierarchical spatial-temporal model**: (1) a city-level forecast captures macro patterns (events, weather, day-of-week); (2) a spatial disaggregation model distributes city demand to hexagonal cells using local features (POI density, historical spatial demand share); (3) a temporal model captures within-day variation.

**Key technical decisions:**
- **Spatial features via H3 hex grid**: Uber pioneered the H3 geospatial indexing library (open-sourced) for consistent spatial aggregation at multiple resolutions. This allows multi-resolution modelling: coarse city-level forecast disaggregated to fine hex-level. See [doc 22](../22-geospatial.md).
- **Event detection**: concerts, sports events, and conferences create demand spikes that overwhelm any time-series model trained on normal data. Uber integrates an event calendar API and uses event presence as a covariate — a XGBoost model trained on event characteristics (capacity, type, end time) predicts the spatial and temporal demand spike shape.
- **Rolling-origin evaluation**: Uber evaluates forecast models using rolling-origin cross-validation — training up to time T, evaluating at T+1 through T+H. This matches deployment exactly and avoids temporal leakage. See [doc 07](../07-time-series-forecasting.md).

**What failed first:** Demand forecasts during major holidays were systematically wrong because holiday patterns don't repeat frequently enough for a time-series model to learn from them. The model "learned" the average day, and holidays looked like average days with unusual residuals. Uber now over-represents holiday data in training by upsampling holiday periods and adding explicit calendar feature embeddings.

**Transferable lesson:** Rare-but-patterned events (holidays, events) require explicit representation — they will not be correctly learnt from their rare occurrences in training data. Oversample them, encode them explicitly, or retrieve reference patterns from the same event in prior years.

---

## Meta — Prophet and the practitioner forecasting problem

**Problem:** Meta (formerly Facebook) needed a forecasting tool that data analysts (not time-series specialists) could use reliably for business metrics: ad revenue, user growth, content engagement. The trade-off: specialist models (ARIMA, state-space) are powerful but require significant expertise to configure; a simpler model used correctly beats a complex model used incorrectly.

**Approach:** Meta open-sourced **Prophet** — a decomposable time-series model: `y(t) = trend(t) + seasonality(t) + holidays(t) + ε`. Each component is separately modelled (piecewise linear trend with changepoints, Fourier series for seasonality, additive holiday effects) and fitted with Stan (Bayesian inference).

**Key technical decisions:**
- **Changepoint detection**: Meta's business metrics undergo structural breaks (product launches, algorithm changes, policy shifts). Prophet detects changepoints in the trend component automatically using a sparse prior on trend change magnitude — the model adapts to breaks rather than fitting a single global trend.
- **Explicit holiday handling**: rather than hoping the model learns holiday effects from sparse data, Prophet takes a user-specified holiday calendar and adds explicit regression terms for each holiday. This is the same lesson as Uber — holidays must be explicit.
- **Uncertainty intervals**: Prophet outputs 95% prediction intervals via posterior sampling from the Stan model. These are *important* for business planning — a forecast without uncertainty is a false promise. See [doc 07](../07-time-series-forecasting.md).
- **Interpretability for non-specialists**: the decomposition (trend + seasonality + holidays) is directly plotted and explained. A non-specialist can inspect and critique each component rather than treating the model as a black box.

**Limitation acknowledged:** Prophet's Fourier seasonality assumes stable seasonal patterns. If seasonality itself changes over time (e.g., COVID changed the day-of-week effect for restaurant traffic), Prophet mishandles it. It is a practitioner tool, not a state-of-the-art research model — and Meta is clear about this in their paper.

**Transferable lesson:** The best forecast model for an organisation is often not the state-of-the-art model — it is the model that practitioners can correctly configure, interpret, and critique. A well-used simple model beats a poorly-used complex model. Tools that expose their components (trend, seasonality, holidays) reduce misconfiguration errors.

---

## DoorDash — delivery ETA with multi-stage uncertainty

**Problem:** DoorDash's ETA (Estimated Time of Arrival) is a promise to the customer: "Your order will arrive at 7:35pm." An ETA that is consistently late is worse than one that is consistently 5 minutes earlier — customers budget around the ETA and are more frustrated by a late delivery than reassured by an early one. The ETA is a multi-stage forecast: (1) restaurant prep time, (2) dasher assignment, (3) pickup to dropoff travel time.

**Approach:** A **cascade of sequential models**: each stage's uncertainty propagates to the next. Stage 1: a gradient boosting model predicts restaurant prep time as a function of order complexity, time of day, restaurant load, and historical restaurant-level prep time. Stage 2: a survival model estimates time-to-dasher-assignment. Stage 3: a routing model estimates drive time (Google Maps API + traffic model). Total ETA = sum of three stage estimates with uncertainty propagated.

**Key technical decisions:**
- **Asymmetric loss**: DoorDash explicitly trains with an asymmetric loss function — underprediction (telling the customer 7:35pm when it's 7:50pm) is penalised more than overprediction. This shifts the predicted ETA to be conservatively late, reducing customer frustration. See [doc 07](../07-time-series-forecasting.md) quantile regression.
- **Quantile regression**: by training separate models for the 50th and 80th percentile of delivery time, DoorDash can show customers a range ("arriving between 7:30 and 7:45") rather than a point estimate, which is more honest and better calibrated.
- **Restaurant-level learning**: prep time varies enormously by restaurant. A model that treats all restaurants identically has high error. A per-restaurant model has insufficient data for new restaurants. The solution is a **hierarchical model** — a global prior with restaurant-level random effects (shrinkage).

**What failed first:** The original routing model used Google Maps ETA directly. Google Maps ETA doesn't account for time-on-foot at pickup (parking, walking to door, checking order) or for dasher behaviour (multiple pickups). Adding dasher-level historical pickup time as a feature reduced this component's error by ~25%.

**Transferable lesson:** Multi-stage prediction pipelines require explicit uncertainty propagation — don't just sum point estimates. Each stage adds variance; the total uncertainty at the end is larger than any individual stage's error. Quantile regression at each stage enables honest uncertainty communication at the final output.

---

## Cross-cutting lessons

1. **Rare events** (holidays, product launches, events) will not be learnt from sparse occurrences. Oversample them, encode them explicitly, or retrieve reference patterns from prior instances of the same event.
2. **Hierarchy** in spatial and organisational forecasting: top-down (disaggregate from aggregate) or bottom-up (aggregate from granular), both with coherence constraints. See [doc 07](../07-time-series-forecasting.md).
3. **Asymmetric loss** encodes the business cost structure. When underprediction is more costly than overprediction (ETA, inventory), train with an asymmetric loss or use a high quantile as your point forecast.
4. **Simple, interpretable models used correctly** (Prophet) often outperform complex models used incorrectly, especially for non-specialist practitioners.
