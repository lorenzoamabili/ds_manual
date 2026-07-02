# Energy & Utilities Case Studies

## DeepMind — data centre cooling optimisation

**Problem:** Google's data centres use ~15% of their total electricity for cooling. Cooling systems are complex, multi-variable control problems with significant thermal inertia — the effect of a cooling action on temperature is delayed by minutes to hours, making manual or rule-based control suboptimal.

**Approach:** DeepMind trained a deep reinforcement learning (RL) agent on historical sensor data from data centre cooling infrastructure. The agent observes ~120 features (temperatures, pump speeds, cooling tower setpoints, IT workload) and outputs recommended setpoints for cooling systems.

**Key technical decisions:**
- **Safety constraints**: the agent cannot directly control the data centre (risk of equipment damage, SLA violation, data loss). Instead, it *recommends* setpoints that human operators review and approve. After a period of supervised operation with human oversight, the constraint was relaxed to allow automated control within defined safe ranges. This is a **human-in-the-loop RL** deployment pattern.
- **Offline evaluation**: RL policy evaluation in production is hard — you can't run the old policy and new policy simultaneously on the same data centre. DeepMind used a simulator trained on historical data to evaluate policy performance before live deployment.
- **Transfer challenge**: a cooling policy trained on one data centre doesn't transfer directly to another — each has different thermal characteristics, equipment, and workload patterns. They fine-tuned per-facility rather than applying a single global policy.

**Result:** 40% reduction in cooling energy (30% reduction in total data centre PUE), sustained over multiple years of deployment. This is the most publicised ML cost-saving result in industry.

**Transferable lesson:** Reinforcement learning for real-world control systems requires safety constraints, human oversight, and simulation-based offline evaluation. The biggest risk is not model underperformance — it's an unsafe action in a physical system. Safety architecture must precede model design.

---

## Ørsted — offshore wind farm power forecasting

**Problem:** Wind power is variable and hard to forecast. Grid operators require power producers to submit day-ahead production schedules; deviations from schedule incur penalty costs. Accurate short-term (0-48 hour) wind power forecasts directly reduce imbalance penalties.

**Approach:** A hybrid **numerical weather prediction + ML** approach. NWP models (ECMWF, GFS) provide physical wind speed forecasts at grid level. An ML model learns the **turbine power curve correction** — the relationship between forecast wind speed and actual turbine output, accounting for turbine wake effects, curtailments, and model biases.

**Key technical decisions:**
- **Wake effect modelling**: offshore wind turbines in arrays create wind shadow (wake) on downstream turbines. A turbine's actual output depends not just on wind speed at its location but on the operating state of upstream turbines. Spatial graph features (which turbines are upwind at the current wind direction) are critical. See [doc 21](../21-graph-and-network-analysis.md).
- **Probabilistic forecasts**: grid operators need not just a point forecast but a confidence interval. A 95% prediction interval allows the operator to set schedule bids that minimise expected imbalance cost. See [doc 07](../07-time-series-forecasting.md).
- **Extreme weather handling**: wind turbines curtail (shut down) at very high wind speeds to prevent damage. The power curve is therefore non-monotone at high speeds. The ML model must learn this curtailment regime — often missed in naive power curve models that assume monotone input-output.

**What failed first:** The first model was trained on normal operating data and had no representation of curtailment events. During a storm with gusts above 25 m/s, the forecast predicted high output (high wind speed → high power), while the actual output was zero (turbines curtailed). The model had never seen curtailment in training. They added curtailment state as an explicit feature from the SCADA system.

**Transferable lesson:** Regime changes — operating modes that rarely occur in training data but are critical in production — require explicit handling. They will not be extrapolated from nominal training data by any model.

---

## Tesla — energy grid dispatch with Megapack

**Problem:** Tesla's Megapack is a utility-scale battery storage system. The business model for grid-connected batteries involves **arbitrage** (charge when electricity is cheap, discharge when it's expensive) and **frequency regulation** (respond within seconds to grid frequency deviations for payment). Optimal dispatch requires short-term price forecasting.

**Approach:** Tesla Autobidder is an ML-powered bidding system: a price forecasting model (gradient boosting on historical AEMO/CAISO prices, weather, demand forecasts) feeds an optimisation engine that determines the charge/discharge schedule to maximise expected revenue subject to battery constraints (state of charge, round-trip efficiency, maximum discharge rate).

**Key technical decisions:**
- **Forecast → optimisation pipeline**: the ML model outputs probabilistic price forecasts; the optimiser uses the full distribution to compute *expected* revenue under different bid strategies. This stochastic optimisation is more robust than optimising for the point forecast. See [doc 12](../12-optimization.md).
- **Model Predictive Control (MPC)** rolling horizon: rather than optimising the full 24-hour schedule at once, the system re-optimises every 5 minutes using updated price forecasts, temperature readings (battery efficiency depends on temperature), and grid signals. This adapts to intraday forecast errors.
- **Frequency regulation** market: a separate model scores the expected payment for offering regulation capacity vs. discharging into the energy market. The two revenue streams compete for the same battery capacity and the optimiser allocates between them.

**What failed first:** In the South Australian grid, extreme price spikes (>$15,000/MWh vs. typical $50-100/MWh) occur rarely but are the primary source of revenue. A model trained on typical price data systematically underforecasted these spikes. They added quantile regression to explicitly model the tail distribution of prices, and an alert system that triggers a manual review when forecast uncertainty is extreme.

**Transferable lesson:** In energy and financial markets, the tail of the distribution (rare extreme events) often accounts for the majority of revenue (or risk). Mean-optimising models are systematically wrong when tail events dominate outcomes. Use quantile regression or probabilistic forecasting.

---

## Cross-cutting lessons

1. **Safety constraints precede model design** in physical control systems. Never let an ML agent take unconstrained real-world actions.
2. **Regime changes** (curtailment, storms, market spikes) are invisible in normal training data. Identify and explicitly model operating modes.
3. **Probabilistic forecasts** are necessary when downstream decisions (bidding, inventory, scheduling) are risk-sensitive.
4. **Forecast + optimise** is the architecture for decision-making problems: ML handles uncertainty; OR handles constraint satisfaction.
