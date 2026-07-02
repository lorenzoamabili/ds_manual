# Manufacturing Case Studies

## Siemens — predictive maintenance on turbine bearings

**Problem:** Unplanned failure of a gas turbine bearing can cause 2-4 weeks of unplanned downtime and $1-5M in damage and lost production. Scheduled maintenance (replace every N hours) is safe but wasteful — most bearings are replaced with significant life remaining.

**Approach:** **Remaining Useful Life (RUL) prediction** from vibration and temperature sensor data. Siemens uses a multi-output LSTM (sequence-to-scalar) trained on NASA CMAPSS-style run-to-failure datasets: sensor readings from installation to failure, with RUL as the regression target. See [Project P5 analog](../../projects/p5_survival_analysis) for the [survival analysis](16-survival-analysis.md) version.

**Key technical decisions:**
- **[Feature engineering](03-data-and-feature-engineering.md) on time series**: raw sensor data is noisy. Standard features: rolling mean, rolling std, inter-sensor ratios (temperature/vibration ratio is more informative than either alone). Frequency-domain features (FFT, wavelet transform) capture bearing fault signatures that time-domain features miss.
- **Health index construction**: rather than directly predicting RUL, construct a **health index** (HI) — a scalar in [0, 1] that represents the bearing's degradation state. RUL is then estimated from the HI trajectory. This decouples the health estimation problem from the RUL regression problem and makes the HI interpretable by maintenance engineers.
- **False-alarm cost asymmetry**: sending a maintenance crew unnecessarily costs £5,000. Missing a failure costs £500,000+. The threshold is set to tolerate many false alarms (high sensitivity) while catching all true failures. See [doc 04](../04-evaluation-and-validation.md) cost-based threshold.

**What failed first:** Early models had high AUC on test data but low precision on live deployments because the *failure rate* in operations was far lower than in the training data (which oversampled run-to-failure events). Classic imbalanced learning failure — the model saw training data where 50% of sequences failed; in production, <5% fail in any given monitoring window. See [doc 13](../13-anomaly-detection.md).

**Transferable lesson:** Predictive maintenance training datasets from "run to failure" experiments have a very different failure rate from operational monitoring contexts. Always [calibrate](04-evaluation-and-validation.md) your model on data that matches the *operational* base rate, not the experimental base rate.

---

## GE Aviation — aircraft engine health management

**Problem:** GE Aviation manages engine health monitoring for commercial airlines. A premature engine removal (before failure) is expensive ($500k+). A missed failure causes an air safety incident. The asymmetry is enormous.

**Approach:** GE's "Digital Twin" approach: a physics-based simulation model of each engine captures the *expected* sensor readings under nominal operating conditions. The ML model learns the **residual** — the difference between simulated expectation and observed sensors. Anomalies are detected in the residual, not in the raw sensor.

**Key technical decisions:**
- **Physics-informed ML**: using physics models as a prior removes most of the variance attributable to operating conditions (altitude, temperature, thrust setting) from the residual. What remains is either noise or fault signal — a much cleaner learning problem.
- **Multi-sensor fusion**: bearing failures manifest as characteristic patterns across *multiple* sensors simultaneously. [Anomaly detection](13-anomaly-detection.md) on each sensor independently misses correlated multi-sensor signatures. [PCA](06-unsupervised-learning.md) on the residual vector detects correlated anomalies. See [doc 13](../13-anomaly-detection.md).
- **Fleet-level learning**: each engine is one unit; run-to-failure data is rare. GE trains across the entire fleet of thousands of engines, sharing parameters while conditioning on engine-type and age. This is essentially the multi-task learning version of the [DeepAR](07-time-series-forecasting.md) idea.

**What failed first:** Early anomaly scores were not interpretable to maintenance engineers. A score of 0.87 tells an engineer nothing; "high residual in sensor channels T3, T4, and vibration at 2× rotational frequency" tells them "potential bearing fault." They added attribution (which sensors drive the anomaly score) as a mandatory output.

**Transferable lesson:** ML anomaly scores without attribution are often rejected by domain experts. For safety-critical applications, the model must explain *which* signals are anomalous, not just *that* something is anomalous.

---

## BMW — quality inspection with computer vision

**Problem:** Visual inspection of paint defects on car bodies at end of assembly line. Manual inspection is slow, expensive, and inconsistent between shifts. Defects must be caught before delivery — a defect found by the customer is 100× more expensive to fix than one caught in-line.

**Approach:** A defect detection [CNN](11-computer-vision.md) (YOLOv5, later [fine-tuned](11-computer-vision.md) EfficientDet) trained on images of car body panels under structured lighting. The model classifies defect type (scratch, inclusion, orange-peel, dent) and localises it with a bounding box. See [doc 11](../11-computer-vision.md).

**Key technical decisions:**
- **Data collection**: defects are rare by design — a well-tuned line has a defect rate of ~2%. They used a combination of (a) real defects collected over 18 months, (b) synthetic defects composited onto clean panels using image augmentation, and (c) [transfer learning](11-computer-vision.md) from a defect detection model trained on a different product line.
- **Structured lighting**: the same defect is invisible under ambient lighting but visible under raking light (light at oblique angle). BMW uses multi-angle lighting rigs to create consistent conditions — this dramatically simplifies the vision problem because the model doesn't have to handle lighting variation.
- **Human-in-the-loop**: the model flags suspected defects; a human inspector makes the final rework decision. This avoids the liability of a fully automated reject decision on a €40,000 car body. The model's job is to focus human attention, not to replace it.

**What failed first:** The model trained on summer production data failed on winter production data — condensation from cold-morning logistics created a surface texture the model classified as defects (false positives). They rebalanced training data by season and added humidity/temperature as metadata features.

**Transferable lesson:** Computer vision in manufacturing is a controlled-environment problem. Invest heavily in *consistent sensing conditions* (structured lighting, controlled background) rather than trying to build a robust model for all conditions. Reducing input variation is a better investment than model complexity.

---

## Cross-cutting lessons

1. **Physics priors** (digital twins, residual modelling) dramatically simplify the ML problem by removing known sources of variance.
2. **Failure rate calibration**: training on run-to-failure experiments oversamples failures vs. operational monitoring. Recalibrate for the operational base rate.
3. **Attribution is not optional** in safety-critical ML. Domain experts need to know *why* the model flagged, not just *that* it flagged.
4. **Sensor data feature engineering** (rolling statistics, frequency domain, inter-sensor ratios) routinely outperforms raw [time-series](07-time-series-forecasting.md) models.
