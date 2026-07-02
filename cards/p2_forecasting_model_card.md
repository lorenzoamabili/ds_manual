# Model Card — P2 Time Series Forecasting (ETS / Holt-Winters)

## Overview
- **Purpose / intended use:** Educational demonstration of rolling-origin backtesting for seasonal forecasting. The lesson: evaluate at the right horizon, against the seasonal-naive baseline, using multiple origins.
- **Out-of-scope uses:** Not for production deployment on the airline series (too old, too few observations). The techniques transfer; this specific model does not.
- **Owner / contact:** ds-manual portfolio project
- **Version / date:** 1.0 · 2024

## Model details
- **Type / architecture:** Exponential Smoothing (ETS) with additive trend + multiplicative seasonality (Holt-Winters). Selected over SARIMA and Prophet by rolling-origin MAPE.
- **Inputs:** Monthly passenger counts (single univariate series, lagged 1-12 periods).
- **Output:** 12-step-ahead point forecast.
- **Training data:** Box-Jenkins airline passengers dataset (1949–1960, 144 observations). Classic benchmark; exhibits multiplicative seasonality and exponential trend.

## Evaluation
- **Validation scheme:** Rolling-origin backtest across 3 origins. Train on history before cutoff; forecast 12 months ahead; roll cutoff forward. No future data touches training.
- **Headline metrics:** ETS MAPE ≈ 3.6% vs. seasonal-naive baseline 8.1% (MASE < 1 confirms we beat naive).
- **Subgroup performance:** N/A — single univariate time series.
- **Calibration:** Not applicable for point forecasts. Prediction intervals available from ETS but not the primary focus.

## Limitations & ethical considerations
- **Single-series, historical benchmark:** the airline dataset is a pedagogical fixture. Real-world forecasting has exogenous shocks (COVID, price changes) that require external regressors.
- **No prediction intervals in primary output:** for decisions under uncertainty, always report intervals alongside point forecasts.
- **Multiplicative model can fail near zero:** if the series ever approaches zero, the multiplicative seasonal component becomes unstable.
- **Human oversight:** N/A (portfolio teaching artefact).

## Maintenance
- **Monitoring:** N/A (historical fixed dataset).
- **Retraining trigger:** N/A.
