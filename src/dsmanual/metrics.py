"""dsmanual — small, tested, reusable utilities shared by the projects.

The manual's docs argue that transformation logic should live in importable,
tested code rather than one-off notebook cells. This package is that principle
made real: every function here is covered by tests/ and imported by the projects.
"""
from __future__ import annotations

import numpy as np

__all__ = ["seasonal_naive", "mape", "mase", "standardised_mean_difference",
           "clip_propensity"]


def seasonal_naive(train, horizon: int, season: int = 12) -> np.ndarray:
    """Forecast by repeating the value from `season` steps ago.

    The baseline every real forecaster must beat (see docs/07). Repeats the last
    full season forward for `horizon` steps.
    """
    train = np.asarray(train, dtype=float)
    if len(train) < season:
        raise ValueError(f"need >= {season} points, got {len(train)}")
    last = train[-season:]
    return np.array([last[i % season] for i in range(horizon)])


def mape(actual, forecast) -> float:
    """Mean Absolute Percentage Error (%). Undefined if any actual == 0."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    if np.any(actual == 0):
        raise ValueError("MAPE undefined when an actual value is 0; use MASE")
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def mase(actual, forecast, train, season: int = 1) -> float:
    """Mean Absolute Scaled Error: error relative to the in-sample naive forecast.

    <1 means the model beats the naive baseline. Scale-free and comparable across
    series, which is why it is the most honest single forecasting number (docs/07).
    """
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    train = np.asarray(train, float)
    denom = np.mean(np.abs(train[season:] - train[:-season]))
    if denom == 0:
        raise ValueError("naive in-sample error is 0; MASE undefined")
    return float(np.mean(np.abs(actual - forecast)) / denom)


def standardised_mean_difference(x_treat, x_control,
                                 w_treat=None, w_control=None) -> float:
    """|Standardised mean difference| between treated and control on one covariate.

    The balance diagnostic used in causal inference (docs/09). Want < 0.1 after
    adjustment. Pass weights to compute the *weighted* (post-IPW) balance.
    """
    xt, xc = np.asarray(x_treat, float), np.asarray(x_control, float)
    mt = np.average(xt, weights=w_treat) if w_treat is not None else xt.mean()
    mc = np.average(xc, weights=w_control) if w_control is not None else xc.mean()
    pooled_sd = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float(abs(mt - mc) / pooled_sd)


def clip_propensity(ps, eps: float = 0.01) -> np.ndarray:
    """Trim propensity scores away from {0,1} so IPW weights don't explode."""
    return np.clip(np.asarray(ps, float), eps, 1 - eps)
