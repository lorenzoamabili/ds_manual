"""Tests for dsmanual.metrics — the shared, project-critical utilities.

These are behavioural tests: known-answer checks, invariants, and the guardrails
that should raise rather than silently return nonsense.
"""
import numpy as np
import pytest

from dsmanual import clip_propensity, mape, mase, seasonal_naive, standardised_mean_difference


# ----------------------------------------------------------------- seasonal_naive
def test_seasonal_naive_repeats_last_season():
    train = [1, 2, 3, 4]          # season=4 -> next 4 repeat the last season
    out = seasonal_naive(train, horizon=4, season=4)
    assert np.allclose(out, [1, 2, 3, 4])

def test_seasonal_naive_wraps_around():
    out = seasonal_naive([10, 20], horizon=5, season=2)
    assert np.allclose(out, [10, 20, 10, 20, 10])

def test_seasonal_naive_too_short_raises():
    with pytest.raises(ValueError):
        seasonal_naive([1, 2], horizon=3, season=4)


# ----------------------------------------------------------------- mape
def test_mape_perfect_forecast_is_zero():
    assert mape([100, 200], [100, 200]) == 0.0

def test_mape_known_value():
    # actual 100, forecast 110 -> 10% error
    assert mape([100], [110]) == pytest.approx(10.0)

def test_mape_zero_actual_raises():
    with pytest.raises(ValueError):
        mape([0, 100], [1, 100])


# ----------------------------------------------------------------- mase
def test_mase_below_one_when_beating_naive():
    train = np.arange(1, 25, dtype=float)          # smooth trend
    actual = np.array([25, 26, 27], dtype=float)
    good = actual.copy()                            # perfect forecast
    assert mase(actual, good, train, season=1) == pytest.approx(0.0)

def test_mase_scale_free():
    train = np.array([1.0, 2, 1, 2, 1, 2])
    a, f = [2.0, 1], [1.0, 2]
    s1 = mase(a, f, train)
    s2 = mase(np.array(a) * 1000, np.array(f) * 1000, train * 1000)
    assert s1 == pytest.approx(s2)                  # scaling data doesn't change MASE


# ----------------------------------------------------------------- SMD
def test_smd_zero_when_identical():
    x = np.array([1.0, 2, 3, 4])
    assert standardised_mean_difference(x, x) == pytest.approx(0.0)

def test_smd_positive_when_shifted():
    treat = np.array([2.0, 3, 4])
    control = np.array([0.0, 1, 2])
    assert standardised_mean_difference(treat, control) > 0

def test_smd_weighting_can_rebalance():
    # control skewed low; up-weighting its high tail should shrink the gap
    treat = np.array([5.0, 6, 7])
    control = np.array([1.0, 2, 7])
    unweighted = standardised_mean_difference(treat, control)
    w = np.array([0.1, 0.1, 5.0])                   # emphasise the high control value
    weighted = standardised_mean_difference(treat, control, w_control=w)
    assert weighted < unweighted


# ----------------------------------------------------------------- clip_propensity
def test_clip_propensity_bounds():
    out = clip_propensity([0.0, 0.5, 1.0], eps=0.01)
    assert out.min() >= 0.01 and out.max() <= 0.99
    assert out[1] == pytest.approx(0.5)             # interior values untouched
