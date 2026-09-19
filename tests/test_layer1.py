import warnings

import numpy as np
import pandas as pd
import pytest

from layer1 import compute_hurst, detect_signal


def test_compute_hurst_flat_series_returns_nan_without_warning():
    flat = np.array([1.0] * 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_hurst(flat)
    assert np.isnan(result)


def test_compute_hurst_normal_series_returns_a_number():
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.standard_normal(200))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compute_hurst(prices)
    assert not np.isnan(result)


def test_detect_signal_flat_series_reports_undefined_regime_without_warning():
    flat_prices = pd.Series([1.0] * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = detect_signal(flat_prices)
    assert result["regime"] == "undefined"
    assert np.isnan(result["hurst_exponent"])
    assert np.isnan(result["autocorrelation"])


def test_detect_signal_normal_series_reports_a_real_regime():
    rng = np.random.default_rng(7)
    prices = pd.Series(100 + np.cumsum(rng.standard_normal(200)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = detect_signal(prices)
    assert result["regime"] in ("trending", "mean_reverting")
    assert not np.isnan(result["hurst_exponent"])
