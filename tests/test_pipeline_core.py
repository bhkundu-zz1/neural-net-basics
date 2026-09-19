import warnings

import pandas as pd
import pytest

import pipeline_core


def _fake_flat_fetch(ticker, lookback):
    prices = pd.Series([1.0] * 100)
    volume = pd.Series([1000] * 100)
    factor_returns = pd.DataFrame({
        "market": [0.001] * 99, "size": [0.001] * 99,
        "value": [0.001] * 99, "momentum": [0.001] * 99,
    })
    return prices, volume, factor_returns


def test_undefined_regime_short_circuits_to_hold(monkeypatch):
    monkeypatch.setattr(pipeline_core, "fetch_prices_and_factors", _fake_flat_fetch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pipeline_core.run_pipeline_for_ticker(
            "VMFXX", shares=100, weights_path="nonexistent_weights.pt", min_confidence=0.75,
        )

    assert result["signal"]["regime"] == "undefined"
    assert result["should_trade"] is False
    assert result["direction"] == "flat"
    assert result["position_size"] == 0.0
    assert result["factor_result"] is None
    assert result["edge_probs"] is None
