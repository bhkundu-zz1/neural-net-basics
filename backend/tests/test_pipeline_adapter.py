import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import warnings

import pandas as pd
import pytest

import pipeline_core
import pipeline_adapter


def _fake_flat_fetch(ticker, lookback):
    prices = pd.Series([1.0] * 100)
    volume = pd.Series([1000] * 100)
    factor_returns = pd.DataFrame({
        "market": [0.001] * 99, "size": [0.001] * 99,
        "value": [0.001] * 99, "momentum": [0.001] * 99,
    })
    return prices, volume, factor_returns


def test_strips_net_and_checkpoint_from_degenerate_result(monkeypatch):
    monkeypatch.setattr(pipeline_core, "fetch_prices_and_factors", _fake_flat_fetch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pipeline_adapter.get_signal(
            "VMFXX", shares=100, lookback="2y", weights_path="nonexistent_weights.pt",
            min_confidence=0.75, net=None, checkpoint=None,
        )

    assert "net" not in result
    assert "checkpoint" not in result
    assert result["should_trade"] is False
    assert result["direction"] == "flat"
    assert result["factor_result"] is None
    assert result["edge_probs"] is None


def test_normal_path_keeps_rich_fields_and_strips_non_serializable(monkeypatch):
    rng_prices = pd.Series([100.0 + i * 0.3 + (i % 5) for i in range(300)])
    rng_volume = pd.Series([1000 + i for i in range(300)])
    factor_returns = pd.DataFrame({
        "market": [0.0005 * ((-1) ** i) for i in range(299)],
        "size": [0.0003 * ((-1) ** i) for i in range(299)],
        "value": [0.0002 * ((-1) ** i) for i in range(299)],
        "momentum": [0.0004 * ((-1) ** i) for i in range(299)],
    })

    def _fake_normal_fetch(ticker, lookback):
        return rng_prices, rng_volume, factor_returns

    monkeypatch.setattr(pipeline_core, "fetch_prices_and_factors", _fake_normal_fetch)

    result = pipeline_adapter.get_signal(
        "NVDA", shares=100, lookback="2y", weights_path="nonexistent_weights.pt",
        min_confidence=0.75, net=None, checkpoint=None,
    )

    assert "net" not in result
    assert "checkpoint" not in result
    assert result["factor_result"] is not None
    assert result["edge_probs"] is not None
    assert set(result["edge_probs"].keys()) == {"long", "short", "flat"}
    assert result["direction"] in ("long", "short")
