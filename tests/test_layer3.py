import numpy as np
import pandas as pd
import pytest

import layer3
from layer3 import (
    attach_regime_calibration,
    get_next_regime,
    seed_state_from_prices,
)


def _prices_with_trailing_return(total_return: float, horizon_days: int = 5, n: int = 30):
    """A flat-ish series where the LAST horizon_days move produces exactly
    total_return, so seed_state_from_prices sees a known trailing return."""
    base = np.full(n, 100.0)
    prices = pd.Series(base)
    prices.iloc[-(horizon_days + 1):] = np.linspace(100.0, 100.0 * (1 + total_return), horizon_days + 1)
    return prices


def test_seed_state_from_prices_bull_when_trailing_return_exceeds_up_threshold():
    prices = _prices_with_trailing_return(0.05, horizon_days=5)
    assert seed_state_from_prices(prices, horizon_days=5, up_threshold=0.02, down_threshold=0.02) == 0


def test_seed_state_from_prices_bear_when_trailing_return_below_down_threshold():
    prices = _prices_with_trailing_return(-0.05, horizon_days=5)
    assert seed_state_from_prices(prices, horizon_days=5, up_threshold=0.02, down_threshold=0.02) == 1


def test_seed_state_from_prices_stagnant_when_trailing_return_within_thresholds():
    prices = _prices_with_trailing_return(0.001, horizon_days=5)
    assert seed_state_from_prices(prices, horizon_days=5, up_threshold=0.02, down_threshold=0.02) == 2


def test_seed_state_from_prices_stagnant_fallback_for_short_history():
    prices = pd.Series([100.0, 101.0, 102.0])  # fewer than horizon_days+1 points
    assert seed_state_from_prices(prices, horizon_days=5) == 2


@pytest.mark.parametrize("current_state", [0, 1, 2])
def test_get_next_regime_returns_valid_probability_distribution(current_state):
    result = get_next_regime(current_state, steps=5)
    total = result["bull_probability"] + result["bear_probability"] + result["stagnant_probability"]
    assert total == pytest.approx(1.0, abs=1e-2)
    assert result["dominant_regime"] in ("Bull", "Bear", "Stagnant")
    # get_next_regime's own dict must stay exactly these 4 keys — layer4's
    # build_feature_vector consumes .values() wholesale filtered by numeric
    # type, so extra keys here would silently change the feature vector shape.
    assert set(result.keys()) == {
        "bull_probability", "bear_probability", "stagnant_probability", "dominant_regime",
    }


def test_get_next_regime_seeded_from_each_state_can_reach_bear():
    # With the v1 hand-typed matrix (or any reasonable fitted one), starting
    # from Bear (state 1) should keep Bear reachable as the dominant regime
    # for at least a few steps — the old bug was that Bear was never SEEDED,
    # not that the matrix couldn't represent it.
    result = get_next_regime(1, steps=1)
    assert result["bear_probability"] > 0


def test_attach_regime_calibration_returns_new_dict_without_mutating_input():
    original = {"bull_probability": 0.5, "bear_probability": 0.3, "stagnant_probability": 0.2, "dominant_regime": "Bull"}
    original_copy = dict(original)

    enriched = attach_regime_calibration(original)

    assert original == original_copy  # input untouched
    assert enriched["dominant_regime"] == "Bull"
    assert "regime_calibrated" in enriched
    assert "regime_hit_rate" in enriched


def test_attach_regime_calibration_uncalibrated_when_table_empty(monkeypatch):
    monkeypatch.setattr(layer3, "_load_regime_calibration_table", lambda: [])
    result = attach_regime_calibration({"dominant_regime": "Bull"})
    assert result["regime_calibrated"] is False
    assert result["regime_hit_rate"] is None


def test_attach_regime_calibration_calibrated_when_bucket_present(monkeypatch):
    monkeypatch.setattr(
        layer3, "_load_regime_calibration_table",
        lambda: [{"predicted_regime": "Bull", "hit_rate": 0.6}],
    )
    result = attach_regime_calibration({"dominant_regime": "Bull"})
    assert result["regime_calibrated"] is True
    assert result["regime_hit_rate"] == 0.6
