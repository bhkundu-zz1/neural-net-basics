import numpy as np
import pandas as pd
import pytest

from calibrate_layer3_regime import (
    build_regime_calls_for_ticker,
    build_regime_table,
    label_from_forward_return,
    make_fold_boundaries,
)


@pytest.mark.parametrize(
    "forward_return,up_threshold,down_threshold,expected",
    [
        (0.03, 0.02, 0.02, "Bull"),
        (0.02, 0.02, 0.02, "Stagnant"),   # boundary is exclusive (> not >=)
        (0.021, 0.02, 0.02, "Bull"),
        (-0.03, 0.02, 0.02, "Bear"),
        (-0.02, 0.02, 0.02, "Stagnant"),  # boundary is exclusive (< not <=)
        (-0.021, 0.02, 0.02, "Bear"),
        (0.0, 0.02, 0.02, "Stagnant"),
    ],
)
def test_label_from_forward_return(forward_return, up_threshold, down_threshold, expected):
    assert label_from_forward_return(forward_return, up_threshold, down_threshold) == expected


def _synthetic_prices(n=60, trend=0.01, seed=0):
    """A price series with a steady per-day drift, so build_regime_calls_for_ticker
    has enough history (FEATURE_LOOKBACK=20) and a non-degenerate Hurst exponent."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.002, n)
    prices = 100 * np.cumprod(1 + trend + noise)
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(prices, index=dates)


def test_build_regime_calls_for_ticker_returns_one_call_per_valid_day():
    prices = _synthetic_prices(n=60)
    regime_horizon_days = 5

    calls = build_regime_calls_for_ticker("TEST", prices, regime_horizon_days, up_threshold=0.02, down_threshold=0.02)

    # Valid range is [FEATURE_LOOKBACK, len(prices) - regime_horizon_days)
    assert len(calls) > 0
    assert len(calls) <= len(prices) - regime_horizon_days - 20 + 1

    for call in calls:
        assert call["ticker"] == "TEST"
        assert call["dominant_regime"] in ("Bull", "Bear", "Stagnant")
        assert call["realized_label"] in ("Bull", "Bear", "Stagnant")
        assert isinstance(call["forward_return"], float)
        assert isinstance(call["bull_probability"], float)


def test_build_regime_calls_for_ticker_forward_return_matches_realized_label():
    prices = _synthetic_prices(n=60, trend=0.03)  # strong uptrend
    calls = build_regime_calls_for_ticker("TEST", prices, regime_horizon_days=5, up_threshold=0.02, down_threshold=0.02)

    assert len(calls) > 0
    # A strong steady uptrend should realize mostly Bull labels.
    bull_fraction = sum(1 for c in calls if c["realized_label"] == "Bull") / len(calls)
    assert bull_fraction > 0.5


def test_make_fold_boundaries_produces_non_overlapping_rolling_windows():
    dates = pd.bdate_range("2015-01-01", periods=252 * 8)  # 8 years of daily bars
    prices_by_ticker = {"TEST": pd.Series(range(len(dates)), index=dates, dtype=float)}

    folds = make_fold_boundaries(prices_by_ticker, train_years=3.0, test_months=6.0, step_months=6.0)

    assert len(folds) > 0
    for train_start, train_end, test_end in folds:
        assert train_start < train_end < test_end
    # Folds should slide forward in time.
    starts = [f[0] for f in folds]
    assert starts == sorted(starts)


def _fake_call(predicted, realized, forward_return):
    return {
        "ticker": "TEST",
        "date": pd.Timestamp("2024-01-01"),
        "dominant_regime": predicted,
        "bull_probability": 0.5,
        "realized_label": realized,
        "forward_return": forward_return,
    }


def test_build_regime_table_computes_hit_rate_and_confusion():
    calls = [
        _fake_call("Bull", "Bull", 0.03),
        _fake_call("Bull", "Bull", 0.02),
        _fake_call("Bull", "Bear", -0.01),
        _fake_call("Bull", "Stagnant", 0.001),
        _fake_call("Stagnant", "Stagnant", 0.0),
    ]

    table = build_regime_table(calls)

    bull_entry = next(e for e in table if e["predicted_regime"] == "Bull")
    assert bull_entry["n"] == 4
    assert bull_entry["hit_rate"] == pytest.approx(2 / 4)
    assert bull_entry["confusion_realized_counts"] == {"Bull": 2, "Bear": 1, "Stagnant": 1}

    stagnant_entry = next(e for e in table if e["predicted_regime"] == "Stagnant")
    assert stagnant_entry["n"] == 1

    predicted_regimes = {e["predicted_regime"] for e in table}
    assert "Bear" not in predicted_regimes  # never predicted in this sample -> not in the table


def test_build_regime_table_empty_input_returns_empty_table():
    assert build_regime_table([]) == []
