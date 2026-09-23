import numpy as np
import pandas as pd
import pytest

from fit_layer3_transition_matrix import (
    count_transitions,
    fit_transition_matrix,
    realized_state_from_trailing_return,
    realized_state_sequence,
)


@pytest.mark.parametrize(
    "trailing_return,up_threshold,down_threshold,expected",
    [
        (0.03, 0.02, 0.02, "Bull"),
        (0.02, 0.02, 0.02, "Stagnant"),
        (-0.03, 0.02, 0.02, "Bear"),
        (-0.02, 0.02, 0.02, "Stagnant"),
        (0.0, 0.02, 0.02, "Stagnant"),
    ],
)
def test_realized_state_from_trailing_return(trailing_return, up_threshold, down_threshold, expected):
    assert realized_state_from_trailing_return(trailing_return, up_threshold, down_threshold) == expected


def test_realized_state_sequence_length_matches_available_history():
    prices = pd.Series(np.linspace(100, 110, 30), index=pd.bdate_range("2024-01-01", periods=30))
    sequence = realized_state_sequence(prices, regime_horizon_days=5, up_threshold=0.02, down_threshold=0.02)
    assert len(sequence) == 30 - 5
    for date, state in sequence:
        assert state in ("Bull", "Bear", "Stagnant")


def test_realized_state_sequence_detects_steady_uptrend_as_mostly_bull():
    prices = pd.Series(100 * np.cumprod(1 + np.full(60, 0.01)), index=pd.bdate_range("2024-01-01", periods=60))
    sequence = realized_state_sequence(prices, regime_horizon_days=5, up_threshold=0.02, down_threshold=0.02)
    bull_fraction = sum(1 for _, s in sequence if s == "Bull") / len(sequence)
    assert bull_fraction > 0.8


def test_count_transitions_counts_bull_to_bull_for_steady_uptrend():
    prices = pd.Series(100 * np.cumprod(1 + np.full(60, 0.01)), index=pd.bdate_range("2024-01-01", periods=60))
    counts, n_transitions = count_transitions({"TEST": prices}, regime_horizon_days=5, up_threshold=0.02, down_threshold=0.02)
    assert n_transitions > 0
    # Bull is state index 0 — a steady uptrend should show most mass on Bull->Bull.
    assert counts[0, 0] == counts.max()


def test_count_transitions_empty_for_too_short_series():
    prices = pd.Series([100.0, 101.0, 102.0], index=pd.bdate_range("2024-01-01", periods=3))
    counts, n_transitions = count_transitions({"TEST": prices}, regime_horizon_days=5, up_threshold=0.02, down_threshold=0.02)
    assert n_transitions == 0
    assert counts.sum() == 0


def test_fit_transition_matrix_rows_sum_to_one():
    counts = np.array([
        [50, 10, 5],
        [8, 40, 12],
        [15, 15, 60],
    ], dtype=float)
    matrix = fit_transition_matrix(counts, alpha=1.0)
    for row in matrix:
        assert row.sum() == pytest.approx(1.0)


def test_fit_transition_matrix_smoothing_avoids_hard_zeros():
    # A row with all-zero counts (state never observed transitioning) should
    # still produce a valid, non-degenerate probability distribution thanks
    # to Laplace smoothing, not a divide-by-zero or all-zero row.
    counts = np.array([
        [50, 10, 5],
        [0, 0, 0],
        [15, 15, 60],
    ], dtype=float)
    matrix = fit_transition_matrix(counts, alpha=1.0)
    assert matrix[1].sum() == pytest.approx(1.0)
    assert all(p > 0 for p in matrix[1])
    assert matrix[1, 0] == pytest.approx(matrix[1, 1]) == pytest.approx(matrix[1, 2])  # uniform when no data
