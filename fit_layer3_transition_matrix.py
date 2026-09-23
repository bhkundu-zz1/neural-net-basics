"""
Empirically fits layer3.py's Bull/Bear/Stagnant transition matrix from
realized price action, replacing the hand-typed constant that shipped in
v1 (see docs/pipeline_guide.md's "layer3's regime label carries no signal"
finding, produced by calibrate_layer3_regime.py).

Two things were wrong with the v1 matrix, both fixed here:
  1. It was invented, not fit from data — transition_matrix in layer3.py had
     no computation behind it at all.
  2. current_state was seeded from layer1's Hurst exponent (trending vs.
     mean-reverting), which measures trend PERSISTENCE, not price DIRECTION
     — a persistent downtrend reads as "trending" and got seeded into Bull
     (state 0). Bear (state 1) was never reachable as a seed at all.

Realized state per day (same definition calibrate_layer3_regime.py already
validates the label against, applied here to the TRAILING window instead of
the forward one, since a transition matrix describes day-to-day state
evolution): over the trailing regime_horizon_days trading days ending at t,
    trailing return > +up_threshold   -> Bull
    trailing return < -down_threshold -> Bear
    otherwise                         -> Stagnant

The matrix is fit by counting realized state[t] -> state[t+regime_horizon_days]
transitions across all tickers/dates and normalizing each row to sum to 1
(Laplace-smoothed with alpha=1 so no transition ever gets a hard zero
probability from limited data). Pooled across the SAME ticker universe and
lookback calibrate_layer3_regime.py uses by default, so the two scripts'
numbers are directly comparable.

Output: layer3_transition_matrix.json — the fitted 3x3 matrix plus metadata
(tickers, lookback, transition counts) recording exactly what it was fit
from. layer3.py loads this at import time, falling back to the original
hand-typed constant if the file is missing (e.g. a fresh checkout before
anyone has run this script).

Usage:
    python fit_layer3_transition_matrix.py --tickers NVDA,ANET,INTC,M --lookback-days 10y \
        --regime-horizon-days 5
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np

from train_layer4 import fetch_prices_and_factors

STATES = ["Bull", "Bear", "Stagnant"]
STATE_INDEX = {name: i for i, name in enumerate(STATES)}


def realized_state_from_trailing_return(trailing_return: float, up_threshold: float, down_threshold: float) -> str:
    if trailing_return > up_threshold:
        return "Bull"
    if trailing_return < -down_threshold:
        return "Bear"
    return "Stagnant"


def realized_state_sequence(prices, regime_horizon_days: int, up_threshold: float, down_threshold: float):
    """
    Returns a list of (date, state) for every day t with regime_horizon_days
    of trailing history available. state[t] is realized from the trailing
    return over (t - regime_horizon_days, t].
    """
    sequence = []
    for i in range(regime_horizon_days, len(prices)):
        trailing_return = float(prices.iloc[i] / prices.iloc[i - regime_horizon_days] - 1.0)
        state = realized_state_from_trailing_return(trailing_return, up_threshold, down_threshold)
        sequence.append((prices.index[i], state))
    return sequence


def count_transitions(prices_by_ticker, regime_horizon_days, up_threshold, down_threshold):
    """
    Counts state[t] -> state[t + regime_horizon_days] transitions (non-overlapping
    with the trailing window used to realize state[t] itself, so a transition
    reflects two genuinely different, non-overlapping windows of price action)
    for every ticker, pooled into one 3x3 count matrix.
    """
    counts = np.zeros((3, 3))
    n_transitions = 0

    for ticker, prices in prices_by_ticker.items():
        sequence = realized_state_sequence(prices, regime_horizon_days, up_threshold, down_threshold)
        state_by_date = {date: state for date, state in sequence}
        dates = [date for date, _ in sequence]

        for i in range(len(dates) - regime_horizon_days):
            from_date = dates[i]
            to_date = dates[i + regime_horizon_days]
            from_state = state_by_date[from_date]
            to_state = state_by_date[to_date]
            counts[STATE_INDEX[from_state], STATE_INDEX[to_state]] += 1
            n_transitions += 1

    return counts, n_transitions


def fit_transition_matrix(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Row-normalizes counts into a transition matrix. Laplace/additive
    smoothing (alpha=1, i.e. add one pseudo-count to every cell before
    normalizing) so a transition never observed in this sample (e.g.
    Bear -> Bull, if Bear itself is rare) still gets a small nonzero
    probability rather than a hard zero baked into the matrix.
    """
    smoothed = counts + alpha
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fit layer3.py's Bull/Bear/Stagnant transition matrix from realized price action."
    )
    parser.add_argument("--tickers", default="NVDA,ANET,INTC,M")
    parser.add_argument("--lookback-days", default="10y")
    parser.add_argument("--regime-horizon-days", type=int, default=5,
                         help="Trailing/forward window used to realize Bull/Bear/Stagnant state "
                              "and the step between transitions. Matches get_next_regime's steps=5.")
    parser.add_argument("--up-threshold", type=float, default=0.02)
    parser.add_argument("--down-threshold", type=float, default=0.02)
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--output", default="layer3_transition_matrix.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"Fetching price history for {tickers}...")
    prices_by_ticker, _, _ = fetch_prices_and_factors(tickers, args.lookback_days)

    print("Counting realized state transitions...")
    counts, n_transitions = count_transitions(
        prices_by_ticker, args.regime_horizon_days, args.up_threshold, args.down_threshold,
    )
    print(f"  {n_transitions} total transitions counted across {len(prices_by_ticker)} tickers.")
    for i, from_state in enumerate(STATES):
        row_total = int(counts[i].sum())
        print(f"  From {from_state} (n={row_total}): " +
              ", ".join(f"{to_state}={int(counts[i, j])}" for j, to_state in enumerate(STATES)))

    matrix = fit_transition_matrix(counts, alpha=args.smoothing_alpha)
    print("\nFitted transition matrix (rows=from, cols=to, order=Bull/Bear/Stagnant):")
    print(matrix)

    output = {
        "transition_matrix": matrix.tolist(),
        "states": STATES,
        "fitted_from": {
            "tickers": tickers,
            "lookback_days": args.lookback_days,
            "regime_horizon_days": args.regime_horizon_days,
            "up_threshold": args.up_threshold,
            "down_threshold": args.down_threshold,
            "smoothing_alpha": args.smoothing_alpha,
            "n_transitions": n_transitions,
            "raw_counts": counts.tolist(),
        },
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved fitted transition matrix to {args.output}")
    print("layer3.py will load this automatically if present.")


if __name__ == "__main__":
    main()
