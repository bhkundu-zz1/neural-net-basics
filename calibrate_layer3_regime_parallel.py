"""
Ticker-level parallel version of calibrate_layer3_regime.py — same
methodology, same output schema, same CLI, just runs each fold's per-ticker
day-by-day walk (build_regime_calls_for_ticker) across a process pool
instead of one ticker at a time on a single core.

Why a separate file rather than editing calibrate_layer3_regime.py in place:
this was written while a real run of the original script was already
in-flight (see docs/pipeline_guide.md's regime-calibration numbers) —
multiprocessing changes worker-count/pickling behavior and is worth keeping
as an isolated, independently-testable variant rather than risking the
running script's fold_checkpoint cache keys or behavior mid-run. Once this
version is validated, it can replace calibrate_layer3_regime.py outright.

Why this is the bottleneck worth parallelizing: build_regime_calls_for_ticker
calls layer1.detect_signal (an np.polyfit-based Hurst exponent estimate) once
per day, per ticker — CPU-bound, pure-Python-loop work with no vectorized
fast path in this codebase. Each ticker's walk is fully independent of every
other ticker's, so it parallelizes with no shared state and no risk of
changing the result versus the sequential version (see
tests/test_calibrate_layer3_regime_parallel.py, which checks identical
output against calibrate_layer3_regime.py's sequential functions on the same
input).

Usage (identical to calibrate_layer3_regime.py, plus --workers):
    python calibrate_layer3_regime_parallel.py --tickers NVDA,ANET,INTC,M --lookback-days 10y \
        --regime-horizon-days 5 --train-years 3 --test-months 6 --step-months 6 --workers 8

--workers defaults to os.cpu_count() (all logical cores). Pass a lower
number to leave headroom for other work on the machine.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from calibrate_layer3_regime import (
    REGIMES,
    build_full_dataset,
    build_regime_calls_for_ticker,
    build_regime_table,
    make_fold_boundaries,
)
from fold_checkpoint import resolve_fold_result


def _ticker_worker(args):
    """
    Top-level (picklable) worker: computes one ticker's regime calls for one
    fold's test window. Takes a single tuple argument (not **kwargs) since
    ProcessPoolExecutor.map pickles positional args one at a time — a single
    tuple keeps the call site simple and avoids a lambda (which isn't
    picklable, and would silently make this fall back to serial execution
    on Windows' spawn-based multiprocessing).
    """
    ticker, prices, regime_horizon_days, up_threshold, down_threshold, steps, train_end, test_end = args
    calls = build_regime_calls_for_ticker(
        ticker, prices, regime_horizon_days, up_threshold, down_threshold, steps,
    )
    return [c for c in calls if train_end <= c["date"] < test_end]


def collect_all_oos_regime_calls_parallel(prices_by_ticker, train_years, test_months, step_months,
                                           regime_horizon_days, up_threshold, down_threshold, run_config,
                                           steps: int = 5, workers: int | None = None):
    """
    Same fold-by-fold structure and fold_checkpoint caching as
    calibrate_layer3_regime.py's collect_all_oos_regime_calls — only the
    inner per-ticker loop is parallelized. A crash or interrupt still
    resumes cleanly at the next un-checkpointed fold; only the WORK inside
    one fold's compute_this_fold is now spread across processes.
    """
    folds = make_fold_boundaries(prices_by_ticker, train_years, test_months, step_months)
    print(f"{len(folds)} rolling folds to collect out-of-sample regime calls from...")

    worker_count = workers or os.cpu_count() or 1
    print(f"Using up to {worker_count} worker processes per fold.")

    all_calls = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for i, (train_start, train_end, test_end) in enumerate(folds, 1):

            def compute_this_fold(train_end=train_end, test_end=test_end):
                tasks = []
                for ticker, prices in prices_by_ticker.items():
                    test_mask = (prices.index >= train_end) & (prices.index < test_end)
                    if test_mask.sum() < 5:
                        continue
                    tasks.append((
                        ticker, prices, regime_horizon_days, up_threshold, down_threshold,
                        steps, train_end, test_end,
                    ))

                fold_calls = []
                for ticker_calls in pool.map(_ticker_worker, tasks):
                    fold_calls.extend(ticker_calls)
                return fold_calls

            fold_calls, from_cache = resolve_fold_result("regimecalib", run_config, i, compute_this_fold)
            all_calls.extend(fold_calls)
            cache_note = " (from checkpoint)" if from_cache else ""
            print(f"  fold {i}: collected {len(fold_calls)} out-of-sample regime calls{cache_note}")

    return all_calls


def main():
    parser = argparse.ArgumentParser(
        description="Empirically calibrate layer3's Bull/Bear/Stagnant regime label against realized "
                     "forward returns (ticker-level parallel version)."
    )
    parser.add_argument("--tickers", default="NVDA,ANET,INTC,M")
    parser.add_argument("--lookback-days", default="10y")
    parser.add_argument("--regime-horizon-days", type=int, default=5,
                         help="Forward-return window used both to realize the Bull/Bear/Stagnant "
                              "label and to match get_next_regime's steps projection horizon.")
    parser.add_argument("--steps", type=int, default=5,
                         help="How many transition-matrix steps get_next_regime projects ahead "
                              "(default 5, matching live inference). Lower this to check whether "
                              "dominant_regime still depends on the seeded starting state, or has "
                              "already mixed to the matrix's stationary distribution.")
    parser.add_argument("--up-threshold", type=float, default=0.02,
                         help="Forward return above which a day is realized as Bull (raw fraction, e.g. 0.02 = 2%%).")
    parser.add_argument("--down-threshold", type=float, default=0.02,
                         help="Forward return below which (negated) a day is realized as Bear.")
    parser.add_argument("--train-years", type=float, default=3.0)
    parser.add_argument("--test-months", type=float, default=6.0)
    parser.add_argument("--step-months", type=float, default=6.0)
    parser.add_argument("--workers", type=int, default=None,
                         help="Number of worker processes per fold (default: os.cpu_count(), all logical cores).")
    parser.add_argument("--output", default="regime_calibration_table.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"Fetching price history for {tickers}...")
    prices_by_ticker = build_full_dataset(tickers, args.lookback_days)

    # Same run_config shape (and therefore the same fold_checkpoint cache keys) as
    # calibrate_layer3_regime.py's sequential version — a fold already completed by
    # either script is reused by the other; workers is deliberately excluded since it
    # doesn't change the RESULT, only how fast it's computed.
    run_config = {
        "tickers": ",".join(sorted(tickers)), "lookback_days": args.lookback_days,
        "regime_horizon_days": args.regime_horizon_days, "steps": args.steps,
        "up_threshold": args.up_threshold, "down_threshold": args.down_threshold,
        "train_years": args.train_years, "test_months": args.test_months,
        "step_months": args.step_months,
    }

    all_calls = collect_all_oos_regime_calls_parallel(
        prices_by_ticker, args.train_years, args.test_months, args.step_months,
        args.regime_horizon_days, args.up_threshold, args.down_threshold, run_config,
        args.steps, args.workers,
    )
    print(f"\nCollected {len(all_calls)} total out-of-sample regime calls across all folds.\n")

    table = build_regime_table(all_calls)

    with open(args.output, "w") as f:
        json.dump(table, f, indent=2, default=str)
    print(f"\nSaved regime calibration table to {args.output}")


if __name__ == "__main__":
    main()
