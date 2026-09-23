"""
Builds an empirical calibration table for layer3's Bull/Bear/Stagnant regime
label: for every out-of-sample historical day where get_next_regime() said
"dominant_regime = Bull" (or Bear/Stagnant), what did the price ACTUALLY do
over the following regime_horizon_days?

Originally written to check the v1 layer3 (hand-typed transition matrix +
Hurst-exponent seeding, which conflated trend persistence with price
direction and never seeded Bear at all) — that run found dominant_regime
predicted "Bull" 100% of the time out-of-sample, with zero discriminative
signal (see docs/pipeline_guide.md). layer3.py has since been fixed
(fit_layer3_transition_matrix.py fits the matrix from realized transitions;
seed_state_from_prices seeds from realized trailing price direction instead
of Hurst). This script now re-validates the FIXED version the same way, and
seeds with the same realized-direction rule it checks the label against, so
seeding/fitting/validation all agree on what "Bull"/"Bear"/"Stagnant" means.
This mirrors build_calibration_table.py's approach for layer4's
win_probability: replace "trust the model's internal label" with "measure
what happened, out-of-sample, when the model said that."

Realized label (over the FORWARD window, to check the prediction against
what actually happens next — seeding itself uses the same threshold rule
but over the TRAILING window, see layer3.seed_state_from_prices): over the
next regime_horizon_days trading days,
    forward return > +up_threshold   -> realized Bull
    forward return < -down_threshold -> realized Bear
    otherwise                        -> realized Stagnant

Walk-forward folds (not one split) are the source of truth, same rationale
as build_calibration_table.py: every fold's test window is genuinely
out-of-sample relative to that fold's "training" period (here, "training"
only matters in the sense of keeping the same rolling train/test split
convention as the rest of the walk-forward tooling — layer3's transition
matrix itself is not fit per fold, since nothing in this script refits it;
run fit_layer3_transition_matrix.py separately to refit it).

Output: regime_calibration_table.json — a list of one entry per predicted
dominant_regime value ("Bull"/"Bear"/"Stagnant"), each with a sample count,
hit rate against the realized label, mean/median conditional forward
return, a 3x3 confusion matrix (predicted x realized), and a one-sample
t-test + bootstrap CI on the conditional forward return (same statistical
pattern backtest_layer4.py's summarize() uses for layer4 P&L).

Usage:
    python calibrate_layer3_regime.py --tickers NVDA,ANET,INTC,M --lookback-days 10y \
        --regime-horizon-days 5 --train-years 3 --test-months 6 --step-months 6
"""

import argparse
import json

import numpy as np
from scipy import stats

from layer1 import detect_signal
from layer3 import get_next_regime, seed_state_from_prices
from train_layer4 import fetch_prices_and_factors
from fold_checkpoint import resolve_fold_result

REGIMES = ["Bull", "Bear", "Stagnant"]
FEATURE_LOOKBACK = 20  # matches layer1/layer4's tail(20) convention — detect_signal needs this much history


def label_from_forward_return(forward_return: float, up_threshold: float, down_threshold: float) -> str:
    if forward_return > up_threshold:
        return "Bull"
    if forward_return < -down_threshold:
        return "Bear"
    return "Stagnant"


def build_regime_calls_for_ticker(ticker: str, prices, regime_horizon_days: int,
                                   up_threshold: float, down_threshold: float, steps: int = 5):
    """
    Walk forward day by day for a single ticker. At each valid day t:
      - dominant_regime predicted from data up to and including t (same
        detect_signal -> current_state seed -> get_next_regime call
        pipeline_core.py makes live)
      - realized label from the actual (t, t+regime_horizon_days] return
    Returns a list of {ticker, date, dominant_regime, bull_probability,
    realized_label, forward_return} dicts, one per valid day.

    steps: how many transition-matrix steps get_next_regime projects ahead.
    Kept as a parameter (rather than hardcoded 5) because a fitted matrix can
    mix close to its stationary distribution well before 5 steps, making
    dominant_regime nearly constant regardless of starting state — this
    script is how that gets checked empirically rather than assumed.
    """
    calls = []
    max_end = len(prices) - regime_horizon_days

    for i in range(FEATURE_LOOKBACK, max_end):
        t_date = prices.index[i]
        price_slice = prices.iloc[: i + 1]

        signal = detect_signal(price_slice)
        if np.isnan(signal["hurst_exponent"]):
            continue

        current_state = seed_state_from_prices(price_slice, regime_horizon_days, up_threshold, down_threshold)
        regime_probs = get_next_regime(current_state, steps=steps)

        forward_return = float(prices.iloc[i + regime_horizon_days] / prices.iloc[i] - 1.0)
        realized_label = label_from_forward_return(forward_return, up_threshold, down_threshold)

        calls.append({
            "ticker": ticker,
            "date": t_date,
            "dominant_regime": regime_probs["dominant_regime"],
            "bull_probability": regime_probs["bull_probability"],
            "realized_label": realized_label,
            "forward_return": forward_return,
        })

    return calls


def build_full_dataset(tickers, lookback_days):
    """Fetches each ticker's price history once. No factor/volume data needed —
    layer3's regime call only depends on price history via layer1's detect_signal."""
    prices_by_ticker, _, _ = fetch_prices_and_factors(tickers, lookback_days)
    return prices_by_ticker


def make_fold_boundaries(prices_by_ticker, train_years, test_months, step_months):
    """Same rolling-window convention as walkforward_layer4.py's make_fold_boundaries,
    applied to raw price dates instead of a pre-built features/labels dataset."""
    all_dates = sorted(set(d for p in prices_by_ticker.values() for d in p.index))
    overall_start, overall_end = all_dates[0], all_dates[-1]

    train_delta = np.timedelta64(int(train_years * 365.25), "D")
    test_delta = np.timedelta64(int(test_months * 30.44), "D")
    step_delta = np.timedelta64(int(step_months * 30.44), "D")

    folds = []
    train_start = np.datetime64(overall_start)
    while True:
        train_end = train_start + train_delta
        test_end = train_end + test_delta
        if test_end > np.datetime64(overall_end):
            break
        folds.append((train_start, train_end, test_end))
        train_start = train_start + step_delta

    return folds


def collect_all_oos_regime_calls(prices_by_ticker, train_years, test_months, step_months,
                                  regime_horizon_days, up_threshold, down_threshold, run_config, steps: int = 5):
    """
    For each rolling fold, computes regime calls for every ticker's test window only
    (the "train" window exists solely to keep the same rolling-split convention as
    walkforward_layer4.py/build_calibration_table.py — layer3's transition matrix and
    layer1's Hurst threshold are both fixed constants, not refit per fold — so a call
    only needs enough preceding history for detect_signal's rolling window, which is
    satisfied by prices from before the test window starts).
    """
    folds = make_fold_boundaries(prices_by_ticker, train_years, test_months, step_months)
    print(f"{len(folds)} rolling folds to collect out-of-sample regime calls from...")

    all_calls = []
    for i, (train_start, train_end, test_end) in enumerate(folds, 1):

        def compute_this_fold(train_end=train_end, test_end=test_end):
            fold_calls = []
            for ticker, prices in prices_by_ticker.items():
                test_mask = (prices.index >= train_end) & (prices.index < test_end)
                if test_mask.sum() < 5:
                    continue
                calls = build_regime_calls_for_ticker(
                    ticker, prices, regime_horizon_days, up_threshold, down_threshold, steps,
                )
                fold_calls.extend(c for c in calls if train_end <= c["date"] < test_end)
            return fold_calls

        fold_calls, from_cache = resolve_fold_result("regimecalib", run_config, i, compute_this_fold)
        all_calls.extend(fold_calls)
        cache_note = " (from checkpoint)" if from_cache else ""
        print(f"  fold {i}: collected {len(fold_calls)} out-of-sample regime calls{cache_note}")

    return all_calls


def build_regime_table(all_calls):
    table = []
    for predicted in REGIMES:
        bucket = [c for c in all_calls if c["dominant_regime"] == predicted]
        n = len(bucket)
        if n == 0:
            print(f"  {predicted}: n=0 — this predicted regime was never seen out-of-sample.")
            continue

        forward_returns = np.array([c["forward_return"] for c in bucket])
        hits = np.array([c["realized_label"] == predicted for c in bucket])
        confusion = {realized: sum(1 for c in bucket if c["realized_label"] == realized) for realized in REGIMES}

        entry = {
            "predicted_regime": predicted,
            "n": n,
            "hit_rate": float(hits.mean()),
            "mean_forward_return": float(forward_returns.mean()),
            "median_forward_return": float(np.median(forward_returns)),
            "confusion_realized_counts": confusion,
        }

        if n >= 2 and forward_returns.std() > 0:
            t_stat, p_value = stats.ttest_1samp(forward_returns, 0.0)
            rng = np.random.default_rng(42)
            n_bootstrap = 10000
            bootstrap_means = np.array([
                rng.choice(forward_returns, size=n, replace=True).mean() for _ in range(n_bootstrap)
            ])
            ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
            entry.update({
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
            })

        table.append(entry)
        print(f"  {predicted}: n={n:5d}  hit_rate={entry['hit_rate']:.1%}  "
              f"mean_fwd_return={entry['mean_forward_return']:.4%}  confusion={confusion}")

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Empirically calibrate layer3's Bull/Bear/Stagnant regime label against realized forward returns."
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
    parser.add_argument("--output", default="regime_calibration_table.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"Fetching price history for {tickers}...")
    prices_by_ticker = build_full_dataset(tickers, args.lookback_days)

    run_config = {
        "tickers": ",".join(sorted(tickers)), "lookback_days": args.lookback_days,
        "regime_horizon_days": args.regime_horizon_days, "steps": args.steps,
        "up_threshold": args.up_threshold, "down_threshold": args.down_threshold,
        "train_years": args.train_years, "test_months": args.test_months,
        "step_months": args.step_months,
    }

    all_calls = collect_all_oos_regime_calls(
        prices_by_ticker, args.train_years, args.test_months, args.step_months,
        args.regime_horizon_days, args.up_threshold, args.down_threshold, run_config, args.steps,
    )
    print(f"\nCollected {len(all_calls)} total out-of-sample regime calls across all folds.\n")

    table = build_regime_table(all_calls)

    with open(args.output, "w") as f:
        json.dump(table, f, indent=2, default=str)
    print(f"\nSaved regime calibration table to {args.output}")


if __name__ == "__main__":
    main()
