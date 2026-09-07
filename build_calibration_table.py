"""
Builds an empirical calibration table mapping the model's confidence
(win_probability) to what actually happens historically at that confidence
level: win rate, average win/loss magnitude, and average directional P&L.

Why: trade_glue.py's win_loss_ratio and edge_bps were hand-picked formulas
invented before any empirical data existed (win_loss_ratio from recent
raw price swings, unrelated to when the model itself is right or wrong;
edge_bps a linear guess at how confidence maps to "edge"). This script
replaces those guesses with real measurements: for each confidence bucket,
what did trades at that confidence level actually do, out-of-sample,
across the walk-forward folds?

Uses walk-forward folds (not one train/test split) as the source of truth,
since every fold's test predictions are genuinely out-of-sample relative to
that fold's training — pooling across 11 independent folds gives a much
more trustworthy empirical sample than one split.

Output: calibration_table.json — a list of buckets, each with edges, a
sample count, empirical win rate, avg win/loss size, and avg directional
P&L. trade_glue.py loads this at runtime and looks up the bucket matching
the model's current confidence.

Usage:
    python build_calibration_table.py --tickers NVDA,ANET,INTC,M --lookback-days 10y \
        --train-years 3 --test-months 6 --step-months 6
"""

import argparse
import json

import numpy as np

from walkforward_layer4 import (
    build_full_dataset, make_fold_boundaries, slice_fold, train_fold, evaluate_fold,
)

BUCKET_EDGES = [0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.01]


def collect_all_oos_trades(per_ticker, train_years, test_months, step_months,
                            hidden_sizes, dropout, weight_decay, lr, batch_size, epochs):
    folds = make_fold_boundaries(per_ticker, train_years, test_months, step_months)
    print(f"{len(folds)} rolling folds to collect out-of-sample trades from...")

    all_trades = []
    for i, (train_start, train_end, test_end) in enumerate(folds, 1):
        sliced = slice_fold(per_ticker, train_start, train_end, test_end)
        if sliced is None:
            continue
        x_train, y_train, test_records = sliced
        n_test = sum(len(r["labels"]) for r in test_records)
        if len(y_train) < 100 or n_test < 20:
            continue

        net, mean, std = train_fold(x_train, y_train, hidden_sizes, dropout, weight_decay, lr, batch_size, epochs)
        # min_confidence=0.0: record EVERY prediction, unfiltered — the calibration
        # table needs coverage across the full confidence range, not just the subset
        # that would have passed should_trade under the OLD (unfixed) glue.
        _, _, _, _, _, _, trades = evaluate_fold(net, mean, std, test_records, min_confidence=0.0)
        all_trades.extend(trades)
        print(f"  fold {i}: collected {len(trades)} out-of-sample predictions")

    return all_trades


def build_table(all_trades):
    confidences = np.array([t["confidence"] for t in all_trades])
    directional_returns = np.array([t["directional_return"] for t in all_trades])

    table = []
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        n = int(mask.sum())
        if n == 0:
            continue

        bucket_returns = directional_returns[mask]
        wins = bucket_returns[bucket_returns > 0]
        losses = bucket_returns[bucket_returns <= 0]

        entry = {
            "conf_low": lo,
            "conf_high": hi,
            "n": n,
            "win_rate": float((bucket_returns > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_loss": float(abs(losses.mean())) if len(losses) > 0 else 0.0,
            "avg_directional_return": float(bucket_returns.mean()),
        }
        entry["win_loss_ratio"] = entry["avg_win"] / entry["avg_loss"] if entry["avg_loss"] > 0 else 10.0
        table.append(entry)
        print(f"  [{lo:.2f},{hi:.2f}) n={n:4d}  win_rate={entry['win_rate']:.1%}  "
              f"avg_win={entry['avg_win']:.3%}  avg_loss={entry['avg_loss']:.3%}  "
              f"win_loss_ratio={entry['win_loss_ratio']:.2f}  avg_dir_return={entry['avg_directional_return']:.4%}")

    return table


def main():
    parser = argparse.ArgumentParser(description="Build an empirical calibration table from walk-forward folds.")
    parser.add_argument("--tickers", default="NVDA,ANET,INTC,M")
    parser.add_argument("--lookback-days", default="10y")
    parser.add_argument("--forward-days", type=int, default=5)
    parser.add_argument("--label-mode", choices=["risk_adjusted", "raw"], default="raw")
    parser.add_argument("--up-threshold", type=float, default=0.02)
    parser.add_argument("--down-threshold", type=float, default=0.02)
    parser.add_argument("--train-years", type=float, default=3.0)
    parser.add_argument("--test-months", type=float, default=6.0)
    parser.add_argument("--step-months", type=float, default=6.0)
    parser.add_argument("--hidden-sizes", default="32,16")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="calibration_table.json")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))

    print(f"Building full dataset for {tickers}...")
    per_ticker = build_full_dataset(
        tickers, args.lookback_days, args.forward_days, args.label_mode,
        args.up_threshold, args.down_threshold,
    )

    all_trades = collect_all_oos_trades(
        per_ticker, args.train_years, args.test_months, args.step_months,
        hidden_sizes, args.dropout, args.weight_decay, args.lr, args.batch_size, args.epochs,
    )
    print(f"\nCollected {len(all_trades)} total out-of-sample predictions across all folds.\n")

    table = build_table(all_trades)

    with open(args.output, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved calibration table to {args.output}")
    print("trade_glue.py will load this automatically if present.")


if __name__ == "__main__":
    main()
