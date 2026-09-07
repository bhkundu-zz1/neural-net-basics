"""
Walk-forward re-fitting: the test everything so far has been missing.

Every result reported to date (48.2% accuracy, 0.10%/trade avg P&L, CI
excluding zero) came from ONE train/test split: train on the first 70% of
each ticker's history, test on the final 15%. That tells you the model
found something in that specific historical arrangement — it does NOT tell
you whether the result is stable across time, or whether it happened to be
a lucky (or unlucky) draw of which period got used as the test set.

This script retrains the model from scratch on a series of ROLLING windows
sliding across the full 10-year history:
    fold 1: train on [year 0, year 3),   test on [year 3.0, year 3.5)
    fold 2: train on [year 0.5, year 3.5), test on [year 3.5, year 4.0)
    ...
and so on across the full history. Each fold is a fresh model — no weights
carry over between folds — trained and evaluated exactly like the original
single-split experiment, just on a different slice of time. If the ~48%
accuracy / positive P&L result is real and stable, most folds should land
in a similar range. If it was a lucky draw, fold results should scatter
widely, including folds with accuracy near or below the 33% baseline and
P&L confidence intervals that include zero.

Usage:
    python walkforward_layer4.py --tickers NVDA,ANET,INTC,M --lookback-days 10y \
        --train-years 3 --test-months 6 --step-months 6
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from layer4 import QuantEdgeNet
from train_layer4 import fetch_prices_and_factors, build_dataset_for_ticker
from backtest_layer4 import compute_trades


def build_full_dataset(tickers, lookback_days, forward_days, label_mode, up_threshold, down_threshold):
    """Builds each ticker's full (features, labels, dates, fwd_returns, entry_prices, prices) once."""
    prices_by_ticker, volume_by_ticker, factors_by_ticker = fetch_prices_and_factors(tickers, lookback_days)

    per_ticker = {}
    for ticker in tickers:
        if ticker not in prices_by_ticker:
            continue
        features, labels, dates, factor_window, fwd_returns, entry_prices = build_dataset_for_ticker(
            ticker, prices_by_ticker[ticker], volume_by_ticker[ticker], factors_by_ticker[ticker],
            forward_days, up_threshold, down_threshold, label_mode,
        )
        if len(labels) < 50:
            print(f"  {ticker}: only {len(labels)} samples total, skipping from walk-forward.")
            continue
        per_ticker[ticker] = {
            "features": features, "labels": labels, "dates": dates,
            "fwd_returns": fwd_returns, "entry_prices": entry_prices,
            "prices": prices_by_ticker[ticker],
        }
        print(f"  {ticker}: {len(labels)} total samples, {dates[0].date()} to {dates[-1].date()}")

    return per_ticker


def make_fold_boundaries(per_ticker, train_years, test_months, step_months):
    """
    Computes (train_start, train_end, test_end) date boundaries for each fold,
    based on the overall min/max dates across all tickers. Folds slide forward
    by step_months until the test window would run past the available data.
    """
    all_dates = sorted(set(d for t in per_ticker.values() for d in t["dates"]))
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


def slice_fold(per_ticker, train_start, train_end, test_end):
    """Slices each ticker's dataset into this fold's train/test sets, then pools across tickers."""
    train_x, train_y, test_records = [], [], []

    for ticker, data in per_ticker.items():
        dates = np.array(data["dates"], dtype="datetime64[ns]")
        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates < test_end)

        if train_mask.sum() < 30 or test_mask.sum() < 5:
            continue

        train_x.append(data["features"][train_mask])
        train_y.append(data["labels"][train_mask])
        test_records.append({
            "ticker": ticker,
            "features": data["features"][test_mask],
            "labels": data["labels"][test_mask],
            "fwd_returns": data["fwd_returns"][test_mask],
            "entry_prices": data["entry_prices"][test_mask],
            "dates": [d for d, m in zip(data["dates"], test_mask) if m],
            "prices": data["prices"],
        })

    if not train_x or not test_records:
        return None

    return np.concatenate(train_x), np.concatenate(train_y), test_records


def train_fold(x_train_raw, y_train_raw, hidden_sizes, dropout, weight_decay, lr, batch_size, epochs):
    """
    Trains one fold's model. No separate val set here (folds are short and data
    is limited) — uses a small internal validation carve-out from train for
    early stopping only, then evaluates on the fold's true held-out test set.
    """
    n = len(y_train_raw)
    val_size = max(int(n * 0.15), 10)
    perm = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    x_train = torch.tensor(x_train_raw[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_train_raw[train_idx], dtype=torch.long)
    x_val = torch.tensor(x_train_raw[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_train_raw[val_idx], dtype=torch.long)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    class_counts = torch.bincount(y_train, minlength=3).float()
    class_weights = class_counts.sum() / (3 * class_counts.clamp_min(1))

    net = QuantEdgeNet(input_features=x_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0
    patience = 10

    for epoch in range(1, epochs + 1):
        net.train()
        permutation = torch.randperm(len(x_train))
        for start in range(0, len(x_train), batch_size):
            idx = permutation[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = criterion(net.layers(xb), yb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = criterion(net.layers(x_val), y_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            break

    net.load_state_dict(best_state)
    net.eval()
    return net, mean, std


def evaluate_fold(net, mean, std, test_records, min_confidence):
    all_trades = []
    all_preds, all_labels = [], []

    for rec in test_records:
        x = (torch.tensor(rec["features"], dtype=torch.float32) - mean.squeeze(0)) / std.squeeze(0)
        with torch.no_grad():
            probs = net(x).numpy()
        preds = probs.argmax(axis=1)
        all_preds.append(preds)
        all_labels.append(rec["labels"])

        trades = compute_trades(
            rec["ticker"], net, mean, std, rec["prices"],
            rec["features"], rec["fwd_returns"], rec["entry_prices"], rec["dates"],
            min_confidence,
        )
        all_trades.extend(trades)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    accuracy = (preds == labels).mean()

    taken = [t for t in all_trades if t["took_trade"]]
    if taken:
        pnl = np.array([t["pnl_pct"] for t in taken])
        win_rate = (pnl > 0).mean()
        avg_pnl = pnl.mean()
    else:
        win_rate, avg_pnl, pnl = None, None, np.array([])

    return accuracy, len(all_trades), len(taken), win_rate, avg_pnl, pnl, all_trades


def main():
    parser = argparse.ArgumentParser(description="Walk-forward re-fitting test for layer4.")
    parser.add_argument("--tickers", default="NVDA,ANET,INTC,M")
    parser.add_argument("--lookback-days", default="10y")
    parser.add_argument("--forward-days", type=int, default=5)
    parser.add_argument("--label-mode", choices=["risk_adjusted", "raw"], default="raw")
    parser.add_argument("--up-threshold", type=float, default=0.02)
    parser.add_argument("--down-threshold", type=float, default=0.02)
    parser.add_argument("--train-years", type=float, default=3.0)
    parser.add_argument("--test-months", type=float, default=6.0)
    parser.add_argument("--step-months", type=float, default=6.0)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--hidden-sizes", default="32,16")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))

    print(f"Building full dataset for {tickers}...")
    per_ticker = build_full_dataset(
        tickers, args.lookback_days, args.forward_days, args.label_mode,
        args.up_threshold, args.down_threshold,
    )

    folds = make_fold_boundaries(per_ticker, args.train_years, args.test_months, args.step_months)
    print(f"\n{len(folds)} rolling folds: train={args.train_years}y, test={args.test_months}mo, "
          f"step={args.step_months}mo\n")

    fold_results = []
    for i, (train_start, train_end, test_end) in enumerate(folds, 1):
        sliced = slice_fold(per_ticker, train_start, train_end, test_end)
        if sliced is None:
            print(f"Fold {i}: insufficient data, skipping.")
            continue
        x_train, y_train, test_records = sliced
        n_test = sum(len(r["labels"]) for r in test_records)
        if len(y_train) < 100 or n_test < 20:
            print(f"Fold {i}: train={len(y_train)}, test={n_test} — too small, skipping.")
            continue

        net, mean, std = train_fold(
            x_train, y_train, hidden_sizes, args.dropout, args.weight_decay,
            args.lr, args.batch_size, args.epochs,
        )
        accuracy, n_signals, n_taken, win_rate, avg_pnl, pnl, _all_trades = evaluate_fold(
            net, mean, std, test_records, args.min_confidence
        )

        train_start_s = str(train_start)[:10]
        test_start_s = str(train_end)[:10]
        test_end_s = str(test_end)[:10]

        if n_taken >= 5:
            t_stat, p_value = stats.ttest_1samp(pnl, 0.0)
            ci_low, ci_high = np.percentile(
                [np.random.default_rng(i).choice(pnl, size=len(pnl), replace=True).mean() for _ in range(2000)],
                [2.5, 97.5],
            )
            ci_excludes_zero = not (ci_low <= 0 <= ci_high)
        else:
            p_value, ci_low, ci_high, ci_excludes_zero = None, None, None, None

        print(f"Fold {i:2d}  train=[{train_start_s}..{test_start_s})  test=[{test_start_s}..{test_end_s})  "
              f"n_train={len(y_train):5d}  n_test={n_test:4d}  accuracy={accuracy:.1%}  "
              f"trades_taken={n_taken:3d}  win_rate={f'{win_rate:.1%}' if win_rate is not None else 'n/a':>6}  "
              f"avg_pnl={f'{avg_pnl:.4%}' if avg_pnl is not None else 'n/a':>9}  "
              f"CI_excl_0={ci_excludes_zero}")

        fold_results.append({
            "fold": i, "accuracy": accuracy, "n_test": n_test, "n_taken": n_taken,
            "win_rate": win_rate, "avg_pnl": avg_pnl, "ci_excludes_zero": ci_excludes_zero,
        })

    if not fold_results:
        print("\nNo folds produced results — check window sizes vs available data.")
        return

    accuracies = np.array([f["accuracy"] for f in fold_results])
    pnls = np.array([f["avg_pnl"] for f in fold_results if f["avg_pnl"] is not None])
    n_significant = sum(1 for f in fold_results if f["ci_excludes_zero"])
    n_with_trades = sum(1 for f in fold_results if f["ci_excludes_zero"] is not None)

    print(f"\n=== Walk-forward summary across {len(fold_results)} folds ===")
    print(f"Accuracy: mean={accuracies.mean():.1%}  std={accuracies.std():.1%}  "
          f"min={accuracies.min():.1%}  max={accuracies.max():.1%}")
    if len(pnls) > 0:
        print(f"Avg P&L/trade (folds with >=5 trades): mean={pnls.mean():.4%}  std={pnls.std():.4%}  "
              f"min={pnls.min():.4%}  max={pnls.max():.4%}")
    print(f"Folds where P&L was statistically significant (CI excludes 0): "
          f"{n_significant}/{n_with_trades} folds with enough trades to test")
    print("\nIf accuracy and P&L are stable and mostly positive/significant across folds, the")
    print("single-split result was likely a real, if modest, pattern. If they scatter widely —")
    print("including folds near/below the 33% baseline or with CIs including zero — the original")
    print("result was likely specific to that one lucky train/test split.")


if __name__ == "__main__":
    main()
