"""
Backtests the trained QuantEdgeNet (layer4) model on its held-out test split,
running its predictions through the actual layer4->layer5 glue (trade_glue.py)
and layer5 Kelly sizing + execution cost + should_trade filter — measuring
simulated P&L, not just classification accuracy.

Why this matters: classification accuracy treats every sample equally, but a
trading signal only has value if it's right on the samples it actually acts
on, after costs. A model can have mediocre accuracy and still be profitable
if it's well-calibrated about when to sit out (predicts "flat" or fails the
should_trade check) — or have OK-looking accuracy and still lose money if its
correct calls are on small moves and its wrong calls are on big ones.

Simplification / limitation: each test-set day is treated as an independent
capital allocation of size `position_size` (from half-Kelly, capped at 2% of
capital per layer5). Since forward_days=5 by default, consecutive daily
signals have overlapping holding periods — this is NOT a realistic single-
account simulation (you can't be in 5 overlapping positions of the same
stock without a larger capital base), it's closer to "if you allocated a
fresh 2%-max slice of capital to every signal independently, what would the
blended return have been." Treat the resulting numbers as a signal-quality
check, not a portfolio backtest.

Usage:
    python backtest_layer4.py --weights layer4_weights.pt
"""

import argparse

import numpy as np
import torch
from scipy import stats

from layer4 import QuantEdgeNet
from layer5 import kelly_position_size, calculate_execution_cost, should_trade
from trade_glue import build_trade_inputs
from train_layer4 import fetch_prices_and_factors, build_dataset_for_ticker, chronological_split


def load_model(weights_path: str):
    checkpoint = torch.load(weights_path, weights_only=False)
    net = QuantEdgeNet(
        input_features=checkpoint["input_features"],
        hidden_sizes=checkpoint.get("hidden_sizes", (128, 64)),
        dropout=checkpoint.get("dropout", 0.2),
    )
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net, checkpoint


def backtest_ticker(ticker, net, mean, std, prices, volume, factor_returns, forward_days, up_threshold,
                     down_threshold, min_confidence, label_mode="risk_adjusted", cost_multiplier=1.0,
                     evaluate_all=False):
    """
    evaluate_all: if True, evaluate on the ticker's ENTIRE dataset instead of just
    its final 15% test split. Use this for a ticker the model was NOT trained on —
    there's no train/val/test distinction to respect since none of this ticker's
    data was ever seen during training. Only use the test-split slice (default)
    for tickers that WERE part of the training universe, to avoid re-evaluating on
    data the model actually trained on.
    """
    features, labels, dates, factor_window, fwd_returns, entry_prices = build_dataset_for_ticker(
        ticker, prices, volume, factor_returns, forward_days, up_threshold, down_threshold, label_mode
    )
    if len(labels) < 20:
        print(f"  {ticker}: too few samples ({len(labels)}) to backtest, skipping.")
        return []

    if evaluate_all:
        test_features = features
        test_fwd_returns = fwd_returns
        test_entry_prices = entry_prices
        test_dates = dates
    else:
        _, _, test_idx = chronological_split(len(labels))
        test_features = features[test_idx]
        test_fwd_returns = fwd_returns[test_idx]
        test_entry_prices = entry_prices[test_idx]
        test_dates = [dates[i] for i in range(len(dates)) if test_idx.start <= i < test_idx.stop]

    return compute_trades(
        ticker, net, mean, std, prices, test_features, test_fwd_returns, test_entry_prices, test_dates,
        min_confidence, cost_multiplier,
    )


def compute_trades(ticker, net, mean, std, prices, features, fwd_returns, entry_prices, dates,
                    min_confidence, cost_multiplier=1.0):
    """
    Runs the model's predictions through the layer4->layer5 glue and layer5's
    sizing/cost/should_trade logic, returning one trade record per sample.
    Shared by backtest_layer4.py (single train/test split) and
    walkforward_layer4.py (many rolling folds) so both use identical P&L logic.
    """
    x = (torch.tensor(features, dtype=torch.float32) - mean.squeeze(0)) / std.squeeze(0)
    with torch.no_grad():
        probs = net(x).numpy()

    trades = []
    for i in range(len(features)):
        edge_probs = {"long": float(probs[i, 0]), "short": float(probs[i, 1]), "flat": float(probs[i, 2])}
        price_history = prices.loc[:dates[i]]
        trade_inputs = build_trade_inputs(edge_probs, price_history)

        position_size = kelly_position_size(
            win_probability=trade_inputs["win_probability"],
            win_loss_ratio=trade_inputs["win_loss_ratio"],
        )
        execution_cost = calculate_execution_cost(
            spread_bps=2.0 * cost_multiplier, market_impact_bps=3.0 * cost_multiplier,
            shares=100, price=float(entry_prices[i]),
        )
        execution_cost_pct = execution_cost / (100 * entry_prices[i])
        clears_cost = should_trade(
            edge_bps=trade_inputs["edge_bps"],
            execution_cost_bps=trade_inputs["execution_cost_bps"] * cost_multiplier,
        )
        clears_confidence = trade_inputs["win_probability"] >= min_confidence
        take_trade = clears_cost and clears_confidence

        actual_return = fwd_returns[i]
        direction_sign = 1 if trade_inputs["direction"] == "long" else -1
        directional_return = direction_sign * actual_return  # signed return BEFORE Kelly sizing/costs —
        # what the calibration table needs: "when the model called this direction at this
        # confidence, how did the raw underlying move actually turn out."

        if take_trade and position_size > 0:
            pnl_pct = position_size * (directional_return - execution_cost_pct)
        else:
            pnl_pct = 0.0  # sat out — no capital at risk, no P&L

        trades.append({
            "ticker": ticker,
            "date": dates[i],
            "direction": trade_inputs["direction"],
            "confidence": trade_inputs["win_probability"],
            "took_trade": take_trade and position_size > 0,
            "position_size": position_size,
            "actual_return": actual_return,
            "directional_return": directional_return,
            "pnl_pct": pnl_pct,
            "calibrated": trade_inputs["calibrated"],
        })

    return trades


def summarize(trades: list[dict]):
    n_total = len(trades)
    taken = [t for t in trades if t["took_trade"]]
    n_taken = len(taken)

    print(f"\n=== Backtest summary ===")
    print(f"Total test-set signals: {n_total}")
    print(f"Signals where should_trade passed and Kelly size > 0: {n_taken} ({n_taken/n_total:.1%})")

    if n_taken == 0:
        print("No trades were taken — the edge_bps never cleared 1.5x execution cost. "
              "This itself is informative: the model's confidence is never high enough "
              "for layer5's cost filter to approve a trade.")
        return

    # NOTE: these trades have overlapping 5-day holding periods (forward_days),
    # so they are NOT sequential returns on one capital base. Compounding them
    # (e.g. (1+r1)*(1+r2)*...) would wildly overstate/understate reality and is
    # deliberately NOT done here — only per-trade, non-compounded statistics
    # are reported. See the module docstring.
    pnl = np.array([t["pnl_pct"] for t in taken])
    wins = pnl > 0
    avg_pnl = pnl.mean()
    median_pnl = np.median(pnl)
    win_rate = wins.mean()
    pnl_std = pnl.std()
    sharpe_like = avg_pnl / pnl_std if pnl_std > 0 else 0.0
    sum_pnl = pnl.sum()  # sum of independent per-slice P&L, NOT a compounded portfolio return

    print(f"Win rate on taken trades: {win_rate:.1%}")
    print(f"Average P&L per taken trade (% of allocated capital for that trade): {avg_pnl:.4%}")
    print(f"Median P&L per taken trade: {median_pnl:.4%}")
    print(f"Sum of per-trade P&L across all {n_taken} taken trades (NOT compounded — see caveat "
          f"above; this is what you'd get if each trade used a separate, independent capital "
          f"slice of size `position_size`): {sum_pnl:.2%}")
    print(f"Per-trade return/volatility ratio (not annualized): {sharpe_like:.3f}")

    # Statistical significance: is avg_pnl distinguishable from zero, or within noise?
    # One-sample t-test (parametric, assumes roughly normal per-trade P&L) plus a
    # bootstrap CI (nonparametric, more trustworthy if the P&L distribution is skewed —
    # likely here since wins/losses aren't symmetric once Kelly sizing and costs are applied).
    t_stat, p_value = stats.ttest_1samp(pnl, 0.0)
    n_bootstrap = 10000
    rng = np.random.default_rng(42)
    bootstrap_means = np.array([
        rng.choice(pnl, size=len(pnl), replace=True).mean() for _ in range(n_bootstrap)
    ])
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

    print(f"\n=== Statistical significance of average P&L per trade ===")
    print(f"One-sample t-test vs 0: t={t_stat:.3f}, p={p_value:.4f} "
          f"({'reject' if p_value < 0.05 else 'CANNOT reject'} 'true avg P&L is 0' at 5% significance)")
    print(f"Bootstrap 95% CI on average P&L per trade ({n_bootstrap} resamples): "
          f"[{ci_low:.4%}, {ci_high:.4%}]")
    if ci_low <= 0 <= ci_high:
        print("  -> The 95% CI includes 0: cannot statistically distinguish this strategy's")
        print("     average P&L from zero. Treat any apparent edge as unconfirmed.")
    else:
        print("  -> The 95% CI excludes 0: the average P&L is statistically distinguishable")
        print("     from zero at this sample size (this does NOT by itself prove the edge is")
        print("     economically meaningful or will persist out of this sample).")

    # Compare against the model's own directional accuracy on taken trades vs actual moves,
    # rather than a compounded buy-and-hold figure (which suffers the same overlap problem).
    buy_hold_pnl = np.array([t["actual_return"] for t in taken])
    print(f"\nFor reference, mean raw forward return on the same {n_taken} dates "
          f"(unsigned, no sizing/costs/direction applied): {buy_hold_pnl.mean():.4%}")
    print("(Comparing this to the model's average P&L above shows whether the model's")
    print(" directional calls + cost filter added value over just being in the market.)")

    by_ticker = {}
    for t in taken:
        by_ticker.setdefault(t["ticker"], []).append(t["pnl_pct"])
    print("\nPer-ticker (taken trades only):")
    for ticker, pnls in by_ticker.items():
        pnls = np.array(pnls)
        print(f"  {ticker}: n={len(pnls)}  win_rate={np.mean(pnls > 0):.1%}  avg_pnl={pnls.mean():.4%}")

    n_calibrated = sum(1 for t in taken if t["calibrated"])
    print(f"\nTaken trades scored via the empirically calibrated glue "
          f"(calibration_table.json): {n_calibrated}/{n_taken} ({n_calibrated / n_taken:.1%})")
    if n_calibrated < n_taken:
        print("=== IMPORTANT: some taken trades fell back to the placeholder layer4->layer5 glue ===")
        print("(no calibration_table.json bucket covered their confidence level) — those trades'")
        print("win_loss_ratio/edge_bps came from hand-picked heuristics, not empirical history.")
        print("See build_calibration_table.py to extend coverage.")


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest a trained layer4 model via layer5 sizing/cost logic.")
    parser.add_argument("--weights", default="layer4_weights.pt")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.33,
        help="Only take trades where win_probability >= this (default 0.33 = no extra filter beyond "
        "should_trade's cost check). Per calibrate_layer4.py, accuracy rises with confidence — try "
        "0.55-0.60 to restrict to the subset the model is actually good at.",
    )
    parser.add_argument(
        "--cost-multiplier",
        type=float,
        default=1.0,
        help="Multiplies trade_glue.py's placeholder spread_bps/market_impact_bps (2bps/3bps, tuned "
        "for a highly liquid large-cap). Use 2.0 or 3.0 to stress-test how fast P&L degrades under "
        "more realistic/conservative cost assumptions.",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated tickers to backtest against, OVERRIDING the checkpoint's training "
        "tickers. Use this to test the model out-of-sample on tickers it never trained on — "
        "combine with --evaluate-all since there's no test split to isolate for an unseen ticker.",
    )
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        help="Evaluate on each ticker's FULL history instead of just its final 15%% test split. "
        "Required for a meaningful out-of-sample check via --tickers (a ticker the model never "
        "trained on has no 'test split' distinction — all of its data is equally out-of-sample). "
        "Do NOT use this for tickers that WERE in the training universe, since it would include "
        "data the model actually trained on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    net, checkpoint = load_model(args.weights)
    trained_tickers = checkpoint.get("tickers", [checkpoint.get("ticker")])
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else trained_tickers
    forward_days = checkpoint["forward_days"]
    up_threshold = checkpoint["up_threshold"]
    down_threshold = checkpoint["down_threshold"]
    label_mode = checkpoint.get("label_mode", "raw")  # checkpoints saved before label_mode existed were raw-return
    mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]

    if args.tickers:
        unseen = [t for t in tickers if t not in trained_tickers]
        if unseen and not args.evaluate_all:
            print(f"WARNING: {unseen} were not in the training universe ({trained_tickers}) but "
                  f"--evaluate-all was not set — this will only evaluate their FINAL 15% by date, "
                  f"which is a smaller and somewhat arbitrary out-of-sample slice. Consider adding "
                  f"--evaluate-all for a full out-of-sample read.")
        print(f"Testing out-of-sample: model trained on {trained_tickers}, now evaluating on {tickers}")

    print(f"Backtesting {args.weights} (trained on {trained_tickers}, forward_days={forward_days}, "
          f"label_mode={label_mode}, up/down thresholds={up_threshold}/{down_threshold}, "
          f"cost_multiplier={args.cost_multiplier}x, evaluating on {tickers})")
    print("Fetching price history for the evaluation tickers...")

    # Use a lookback long enough to reconstruct the same ~10y history the model trained on
    prices_by_ticker, volume_by_ticker, factors_by_ticker = fetch_prices_and_factors(tickers, "10y")

    all_trades = []
    for ticker in tickers:
        if ticker not in prices_by_ticker:
            continue
        trades = backtest_ticker(
            ticker, net, mean, std,
            prices_by_ticker[ticker], volume_by_ticker[ticker], factors_by_ticker[ticker],
            forward_days, up_threshold, down_threshold, args.min_confidence, label_mode,
            args.cost_multiplier, args.evaluate_all,
        )
        all_trades.extend(trades)

    if not all_trades:
        print("No backtestable samples produced — check ticker data availability.")
        return

    summarize(all_trades)


if __name__ == "__main__":
    main()
