"""
Runs layer1 -> layer2 -> layer3 -> layer4 -> layer5 end to end for a single ticker.

IMPORTANT CAVEATS (read before trusting any output) — see docs/pipeline_guide.md
for the full research history and findings this pipeline is built on:
  - The factor set here (market, size, value, momentum) is a crude proxy
    built from single ETFs, not a real Fama-French/Carhart dataset.
  - layer4_weights_rawreturn.pt (the default weights loaded below) showed a
    small, statistically significant, out-of-sample edge across 11 walk-
    forward folds and 4 tickers never seen in training (JPM, MRNA, QQQ) —
    see docs/pipeline_guide.md for the numbers. This is a research finding,
    not validation for live trading: the edge is thin (~0.10-0.14% per
    trade vs several-percent raw market moves on the same dates), the
    "short" class is a persistent weak spot, and position sizing here has
    no portfolio-level risk controls (correlation across concurrent
    positions, max drawdown limits, etc.).
  - trade_glue.py now uses an empirically calibrated confidence -> win/loss
    lookup (calibration_table.json, built from the same 11 walk-forward
    folds) instead of hand-picked formulas — see build_calibration_table.py.
    --min-confidence 0.75 is the operating point validated against that
    calibration; lower thresholds trade more often at lower quality.
  - This script is for running the research pipeline and inspecting its
    output, not an automated trading system.
"""

import argparse

from pipeline_core import run_pipeline_for_ticker

DEFAULT_WEIGHTS = "layer4_weights_nasdaq100.pt"
DEFAULT_MIN_CONFIDENCE = 0.75
DEFAULT_SHARES = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Run the layer1-5 signal pipeline for a ticker.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker symbol to analyze (default: NVDA)")
    parser.add_argument(
        "--lookback-days",
        default="2y",
        help="Price history window as a yfinance period string, e.g. 6mo, 1y, 2y (default: 2y)",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Trained layer4 weights to load (default: {DEFAULT_WEIGHTS}). Falls back to an "
        "untrained model if the file doesn't exist.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Only flag should_trade=True when win_probability clears this AND the cost check "
        f"passes (default: {DEFAULT_MIN_CONFIDENCE}, the threshold validated in the calibration "
        f"sweep — see docs/pipeline_guide.md).",
    )
    parser.add_argument(
        "--shares",
        type=int,
        default=DEFAULT_SHARES,
        help=f"Share count to use for the execution-cost estimate (default: {DEFAULT_SHARES}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()
    lookback = args.lookback_days

    print(f"=== Running layer1-5 pipeline for {ticker} (lookback={lookback}) ===\n")

    result = run_pipeline_for_ticker(
        ticker,
        shares=args.shares,
        lookback=lookback,
        weights_path=args.weights,
        min_confidence=args.min_confidence,
    )

    print(f"Loaded {result['n_price_points']} price points, {result['n_factor_rows']} factor return rows.\n")

    # Layer 1: signal detection
    print("[layer1] Signal detection:")
    for k, v in result["signal"].items():
        print(f"    {k}: {v}")

    # Layer 2: factor decomposition
    print(f"\n[layer2] Factor decomposition ({ticker}):")
    for k, v in result["factor_result"].items():
        print(f"    {k}: {v}")

    # Layer 3: Markov regime
    print(f"\n[layer3] Regime forecast (5 steps ahead):")
    for k, v in result["regime_probs"].items():
        print(f"    {k}: {v}")

    # Layer 4: neural net edge
    print(f"\n[layer4] QuantEdgeNet output ({result['edge_label']}):")
    for k, v in result["edge_probs"].items():
        print(f"    {k}: {v:.4f}")

    # Layer 5: Kelly sizing + execution cost filter
    trade_inputs = result["trade_inputs"]
    glue_label = "empirically calibrated (calibration_table.json)" if trade_inputs.get("calibrated") else \
        "FALLBACK heuristic — no calibration_table.json found, see build_calibration_table.py"
    print(f"\n[layer4->layer5 glue] Trade inputs ({glue_label}):")
    for k, v in trade_inputs.items():
        print(f"    {k}: {v}")

    print(f"\n[layer5] Kelly + execution:")
    print(f"    direction: {result['direction']}")
    print(f"    half-Kelly position size (fraction of capital): {result['position_size']:.4f}")
    print(f"    execution cost for {result['shares']} shares @ ${result['last_price']:.2f}: ${result['execution_cost']:.2f}")
    print(f"    clears cost check (should_trade): {result['clears_cost']}")
    print(f"    clears confidence >= {args.min_confidence}: {result['clears_confidence']}")
    print(f"    should_trade: {result['should_trade']}")

    print(f"\n=== NOTE: layer4 status: {result['edge_label']} ===")


if __name__ == "__main__":
    main()
