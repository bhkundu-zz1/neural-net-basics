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
import os

import pandas as pd
import torch
import yfinance as yf

from layer1 import detect_signal
from layer2 import factor_decompose
from layer3 import get_next_regime
from layer4 import QuantEdgeNet, build_feature_vector
from layer5 import kelly_position_size, calculate_execution_cost, should_trade
from trade_glue import build_trade_inputs

DEFAULT_WEIGHTS = "layer4_weights_rawreturn.pt"
DEFAULT_MIN_CONFIDENCE = 0.75


def fetch_prices_and_factors(ticker: str, lookback: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Downloads adjusted close prices + volume for the target ticker plus proxy factors:
      market   -> SPY  (broad market)
      size     -> IWM  (small-cap, proxy for size factor)
      value    -> IWD  (Russell 1000 Value, proxy for value factor)
      momentum -> MTUM (iShares Momentum ETF, proxy for momentum factor)
    These are liquid, easy-to-fetch proxies — NOT the academic Fama-French/
    Carhart factor returns. Good enough for wiring the pipeline, not for
    production factor attribution.
    """
    tickers = [ticker, "SPY", "IWM", "IWD", "MTUM"]
    data = yf.download(tickers, period=lookback, auto_adjust=True, progress=False)
    close = data["Close"].dropna()
    volume = data["Volume"][ticker].reindex(close.index)

    prices = close[ticker]
    factor_prices = close[["SPY", "IWM", "IWD", "MTUM"]]
    factor_returns = factor_prices.pct_change().dropna()
    factor_returns.columns = ["market", "size", "value", "momentum"]

    return prices, volume, factor_returns


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
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()
    lookback = args.lookback_days

    print(f"=== Running layer1-5 pipeline for {ticker} (lookback={lookback}) ===\n")

    prices, volume, factor_returns = fetch_prices_and_factors(ticker, lookback)
    returns = prices.pct_change().dropna().to_frame(name=ticker)
    print(f"Loaded {len(prices)} price points, {len(factor_returns)} factor return rows.\n")

    # Layer 1: signal detection
    signal = detect_signal(prices)
    print("[layer1] Signal detection:")
    for k, v in signal.items():
        print(f"    {k}: {v}")

    # Layer 2: factor decomposition
    factor_result = factor_decompose(returns, factor_returns)[ticker]
    print(f"\n[layer2] Factor decomposition ({ticker}):")
    for k, v in factor_result.items():
        print(f"    {k}: {v}")

    # Layer 3: Markov regime — seed current state from layer1's regime call
    current_state = 0 if signal["regime"] == "trending" else 2
    regime_probs = get_next_regime(current_state, steps=5)
    print(f"\n[layer3] Regime forecast (5 steps ahead):")
    for k, v in regime_probs.items():
        print(f"    {k}: {v}")

    # Layer 4: neural net edge
    feature_vector = build_feature_vector(prices, volume, factor_result["betas"], regime_probs)

    weights_path = args.weights
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, weights_only=False)
        net = QuantEdgeNet(
            input_features=checkpoint["input_features"],
            hidden_sizes=checkpoint.get("hidden_sizes", (128, 64)),
            dropout=checkpoint.get("dropout", 0.2),
        )
        net.load_state_dict(checkpoint["state_dict"])
        mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
        x = (torch.tensor(feature_vector, dtype=torch.float32) - mean.squeeze(0)) / std.squeeze(0)
        trained_tickers = checkpoint.get("tickers", [checkpoint.get("ticker")])
        edge_label = (
            f"trained weights from {weights_path} (tickers={trained_tickers}) — see "
            f"docs/pipeline_guide.md for the walk-forward/out-of-sample results behind this model"
        )
    else:
        net = QuantEdgeNet(input_features=feature_vector.shape[0])
        x = torch.tensor(feature_vector, dtype=torch.float32)
        edge_label = f"UNTRAINED weights ({weights_path} not found) — noise, not a real edge"

    net.eval()
    with torch.no_grad():
        raw_probs = net(x).numpy()
    edge_probs = {"long": float(raw_probs[0]), "short": float(raw_probs[1]), "flat": float(raw_probs[2])}
    print(f"\n[layer4] QuantEdgeNet output ({edge_label}):")
    for k, v in edge_probs.items():
        print(f"    {k}: {v:.4f}")

    # Layer 5: Kelly sizing + execution cost filter
    trade_inputs = build_trade_inputs(edge_probs, prices)
    glue_label = "empirically calibrated (calibration_table.json)" if trade_inputs.get("calibrated") else \
        "FALLBACK heuristic — no calibration_table.json found, see build_calibration_table.py"
    print(f"\n[layer4->layer5 glue] Trade inputs ({glue_label}):")
    for k, v in trade_inputs.items():
        print(f"    {k}: {v}")

    position_size = kelly_position_size(
        win_probability=trade_inputs["win_probability"],
        win_loss_ratio=trade_inputs["win_loss_ratio"],
    )
    execution_cost = calculate_execution_cost(
        spread_bps=2.0,
        market_impact_bps=3.0,
        shares=100,
        price=float(prices.iloc[-1]),
    )
    clears_cost = should_trade(
        edge_bps=trade_inputs["edge_bps"],
        execution_cost_bps=trade_inputs["execution_cost_bps"],
    )
    clears_confidence = trade_inputs["win_probability"] >= args.min_confidence
    trade_decision = clears_cost and clears_confidence

    print(f"\n[layer5] Kelly + execution:")
    print(f"    direction: {trade_inputs['direction']}")
    print(f"    half-Kelly position size (fraction of capital): {position_size:.4f}")
    print(f"    execution cost for 100 shares @ ${prices.iloc[-1]:.2f}: ${execution_cost:.2f}")
    print(f"    clears cost check (should_trade): {clears_cost}")
    print(f"    clears confidence >= {args.min_confidence}: {clears_confidence}")
    print(f"    should_trade: {trade_decision}")

    print(f"\n=== NOTE: layer4 status: {edge_label} ===")


if __name__ == "__main__":
    main()
