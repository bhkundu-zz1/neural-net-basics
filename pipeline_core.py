"""
Shared layer1 -> layer2 -> layer3 -> layer4 -> layer5 wiring for a single ticker.

Extracted from run_pipeline.py so both the single-ticker CLI (run_pipeline.py)
and the portfolio CLI (run_portfolio.py) call one shared implementation instead
of maintaining two copies. See run_pipeline.py's module docstring for the full
caveats about what this pipeline's output does and does not mean.
"""

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


def load_model(weights_path: str) -> tuple[QuantEdgeNet | None, dict | None]:
    """
    Loads a trained layer4 checkpoint if weights_path exists, returning
    (net, checkpoint). Returns (None, None) if the file doesn't exist, so
    callers fall back to building an untrained model per-ticker (an untrained
    net needs the ticker's feature vector shape, which isn't known here).
    """
    if not os.path.exists(weights_path):
        return None, None

    checkpoint = torch.load(weights_path, weights_only=False)
    net = QuantEdgeNet(
        input_features=checkpoint["input_features"],
        hidden_sizes=checkpoint.get("hidden_sizes", (128, 64)),
        dropout=checkpoint.get("dropout", 0.2),
    )
    net.load_state_dict(checkpoint["state_dict"])
    return net, checkpoint


def run_pipeline_for_ticker(
    ticker: str,
    shares: int,
    lookback: str = "2y",
    weights_path: str = "layer4_weights_nasdaq100.pt",
    min_confidence: float = 0.75,
    net: QuantEdgeNet | None = None,
    checkpoint: dict | None = None,
) -> dict:
    """
    Runs the full layer1-5 pipeline for one ticker and returns a dict of
    results (no printing — callers decide how to display or aggregate this).

    net/checkpoint: pass a pre-loaded model (from load_model()) to skip
    reloading the weights file for every ticker, e.g. when scoring many
    tickers against the same weights in a portfolio run. If both are None,
    the weights are loaded fresh via load_model(weights_path).
    """
    prices, volume, factor_returns = fetch_prices_and_factors(ticker, lookback)
    returns = prices.pct_change().dropna().to_frame(name=ticker)
    last_price = float(prices.iloc[-1])

    # Layer 1: signal detection
    signal = detect_signal(prices)

    # Layer 2: factor decomposition
    factor_result = factor_decompose(returns, factor_returns)[ticker]

    # Layer 3: Markov regime — seed current state from layer1's regime call
    current_state = 0 if signal["regime"] == "trending" else 2
    regime_probs = get_next_regime(current_state, steps=5)

    # Layer 4: neural net edge
    feature_vector = build_feature_vector(prices, volume, factor_result["betas"], regime_probs)

    if net is None:
        net, checkpoint = load_model(weights_path)

    if net is not None:
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
        trained_tickers = None
        edge_label = f"UNTRAINED weights ({weights_path} not found) — noise, not a real edge"

    net.eval()
    with torch.no_grad():
        raw_probs = net(x).numpy()
    edge_probs = {"long": float(raw_probs[0]), "short": float(raw_probs[1]), "flat": float(raw_probs[2])}

    # Layer 5: Kelly sizing + execution cost filter
    trade_inputs = build_trade_inputs(edge_probs, prices)

    position_size = kelly_position_size(
        win_probability=trade_inputs["win_probability"],
        win_loss_ratio=trade_inputs["win_loss_ratio"],
    )
    execution_cost = calculate_execution_cost(
        spread_bps=2.0,
        market_impact_bps=3.0,
        shares=shares,
        price=last_price,
    )
    clears_cost = should_trade(
        edge_bps=trade_inputs["edge_bps"],
        execution_cost_bps=trade_inputs["execution_cost_bps"],
    )
    clears_confidence = trade_inputs["win_probability"] >= min_confidence
    trade_decision = clears_cost and clears_confidence

    return {
        "ticker": ticker,
        "shares": shares,
        "last_price": last_price,
        "position_value": shares * last_price,
        "n_price_points": len(prices),
        "n_factor_rows": len(factor_returns),
        "signal": signal,
        "factor_result": factor_result,
        "regime_probs": regime_probs,
        "edge_probs": edge_probs,
        "edge_label": edge_label,
        "trained_tickers": trained_tickers,
        "trade_inputs": trade_inputs,
        "direction": trade_inputs["direction"],
        "win_probability": trade_inputs["win_probability"],
        "position_size": position_size,
        "execution_cost": execution_cost,
        "edge_bps": trade_inputs["edge_bps"],
        "execution_cost_bps": trade_inputs["execution_cost_bps"],
        "clears_cost": clears_cost,
        "clears_confidence": clears_confidence,
        "should_trade": trade_decision,
        "net": net,
        "checkpoint": checkpoint,
    }
