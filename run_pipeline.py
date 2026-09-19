"""
Runs layer1 -> layer2 -> layer3 -> layer4 -> layer5 end to end for a single ticker
and prints a friendly BUY/SELL/HOLD summary with a plain-English explanation
(via a self-hosted LLM if configured — see llm_narration.py and .env.example
— falling back to a templated explanation otherwise). Pass --verbose to also
see the full layer-by-layer technical breakdown.

IMPORTANT CAVEATS (read before trusting any output) — see docs/pipeline_guide.md
for the full research history and findings this pipeline is built on:
  - The factor set here (market, size, value, momentum) is a crude proxy
    built from single ETFs, not a real Fama-French/Carhart dataset.
  - layer4_weights_expanded.pt (the default weights loaded below, trained
    across 124 tickers) showed a small, statistically significant,
    out-of-sample edge across 11 walk-forward folds and 5 tickers never
    seen in training (JPM, WMT, JNJ, XOM, DIS) — see docs/pipeline_guide.md
    for the numbers. This is a research finding, not validation for live
    trading: the edge is thin (~0.02-0.04% per trade vs several-percent raw
    market moves on the same dates), the "short" class is a persistent weak
    spot, and position sizing here has no portfolio-level risk controls
    (correlation across concurrent positions, max drawdown limits, etc.).
  - trade_glue.py now uses an empirically calibrated confidence -> win/loss
    lookup (calibration_table.json, rebuilt from the same 124-ticker walk-
    forward folds) instead of hand-picked formulas — see
    build_calibration_table.py. --min-confidence 0.75 is the operating
    point validated against that calibration; lower thresholds trade more
    often at lower quality.
  - This script is for running the research pipeline and inspecting its
    output, not an automated trading system.
"""

import argparse
import sys

import llm_narration
from pipeline_core import run_pipeline_for_ticker
from portfolio_glue import map_action

# Windows consoles often default to a codepage (e.g. cp1252) that can't encode
# everything an LLM-generated explanation might contain (curly quotes, narrow
# no-break spaces, etc.) or the em-dashes used in this file's own docstrings.
# Reconfigure stdout to UTF-8, falling back to character replacement instead
# of crashing, rather than trying to purge every non-ASCII character by hand.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_WEIGHTS = "layer4_weights_expanded.pt"
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print the full layer1-5 technical breakdown, in addition to the summary.",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip the LLM-generated explanation and print a plain templated one instead "
        "(no network call). Useful if the LLM endpoint is unavailable or you want a fast run.",
    )
    return parser.parse_args()


VERDICT_MARKER = {"Buy": "[BUY]", "Sell": "[SELL]", "Hold": "[HOLD]"}


def fallback_explanation(ticker: str, verdict: str, result: dict, min_confidence: float) -> str:
    """A plain, templated explanation used when the LLM narration is unavailable or skipped."""
    win_prob = result["win_probability"]
    if verdict == "Hold":
        if not result["clears_confidence"]:
            return (
                f"{ticker}'s model confidence ({win_prob:.1%}) didn't clear the "
                f"{min_confidence:.0%} threshold required to act, so no new position is "
                f"recommended today."
            )
        return (
            f"{ticker} cleared the confidence bar but not the cost check — the estimated "
            f"edge ({result['edge_bps']:.1f} bps) wasn't enough to justify trading costs "
            f"({result['execution_cost_bps']:.1f} bps), so no new position is recommended."
        )
    action_word = "opening a long" if verdict == "Buy" else "opening a short"
    return (
        f"The model favors {result['direction']} on {ticker} with {win_prob:.1%} calibrated "
        f"confidence (threshold: {min_confidence:.0%}), clearing both the confidence and cost "
        f"checks. This suggests {action_word} position, sized at "
        f"{result['position_size']*100:.2f}% of capital (half-Kelly)."
    )


def print_summary(ticker: str, result: dict, args) -> str:
    verdict = map_action(result["should_trade"], result["direction"])
    marker = VERDICT_MARKER[verdict]

    print(f"{marker} {ticker}: {verdict.upper()}\n")

    explanation = None
    if not args.no_explain:
        explanation = llm_narration.narrate({
            "ticker": ticker,
            "verdict": verdict,
            "direction": result["direction"],
            "win_probability": result["win_probability"],
            "min_confidence": args.min_confidence,
            "edge_bps": result["edge_bps"],
            "execution_cost_bps": result["execution_cost_bps"],
            "clears_cost": result["clears_cost"],
            "clears_confidence": result["clears_confidence"],
            "regime": result["regime_probs"].get("dominant_regime", "unknown"),
            "in_training_universe": result["trained_tickers"] is not None
            and ticker in result["trained_tickers"],
        })

    if explanation is None:
        explanation = fallback_explanation(ticker, verdict, result, args.min_confidence)

    print(explanation)
    print(
        f"\n(Confidence: {result['win_probability']:.1%} | Threshold: {args.min_confidence:.0%} | "
        f"Estimated edge: {result['edge_bps']:.1f} bps | Position size if acted on: "
        f"{result['position_size']*100:.2f}% of capital)"
    )
    if result["trained_tickers"] is not None and ticker not in result["trained_tickers"]:
        print(f"\nNote: {ticker} was not in this model's training universe — treat this signal "
              f"with extra skepticism.")
    print(
        "\nThis is a research signal, not investment advice — see docs/pipeline_guide.md "
        "for what this pipeline does and does not validate."
    )
    return verdict


def print_technical_detail(ticker: str, result: dict, args):
    print(f"\n{'='*60}\nTechnical detail (--verbose)\n{'='*60}")
    print(f"\nLoaded {result['n_price_points']} price points, {result['n_factor_rows']} factor return rows.")

    print("\n[layer1] Signal detection:")
    for k, v in result["signal"].items():
        print(f"    {k}: {v}")

    if result["factor_result"] is None:
        print(f"\n[layer2-5] Skipped: {result['edge_label']}")
        return

    print(f"\n[layer2] Factor decomposition ({ticker}):")
    for k, v in result["factor_result"].items():
        print(f"    {k}: {v}")

    print(f"\n[layer3] Regime forecast (5 steps ahead):")
    for k, v in result["regime_probs"].items():
        print(f"    {k}: {v}")

    print(f"\n[layer4] QuantEdgeNet output ({result['edge_label']}):")
    for k, v in result["edge_probs"].items():
        print(f"    {k}: {v:.4f}")

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

    print(f"\n=== layer4 status: {result['edge_label']} ===")


def main():
    args = parse_args()
    ticker = args.ticker.upper()
    lookback = args.lookback_days

    result = run_pipeline_for_ticker(
        ticker,
        shares=args.shares,
        lookback=lookback,
        weights_path=args.weights,
        min_confidence=args.min_confidence,
    )

    print_summary(ticker, result, args)

    if args.verbose:
        print_technical_detail(ticker, result, args)


if __name__ == "__main__":
    main()
