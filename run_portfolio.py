"""
Runs the layer1-5 pipeline (via pipeline_core.run_pipeline_for_ticker) across
every position in a portfolio CSV and reports a buy/sell/hold recommendation
per position, plus a portfolio-level summary.

CSV schema: Account Number, Investment Name, Symbol, Shares (Share Price/Total
Value are tolerated if present but ignored — this tool prices every position
live via yfinance, not from the CSV's own price/value columns, since those go
stale the moment the CSV is exported).

IMPORTANT CAVEATS — see docs/portfolio_guide.md for the full picture:
  - Tickers outside the loaded weights' training universe are scored anyway
    and flagged "out-of-sample / unvalidated for this model" rather than
    skipped — treat their signal with extra skepticism.
  - Buy/Sell/Hold mapping has no concept of "exit an existing position" — it
    only reflects whether the model currently favors opening a long or short.
    should_trade=False always means Hold, even for a position you already own.
  - The portfolio exposure cap is a simple proportional scale-down of BUY
    sizes, not a correlation-aware risk model.
  - Execution cost assumptions (2bps spread / 3bps market impact) are tuned
    for liquid large-caps and likely understate slippage for thinner names.
  - This is a research tool, not an automated trading system.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import llm_narration
from pipeline_core import load_model
from portfolio_glue import dedupe_positions, load_portfolio_csv, run_portfolio_scan

# See run_pipeline.py for why: avoids crashing on characters (em-dashes,
# LLM-generated punctuation, etc.) that a Windows console's default codepage
# can't encode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_WEIGHTS = "layer4_weights_expanded.pt"
DEFAULT_MIN_CONFIDENCE = 0.75
DEFAULT_MAX_PORTFOLIO_RISK = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run the layer1-5 pipeline across a portfolio CSV.")
    parser.add_argument("--csv", required=True, help="Path to the portfolio holdings CSV.")
    parser.add_argument(
        "--lookback-days",
        default="2y",
        help="Price history window as a yfinance period string, e.g. 6mo, 1y, 2y (default: 2y)",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Trained layer4 weights to load for every position (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Only flag should_trade=True when win_probability clears this (default: {DEFAULT_MIN_CONFIDENCE}).",
    )
    parser.add_argument(
        "--max-portfolio-risk",
        type=float,
        default=DEFAULT_MAX_PORTFOLIO_RISK,
        help=f"Cap on total recommended BUY exposure as a fraction of capital "
        f"(default: {DEFAULT_MAX_PORTFOLIO_RISK} = 100%%). BUY sizes are scaled "
        f"down proportionally if their sum exceeds this.",
    )
    parser.add_argument(
        "--output",
        default="portfolio_report.json",
        help="Path to write the machine-readable JSON report (default: portfolio_report.json). "
        "A CSV report is also written alongside it with the same basename.",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip the LLM-generated portfolio summary (no network call).",
    )
    return parser.parse_args()


def run_portfolio(args):
    df = load_portfolio_csv(args.csv)
    positions = dedupe_positions(df)

    net, checkpoint = load_model(args.weights)

    return run_portfolio_scan(
        positions, net, checkpoint, args.weights, args.lookback_days,
        args.min_confidence, args.max_portfolio_risk,
    )


def note_for(result: dict) -> str:
    notes = []
    if not result["in_training_universe"]:
        notes.append("out-of-sample / unvalidated for this model")
    if result.get("capped"):
        notes.append("size scaled down by portfolio exposure cap")
    return "; ".join(notes)


def fallback_portfolio_summary(results, summary) -> str:
    """Plain templated summary used when the LLM is unavailable or skipped."""
    counts = {"Buy": 0, "Sell": 0, "Hold": 0}
    for r in results:
        counts[r["action"]] += 1
    flagged = [r["ticker"] for r in results if r["action"] in ("Buy", "Sell") and not r["in_training_universe"]]

    parts = [
        f"{len(results)} position(s) scanned: {counts['Buy']} Buy, {counts['Sell']} Sell, "
        f"{counts['Hold']} Hold."
    ]
    if flagged:
        parts.append(f"Active signals on out-of-sample tickers (extra scrutiny warranted): {', '.join(flagged)}.")
    if summary["total_buy_exposure_before_cap"] > summary["max_portfolio_risk"]:
        parts.append(
            f"BUY exposure before capping ({summary['total_buy_exposure_before_cap']*100:.1f}%) exceeded "
            f"the {summary['max_portfolio_risk']*100:.0f}% cap and was scaled down."
        )
    return " ".join(parts)


def print_report(results, errors, summary, explain=True):
    print("=== Portfolio pipeline report ===\n")

    if results:
        narration = llm_narration.narrate_portfolio(results, summary) if explain else None
        if narration is None:
            narration = fallback_portfolio_summary(results, summary)
        print(narration)
        print()

    header = f"{'Symbol':<8}{'Shares':>10}{'Price':>10}{'Value':>14}{'Direction':>10}{'Action':>7}{'Size%':>8}{'In-univ?':>9}  Notes"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["ticker"]):
        print(
            f"{r['ticker']:<8}{r['shares']:>10}{r['last_price']:>10.2f}{r['position_value']:>14,.2f}"
            f"{r['direction']:>10}{r['action']:>7}{r['position_size']*100:>7.2f}%"
            f"{'Yes' if r['in_training_universe'] else 'No':>9}  {note_for(r)}"
        )

    print(f"\nTotal portfolio value (live-priced): ${summary['total_portfolio_value']:,.2f}")
    print(f"Total BUY exposure before cap: {summary['total_buy_exposure_before_cap']*100:.2f}%")
    print(f"Total BUY exposure after cap:  {summary['total_buy_exposure_after_cap']*100:.2f}% "
          f"(cap: {summary['max_portfolio_risk']*100:.0f}%)")

    capped = [r["ticker"] for r in results if r.get("capped")]
    if capped:
        print(f"Capped positions: {', '.join(capped)}")

    out_of_universe = [r["ticker"] for r in results if not r["in_training_universe"]]
    if out_of_universe:
        print(f"Out-of-universe (unvalidated) tickers: {', '.join(out_of_universe)}")

    if errors:
        print(f"\nWARNING: {len(errors)} symbol(s) could not be evaluated:")
        for e in errors:
            print(f"    {e['symbol']}: {e['reason']}")


def write_markdown_report(results, errors, summary, md_path, generated_at):
    lines = [
        "# Portfolio pipeline report",
        "",
        f"Generated: {generated_at}",
        "",
        "| Symbol | Shares | Price | Value | Direction | Action | Size % | In-universe? | Notes |",
        "|---|---:|---:|---:|---|---|---:|---|---|",
    ]
    for r in sorted(results, key=lambda r: r["ticker"]):
        lines.append(
            f"| {r['ticker']} | {r['shares']} | {r['last_price']:.2f} | {r['position_value']:,.2f} "
            f"| {r['direction']} | {r['action']} | {r['position_size']*100:.2f}% "
            f"| {'Yes' if r['in_training_universe'] else 'No'} | {note_for(r)} |"
        )

    lines += [
        "",
        f"**Total portfolio value (live-priced):** ${summary['total_portfolio_value']:,.2f}",
        "",
        f"**Total BUY exposure:** {summary['total_buy_exposure_before_cap']*100:.2f}% before cap -> "
        f"{summary['total_buy_exposure_after_cap']*100:.2f}% after cap "
        f"(cap: {summary['max_portfolio_risk']*100:.0f}%)",
    ]

    capped = [r["ticker"] for r in results if r.get("capped")]
    if capped:
        lines += ["", f"**Capped positions:** {', '.join(capped)}"]

    out_of_universe = [r["ticker"] for r in results if not r["in_training_universe"]]
    if out_of_universe:
        lines += ["", f"**Out-of-universe (unvalidated) tickers:** {', '.join(out_of_universe)}"]

    if errors:
        lines += ["", f"**WARNING: {len(errors)} symbol(s) could not be evaluated:**", ""]
        for e in errors:
            lines.append(f"- {e['symbol']}: {e['reason']}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_reports(results, errors, summary, output_path):
    serializable = [
        {k: v for k, v in r.items() if k not in ("net", "checkpoint", "trade_inputs", "signal", "factor_result", "regime_probs", "edge_probs")}
        for r in results
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "generated_at": generated_at,
        "positions": serializable,
        "errors": errors,
        "summary": summary,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote JSON report to {output_path}")

    base_path = output_path.rsplit(".", 1)[0]

    csv_path = base_path + ".csv"
    import pandas as pd
    pd.DataFrame(serializable).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Wrote CSV report to {csv_path}")

    md_path = base_path + ".md"
    write_markdown_report(results, errors, summary, md_path, generated_at)
    print(f"Wrote Markdown report to {md_path}")


def main():
    args = parse_args()
    results, errors, summary = run_portfolio(args)
    print_report(results, errors, summary, explain=not args.no_explain)
    write_reports(results, errors, summary, args.output)


if __name__ == "__main__":
    main()
