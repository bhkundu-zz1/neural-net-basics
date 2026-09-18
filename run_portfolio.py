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
from datetime import datetime, timezone

from pipeline_core import load_model, run_pipeline_for_ticker
from portfolio_glue import apply_portfolio_exposure_cap, load_portfolio_csv, map_action

DEFAULT_WEIGHTS = "layer4_weights_nasdaq100.pt"
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
    return parser.parse_args()


def dedupe_positions(df):
    return (
        df.groupby("Symbol", as_index=False)
        .agg(
            shares=("Shares", "sum"),
            accounts=("Account Number", lambda s: sorted(set(s))),
        )
    )


def run_portfolio(args):
    df = load_portfolio_csv(args.csv)
    positions = dedupe_positions(df)

    net, checkpoint = load_model(args.weights)
    trained_tickers = checkpoint.get("tickers") if checkpoint else None

    results = []
    errors = []
    for _, row in positions.iterrows():
        ticker = row["Symbol"]
        try:
            result = run_pipeline_for_ticker(
                ticker,
                shares=int(row["shares"]),
                lookback=args.lookback_days,
                weights_path=args.weights,
                min_confidence=args.min_confidence,
                net=net,
                checkpoint=checkpoint,
            )
        except Exception as exc:
            errors.append({"symbol": ticker, "reason": str(exc)})
            continue

        result["in_training_universe"] = trained_tickers is not None and ticker in trained_tickers
        result["accounts"] = row["accounts"]
        result["action"] = map_action(result["should_trade"], result["direction"])
        if result["action"] != "Buy":
            # Kelly sizing only means something for a position we're actually
            # recommending opening; Hold/Sell display 0% rather than the raw
            # (unused) Kelly fraction the pipeline computed for this ticker.
            result["position_size"] = 0.0
        results.append(result)

    buy_positions = [
        {"ticker": r["ticker"], "position_size": r["position_size"]}
        for r in results
        if r["action"] == "Buy"
    ]
    scaled_buys, total_buy_before_cap, total_buy_after_cap = apply_portfolio_exposure_cap(
        buy_positions, max_portfolio_risk=args.max_portfolio_risk
    )
    scaled_by_ticker = {p["ticker"]: p for p in scaled_buys}
    for r in results:
        scaled = scaled_by_ticker.get(r["ticker"])
        if scaled is not None:
            r["position_size"] = scaled["position_size"]
            r["capped"] = scaled["capped"]
        else:
            r["capped"] = False

    total_portfolio_value = sum(r["position_value"] for r in results)

    return results, errors, {
        "total_portfolio_value": total_portfolio_value,
        "total_buy_exposure_before_cap": total_buy_before_cap,
        "total_buy_exposure_after_cap": total_buy_after_cap,
        "max_portfolio_risk": args.max_portfolio_risk,
    }


def note_for(result: dict) -> str:
    notes = []
    if not result["in_training_universe"]:
        notes.append("out-of-sample / unvalidated for this model")
    if result.get("capped"):
        notes.append("size scaled down by portfolio exposure cap")
    return "; ".join(notes)


def print_report(results, errors, summary):
    print("=== Portfolio pipeline report ===\n")
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
    print_report(results, errors, summary)
    write_reports(results, errors, summary, args.output)


if __name__ == "__main__":
    main()
