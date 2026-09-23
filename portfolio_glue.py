"""
Pure helper functions for run_portfolio.py (and the /api/portfolio/signal
backend route): CSV ingestion, position dedup, per-ticker pipeline scanning,
portfolio-level exposure capping, and buy/sell/hold mapping. Kept separate
from run_portfolio.py's CLI/file-report orchestration so these are
unit-testable, and reusable from the API, without touching yfinance or torch
at import time.
"""

import io

import pandas as pd

REQUIRED_COLUMNS = {"Account Number", "Investment Name", "Symbol", "Shares"}


def _clean_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles a trailing comma in the header (e.g. "...,Total Value,"), which
    pandas turns into an all-NaN "Unnamed: N" column.
    """
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing required columns: {sorted(missing)}")

    df["Symbol"] = df["Symbol"].str.strip().str.upper()
    return df


def load_portfolio_csv(path: str) -> pd.DataFrame:
    """
    Reads a portfolio holdings CSV (Account Number, Investment Name, Symbol,
    Shares, plus any other columns, e.g. Share Price/Total Value, which are
    tolerated but not required or used for valuation — this tool prices
    positions live, not from the CSV's own price/value columns).
    """
    return _clean_portfolio_df(pd.read_csv(path))


def parse_portfolio_csv(content: bytes) -> pd.DataFrame:
    """Same as load_portfolio_csv, but from in-memory bytes (e.g. an uploaded file)."""
    return _clean_portfolio_df(pd.read_csv(io.BytesIO(content)))


def dedupe_positions(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Symbol", as_index=False)
        .agg(
            shares=("Shares", "sum"),
            accounts=("Account Number", lambda s: sorted(set(s))),
        )
    )


def apply_portfolio_exposure_cap(
    buy_positions: list[dict],
    max_portfolio_risk: float = 1.0,
) -> tuple[list[dict], float, float]:
    """
    Scales down BUY-side position sizes proportionally if their sum exceeds
    max_portfolio_risk (fraction of capital, default 1.0 = 100%). Each dict in
    buy_positions must have a "position_size" key (fraction of capital).

    Returns (scaled_positions, total_before_cap, total_after_cap). Each
    returned dict gains a "capped" bool. Sell/Hold positions are not passed
    into this function at all — there is nothing here for them to interact with.
    """
    total = sum(p["position_size"] for p in buy_positions)
    if total <= max_portfolio_risk or total == 0:
        return [{**p, "capped": False} for p in buy_positions], total, total

    scale = max_portfolio_risk / total
    scaled = [
        {**p, "position_size": p["position_size"] * scale, "capped": True}
        for p in buy_positions
    ]
    return scaled, total, max_portfolio_risk


def map_action(should_trade: bool, direction: str) -> str:
    """
    Maps the pipeline's should_trade/direction output onto Buy/Sell/Hold.

    This does not consider whether the position is already held — should_trade
    reflects whether there's enough edge to OPEN a position, not whether
    there's a reason to EXIT one. should_trade=False always maps to Hold,
    whether this is a prospective new position or an existing holding with no
    fresh signal. Treat "Sell" here as "the model favors the short direction
    with enough confidence," not as validated exit logic for a long holding.
    """
    if not should_trade:
        return "Hold"
    return "Buy" if direction == "long" else "Sell"


def run_portfolio_scan(
    positions: pd.DataFrame,
    net,
    checkpoint: dict | None,
    weights_path: str,
    lookback: str,
    min_confidence: float,
    max_portfolio_risk: float = 1.0,
) -> tuple[list[dict], list[dict], dict]:
    """
    Runs the layer1-5 pipeline over each deduped position (as returned by
    dedupe_positions) and applies the same action-mapping/exposure-cap steps
    run_portfolio.py's CLI report uses. Shared by run_portfolio.py and the
    /api/portfolio/signal backend route so both give identical results for
    the same CSV.

    Returns (results, errors, summary) — see run_portfolio.py's run_portfolio
    for the shape of each.
    """
    from pipeline_core import run_pipeline_for_ticker

    trained_tickers = checkpoint.get("tickers") if checkpoint else None

    results = []
    errors = []
    for _, row in positions.iterrows():
        ticker = row["Symbol"]
        try:
            result = run_pipeline_for_ticker(
                ticker,
                shares=int(row["shares"]),
                lookback=lookback,
                weights_path=weights_path,
                min_confidence=min_confidence,
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
        buy_positions, max_portfolio_risk=max_portfolio_risk
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
        "max_portfolio_risk": max_portfolio_risk,
    }
