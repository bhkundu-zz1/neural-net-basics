"""
Pure helper functions for run_portfolio.py: CSV ingestion, portfolio-level
exposure capping, and buy/sell/hold mapping. Kept separate from
run_portfolio.py's I/O/orchestration so these are unit-testable without
touching yfinance or torch.
"""

import pandas as pd

REQUIRED_COLUMNS = {"Account Number", "Investment Name", "Symbol", "Shares"}


def load_portfolio_csv(path: str) -> pd.DataFrame:
    """
    Reads a portfolio holdings CSV (Account Number, Investment Name, Symbol,
    Shares, plus any other columns, e.g. Share Price/Total Value, which are
    tolerated but not required or used for valuation — this tool prices
    positions live, not from the CSV's own price/value columns).

    Handles a trailing comma in the header (e.g. "...,Total Value,"), which
    pandas turns into an all-NaN "Unnamed: N" column.
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df = df.dropna(axis=1, how="all")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing required columns: {sorted(missing)}")

    df["Symbol"] = df["Symbol"].str.strip().str.upper()
    return df


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
