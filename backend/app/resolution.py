"""
Resolves open paper trades once their holding horizon has passed.

Horizon is measured in TRADING days, not calendar days — mirrors
train_layer4.py's build_dataset_for_ticker, which computes forward returns
as prices.iloc[i + forward_days] (a positional offset into a trading-day-
indexed series), and backtest_layer4.py's compute_trades, which derives
directional_return as direction_sign * actual_return. This module applies
the same two patterns to a live paper trade instead of a historical sample.
"""

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

RESOLUTION_LOOKBACK = "3mo"


def fetch_price_series(ticker: str) -> pd.Series:
    data = yf.download(ticker, period=RESOLUTION_LOOKBACK, auto_adjust=True, progress=False)
    return data["Close"][ticker].dropna()


def _entry_index(prices: pd.Series, entry_date: str) -> int | None:
    entry_ts = pd.Timestamp(entry_date)
    exact = prices.index.get_indexer([entry_ts])[0]
    if exact != -1:
        return exact
    pos = prices.index.searchsorted(entry_ts)
    if pos >= len(prices):
        return None
    return int(pos)


def resolve_trade(trade_doc: dict, price_series: pd.Series) -> dict:
    """
    Takes one open trade document and a trading-day-indexed price series for
    its ticker; returns an updated copy of the document. If the trade isn't
    eligible yet (not enough trading days have elapsed since entry_date), the
    original document is returned unchanged (still "open").

    v1 simplification: only entry-side execution cost is modeled (already
    stored on the trade at placement) — exit-side slippage is not charged.
    An exactly-zero directional return is treated as "lost" (an arbitrary
    but explicit tie-break, not a silent default).
    """
    doc = dict(trade_doc)
    horizon = doc["horizon_trading_days"]

    idx = _entry_index(price_series, doc["entry_date"])
    if idx is None or len(price_series) <= idx + horizon:
        return doc  # not enough trading days have passed yet — leave open

    entry_price = doc["entry_price"]
    exit_price = float(price_series.iloc[idx + horizon])
    resolution_date = price_series.index[idx + horizon]

    actual_return = (exit_price - entry_price) / entry_price
    direction_sign = 1 if doc["direction"] == "long" else -1
    directional_return = direction_sign * actual_return

    pnl_pct = doc["position_size"] * directional_return
    pnl = pnl_pct * doc["position_value"]

    doc.update({
        "status": "won" if directional_return > 0 else "lost",
        "resolution_date": str(resolution_date.date()),
        "exit_price": exit_price,
        "actual_return": actual_return,
        "pnl_pct": pnl_pct,
        "pnl": pnl,
        "resolution_note": None,
    })
    return doc


def resolve_open_trades(open_trades: list[dict], as_of: str | None = None) -> dict:
    """
    Groups open trades by ticker, fetches each ticker's price series once,
    and resolves whichever trades are eligible. `as_of` is accepted for
    tests only (unused here — resolution eligibility is driven entirely by
    how much real trading-day history yfinance returns, not a simulated
    clock); passing it does not change behavior in this implementation.
    """
    by_ticker: dict[str, list[dict]] = {}
    for trade in open_trades:
        by_ticker.setdefault(trade["ticker"], []).append(trade)

    resolved, still_pending, errors = [], [], []

    for ticker, trades in by_ticker.items():
        try:
            prices = fetch_price_series(ticker)
        except Exception as exc:
            errors.append({"ticker": ticker, "detail": str(exc)})
            continue

        for trade in trades:
            updated = resolve_trade(trade, prices)
            if updated["status"] == "open":
                still_pending.append({"ticker": ticker, "entry_date": trade["entry_date"]})
            else:
                resolved.append(updated)

    return {"resolved": resolved, "still_pending": still_pending, "errors": errors}
