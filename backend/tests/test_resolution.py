import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

import pandas as pd
import pytest

import resolution


def _price_series(dates, prices):
    return pd.Series(prices, index=pd.DatetimeIndex(dates))


def _base_trade(**overrides):
    trade = {
        "_id": "t1",
        "ticker": "NVDA",
        "direction": "long",
        "entry_date": "2026-01-05",
        "entry_price": 100.0,
        "position_size": 0.02,
        "position_value": 10000.0,
        "horizon_trading_days": 5,
        "status": "open",
        "resolution_date": None,
        "exit_price": None,
        "actual_return": None,
        "pnl": None,
        "pnl_pct": None,
    }
    trade.update(overrides)
    return trade


def test_eligible_long_trade_wins_when_price_rises():
    # 2026-01-05 is a Monday; 5 TRADING days later skips the following
    # weekend, landing on 2026-01-12 (a Monday) — 7 calendar days but
    # exactly 5 trading-day positions later, proving the offset is
    # positional, not calendar+5.
    dates = pd.date_range("2026-01-05", periods=10, freq="B")  # business days
    prices = [100.0, 101, 102, 103, 104, 110, 106, 107, 108, 109]
    series = _price_series(dates, prices)

    trade = _base_trade(direction="long", entry_price=100.0)
    result = resolution.resolve_trade(trade, series)

    assert result["status"] == "won"
    assert result["exit_price"] == 110.0
    assert result["actual_return"] == pytest.approx(0.10)
    assert result["pnl_pct"] == pytest.approx(0.02 * 0.10)
    assert result["pnl"] == pytest.approx(10000.0 * 0.02 * 0.10)


def test_eligible_short_trade_wins_when_price_falls():
    dates = pd.date_range("2026-01-05", periods=10, freq="B")
    prices = [100.0, 99, 98, 97, 96, 90, 91, 92, 93, 94]
    series = _price_series(dates, prices)

    trade = _base_trade(direction="short", entry_price=100.0)
    result = resolution.resolve_trade(trade, series)

    assert result["status"] == "won"
    assert result["actual_return"] == pytest.approx(-0.10)
    # directional_return = -1 * -0.10 = +0.10 for a short that fell
    assert result["pnl_pct"] == pytest.approx(0.02 * 0.10)


def test_eligible_long_trade_loses_when_price_falls():
    dates = pd.date_range("2026-01-05", periods=10, freq="B")
    prices = [100.0, 99, 98, 97, 96, 95, 94, 93, 92, 91]
    series = _price_series(dates, prices)

    trade = _base_trade(direction="long", entry_price=100.0)
    result = resolution.resolve_trade(trade, series)

    assert result["status"] == "lost"
    assert result["pnl_pct"] < 0


def test_not_yet_eligible_stays_open():
    # Only 3 trading days exist after entry — horizon is 5, so this trade
    # must remain open rather than resolving on incomplete data.
    dates = pd.date_range("2026-01-05", periods=4, freq="B")
    prices = [100.0, 101, 102, 103]
    series = _price_series(dates, prices)

    trade = _base_trade(entry_date="2026-01-05", horizon_trading_days=5)
    result = resolution.resolve_trade(trade, series)

    assert result["status"] == "open"
    assert result["exit_price"] is None
    assert result["resolution_date"] is None


def test_weekend_gap_uses_trading_day_offset_not_calendar_days():
    # Entry on a Friday; the very next VALUE in the series is the following
    # Monday (weekend skipped by the trading-day index itself, as a real
    # yfinance series would be) — horizon=1 should resolve to that Monday's
    # price, not "Friday + 1 calendar day" (Saturday, which doesn't exist
    # in the index at all).
    dates = pd.to_datetime(["2026-01-09", "2026-01-12", "2026-01-13"])  # Fri, Mon, Tue
    prices = [100.0, 105.0, 106.0]
    series = pd.Series(prices, index=pd.DatetimeIndex(dates))

    trade = _base_trade(entry_date="2026-01-09", horizon_trading_days=1, entry_price=100.0)
    result = resolution.resolve_trade(trade, series)

    assert result["status"] == "won"
    assert result["exit_price"] == 105.0
    assert result["resolution_date"] == "2026-01-12"


def test_resolve_open_trades_groups_by_ticker_and_reports_pending_and_resolved():
    dates = pd.date_range("2026-01-05", periods=10, freq="B")
    winning_prices = [100.0, 101, 102, 103, 104, 110, 106, 107, 108, 109]
    series = _price_series(dates, winning_prices)

    call_count = {"n": 0}

    def fake_fetch(ticker):
        call_count["n"] += 1
        return series

    import resolution as res_mod
    orig = res_mod.fetch_price_series
    res_mod.fetch_price_series = fake_fetch
    try:
        trades = [
            _base_trade(**{"_id": "a", "ticker": "NVDA", "entry_date": "2026-01-05"}),
            _base_trade(**{"_id": "b", "ticker": "NVDA", "entry_date": "2026-01-05"}),
            _base_trade(**{"_id": "c", "ticker": "NVDA", "entry_date": "2026-01-05", "horizon_trading_days": 20}),
        ]
        result = res_mod.resolve_open_trades(trades)
    finally:
        res_mod.fetch_price_series = orig

    assert call_count["n"] == 1  # one ticker, one fetch — grouped, not per-trade
    assert len(result["resolved"]) == 2
    assert len(result["still_pending"]) == 1
    assert result["errors"] == []
