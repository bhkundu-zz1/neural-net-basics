import io

import pandas as pd
import pytest

import pipeline_core
from portfolio_glue import (
    apply_portfolio_exposure_cap,
    dedupe_positions,
    load_portfolio_csv,
    map_action,
    parse_portfolio_csv,
    run_portfolio_scan,
)


def test_load_portfolio_csv_drops_trailing_unnamed_column():
    csv_text = (
        "Account Number,Investment Name,Symbol,Shares,Share Price,Total Value,\n"
        "abcd001,ADVANCED MICRO DEVICES INC,AMD,25,465.58,11639.5,\n"
    )
    df = load_portfolio_csv(io.StringIO(csv_text))
    assert list(df.columns) == [
        "Account Number", "Investment Name", "Symbol", "Shares", "Share Price", "Total Value",
    ]


def test_load_portfolio_csv_missing_required_column_raises():
    csv_text = "Account Number,Investment Name,Shares\nabcd001,ADVANCED MICRO DEVICES INC,25\n"
    with pytest.raises(ValueError, match="missing required columns"):
        load_portfolio_csv(io.StringIO(csv_text))


def test_load_portfolio_csv_uppercases_symbol():
    csv_text = "Account Number,Investment Name,Symbol,Shares\nabcd001,NVIDIA CORP,nvda,10\n"
    df = load_portfolio_csv(io.StringIO(csv_text))
    assert df.loc[0, "Symbol"] == "NVDA"


def test_load_portfolio_csv_without_price_columns():
    csv_text = "Account Number,Investment Name,Symbol,Shares\nabcd001,NVIDIA CORP,NVDA,10\n"
    df = load_portfolio_csv(io.StringIO(csv_text))
    assert df.loc[0, "Shares"] == 10


@pytest.mark.parametrize(
    "position_sizes,expected_total_after,expected_capped",
    [
        ([0.2, 0.3], 0.5, False),
        ([0.6, 0.6], 1.0, True),
    ],
)
def test_apply_portfolio_exposure_cap(position_sizes, expected_total_after, expected_capped):
    buy_positions = [{"ticker": f"T{i}", "position_size": s} for i, s in enumerate(position_sizes)]
    scaled, total_before, total_after = apply_portfolio_exposure_cap(buy_positions, max_portfolio_risk=1.0)

    assert total_before == pytest.approx(sum(position_sizes))
    assert total_after == pytest.approx(expected_total_after)
    assert all(p["capped"] == expected_capped for p in scaled)
    if expected_capped:
        assert sum(p["position_size"] for p in scaled) == pytest.approx(1.0)
    else:
        assert [p["position_size"] for p in scaled] == position_sizes


def test_apply_portfolio_exposure_cap_empty_list():
    scaled, total_before, total_after = apply_portfolio_exposure_cap([], max_portfolio_risk=1.0)
    assert scaled == []
    assert total_before == 0
    assert total_after == 0


@pytest.mark.parametrize(
    "should_trade,direction,expected",
    [
        (False, "long", "Hold"),
        (False, "short", "Hold"),
        (True, "long", "Buy"),
        (True, "short", "Sell"),
    ],
)
def test_map_action(should_trade, direction, expected):
    assert map_action(should_trade, direction) == expected


def test_parse_portfolio_csv_from_bytes_matches_load_from_path():
    csv_text = "Account Number,Investment Name,Symbol,Shares\nabcd001,NVIDIA CORP,nvda,10\n"
    df = parse_portfolio_csv(csv_text.encode("utf-8"))
    assert df.loc[0, "Symbol"] == "NVDA"
    assert df.loc[0, "Shares"] == 10


def test_dedupe_positions_sums_shares_and_collects_accounts():
    df = pd.DataFrame({
        "Account Number": ["a1", "a2"],
        "Investment Name": ["NVIDIA CORP", "NVIDIA CORP"],
        "Symbol": ["NVDA", "NVDA"],
        "Shares": [10, 5],
    })
    positions = dedupe_positions(df)
    assert len(positions) == 1
    assert positions.loc[0, "shares"] == 15
    assert positions.loc[0, "accounts"] == ["a1", "a2"]


def _fake_pipeline_result(ticker, shares, direction, should_trade, position_size=0.1):
    return {
        "ticker": ticker,
        "shares": shares,
        "last_price": 100.0,
        "position_value": shares * 100.0,
        "direction": direction,
        "win_probability": 0.8,
        "position_size": position_size,
        "should_trade": should_trade,
    }


def test_run_portfolio_scan_maps_actions_and_caps_exposure(monkeypatch):
    def fake_run_pipeline_for_ticker(ticker, shares, lookback, weights_path, min_confidence, net, checkpoint):
        if ticker == "NVDA":
            return _fake_pipeline_result(ticker, shares, "long", True, position_size=0.6)
        if ticker == "AMD":
            return _fake_pipeline_result(ticker, shares, "long", True, position_size=0.6)
        return _fake_pipeline_result(ticker, shares, "short", False, position_size=0.3)

    monkeypatch.setattr(pipeline_core, "run_pipeline_for_ticker", fake_run_pipeline_for_ticker)

    positions = pd.DataFrame({
        "Symbol": ["NVDA", "AMD", "MSFT"],
        "shares": [10, 20, 5],
        "accounts": [["a1"], ["a1"], ["a1"]],
    })

    results, errors, summary = run_portfolio_scan(
        positions, net=None, checkpoint=None, weights_path="weights.pt",
        lookback="2y", min_confidence=0.75, max_portfolio_risk=1.0,
    )

    assert errors == []
    actions = {r["ticker"]: r["action"] for r in results}
    assert actions == {"NVDA": "Buy", "AMD": "Buy", "MSFT": "Hold"}

    msft = next(r for r in results if r["ticker"] == "MSFT")
    assert msft["position_size"] == 0.0  # Hold positions are zeroed out

    # NVDA + AMD sum to 1.2 > max_portfolio_risk=1.0, so both get scaled down
    assert summary["total_buy_exposure_before_cap"] == pytest.approx(1.2)
    assert summary["total_buy_exposure_after_cap"] == pytest.approx(1.0)
    assert all(r["capped"] for r in results if r["action"] == "Buy")


def test_run_portfolio_scan_collects_errors_without_failing_other_tickers(monkeypatch):
    def fake_run_pipeline_for_ticker(ticker, shares, lookback, weights_path, min_confidence, net, checkpoint):
        if ticker == "BADTICKER":
            raise ValueError("no data found")
        return _fake_pipeline_result(ticker, shares, "long", True)

    monkeypatch.setattr(pipeline_core, "run_pipeline_for_ticker", fake_run_pipeline_for_ticker)

    positions = pd.DataFrame({
        "Symbol": ["NVDA", "BADTICKER"],
        "shares": [10, 5],
        "accounts": [["a1"], ["a1"]],
    })

    results, errors, summary = run_portfolio_scan(
        positions, net=None, checkpoint=None, weights_path="weights.pt",
        lookback="2y", min_confidence=0.75,
    )

    assert len(results) == 1
    assert results[0]["ticker"] == "NVDA"
    assert len(errors) == 1
    assert errors[0]["symbol"] == "BADTICKER"
