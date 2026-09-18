import io

import pandas as pd
import pytest

from portfolio_glue import apply_portfolio_exposure_cap, load_portfolio_csv, map_action


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
