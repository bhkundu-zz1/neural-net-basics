"""
Shared layer4 -> layer5 glue: converts QuantEdgeNet's 3-way softmax output
into the inputs layer5.kelly_position_size / should_trade expect.

If calibration_table.json exists (built by build_calibration_table.py from
out-of-sample walk-forward trades), win_loss_ratio and edge_bps are looked
up from EMPIRICAL history at the model's actual confidence level, instead
of the hand-picked formulas below. See build_calibration_table.py's
docstring for why: the original formulas were invented before any
out-of-sample data existed, and are unrelated to what the model actually
does when right or wrong.

Without a calibration table, falls back to the original placeholder
heuristics (not derived from any backtest) — see backtest_layer4.py for how
to evaluate whether either version has financial value.
"""

import json
import os

import numpy as np
import pandas as pd

_CALIBRATION_TABLE_PATH = "calibration_table.json"
_calibration_table_cache = None


def _load_calibration_table():
    global _calibration_table_cache
    if _calibration_table_cache is not None:
        return _calibration_table_cache
    if os.path.exists(_CALIBRATION_TABLE_PATH):
        with open(_CALIBRATION_TABLE_PATH) as f:
            _calibration_table_cache = json.load(f)
    else:
        _calibration_table_cache = []
    return _calibration_table_cache


def _lookup_calibration_bucket(confidence: float, table: list[dict]) -> dict | None:
    for entry in table:
        if entry["conf_low"] <= confidence < entry["conf_high"]:
            return entry
    return None


def build_trade_inputs(edge_probs: dict, prices: pd.Series) -> dict:
    """
      win_probability = margin of the chosen direction over its runner-up class,
                        rescaled to [0.5, 1.0] — reflects how much the softmax
                        favors this direction over the closest alternative,
                        not just its raw top-class probability (which ignores
                        whether the runner-up is a near-tie or a distant third).
      win_loss_ratio, edge_bps = looked up from calibration_table.json at this
                        confidence level if available (empirical, out-of-sample);
                        otherwise fall back to the original placeholder formulas.
      execution_cost_bps = spread + market impact assumption for a liquid
                        large-cap stock (placeholder constants; unrelated to
                        the calibration table, since execution costs aren't a
                        function of model confidence).
    """
    long_p, short_p, flat_p = edge_probs["long"], edge_probs["short"], edge_probs["flat"]
    direction = "long" if long_p >= short_p else "short"
    chosen, runner_up = max(long_p, short_p), min(long_p, short_p)

    # Margin-based confidence: a coin-flip between long/short (chosen==runner_up)
    # maps to 0.5; complete certainty (runner_up==0) maps to 1.0. This is more
    # informative than max(long_p, short_p) alone, which can't distinguish "long
    # barely beats short" from "long dominates a near-zero short probability."
    margin = chosen - runner_up
    win_probability = 0.5 + margin / 2

    calibration_table = _load_calibration_table()
    bucket = _lookup_calibration_bucket(win_probability, calibration_table)

    if bucket is not None:
        win_loss_ratio = bucket["win_loss_ratio"]
        # edge_bps from the bucket's actual measured average directional return,
        # not a linear guess — this is what trades at this confidence level have
        # historically been worth, out-of-sample, before position sizing/costs.
        edge_bps = bucket["avg_directional_return"] * 10_000
    else:
        # Fallback: original placeholder heuristics (no calibration data available).
        recent_returns = prices.pct_change().tail(20).dropna()
        upside = recent_returns[recent_returns > 0].mean()
        downside = abs(recent_returns[recent_returns < 0].mean())
        win_loss_ratio = max(upside / downside, 0.1) if downside and not np.isnan(downside) else 1.0
        edge_bps = (win_probability - 0.5) * 200

    spread_bps = 2.0          # placeholder: assumes a liquid large-cap, tight spread
    market_impact_bps = 3.0   # placeholder: assumes modest order size
    execution_cost_bps = spread_bps + market_impact_bps

    return {
        "direction": direction,
        "win_probability": win_probability,
        "win_loss_ratio": win_loss_ratio,
        "edge_bps": edge_bps,
        "execution_cost_bps": execution_cost_bps,
        "flat_probability": flat_p,
        "calibrated": bucket is not None,
    }
