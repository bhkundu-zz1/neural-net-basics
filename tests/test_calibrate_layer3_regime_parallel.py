import numpy as np
import pandas as pd
import pytest

from calibrate_layer3_regime import build_full_dataset, collect_all_oos_regime_calls
from calibrate_layer3_regime_parallel import collect_all_oos_regime_calls_parallel


def _synthetic_prices(n=800, trend=0.0003, seed=0):
    """Enough history (n days) to span a few walk-forward folds at small
    train/test windows, mirroring test_calibrate_layer3_regime.py's fixture
    style but longer, so make_fold_boundaries produces more than one fold."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, n)
    prices = 100 * np.cumprod(1 + trend + noise)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(prices, index=dates)


@pytest.fixture
def prices_by_ticker():
    return {
        "AAA": _synthetic_prices(seed=1),
        "BBB": _synthetic_prices(seed=2, trend=-0.0002),
        "CCC": _synthetic_prices(seed=3, trend=0.0),
    }


def _sort_key(call):
    return (call["ticker"], str(call["date"]))


def test_parallel_matches_sequential_results(prices_by_ticker, tmp_path, monkeypatch):
    # Both scripts write per-fold checkpoints under the same script_name
    # ("regimecalib"), keyed by run_config — use a tmp CHECKPOINT_DIR so this
    # test doesn't read/write the real .fold_checkpoints/ used by actual runs,
    # and so run_config-based cache hits can't hide a real behavioral difference.
    import fold_checkpoint
    monkeypatch.setattr(fold_checkpoint, "CHECKPOINT_DIR", str(tmp_path))

    train_years, test_months, step_months = 1.0, 3.0, 3.0
    regime_horizon_days, up_threshold, down_threshold, steps = 5, 0.02, 0.02, 1

    run_config_seq = {
        "tickers": "AAA,BBB,CCC", "lookback_days": "test", "regime_horizon_days": regime_horizon_days,
        "steps": steps, "up_threshold": up_threshold, "down_threshold": down_threshold,
        "train_years": train_years, "test_months": test_months, "step_months": step_months,
        "variant": "sequential",
    }
    run_config_par = {**run_config_seq, "variant": "parallel"}  # distinct cache key from the sequential run

    sequential_calls = collect_all_oos_regime_calls(
        prices_by_ticker, train_years, test_months, step_months,
        regime_horizon_days, up_threshold, down_threshold, run_config_seq, steps,
    )
    parallel_calls = collect_all_oos_regime_calls_parallel(
        prices_by_ticker, train_years, test_months, step_months,
        regime_horizon_days, up_threshold, down_threshold, run_config_par, steps, workers=2,
    )

    assert len(sequential_calls) > 0
    assert len(parallel_calls) > 0

    seq_sorted = sorted(sequential_calls, key=_sort_key)
    par_sorted = sorted(parallel_calls, key=_sort_key)

    assert len(seq_sorted) == len(par_sorted)
    for seq_call, par_call in zip(seq_sorted, par_sorted):
        assert seq_call["ticker"] == par_call["ticker"]
        assert seq_call["date"] == par_call["date"]
        assert seq_call["dominant_regime"] == par_call["dominant_regime"]
        assert seq_call["realized_label"] == par_call["realized_label"]
        assert seq_call["forward_return"] == pytest.approx(par_call["forward_return"])
        assert seq_call["bull_probability"] == pytest.approx(par_call["bull_probability"])


def test_parallel_run_config_excludes_workers_from_cache_key():
    # workers must not appear in run_config, otherwise --workers 4 vs --workers 8
    # on an otherwise-identical run would miss the cache and recompute everything —
    # defeating the whole point of sharing fold_checkpoint results across runs.
    import calibrate_layer3_regime_parallel as m
    import inspect
    source = inspect.getsource(m.main)
    assert '"workers"' not in source.split("run_config = {")[1].split("}")[0]
