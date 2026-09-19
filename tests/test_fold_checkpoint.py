import os

import fold_checkpoint


def test_fold_key_same_config_same_key_regardless_of_dict_order():
    config_a = {"tickers": "AAPL,MSFT", "epochs": 100, "lr": 0.001}
    config_b = {"lr": 0.001, "epochs": 100, "tickers": "AAPL,MSFT"}
    assert fold_checkpoint.fold_key("walkforward", config_a) == fold_checkpoint.fold_key("walkforward", config_b)


def test_fold_key_different_config_gives_different_key():
    base = {"tickers": "AAPL,MSFT", "epochs": 100, "lr": 0.001}
    changed = {"tickers": "AAPL,MSFT", "epochs": 50, "lr": 0.001}
    assert fold_checkpoint.fold_key("walkforward", base) != fold_checkpoint.fold_key("walkforward", changed)


def test_fold_key_different_script_name_gives_different_key():
    config = {"tickers": "AAPL,MSFT", "epochs": 100}
    assert fold_checkpoint.fold_key("walkforward", config) != fold_checkpoint.fold_key("calibtable", config)


def test_save_then_load_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(fold_checkpoint, "CHECKPOINT_DIR", str(tmp_path))
    config = {"tickers": "AAPL", "epochs": 10}
    path = fold_checkpoint.checkpoint_path("walkforward", config, 1)

    obj = {"fold": 1, "accuracy": 0.51}
    fold_checkpoint.save_fold_checkpoint(path, obj)
    loaded = fold_checkpoint.load_fold_checkpoint(path)

    assert loaded == obj


def test_save_is_atomic_no_leftover_tmp_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fold_checkpoint, "CHECKPOINT_DIR", str(tmp_path))
    config = {"tickers": "AAPL", "epochs": 10}
    path = fold_checkpoint.checkpoint_path("walkforward", config, 1)

    fold_checkpoint.save_fold_checkpoint(path, {"fold": 1})

    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_load_missing_checkpoint_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(fold_checkpoint, "CHECKPOINT_DIR", str(tmp_path))
    config = {"tickers": "AAPL", "epochs": 10}
    path = fold_checkpoint.checkpoint_path("walkforward", config, 1)

    assert fold_checkpoint.load_fold_checkpoint(path) is None


def test_resolve_fold_result_calls_compute_fn_on_cache_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(fold_checkpoint, "CHECKPOINT_DIR", str(tmp_path))
    config = {"tickers": "AAPL", "epochs": 10}
    calls = []

    def compute():
        calls.append(1)
        return {"fold": 1, "accuracy": 0.5}

    result, from_cache = fold_checkpoint.resolve_fold_result("walkforward", config, 1, compute)

    assert result == {"fold": 1, "accuracy": 0.5}
    assert from_cache is False
    assert len(calls) == 1


def test_resolve_fold_result_skips_compute_fn_on_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(fold_checkpoint, "CHECKPOINT_DIR", str(tmp_path))
    config = {"tickers": "AAPL", "epochs": 10}
    calls = []

    def compute():
        calls.append(1)
        return {"fold": 1, "accuracy": 0.5}

    fold_checkpoint.resolve_fold_result("walkforward", config, 1, compute)
    result, from_cache = fold_checkpoint.resolve_fold_result("walkforward", config, 1, compute)

    assert result == {"fold": 1, "accuracy": 0.5}
    assert from_cache is True
    assert len(calls) == 1
