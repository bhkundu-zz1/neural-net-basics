"""
Per-fold checkpointing for walk-forward validation scripts (walkforward_layer4.py,
build_calibration_table.py). Both retrain a fresh model for each of ~11 rolling-
window folds with no persistence between folds — a crash partway through means
redoing every completed fold's training from scratch on re-run.

Mirrors train_layer4.py's existing CACHE_DIR / _cache_path / _load_cache /
_save_cache pattern (atomic tmp-file-then-replace writes, SHA1-hashed config
keys so a changed setting produces a fresh key rather than silently reusing
stale results) — just keyed per (script_name, fold_index, full_run_config)
instead of per (kind, ticker, key_parts).

One checkpoint file per fold, not one growing file: if fold 8 finishes and
fold 9 crashes mid-training, fold 8's file already landed via atomic
tmp+rename, and fold 9 never wrote a partial file because nothing is written
until the fold succeeds. Resume is automatic — re-run the same command and
folds with an existing checkpoint are skipped.
"""

import hashlib
import os
import pickle

CHECKPOINT_DIR = ".fold_checkpoints"


def fold_key(script_name: str, config: dict) -> str:
    """SHA1 hash of script_name + sorted config items, so key order never matters
    and any changed setting (tickers, hyperparams, date-window params) produces a
    different key rather than reusing a checkpoint built under different settings."""
    parts = [script_name] + [f"{k}={v}" for k, v in sorted(config.items())]
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    return digest


def checkpoint_path(script_name: str, config: dict, fold_index: int) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_key = fold_key(script_name, config)
    return os.path.join(CHECKPOINT_DIR, f"{script_name}_{run_key}_fold{fold_index:02d}.pkl")


def load_fold_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_fold_checkpoint(path, obj) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def resolve_fold_result(script_name: str, config: dict, fold_index: int, compute_fn):
    """Loads fold_index's checkpoint if present; otherwise calls compute_fn() to
    produce the result, saves it, and returns it. compute_fn is only invoked on
    a cache miss, so re-running a crashed multi-fold script skips every fold
    that already completed."""
    path = checkpoint_path(script_name, config, fold_index)
    cached = load_fold_checkpoint(path)
    if cached is not None:
        return cached, True

    result = compute_fn()
    save_fold_checkpoint(path, result)
    return result, False
