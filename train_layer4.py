"""
Trains QuantEdgeNet (layer4) on historical data, pooled across one or more tickers.

Labeling rule (risk-adjusted forward-return threshold):
  Look FORWARD_DAYS trading days ahead. Divide the cumulative return over
  that window by the ticker's trailing 20-day realized volatility (as of
  day t) to get a risk-adjusted move size. If that risk-adjusted return is
  > +UP_THRESHOLD, label = long. If < -DOWN_THRESHOLD, label = short.
  Otherwise label = flat. UP_THRESHOLD/DOWN_THRESHOLD are now expressed in
  "standard deviations of recent volatility," not raw percent — e.g. 0.5
  means "moved more than half a recent-vol-unit."

  Why risk-adjusted: a fixed raw-percent threshold (e.g. 2%) is a much
  easier bar for a high-volatility stock (NVDA) to clear than a low-
  volatility one (M), so the model can pick up "which stocks are just more
  volatile" rather than a real, transferable timing signal — this was the
  likely explanation for why "long" predictions (59.9% accuracy) were
  meaningfully better than "short" (43.2%) in the raw-return version: several
  of the pooled tickers trended upward over the sample period. Risk-
  adjusting the label removes that confound.

Feature construction at each day t uses ONLY data available up to and
including day t (rolling factor betas computed on a trailing window, regime
state seeded from layer1's read as of day t) — labels come strictly from
days after t. This avoids look-ahead bias in the features; it does NOT
make the labels or the resulting model validated for live trading.

Multi-ticker pooling: samples from each ticker are built independently
(each ticker gets its own rolling factor regression and its own
chronological train/val/test split), then the train/val/test sets are
concatenated across tickers. Splitting per-ticker before pooling avoids a
ticker with a shorter history leaking entirely into one split.

Adaptive factor window: tickers with less than FACTOR_WINDOW days of
history (e.g. a recent IPO/SPAC) use a shorter trailing window, down to
MIN_FACTOR_WINDOW. Their factor betas are noisier as a result — this is
reported per ticker so it isn't silently hidden in the pooled numbers.

Usage:
    python train_layer4.py --tickers NVDA,ANET,INTC,M,SPCX --lookback-days 10y \
        --forward-days 5 --up-threshold 0.6 --down-threshold 0.6
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf

from layer1 import detect_signal
from layer2 import factor_decompose
from layer3 import get_next_regime
from layer4 import QuantEdgeNet, build_feature_vector

FACTOR_WINDOW = 252     # preferred trailing days for rolling factor betas
MIN_FACTOR_WINDOW = 20  # floor for tickers with limited history
FEATURE_LOOKBACK = 20   # matches build_feature_vector's tail(20) usage in layer1/layer4
LONG, SHORT, FLAT = 0, 1, 2


def fetch_prices_and_factors(tickers: list[str], lookback: str) -> tuple[dict, dict, dict]:
    """
    Downloads prices/volume for each ticker and shared factor proxies (SPY/IWM/IWD/MTUM).
    Each ticker's series is aligned only against dates where BOTH the ticker and the
    factor proxies have data — tickers with a later start date (e.g. a recent listing)
    naturally get a shorter, more recent history rather than being dropped or padded.
    Returns per-ticker dicts: {ticker: pd.Series prices}, {ticker: pd.Series volume},
    {ticker: pd.DataFrame factor_returns}.
    """
    all_symbols = list(dict.fromkeys([*tickers, "SPY", "IWM", "IWD", "MTUM"]))
    data = yf.download(all_symbols, period=lookback, auto_adjust=True, progress=False)
    close = data["Close"]
    volume = data["Volume"]

    factor_prices_full = close[["SPY", "IWM", "IWD", "MTUM"]].dropna()
    factor_returns_full = factor_prices_full.pct_change().dropna()
    factor_returns_full.columns = ["market", "size", "value", "momentum"]

    prices_by_ticker, volume_by_ticker, factors_by_ticker = {}, {}, {}
    for ticker in tickers:
        ticker_close = close[ticker].dropna()
        common_index = ticker_close.index.intersection(factor_returns_full.index)
        if len(common_index) < MIN_FACTOR_WINDOW + FEATURE_LOOKBACK:
            print(f"  WARNING: {ticker} has only {len(common_index)} usable overlapping days — skipping.")
            continue
        prices_by_ticker[ticker] = ticker_close.loc[ticker_close.index.union(common_index)].sort_index()
        volume_by_ticker[ticker] = volume[ticker].reindex(prices_by_ticker[ticker].index)
        factors_by_ticker[ticker] = factor_returns_full.loc[common_index]

    return prices_by_ticker, volume_by_ticker, factors_by_ticker


def label_from_forward_return(risk_adjusted_return: float, up_threshold: float, down_threshold: float) -> int:
    if risk_adjusted_return > up_threshold:
        return LONG
    if risk_adjusted_return < -down_threshold:
        return SHORT
    return FLAT


def build_dataset_for_ticker(
    ticker: str,
    prices: pd.Series,
    volume: pd.Series,
    factor_returns: pd.DataFrame,
    forward_days: int,
    up_threshold: float,
    down_threshold: float,
    label_mode: str = "risk_adjusted",
):
    """
    Walk forward day by day for a single ticker. At each valid day t:
      - features built from data up to and including t (rolling factor
        window ending at t, sized adaptively to available history; regime
        seeded from layer1's read as of t)
      - label built from the return over (t, t+forward_days]
    Returns (features, labels, dates, factor_window, forward_returns, entry_prices).
    forward_returns/entry_prices are the raw (t, t+forward_days] return and the
    price at t for each sample — not used in training, but needed for backtesting
    actual P&L rather than just classification accuracy.
    """
    returns = prices.pct_change().dropna().to_frame(name=ticker)
    common_index = returns.index.intersection(factor_returns.index)
    returns = returns.loc[common_index]
    factor_returns = factor_returns.loc[common_index]

    # Reserve at least MIN_WALK_STEPS days after the factor window so the walk-forward
    # loop can actually produce more than a token number of samples.
    MIN_WALK_STEPS = 10
    usable_days = len(common_index) - forward_days
    factor_window = min(FACTOR_WINDOW, usable_days - MIN_WALK_STEPS)
    if factor_window < MIN_FACTOR_WINDOW:
        return np.empty((0,)), np.empty((0,)), [], 0, np.empty((0,)), np.empty((0,))
    if factor_window < FACTOR_WINDOW:
        print(f"  {ticker}: only {len(common_index)} days available — using a "
              f"{factor_window}-day factor window instead of the standard {FACTOR_WINDOW} "
              f"(betas will be noisier; ~{usable_days - factor_window} samples possible).")

    features_list, labels_list, dates_list, fwd_return_list, entry_price_list = [], [], [], [], []
    min_start = factor_window
    max_end = len(common_index) - forward_days

    for i in range(min_start, max_end):
        t_date = common_index[i]

        window_returns = returns.iloc[i - factor_window : i + 1]
        window_factors = factor_returns.iloc[i - factor_window : i + 1]

        try:
            factor_result = factor_decompose(window_returns, window_factors)[ticker]
        except Exception:
            continue  # skip windows where the regression is degenerate

        price_slice = prices.loc[:t_date]
        volume_slice = volume.loc[:t_date]
        if len(price_slice) < FEATURE_LOOKBACK + 1:
            continue

        signal = detect_signal(price_slice)
        if np.isnan(signal["z_score_latest"]) or np.isnan(signal["hurst_exponent"]):
            continue

        current_state = 0 if signal["regime"] == "trending" else 2
        regime_probs = get_next_regime(current_state, steps=5)

        try:
            feature_vector = build_feature_vector(price_slice, volume_slice, factor_result["betas"], regime_probs)
        except Exception:
            continue

        if np.isnan(feature_vector).any():
            continue

        fwd_return = prices.iloc[i + forward_days] / prices.iloc[i] - 1.0

        if label_mode == "risk_adjusted":
            # Risk-adjust using trailing realized vol as of day t (no look-ahead: uses
            # only returns up to and including t, same window as build_feature_vector).
            trailing_vol = returns[ticker].loc[:t_date].tail(FEATURE_LOOKBACK).std()
            if not trailing_vol or np.isnan(trailing_vol) or trailing_vol == 0:
                continue
            label_input = fwd_return / (trailing_vol * np.sqrt(forward_days))
        elif label_mode == "raw":
            label_input = fwd_return
        else:
            raise ValueError(f"Unknown label_mode: {label_mode!r} (expected 'risk_adjusted' or 'raw')")

        label = label_from_forward_return(label_input, up_threshold, down_threshold)

        features_list.append(feature_vector)
        labels_list.append(label)
        dates_list.append(t_date)
        fwd_return_list.append(fwd_return)
        entry_price_list.append(prices.iloc[i])

    if not features_list:
        return np.empty((0,)), np.empty((0,)), [], factor_window, np.empty((0,)), np.empty((0,))

    return (
        np.stack(features_list), np.array(labels_list), dates_list, factor_window,
        np.array(fwd_return_list), np.array(entry_price_list),
    )


def chronological_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def build_pooled_dataset(
    tickers: list[str],
    prices_by_ticker: dict,
    volume_by_ticker: dict,
    factors_by_ticker: dict,
    forward_days: int,
    up_threshold: float,
    down_threshold: float,
    label_mode: str = "risk_adjusted",
):
    """
    Builds a dataset per ticker, splits each chronologically, then pools the
    splits across tickers. Splitting before pooling ensures a ticker with a
    shorter or more recent history doesn't leak entirely into one split.
    """
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    summary = []

    for ticker in tickers:
        if ticker not in prices_by_ticker:
            continue
        features, labels, dates, factor_window, _fwd_returns, _entry_prices = build_dataset_for_ticker(
            ticker,
            prices_by_ticker[ticker],
            volume_by_ticker[ticker],
            factors_by_ticker[ticker],
            forward_days,
            up_threshold,
            down_threshold,
            label_mode,
        )
        if len(labels) == 0:
            print(f"  {ticker}: produced 0 samples — skipping.")
            continue

        if len(labels) < 20:
            print(f"  {ticker}: only {len(labels)} samples — too few for a meaningful "
                  f"train/val/test split, skipping this ticker entirely.")
            continue

        train_idx, val_idx, test_idx = chronological_split(len(labels))
        train_x.append(features[train_idx]); train_y.append(labels[train_idx])
        val_x.append(features[val_idx]); val_y.append(labels[val_idx])
        test_x.append(features[test_idx]); test_y.append(labels[test_idx])

        summary.append((ticker, len(labels), dates[0].date(), dates[-1].date(), factor_window))

    print("\nPer-ticker sample counts:")
    for ticker, n, start, end, factor_window in summary:
        print(f"  {ticker}: {n} samples ({start} to {end}), factor_window={factor_window}")

    return (
        np.concatenate(train_x), np.concatenate(train_y),
        np.concatenate(val_x), np.concatenate(val_y),
        np.concatenate(test_x), np.concatenate(test_y),
    )


def train(
    x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw,
    epochs: int,
    lr: float,
    batch_size: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    weight_decay: float,
    patience: int,
):
    x_train = torch.tensor(x_train_raw, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.long)
    x_val = torch.tensor(x_val_raw, dtype=torch.float32)
    y_val = torch.tensor(y_val_raw, dtype=torch.long)
    x_test = torch.tensor(x_test_raw, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.long)

    # Normalize features using train-set statistics only (no look-ahead into val/test)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    class_counts = torch.bincount(y_train, minlength=3).float()
    class_weights = (class_counts.sum() / (3 * class_counts.clamp_min(1)))
    print(f"\nTrain class counts (long/short/flat): {class_counts.tolist()}")
    print(f"Class weights applied to loss: {class_weights.tolist()}")

    net = QuantEdgeNet(input_features=x_train.shape[1], hidden_sizes=hidden_sizes, dropout=dropout)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        net.train()
        permutation = torch.randperm(len(x_train))
        epoch_loss = 0.0
        for start in range(0, len(x_train), batch_size):
            idx = permutation[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            # QuantEdgeNet.forward applies softmax; use the pre-softmax logits for CE loss stability
            logits = net.layers(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= len(x_train)

        net.eval()
        with torch.no_grad():
            val_logits = net.layers(x_val)
            val_loss = criterion(val_logits, y_val).item()
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        test_logits = net.layers(x_test)
        test_loss = criterion(test_logits, y_test).item()
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()
        test_preds = test_logits.argmax(dim=1)

    print(f"\nBest val_loss={best_val_loss:.4f}")
    print(f"Test loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")
    print(f"Test label distribution:      {torch.bincount(y_test, minlength=3).tolist()}")
    print(f"Test prediction distribution: {torch.bincount(test_preds, minlength=3).tolist()}")
    print("(Note: with 3 classes, a model that always predicts the majority class can still")
    print(" score well on accuracy — compare the distributions above, not just accuracy.)")

    return net, mean, std


def parse_args():
    parser = argparse.ArgumentParser(description="Train QuantEdgeNet (layer4) on historical data.")
    parser.add_argument("--tickers", default="NVDA", help="Comma-separated tickers, e.g. NVDA,ANET,INTC,M,SPCX")
    parser.add_argument("--lookback-days", default="10y", help="yfinance period string, e.g. 5y, 10y")
    parser.add_argument("--forward-days", type=int, default=5, help="Forward window (trading days) for the label")
    parser.add_argument(
        "--label-mode", choices=["risk_adjusted", "raw"], default="risk_adjusted",
        help="'risk_adjusted': forward return / trailing realized vol, thresholds in vol-units. "
        "'raw': forward return directly, thresholds in raw percent (e.g. 0.02 = 2%%).",
    )
    parser.add_argument(
        "--up-threshold", type=float, default=0.6,
        help="Threshold above which label=long. Units depend on --label-mode: vol-units for "
        "risk_adjusted (0.6 default gives balanced classes), raw percent for raw (e.g. 0.02).",
    )
    parser.add_argument(
        "--down-threshold", type=float, default=0.6,
        help="Threshold below which (negated) label=short. Same units as --up-threshold.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs (early stopping usually ends sooner)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--hidden-sizes",
        default="32,16",
        help="Comma-separated hidden layer sizes, e.g. 32,16 (smaller than layer4's original "
        "128,64 -- with a few thousand samples the bigger network overfits almost immediately)",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs")
    parser.add_argument("--output", default="layer4_weights.pt", help="Path to save trained weights + norm stats")
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"Fetching {args.lookback_days} of data for: {', '.join(tickers)}...")
    prices_by_ticker, volume_by_ticker, factors_by_ticker = fetch_prices_and_factors(tickers, args.lookback_days)

    print("\nBuilding walk-forward feature/label dataset per ticker (this recomputes rolling "
          "factor regressions and signal detection at every step, so it can take a while)...")
    x_train, y_train, x_val, y_val, x_test, y_test = build_pooled_dataset(
        tickers, prices_by_ticker, volume_by_ticker, factors_by_ticker,
        args.forward_days, args.up_threshold, args.down_threshold, args.label_mode,
    )
    total = len(y_train) + len(y_val) + len(y_test)
    print(f"\nPooled dataset: {total} samples ({len(y_train)} train / {len(y_val)} val / {len(y_test)} test)")

    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(","))
    net, mean, std = train(
        x_train, y_train, x_val, y_val, x_test, y_test,
        args.epochs, args.lr, args.batch_size, hidden_sizes, args.dropout, args.weight_decay, args.patience,
    )

    torch.save(
        {
            "state_dict": net.state_dict(),
            "input_features": x_train.shape[1],
            "hidden_sizes": hidden_sizes,
            "dropout": args.dropout,
            "feature_mean": mean,
            "feature_std": std,
            "tickers": tickers,
            "forward_days": args.forward_days,
            "label_mode": args.label_mode,
            "up_threshold": args.up_threshold,
            "down_threshold": args.down_threshold,
        },
        args.output,
    )
    print(f"\nSaved trained weights + normalization stats to {args.output}")
    print("Load these in run_pipeline.py before calling QuantEdgeNet on live features —")
    print("otherwise you're back to random weights.")


if __name__ == "__main__":
    main()
