"""
Trains QuantEdgeNet (layer4) using CROSS-SECTIONAL ranking labels instead of
absolute-direction labels.

Why: prior versions (train_layer4.py) predicted each stock's own forward
direction independently. Across raw-return and risk-adjusted labeling, that
approach showed either an illusory lift (driven by which stocks happened to
trend upward over the sample period) or near-zero real skill once that
confound was removed. Cross-sectional ranking asks a different, often more
learnable question: "will this stock do better or worse than its peers over
the next N days," which factors out common market-wide moves (a "rising
tide lifts all boats" day no longer looks like universal skill) and asks the
model to find genuine relative-value information instead.

Labeling rule (cross-sectional tercile ranking):
  For each date t, compute every ticker's forward N-day return. Rank all
  tickers that have a valid return on that date. The top third (by rank)
  are labeled long, the bottom third short, the middle third flat. This
  label depends on ALL tickers' returns on date t, not just one ticker's
  own history — a fundamentally different question than before.

No look-ahead: features for (ticker, date=t) still use only data up to and
including t. The cross-sectional ranking itself necessarily uses forward
data (by construction, like all the labels in this pipeline) — this remains
a label, not a feature, so it's not used at prediction time.

Split: chronological by DATE (not by ticker), since the label for every
ticker on a given date depends on that date's whole cross-section. Splitting
by ticker would let, e.g., NVDA's Tuesday sit in train while AAPL's Tuesday
(which shares the same ranking computation) sits in test — a subtle leak.

Usage:
    python train_layer4_xsection.py --tickers NVDA,ANET,INTC,M,AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,XOM,CVX,JNJ,PFE,KO,PEP,WMT,HD,DIS,BA,CAT,GE,T,VZ \
        --lookback-days 10y --forward-days 5
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

FACTOR_WINDOW = 252
MIN_FACTOR_WINDOW = 20
FEATURE_LOOKBACK = 20
LONG, SHORT, FLAT = 0, 1, 2


def fetch_universe(tickers: list[str], lookback: str):
    all_symbols = list(dict.fromkeys([*tickers, "SPY", "IWM", "IWD", "MTUM"]))
    data = yf.download(all_symbols, period=lookback, auto_adjust=True, progress=False)
    close = data["Close"].dropna()
    volume = data["Volume"]

    factor_returns = close[["SPY", "IWM", "IWD", "MTUM"]].pct_change().dropna()
    factor_returns.columns = ["market", "size", "value", "momentum"]

    ticker_prices = close[tickers]
    ticker_volume = volume[tickers].reindex(ticker_prices.index)

    common_index = ticker_prices.index.intersection(factor_returns.index)
    return ticker_prices.loc[common_index], ticker_volume.loc[common_index], factor_returns.loc[common_index]


def compute_cross_sectional_labels(prices: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    """
    For each date t and ticker, computes the forward N-day return and its
    rank-percentile among all tickers with a valid return on that date.
    Returns a DataFrame indexed like prices, columns=tickers, values=rank
    percentile in [0, 1] (NaN where forward data is unavailable, e.g. the
    last `forward_days` dates).
    """
    fwd_returns = prices.shift(-forward_days) / prices - 1.0
    # rank across columns (tickers) per row (date); pct=True gives percentile in (0, 1]
    rank_pct = fwd_returns.rank(axis=1, pct=True)
    return fwd_returns, rank_pct


def label_from_rank(rank_pct: float) -> int:
    if rank_pct >= 2 / 3:
        return LONG
    if rank_pct <= 1 / 3:
        return SHORT
    return FLAT


def build_xsection_dataset(tickers, prices, volume, factor_returns, forward_days):
    """
    Walk forward date by date (not ticker by ticker). At each date t with
    enough trailing history and enough forward dates remaining:
      - compute the cross-sectional rank label for every ticker using
        forward returns from t to t+forward_days
      - build each ticker's feature vector using only data up to t
    Returns (features, labels, dates, tickers_per_sample, forward_returns, entry_prices).
    """
    fwd_returns, rank_pct = compute_cross_sectional_labels(prices, forward_days)

    min_start = FACTOR_WINDOW
    max_end = len(prices) - forward_days

    features_list, labels_list, dates_list, ticker_list = [], [], [], []
    fwd_return_list, entry_price_list = [], []

    print(f"Walking forward over {max_end - min_start} dates x {len(tickers)} tickers "
          f"(~{(max_end - min_start) * len(tickers)} feature builds — this will take a while)...")

    for i in range(min_start, max_end):
        t_date = prices.index[i]
        window_factors = factor_returns.iloc[i - FACTOR_WINDOW : i + 1]

        for ticker in tickers:
            rank = rank_pct.loc[t_date, ticker]
            if np.isnan(rank):
                continue

            ticker_prices_full = prices[ticker]
            price_slice = ticker_prices_full.iloc[: i + 1]
            if len(price_slice) < max(FEATURE_LOOKBACK + 1, FACTOR_WINDOW + 1):
                continue

            window_returns = price_slice.pct_change().iloc[-(FACTOR_WINDOW + 1):].to_frame(name=ticker)
            aligned_factors = window_factors.reindex(window_returns.index).dropna()
            aligned_returns = window_returns.reindex(aligned_factors.index)
            if len(aligned_returns) < MIN_FACTOR_WINDOW:
                continue

            try:
                factor_result = factor_decompose(aligned_returns, aligned_factors)[ticker]
            except Exception:
                continue

            signal = detect_signal(price_slice)
            if np.isnan(signal["z_score_latest"]) or np.isnan(signal["hurst_exponent"]):
                continue

            current_state = 0 if signal["regime"] == "trending" else 2
            regime_probs = get_next_regime(current_state, steps=5)

            volume_slice = volume[ticker].iloc[: i + 1]
            try:
                feature_vector = build_feature_vector(price_slice, volume_slice, factor_result["betas"], regime_probs)
            except Exception:
                continue

            if np.isnan(feature_vector).any():
                continue

            label = label_from_rank(rank)

            features_list.append(feature_vector)
            labels_list.append(label)
            dates_list.append(t_date)
            ticker_list.append(ticker)
            fwd_return_list.append(fwd_returns.loc[t_date, ticker])
            entry_price_list.append(ticker_prices_full.loc[t_date])

        if (i - min_start) % 100 == 0:
            print(f"  processed date {i - min_start}/{max_end - min_start} ({t_date.date()}), "
                  f"{len(labels_list)} samples so far")

    return (
        np.stack(features_list), np.array(labels_list), dates_list, ticker_list,
        np.array(fwd_return_list), np.array(entry_price_list),
    )


def chronological_split_by_date(dates: list, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Splits sample indices by DATE, not by row order — necessary because
    multiple tickers share the same date and must land in the same split.
    """
    unique_dates = sorted(set(dates))
    n = len(unique_dates)
    train_end_date = unique_dates[int(n * train_frac)]
    val_end_date = unique_dates[int(n * (train_frac + val_frac))]

    dates_arr = np.array(dates)
    train_mask = dates_arr < train_end_date
    val_mask = (dates_arr >= train_end_date) & (dates_arr < val_end_date)
    test_mask = dates_arr >= val_end_date
    return train_mask, val_mask, test_mask


def train(x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw,
          epochs, lr, batch_size, hidden_sizes, dropout, weight_decay, patience):
    x_train = torch.tensor(x_train_raw, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.long)
    x_val = torch.tensor(x_val_raw, dtype=torch.float32)
    y_val = torch.tensor(y_val_raw, dtype=torch.long)
    x_test = torch.tensor(x_test_raw, dtype=torch.float32)
    y_test = torch.tensor(y_test_raw, dtype=torch.long)

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    class_counts = torch.bincount(y_train, minlength=3).float()
    class_weights = class_counts.sum() / (3 * class_counts.clamp_min(1))
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
    print("(Cross-sectional labels are roughly balanced by construction (~1/3 each),")
    print(" so unlike prior versions, a majority-class baseline here is ~33%, not skewed.)")

    return net, mean, std


def parse_args():
    parser = argparse.ArgumentParser(description="Train QuantEdgeNet using cross-sectional ranking labels.")
    parser.add_argument(
        "--tickers",
        default="NVDA,ANET,INTC,M,AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,XOM,CVX,JNJ,PFE,KO,PEP,WMT,HD,DIS,BA,CAT,GE,T,VZ",
        help="Comma-separated universe of tickers to rank against each other",
    )
    parser.add_argument("--lookback-days", default="10y")
    parser.add_argument("--forward-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-sizes", default="32,16")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", default="layer4_weights_xsection.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"Fetching {args.lookback_days} of data for {len(tickers)} tickers: {', '.join(tickers)}")
    prices, volume, factor_returns = fetch_universe(tickers, args.lookback_days)
    print(f"Aligned universe: {len(prices)} common trading days across all {len(tickers)} tickers.")

    features, labels, dates, sample_tickers, fwd_returns, entry_prices = build_xsection_dataset(
        tickers, prices, volume, factor_returns, args.forward_days
    )
    print(f"\nBuilt {len(labels)} total (ticker, date) samples.")

    train_mask, val_mask, test_mask = chronological_split_by_date(dates)
    x_train, y_train = features[train_mask], labels[train_mask]
    x_val, y_val = features[val_mask], labels[val_mask]
    x_test, y_test = features[test_mask], labels[test_mask]
    print(f"Split: {len(y_train)} train / {len(y_val)} val / {len(y_test)} test (split by date, not row order)")

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
            "labeling": "cross_sectional_tercile",
        },
        args.output,
    )
    print(f"\nSaved trained weights to {args.output}")


if __name__ == "__main__":
    main()
