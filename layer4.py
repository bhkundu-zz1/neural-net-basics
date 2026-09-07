import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from layer1 import detect_signal

class QuantEdgeNet(nn.Module):
    def __init__(self, input_features: int = 50, hidden_sizes: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        dims = [input_features, *hidden_sizes]
        blocks = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            blocks += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
        blocks.append(nn.Linear(dims[-1], 3))    # Output: [long, short, flat] probabilities
        self.layers = nn.Sequential(*blocks)
    
    def forward(self, x):
        return torch.softmax(self.layers(x), dim=-1)

def compute_rsi(prices: pd.Series, window: int = 14) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean().iloc[-1]
    avg_loss = loss.rolling(window).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal_window: int = 9) -> tuple[float, float]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line.iloc[-1], (macd_line.iloc[-1] - signal_line.iloc[-1])


def compute_obv(prices: pd.Series, volume: pd.Series) -> float:
    direction = np.sign(prices.diff().fillna(0))
    obv = (direction * volume).cumsum()
    # Normalize by trailing average volume so OBV is comparable across tickers/time
    avg_volume = volume.tail(20).mean()
    return obv.iloc[-1] / avg_volume if avg_volume else 0.0


# Training signal: combine factor scores + regime probs + price/volume technicals
def build_feature_vector(prices, volume, factors, regime_probs):
    """
    prices: pd.Series of asset prices
    volume: pd.Series of asset trading volume, same index as prices
    factors: flat dict of factor exposures for one asset, e.g. factor_decompose(...)[asset]["betas"]
    regime_probs: dict returned by layer3.get_next_regime
    """
    signal = detect_signal(prices)
    regime_prob_values = [v for v in regime_probs.values() if isinstance(v, (int, float, np.floating, np.integer))]

    returns = prices.pct_change()
    realized_vol_20 = returns.tail(20).std()
    return_5d = prices.iloc[-1] / prices.iloc[-6] - 1.0 if len(prices) > 5 else 0.0
    return_10d = prices.iloc[-1] / prices.iloc[-11] - 1.0 if len(prices) > 10 else 0.0
    rsi_14 = compute_rsi(prices)
    macd_line, macd_hist = compute_macd(prices)

    volume_aligned = volume.reindex(prices.index).ffill()
    volume_trend = volume_aligned.tail(5).mean() / volume_aligned.tail(20).mean() if volume_aligned.tail(20).mean() else 1.0
    obv_normalized = compute_obv(prices, volume_aligned)

    return np.concatenate([
        returns.tail(20).values,   # Last 20 returns
        list(factors.values()),                 # Factor exposures
        regime_prob_values,                     # Markov regime probs (numeric only — drops "dominant_regime" label)
        [signal["hurst_exponent"],
         signal["z_score_latest"],
         realized_vol_20,
         return_5d,
         return_10d,
         rsi_14,
         macd_line,
         macd_hist,
         volume_trend,
         obv_normalized]
    ])