import numpy as np
import pandas as pd
# Removed unused import of scipy.stats to avoid import errors when scipy is unavailable

def detect_signal(prices: pd.Series, lookback: int = 20) -> dict:
    returns = prices.pct_change().dropna()
    
    # Autocorrelation — is today's return predictive of tomorrow's?
    acf_1 = returns.autocorr(lag=1)
    
    # Z-score deviation from rolling mean
    rolling_mean = returns.rolling(lookback).mean()
    rolling_std  = returns.rolling(lookback).std()
    z_score      = (returns - rolling_mean) / rolling_std
    
    # Hurst exponent — is this series trending or mean-reverting?
    hurst = compute_hurst(prices.values)
    
    return {
        "autocorrelation": acf_1,
        "z_score_latest": z_score.iloc[-1],
        "hurst_exponent": hurst,   # <0.5 = mean-reverting, >0.5 = trending
        "regime": "trending" if hurst > 0.55 else "mean_reverting"
    }

def compute_hurst(ts):
    lags = range(2, 20)
    tau  = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    m    = np.polyfit(np.log(lags), np.log(tau), 1)
    return m[0] * 2.0