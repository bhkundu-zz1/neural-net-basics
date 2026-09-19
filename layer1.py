import numpy as np
import pandas as pd
# Removed unused import of scipy.stats to avoid import errors when scipy is unavailable

def detect_signal(prices: pd.Series, lookback: int = 20) -> dict:
    returns = prices.pct_change().dropna()

    # Autocorrelation — is today's return predictive of tomorrow's? Undefined
    # for a flat-priced instrument (e.g. a money market fund pinned at
    # $1.00): zero-variance returns make the correlation's denominator zero.
    # pandas returns NaN correctly but warns noisily doing so — check first
    # and skip the call entirely rather than suppress the warning.
    acf_1 = returns.autocorr(lag=1) if returns.std() > 0 else np.nan

    # Z-score deviation from rolling mean
    rolling_mean = returns.rolling(lookback).mean()
    rolling_std  = returns.rolling(lookback).std()
    z_score      = (returns - rolling_mean) / rolling_std

    # Hurst exponent — is this series trending or mean-reverting? Undefined
    # for a series with (near-)zero price variance at every lag.
    hurst = compute_hurst(prices.values)

    if np.isnan(hurst):
        regime = "undefined"
    else:
        regime = "trending" if hurst > 0.55 else "mean_reverting"

    return {
        "autocorrelation": acf_1,
        "z_score_latest": z_score.iloc[-1],
        "hurst_exponent": hurst,   # <0.5 = mean-reverting, >0.5 = trending, NaN = undefined
        "regime": regime
    }

def compute_hurst(ts):
    lags = range(2, 20)
    tau = np.array([np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags])

    # A flat/near-constant price series (e.g. a money market fund pinned at
    # $1.00) has ~zero variance at every lag, making tau all zeros — log(0)
    # is undefined and the resulting fit is meaningless. Report NaN instead
    # of silently fitting garbage through -inf inputs.
    if np.any(tau <= 0):
        return np.nan

    m = np.polyfit(np.log(lags), np.log(np.sqrt(tau)), 1)
    return m[0] * 2.0