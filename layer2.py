import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def factor_decompose(returns: pd.DataFrame, factors: pd.DataFrame) -> dict:
    """
    returns: DataFrame of asset returns, one column per asset
    factors: DataFrame of factor returns (market, size, value, momentum, etc.)
    """
    aligned = factors.join(returns, how="inner", lsuffix="_factor", rsuffix="_asset").dropna()
    x = aligned[factors.columns]
    y = aligned[returns.columns]

    model = LinearRegression()
    model.fit(x, y)

    alpha = np.atleast_1d(model.intercept_)          # Return unexplained by factors = pure edge
    betas = np.atleast_2d(model.coef_)                # Factor exposures, one row per asset
    preds = model.predict(x)

    results = {}
    for i, asset in enumerate(returns.columns):
        asset_alpha = alpha[i]
        asset_r2 = r2_score(y.iloc[:, i], preds[:, i]) if preds.ndim > 1 else model.score(x, y)
        results[asset] = {
            "alpha": asset_alpha,    # This is what you're hunting — is alpha > 0?
            "betas": dict(zip(factors.columns, betas[i])),
            "r_squared": asset_r2,
            "edge_quality": "strong" if abs(asset_alpha) > 0.001 and asset_r2 > 0.3 else "weak"
        }

    return results

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
