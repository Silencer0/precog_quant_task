import pandas as pd
import numpy as np

from typing import Any
from typing import cast

AutoReg: Any | None
ARIMA: Any | None

try:
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
except ModuleNotFoundError:  # optional dependency
    AutoReg = None
    ARIMA = None


def _require_statsmodels() -> None:
    if AutoReg is None or ARIMA is None:
        raise ModuleNotFoundError(
            "statsmodels is required for AR/ARMA smoothing. Install it or skip these filters."
        )


def smooth_ar(series, lags=5, stride=20):
    """
    Autoregressive Model smoothing (Causal).
    Fits the model on an expanding window and predicts the next value.
    Refits every `stride` steps for performance.
    """
    _require_statsmodels()
    autoreg = cast(Any, AutoReg)
    s = series.ffill().bfill()
    n = len(s)
    if n <= lags:
        return s

    s_np = s.to_numpy()
    res = np.zeros(n)
    res[:lags] = s_np[:lags]
    
    last_model = None
    for i in range(lags, n):
        # Refit every `stride` steps
        if i % stride == 0 or last_model is None:
            model = autoreg(s_np[:i], lags=lags).fit()
            last_model = model
            
        # Extract coefficients for the last p lags + intercept
        params = last_model.params # [intercept, L1, L2, ..., Lp]
        # Prediction: intercept + sum(param_j * s_{i-j})
        pred = params[0] + np.dot(params[1:], s_np[i-lags:i][::-1])
        res[i] = pred

    return pd.Series(res, index=series.index)


def smooth_arma(series, order=(2, 0, 2), stride=50):
    """
    ARMA Model smoothing (Causal).
    Fits the model on an expanding window and predicts the next value.
    Refits every `stride` steps for performance.
    """
    _require_statsmodels()
    arima = cast(Any, ARIMA)
    s = series.ffill().bfill()
    n = len(s)
    if n <= max(order):
        return s

    s_np = s.to_numpy()
    res = np.zeros(n)
    res[:max(order)] = s_np[:max(order)]
    
    model_fit = None
    for i in range(max(order), n):
        if i % stride == 0 or model_fit is None:
            try:
                model_fit = arima(s_np[:i], order=order).fit()
            except:
                pass
        
        if model_fit is not None:
            # Statsmodels ARIMA forecast(1) uses the last available data
            pred = model_fit.forecast(steps=1)[0]
            res[i] = pred
        else:
            res[i] = s_np[i]

    return pd.Series(res, index=series.index)
