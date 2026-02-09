import numpy as np
import pandas as pd


def smooth_kalman(series, r=0.1, q=0.01):
    """
    Simple 1D Kalman Filter.
    R: Measurement variance (uncertainty in data)
    Q: Process variance (uncertainty in model)
    """
    s = series.ffill().bfill().values
    n = len(s)
    x_hat = np.zeros(n)
    p = np.zeros(n)

    x_hat[0] = s[0]
    p[0] = 1.0

    for k in range(1, n):
        # Prediction
        x_hat_minus = x_hat[k - 1]
        p_minus = p[k - 1] + q

        # Update
        k_gain = p_minus / (p_minus + r)
        x_hat[k] = x_hat_minus + k_gain * (s[k] - x_hat_minus)
        p[k] = (1 - k_gain) * p_minus

    return pd.Series(x_hat, index=series.index)


def smooth_wiener(series, mysize=5):
    """
    Causal Wiener filter for noise reduction.
    Uses trailing windows to estimate local mean and variance, and an
    expanding window to estimate noise variance.
    """
    s = series.ffill().bfill()
    # Use trailing statistics instead of centered ones
    rolling = s.rolling(window=mysize, min_periods=1)
    local_mean = rolling.mean()
    local_var = rolling.var(ddof=0)
    
    # Estimate noise via expanding average of local variance (causal)
    noise_est = local_var.expanding().mean()
    
    # G(t) = max(0, var(t) - noise) / var(t)
    denom = local_var.replace(0.0, np.nan)
    gain = (local_var - noise_est).clip(lower=0.0) / denom
    
    # Filtered output: mean + gain * (obs - mean)
    res = local_mean + gain.fillna(0.0) * (s - local_mean)
    return res.fillna(local_mean).astype(float)
