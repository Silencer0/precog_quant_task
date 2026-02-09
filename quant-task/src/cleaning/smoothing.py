import pandas as pd
import numpy as np

try:
    from scipy.signal import savgol_filter
except ModuleNotFoundError:  # optional dependency
    savgol_filter = None


def smooth_sma(series, window=5):
    """Simple Moving Average (trailing)."""
    s = series.ffill().bfill()
    return s.rolling(window=window, center=False).mean().ffill().bfill()


def smooth_ema(series, span=5):
    """Exponential Moving Average."""
    s = series.ffill().bfill()
    return s.ewm(span=span).mean()


def smooth_wma(series, window=5):
    """Weighted Moving Average."""
    s = series.ffill().bfill()
    weights = np.arange(1, window + 1)
    return (
        s.rolling(window)
        .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        .ffill()
        .bfill()
    )


def smooth_savgol(series, window=11, polyorder=2):
    """
    Causal (Trailing) Savitzky-Golay Filter.
    Smoothes data by fitting a polynomial to a trailing window and taking the last point.
    Optimized via FIR convolution (lfilter).
    """
    from scipy.signal import savgol_coeffs, lfilter
    s = series.ffill().bfill().values
    if window % 2 == 0:
        window += 1

    # Causal coefficients (last point of the fit)
    coeffs = savgol_coeffs(window, polyorder, pos=window-1)
    
    # Convolution with lfilter is 1000x faster than rolling.apply
    res = lfilter(coeffs, 1.0, s)
    return pd.Series(res, index=series.index)
