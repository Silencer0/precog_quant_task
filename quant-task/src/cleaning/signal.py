import numpy as np
import pandas as pd

try:
    from scipy.fft import fft, ifft
except ModuleNotFoundError:  # optional dependency
    from numpy.fft import fft, ifft


def smooth_spectral(series, threshold=0.1, window=None):
    """
    Causal Spectral (Low-pass) Smoothing using a Butterworth Filter.
    This replaces the rolling FFT with a much faster recursive filter
    that achieves the same goal without look-ahead bias.
    """
    from scipy.signal import butter, lfilter
    s = series.ffill().bfill().values
    
    # Butterworth low-pass: threshold acts as the normalized cutoff frequency (0 to 1)
    # 0.1 means 10% of the Nyquist frequency.
    b, a = butter(4, threshold, btype='low')
    res = lfilter(b, a, s)
    
    return pd.Series(res, index=series.index)


def lattice_filter(series, k_coeffs=None):
    """
    Levinson-Durbin based Lattice Filter (Causal).
    Uses pd.Series.shift instead of np.roll to prevent circular leakage.
    """
    s = series.ffill().bfill()
    if k_coeffs is None:
        k_coeffs = [0.1, 0.05, 0.02]

    f = s.copy()
    b = s.copy()

    for k in k_coeffs:
        f_prev = f.copy()
        # Use shift(1) instead of np.roll to ensure causality and no wrap-around
        b_shifted = b.shift(1).fillna(0)
        f = f_prev + k * b_shifted
        b = b_shifted + k * f_prev

    return f


def adaptive_lms(series, mu=0.0001, n_taps=5):
    """Least Mean Squares (LMS) Adaptive Filter (Causal)."""
    s_orig = series.ffill().bfill()
    # Causal standardization using an expanding window to prevent lookahead bias
    exp_mean = s_orig.expanding().mean()
    exp_std = s_orig.expanding().std().replace(0.0, 1.0).fillna(1.0)
    s = ((s_orig - exp_mean) / (exp_std + 1e-9)).values

    n = len(s)
    w = np.zeros(n_taps)
    y = np.zeros(n)

    for i in range(n_taps, n):
        x = s[i - n_taps : i][::-1]
        y[i] = np.dot(w, x)
        error = s[i] - y[i]
        w = w + 2 * mu * error * x

    # Unstandardize using original causal parameters
    res = y * exp_std.values + exp_mean.values
    return pd.Series(res, index=series.index)
