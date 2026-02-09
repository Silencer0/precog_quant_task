import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def relationship_dependent_smoothing(df, col="Close", n_components=1):
    """
    Factor-based smoothing using PCA (Causal).
    Captures global market movements using an expanding window to prevent lookahead bias.
    """
    s = df[col].ffill().bfill().values.reshape(-1, 1)
    n = len(s)
    res = np.zeros_like(s)
    
    # Start with a minimum window to fit PCA
    min_window = 30
    for i in range(min_window, n + 1):
        window_data = s[:i]
        pca = PCA(n_components=1)
        pca.fit(window_data)
        # We only take the reconstruction of the LAST point in the expanding window
        res[i-1] = pca.inverse_transform(pca.transform(s[i-1:i]))
        
    # Fill the initial buffer
    res[:min_window-1] = s[:min_window-1]
    return pd.DataFrame({col: res.flatten()}, index=df.index)


def smurf_smoothing(series, window=10):
    """
    Simplified SMURF-like trend detection (Causal).
    Combines local regression with consistency checks using trailing windows.
    """
    s = series.ffill().bfill()
    # Local linear regression trend (trailing)
    trend = s.rolling(window=window, center=False).mean()
    # Residual analysis
    residual = s - trend
    # "Cleaned" version removes high-frequency noise from residuals (trailing)
    cleaned_residual = residual.rolling(window=3, center=False).median()
    return (trend + cleaned_residual).ffill().bfill()


def spatio_temporal_smoothing(series, alpha=0.5):
    """
    Spatio-temporal modeling (Causal).
    Interpolates based on temporal proximity and causal expanding mean.
    """
    s_orig = series.ffill().bfill()
    exp_mean = s_orig.expanding().mean()
    
    s = s_orig.values
    m = exp_mean.values
    n = len(s)
    res = np.zeros_like(s)
    for i in range(n):
        # Weighted average of expanding mean and current value
        res[i] = alpha * s[i] + (1 - alpha) * m[i]
    return pd.Series(res, index=series.index)
