import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor


def detect_dbscan(series, eps=0.5, min_samples=5):
    """
    Detects outliers using DBSCAN clustering (Causal).
    Identifies points in low-density regions using causal expanding normalization.
    """
    s_orig = series.ffill().bfill()
    exp_mean = s_orig.expanding().mean()
    exp_std = s_orig.expanding().std().fillna(1.0)
    
    s_norm = ((s_orig - exp_mean) / (exp_std + 1e-9)).values.reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(s_norm)
    return pd.Series(db.labels_ == -1, index=series.index)


def detect_lof(series, n_neighbors=20, contamination=0.01):
    """
    Detects outliers using Local Outlier Factor.
    Compares local density of a point to its neighbors.
    """
    s = series.ffill().bfill().values.reshape(-1, 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(s)
    return pd.Series(y_pred == -1, index=series.index)


def detect_window_anomaly(series, window=20, sigma=3):
    """
    Window-based anomaly detection using rolling Z-score (Causal).
    Calculates statistical deviation within a trailing moving window.
    """
    rolling_mean = series.rolling(window=window, center=False).mean()
    rolling_std = series.rolling(window=window, center=False).std()
    z_score = (series - rolling_mean) / (rolling_std + 1e-9)
    return z_score.abs() > sigma


def detect_abnormal_sequence(series, window=50, threshold=2.0):
    """
    Detects abnormal sequences based on rolling variance spikes (Causal).
    Identifies periods of unusual volatility using trailing metrics.
    """
    rolling_var = series.rolling(window=window, center=False).var()
    expanding_var = series.expanding().var().fillna(1.0)
    return rolling_var > (threshold * expanding_var)
