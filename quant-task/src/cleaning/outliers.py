import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_3sigma(series):
    """Detects outliers using 3-sigma rule (Causal)."""
    exp_mean = series.expanding().mean()
    exp_std = series.expanding().std().fillna(1.0)
    return (series - exp_mean).abs() > (3 * exp_std)


def detect_iqr(series):
    """Detects outliers using Interquartile Range (Causal)."""
    # Use expanding quantiles for causality
    q1 = series.expanding().quantile(0.25)
    q3 = series.expanding().quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)


def detect_isolation_forest(series, contamination=0.01):
    """
    Detects outliers using Isolation Forest (Causal).
    Fits the model on an expanding window.
    """
    s_orig = series.ffill().bfill().values
    n = len(s_orig)
    res = np.zeros(n, dtype=bool)
    
    # Isolation Forest needs a minimum number of samples
    min_samples = 50
    for i in range(min_samples, n + 1):
        window_data = s_orig[:i].reshape(-1, 1)
        clf = IsolationForest(contamination=contamination, random_state=42)
        # We only predict the LAST point in the expanding set
        labels = clf.fit_predict(window_data)
        res[i-1] = (labels[-1] == -1)
        
    return pd.Series(res, index=series.index)


def detect_lof(series, n_neighbors=20, contamination=0.01):
    """
    Detects outliers using Local Outlier Factor (Causal).
    Fits the model on an expanding window.
    """
    s_orig = series.ffill().bfill().values
    n = len(s_orig)
    res = np.zeros(n, dtype=bool)
    
    min_samples = 40
    for i in range(min_samples, n + 1):
        window_data = s_orig[:i].reshape(-1, 1)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        # LOF is slightly different, fit_predict on the whole window
        labels = clf.fit_predict(window_data)
        res[i-1] = (labels[-1] == -1)
        
    return pd.Series(res, index=series.index)


def remove_outliers(df, detection_func, col="Close"):
    """Replaces outliers with NaN so they can be imputed later."""
    df_clean = df.copy()
    outliers = detection_func(df[col])
    df_clean.loc[outliers, col] = np.nan
    return df_clean
