import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def masked_mse(original_df, cleaning_func, col="Close", mask_fraction=0.1, seed=42):
    """
    Calculates MSE by masking a portion of valid data, applying cleaning_func,
    and comparing reconstruction to ground truth.
    """
    np.random.seed(seed)

    series = original_df[col].copy()
    valid_indices = series.index[series.notna()]

    if len(valid_indices) < 10:
        return np.nan

    mask_size = int(len(valid_indices) * mask_fraction)
    masked_indices = np.random.choice(valid_indices, size=mask_size, replace=False)

    corrupted_df = original_df.copy()
    corrupted_df.loc[masked_indices, col] = np.nan

    try:
        cleaned_df = cleaning_func(corrupted_df)
    except Exception as e:
        print(f"Error in cleaning function: {e}")
        return np.nan

    y_true = original_df.loc[masked_indices, col]
    y_pred = cleaned_df.loc[masked_indices, col]

    mask = y_pred.notna()
    if mask.sum() == 0:
        return np.inf

    return mean_squared_error(y_true[mask], y_pred[mask])


def calculate_advanced_metrics(
    original_df, cleaning_func, col="Close", mask_fraction=0.1, seed=42
):
    """
    Calculates RMSE, Lag, and Root Point Ratio (RPR) for a cleaning function.
    """
    np.random.seed(seed)
    series = original_df[col].copy()

    # 1. RMSE (Root Mean Squared Error)
    # We use the same masking logic as masked_mse
    valid_indices = series.index[series.notna()]
    if len(valid_indices) < 10:
        return {"RMSE": np.nan, "LAG": np.nan, "RPR": np.nan}

    mask_size = int(len(valid_indices) * mask_fraction)
    masked_indices = np.random.choice(valid_indices, size=mask_size, replace=False)

    corrupted_df = original_df.copy()
    corrupted_df.loc[masked_indices, col] = np.nan

    try:
        cleaned_df = cleaning_func(corrupted_df)
        y_pred_series = cleaned_df[col]
    except Exception as e:
        print(f"Error: {e}")
        return {"RMSE": np.inf, "LAG": np.inf, "RPR": np.inf}

    y_true = original_df.loc[masked_indices, col]
    y_pred = cleaned_df.loc[masked_indices, col]

    mask = y_pred.notna()
    if mask.sum() == 0:
        return {"RMSE": np.inf, "LAG": np.inf, "RPR": np.inf}

    mse = mean_squared_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mse)

    # 2. LAG (Time Delay)
    # We find the cross-correlation shift between original and smoothed signal
    # Only meaningful if we have a "clean" signal, but here we compare to original "noisy" signal
    # Ideally, LAG should be 0.
    # We'll use cross-correlation on valid data points
    clean_vals = y_pred_series.ffill().bfill().values
    orig_vals = series.ffill().bfill().values

    # Normalize
    clean_norm = (clean_vals - np.mean(clean_vals)) / np.std(clean_vals)
    orig_norm = (orig_vals - np.mean(orig_vals)) / np.std(orig_vals)

    correlation = np.correlate(clean_norm, orig_norm, mode="full")
    lags = np.arange(-len(clean_norm) + 1, len(clean_norm))
    lag = lags[np.argmax(correlation)]

    # 3. Root Point Ratio (RPR)
    # Measures "smoothness" by counting inflection points
    # Ratio = (Inflections in Cleaned) / (Inflections in Original)
    # Lower is smoother.
    def count_inflections(arr):
        diffs = np.diff(arr)
        # Signs of differences
        signs = np.sign(diffs)
        # Sign changes
        sign_changes = np.diff(signs)
        return np.count_nonzero(sign_changes)

    orig_inflections = count_inflections(orig_vals)
    clean_inflections = count_inflections(clean_vals)

    rpr = clean_inflections / (orig_inflections + 1e-9)

    return {
        "RMSE": rmse,
        "LAG": abs(lag),  # Magnitude of delay
        "RPR": rpr,
    }


def calculate_smoothness(series):
    """
    Measures smoothness as the standard deviation of the second difference.
    Lower is smoother.
    """
    return series.diff().diff().std()
