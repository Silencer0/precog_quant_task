import pandas as pd
import numpy as np


def impute_mean(df, col="Close"):
    """Mean imputation."""
    df_clean = df.copy()
    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    return df_clean


def impute_interpolation(df, col="Close", method="linear"):
    """Linear or spline interpolation."""
    df_clean = df.copy()
    df_clean[col] = df_clean[col].interpolate(method=method)
    # Fill remaining at edges
    df_clean[col] = df_clean[col].ffill().bfill()
    return df_clean


def impute_ffill(df, col="Close"):
    """Forward fill (Common for stock prices)."""
    df_clean = df.copy()
    df_clean[col] = df_clean[col].ffill().bfill()
    return df_clean


def impute_bfill(df, col="Close"):
    """Backward fill."""
    df_clean = df.copy()
    df_clean[col] = df_clean[col].bfill().ffill()
    return df_clean
