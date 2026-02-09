import pandas as pd
import numpy as np


def check_ohlc_integrity(df):
    """
    Checks if High is the maximum and Low is the minimum of the day.
    Returns a boolean series where True indicates integrity issues.
    """
    # High must be >= Open, Close, and Low
    issue_high = (
        (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["High"] < df["Low"])
    )
    # Low must be <= Open, Close, and High
    issue_low = (
        (df["Low"] > df["Open"]) | (df["Low"] > df["Close"]) | (df["Low"] > df["High"])
    )

    return issue_high | issue_low


def fix_ohlc_integrity(df):
    """
    Fixes OHLC integrity by adjusting High and Low to match Open/Close extremes if violated.
    """
    df_clean = df.copy()

    # Correct High
    df_clean["High"] = df_clean[["Open", "High", "Low", "Close"]].max(axis=1)
    # Correct Low
    df_clean["Low"] = df_clean[["Open", "High", "Low", "Close"]].min(axis=1)

    return df_clean


def remove_duplicates(df):
    """
    Removes duplicate records based on index (Date).
    Keeps the first occurrence.
    """
    return df[~df.index.duplicated(keep="first")]


def check_hash_consistency(df):
    """
    Placeholder for hash checks if multiple sources were present.
    In this case, we just check for identical rows.
    """
    return df.duplicated()
