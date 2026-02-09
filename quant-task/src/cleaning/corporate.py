import pandas as pd
import numpy as np


def detect_splits(df, threshold=0.4, col="Close"):
    """
    Detects potential stock splits by looking for large overnight drops
    that aren't followed by a recovery (unlike typical volatility).
    threshold: drop factor (e.g., 0.5 for 2:1 split).
    """
    returns = df[col].pct_change()
    # A 2:1 split looks like a -50% return
    # A 7:1 split looks like a -85% return
    potential_splits = returns[returns < -threshold]

    # Simple heuristic: if it's a split, Volume usually increases or stays high
    # but the price stays at the new level.
    return potential_splits


def apply_split_adjustment(df, split_dates, factor=2.0, col="Close"):
    """
    Adjusts historical prices for a known split date.
    """
    df_adj = df.copy()
    for date in split_dates:
        df_adj.loc[df_adj.index < date, col] = (
            df_adj.loc[df_adj.index < date, col] / factor
        )
        # Also adjust Open, High, Low
        for c in ["Open", "High", "Low"]:
            if c in df_adj.columns:
                df_adj.loc[df_adj.index < date, c] = (
                    df_adj.loc[df_adj.index < date, c] / factor
                )
    return df_adj
