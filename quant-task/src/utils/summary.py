import pandas as pd
import numpy as np
import os


def generate_data_quality_report(data_dict):
    """
    Generates a summary of data quality for all assets.
    """
    stats = []
    for aid, df in data_dict.items():
        n_rows = len(df)
        missing = df.isna().sum()
        zeros = (df == 0).sum()

        # Check OHLC integrity issues
        from src.cleaning.integrity import check_ohlc_integrity

        integrity_issues = check_ohlc_integrity(df).sum()

        stat = {
            "Asset": aid,
            "Rows": n_rows,
            "Missing_Total": missing.sum(),
            "Missing_Pct": (missing.sum() / (n_rows * df.shape[1])) * 100,
            "Zeros_Total": zeros.sum(),
            "Integrity_Issues": integrity_issues,
            "Start_Date": df.index.min(),
            "End_Date": df.index.max(),
        }
        # Add column-specific missing counts
        for col in df.columns:
            stat[f"Missing_{col}"] = missing[col]

        stats.append(stat)

    return pd.DataFrame(stats)
