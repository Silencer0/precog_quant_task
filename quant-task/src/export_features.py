import os
import sys
from glob import glob

from typing import cast

import numpy as np
import pandas as pd


# Allow running as a script: `python src/export_features.py`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.core import (
    accumulation_distribution_line,
    adx,
    aroon,
    atr,
    bollinger_bands,
    cci,
    cumulative_return,
    difference,
    ema_ratio,
    excess_return,
    fib_retracement_ratio,
    fib_levels,
    heiken_ashi,
    ichimoku,
    lag,
    log_return,
    macd,
    market_structure_signals,
    obv,
    roc,
    rmacd,
    rolling_minmax_scale,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_var,
    rolling_zscore,
    rsi,
    sma_ratio,
    simple_return,
    stochastic_oscillator,
    wrobv,
)

from src.cleaning.filters import smooth_kalman, smooth_wiener
from src.cleaning.signal import adaptive_lms, lattice_filter, smooth_spectral
from src.cleaning.smoothing import smooth_ema, smooth_savgol, smooth_sma, smooth_wma


def _load_cleaned_asset_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df


def extract_daily_features(
    df: pd.DataFrame,
    *,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Extract daily features from a single-asset OHLCV DataFrame.

    Expected columns: Open, High, Low, Close, Volume.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    open_col = df["Open"]
    if isinstance(open_col, pd.DataFrame):
        raise ValueError("Open column must be 1D")
    open_ = open_col.astype(float)

    high_col = df["High"]
    if isinstance(high_col, pd.DataFrame):
        raise ValueError("High column must be 1D")
    high = high_col.astype(float)

    low_col = df["Low"]
    if isinstance(low_col, pd.DataFrame):
        raise ValueError("Low column must be 1D")
    low = low_col.astype(float)

    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        raise ValueError("Close column must be 1D")
    close = close_col.astype(float)

    volume_col = df["Volume"]
    if isinstance(volume_col, pd.DataFrame):
        raise ValueError("Volume column must be 1D")
    volume = volume_col.astype(float)

    def _as_series(x: object, label: str) -> pd.Series:
        if isinstance(x, pd.DataFrame):
            raise ValueError(f"{label} must be a 1D series")
        if isinstance(x, pd.Series):
            return x
        return pd.Series(x, index=close.index)

    feats: dict[str, pd.Series | pd.DataFrame] = {}

    # Returns
    feats["ret_1d"] = simple_return(close)
    feats["logret_1d"] = log_return(close)
    feats["excess_ret_1d"] = excess_return(
        feats["ret_1d"],
        risk_free_rate_annual=risk_free_rate_annual,
        periods_per_year=periods_per_year,
        compounding="discrete",
    )

    # Lag features (momentum)
    feats["logret_lag_1"] = lag(feats["logret_1d"], 1)
    feats["logret_lag_5"] = lag(feats["logret_1d"], 5)
    feats["ret_lag_1"] = lag(feats["ret_1d"], 1)
    feats["ret_lag_5"] = lag(feats["ret_1d"], 5)

    # Multi-horizon returns
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_21d"] = close.pct_change(21)
    log_close = pd.Series(np.log(close.to_numpy()), index=close.index)
    feats["logret_5d"] = log_close.diff(5)
    feats["logret_21d"] = log_close.diff(21)

    # Cumulative returns (rolling)
    feats["cumret_5d"] = cumulative_return(feats["ret_1d"], 5)
    feats["cumret_21d"] = cumulative_return(feats["ret_1d"], 21)

    # Differencing (trend removal on semilog scale)
    feats["diff_log_close_1"] = difference(log_close, 1)
    feats["diff_log_close_2"] = difference(log_close, 2)

    # Rolling statistics on log returns
    for w in (5, 10, 20, 60):
        feats[f"logret_roll_mean_{w}"] = rolling_mean(feats["logret_1d"], w)
        feats[f"logret_roll_var_{w}"] = rolling_var(feats["logret_1d"], w)
        feats[f"logret_roll_std_{w}"] = rolling_std(feats["logret_1d"], w)
        feats[f"logret_roll_min_{w}"] = rolling_min(feats["logret_1d"], w)
        feats[f"logret_roll_max_{w}"] = rolling_max(feats["logret_1d"], w)

    # Heiken Ashi
    ha_df = heiken_ashi(open_, high, low, close)
    feats["ha"] = ha_df

    # Moving Averages (sma_5 is already in filter_series, but let's add 20 and 50 here)
    feats["sma_20"] = smooth_sma(close, window=20)
    feats["sma_50"] = smooth_sma(close, window=50)

    # Rolling statistics on volume
    feats["log_volume"] = volume.replace(0.0, np.nan).apply(np.log)
    for w in (5, 20, 60):
        feats[f"volume_roll_mean_{w}"] = rolling_mean(volume, w)
        feats[f"volume_roll_std_{w}"] = rolling_std(volume, w)
        feats[f"volume_zscore_{w}"] = rolling_zscore(volume, w)

    # Feature transformations (per-asset, rolling)
    feats["logret_zscore_20"] = rolling_zscore(feats["logret_1d"], 20)
    feats["close_minmax_20"] = rolling_minmax_scale(close, 20)
    feats["volume_minmax_20"] = rolling_minmax_scale(volume, 20)

    # Volatility indicators
    feats["atr_14"] = atr(high, low, close, window=14)
    feats["realized_vol_20"] = rolling_std(feats["logret_1d"], 20) * np.sqrt(252.0)

    # Technical indicators
    feats["rsi_14"] = rsi(close, window=14)

    # Paper-aligned EMA/SMA ratio features (arXiv:2412.15448)
    feats["sma_ratio_20"] = sma_ratio(close, window=20)
    feats["ema_ratio_20"] = ema_ratio(close, span=20)

    macd_df = macd(close, fast=12, slow=26, signal=9)
    feats["macd"] = macd_df
    macd_line = cast(pd.Series, macd_df["macd"])
    macd_sig = cast(pd.Series, macd_df["macd_signal"])
    feats["rmacd_12_26_9"] = rmacd(macd_line, macd_sig)

    feats["bb"] = bollinger_bands(close, window=20, k=2.0)
    feats["roc_10"] = roc(close, window=10)
    feats["stoch"] = stochastic_oscillator(high, low, close, k_window=14, d_window=3)
    feats["cci_20"] = cci(high, low, close, window=20)

    # Volume-based indicators
    feats["obv"] = obv(close, volume)
    feats["obv_roc_10"] = roc(feats["obv"], window=10)
    feats["wrobv_20"] = wrobv(close, volume, window=20)
    feats["ad_line"] = accumulation_distribution_line(high, low, close, volume)

    # Trend strength indicators
    feats["adx"] = adx(high, low, close, window=14)
    feats["aroon"] = aroon(high, low, window=25)

    # Cloud / support-resistance style indicators
    feats["ichimoku"] = ichimoku(high, low, close)
    feats["fib"] = fib_levels(high, low, window=60)
    feats["fib_retr_60"] = fib_retracement_ratio(high, low, close, window=60)

    # Market structure event indicators (BOS / CHoCH / MSS)
    ms = market_structure_signals(high, low, close, swing_window=3)
    feats["bos"] = cast(pd.Series, ms["bos"])
    feats["choch"] = cast(pd.Series, ms["choch"])
    feats["mss"] = cast(pd.Series, ms["mss"])

    # Filter-based features from the cleaning pipeline (computed as features, not as cleaned data)
    filter_series: dict[str, pd.Series] = {}
    filter_series["sma_5"] = _as_series(smooth_sma(close, window=5), "sma_5")
    filter_series["ema_12"] = _as_series(smooth_ema(close, span=12), "ema_12")
    filter_series["wma_5"] = _as_series(smooth_wma(close, window=5), "wma_5")
    filter_series["savgol_11_2"] = _as_series(
        smooth_savgol(close, window=11, polyorder=2), "savgol_11_2"
    )
    filter_series["kalman_r0p1_q0p01"] = _as_series(
        smooth_kalman(close, r=0.1, q=0.01), "kalman_r0p1_q0p01"
    )
    filter_series["wiener_5"] = _as_series(smooth_wiener(close, mysize=5), "wiener_5")
    filter_series["spectral_0p1"] = _as_series(
        smooth_spectral(close, threshold=0.1), "spectral_0p1"
    )
    filter_series["lms_mu1e-4_taps5"] = _as_series(
        adaptive_lms(close, mu=0.0001, n_taps=5), "lms_mu1e-4_taps5"
    )
    filter_series["lattice_demo"] = _as_series(
        lattice_filter(close, k_coeffs=[0.1, 0.05, 0.02]), "lattice_demo"
    )

    # NOTE: statsmodels AR/ARMA filters were removed from the feature set because they
    # frequently emit convergence/non-stationarity warnings on stock series.

    for name, filt in filter_series.items():
        feats[f"filt_close_{name}"] = filt
        feats[f"filt_resid_{name}"] = close - filt
        feats[f"filt_logret_{name}"] = log_return(filt)

    # Assemble into a DataFrame
    out_parts: list[pd.DataFrame] = []
    for name, obj in feats.items():
        if isinstance(obj, pd.DataFrame):
            renamed = obj.copy()
            renamed.columns = [f"{name}_{c}" for c in renamed.columns]
            out_parts.append(renamed)
        else:
            out_parts.append(obj.rename(name).to_frame())

    features_df = pd.concat(out_parts, axis=1)

    # Keep a small set of raw columns for context (not smoothed/filtered)
    features_df["close"] = close
    features_df["volume"] = volume

    return features_df


def export_feature_data(
    *,
    cleaned_dir: str = "dataset/cleaned",
    output_dir: str = "dataset/features",
    risk_free_rate_annual: float = 0.0,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    asset_files = sorted(glob(os.path.join(cleaned_dir, "Asset_*.csv")))
    if not asset_files:
        raise FileNotFoundError(f"No asset CSVs found in {cleaned_dir!r}")

    all_rows: list[pd.DataFrame] = []
    for path in asset_files:
        aid = os.path.basename(path).replace(".csv", "")
        df = _load_cleaned_asset_csv(path)
        feats = extract_daily_features(df, risk_free_rate_annual=risk_free_rate_annual)
        feats.to_csv(os.path.join(output_dir, f"{aid}.csv"), index=True)

        feats_parquet = feats.copy()
        feats_parquet["Asset_ID"] = aid
        all_rows.append(feats_parquet)

    final_df = pd.concat(all_rows)

    parquet_path = os.path.join(output_dir, "all_features.parquet")
    try:
        final_df.to_parquet(parquet_path)
    except ImportError:
        # Parquet support is optional in pandas (requires pyarrow or fastparquet).
        # Fall back to a combined CSV so the export still completes.
        csv_path = os.path.join(output_dir, "all_features.csv")
        final_df.to_csv(csv_path, index=True)
        print(
            "Parquet engine missing (pyarrow/fastparquet). "
            f"Wrote combined CSV instead: {csv_path}"
        )


if __name__ == "__main__":
    export_feature_data()
