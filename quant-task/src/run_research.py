import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.loader import load_asset_data, get_data_dir
from src.utils.summary import generate_data_quality_report
from src.utils.metrics import masked_mse, calculate_advanced_metrics
from src.cleaning.imputation import impute_mean, impute_interpolation, impute_ffill
from src.cleaning.outliers import (
    detect_3sigma,
    detect_iqr,
    detect_isolation_forest,
    detect_lof,
    remove_outliers,
)
from src.cleaning.smoothing import smooth_sma, smooth_ema, smooth_wma, smooth_savgol
from src.cleaning.anomaly import (
    detect_dbscan,
    detect_lof,
    detect_window_anomaly,
    detect_abnormal_sequence,
)
from src.cleaning.statistical import (
    impute_mle,
    bayesian_smoothing,
    markov_smoothing,
    hmm_smoothing,
    em_imputation,
)
from src.cleaning.advanced import (
    relationship_dependent_smoothing,
    smurf_smoothing,
    spatio_temporal_smoothing,
)
from src.cleaning.corporate import detect_splits
from src.cleaning.ts_models import smooth_ar, smooth_arma
from src.cleaning.filters import smooth_kalman, smooth_wiener
from src.cleaning.signal import smooth_spectral, adaptive_lms, lattice_filter
from src.cleaning.integrity import fix_ohlc_integrity, remove_duplicates
from src.models.lstm import train_lstm_cleaner
from src.models.gan import gan_imputation


def run_research():
    print("Starting Research Pipeline...")
    data_dir = get_data_dir()
    data = load_asset_data(data_dir)
    print(f"Loaded {len(data)} assets.")

    # 1. Data Quality Report
    quality_report = generate_data_quality_report(data)
    quality_report.to_csv("research_quality_report.csv", index=False)
    print("Data quality report generated.")

    # 2. Imputation Benchmarking (on a sample asset)
    sample_aid = "Asset_001"
    sample_df = data[sample_aid]

    imputation_methods = {
        "Mean": lambda df: impute_mean(df),
        "Interpolation": lambda df: impute_interpolation(df),
        "Forward_Fill": lambda df: impute_ffill(df),
        "GAN": lambda df: gan_imputation(df),
        "MLE": lambda df: impute_mle(df),
        "EM": lambda df: em_imputation(df),
    }

    imputation_results = {}
    for name, func in imputation_methods.items():
        metrics = calculate_advanced_metrics(sample_df, func)
        imputation_results[name] = metrics
        print(f"Imputation {name} Metrics: {metrics}")

    # 3. Smoothing Benchmarking
    smoothing_methods = {
        "SMA": lambda s: smooth_sma(s),
        "EMA": lambda s: smooth_ema(s),
        "WMA": lambda s: smooth_wma(s),
        "SavGol": lambda s: smooth_savgol(s),
        "AR": lambda s: smooth_ar(s),
        "ARMA": lambda s: smooth_arma(s),
        "Kalman": lambda s: smooth_kalman(s),
        "Wiener": lambda s: smooth_wiener(s),
        "Spectral": lambda s: smooth_spectral(s),
        "LMS": lambda s: adaptive_lms(s),
        "Lattice": lambda s: lattice_filter(s),
        "Bayesian": lambda s: bayesian_smoothing(s),
        "Markov": lambda s: markov_smoothing(s),
        "HMM": lambda s: hmm_smoothing(s),
        "SMURF": lambda s: smurf_smoothing(s),
        "SpatioTemporal": lambda s: spatio_temporal_smoothing(s),
        "Relationship": lambda df: relationship_dependent_smoothing(df),
        "LSTM": lambda s: train_lstm_cleaner(s, epochs=10),  # Small epochs for speed
    }

    smoothing_results = {}
    for name, func in smoothing_methods.items():
        # Wrapper to make smoothing functions return DataFrame for metrics calc
        if name == "Relationship":
            metrics = calculate_advanced_metrics(sample_df, func)
        else:
            metrics = calculate_advanced_metrics(
                sample_df,
                lambda df: pd.DataFrame({"Close": func(df["Close"])}, index=df.index),
            )
        smoothing_results[name] = metrics
        print(f"Smoothing {name} Metrics: {metrics}")

    # 4. Outlier Detection on All Assets
    print("Running outlier detection on all assets...")
    outlier_report = []
    for aid, df in data.items():
        close = df["Close"].ffill().bfill()
        outlier_report.append(
            {
                "Asset": aid,
                "3Sigma": detect_3sigma(close).sum(),
                "IQR": detect_iqr(close).sum(),
                "IsolationForest": detect_isolation_forest(close).sum(),
                "LOF": detect_lof(close).sum(),
                "DBSCAN": detect_dbscan(close).sum(),
                "WindowAnomaly": detect_window_anomaly(close).sum(),
                "AbnormalSequence": detect_abnormal_sequence(close).sum(),
            }
        )
    outlier_df = pd.DataFrame(outlier_report)
    outlier_df.to_csv("research_outlier_report.csv", index=False)
    print("Outlier detection report generated.")

    # 5. Corporate Action Detection (Splits) on All Assets
    print("Running corporate action detection on all assets...")
    corporate_report = []
    for aid, df in data.items():
        splits = detect_splits(df)
        corporate_report.append(
            {
                "Asset": aid,
                "Potential_Splits": len(splits),
            }
        )
    corporate_df = pd.DataFrame(corporate_report)
    corporate_df.to_csv("research_corporate_report.csv", index=False)
    print("Corporate action report generated.")

    # Save results - Now saving multiple metrics columns
    # Structure: Index=Method, Columns=[RMSE, LAG, RPR]
    imp_df = pd.DataFrame(imputation_results).T
    smooth_df = pd.DataFrame(smoothing_results).T

    imp_df.to_csv("research_imputation_metrics.csv")
    smooth_df.to_csv("research_smoothing_metrics.csv")
    print("Algorithm comparison metrics saved.")


if __name__ == "__main__":
    run_research()
