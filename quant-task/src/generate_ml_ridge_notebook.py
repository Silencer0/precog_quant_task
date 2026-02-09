import json
import os
from typing import Any, Dict, List


def _source_lines(text: str) -> List[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def _md(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text),
    }


def _code(code: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(code),
    }


def create_notebook() -> None:
    cells: List[Dict[str, Any]] = []

    cells.append(
        _md(
            "# Ridge Regression with Heiken Ashi & Trend Features\n"
            "This notebook trains a Ridge Regression model to predict next-day returns using:\n"
            "- **Heiken Ashi**: ha_open, ha_high, ha_low, ha_close (ratios or raw values)\n"
            "- **Market Structure**: bos, choch, mss\n"
            "- **SMA**: sma_20, sma_50 ratios\n"
            "- **Indicators**: rsi_14, macd histogram\n"
            "\n"
            "We also explore custom regularization where Heiken Ashi features are penalized less.\n"
        )
    )

    cells.append(_md("## 0) Setup\n"))

    cells.append(
        _code(
            "import os, sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.linear_model import Ridge\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.pipeline import Pipeline\n"
            "\n"
            "# --- Setup correct working directory (ROOT) ---\n"
            "if os.getcwd().endswith('notebooks'):\n"
            "    os.chdir('..')\n"
            "\n"
            "ROOT = os.getcwd()\n"
            "if ROOT not in sys.path:\n"
            "    sys.path.insert(0, ROOT)\n"
            "\n"
            "from src.backtester.data import align_close_prices\n"
            "from src.backtester.engine import BacktestConfig, run_backtest\n"
            "from src.backtester.report import compute_backtest_report\n"
            "from src.backtester.models import EqualWeightAllocator\n"
            "\n"
            "FEATURES_PARQUET_PATH = 'dataset/features/all_features.parquet'\n"
            "TARGET_COL = 'ret_1d'\n"
            "TARGET_FWD_COL = 'y_ret_1d_fwd'\n"
        )
    )

    cells.append(_md("## 1) Data Loading & Feature Selection\n"))

    cells.append(
        _code(
            "df = pd.read_parquet(FEATURES_PARQUET_PATH)\n"
            "if 'Date' in df.columns:\n"
            "    df['Date'] = pd.to_datetime(df['Date'])\n"
            "    df = df.set_index('Date')\n"
            "\n"
            "# Create Target\n"
            "df[TARGET_FWD_COL] = df.groupby('Asset_ID', sort=False)[TARGET_COL].shift(-1)\n"
            "df = df.dropna(subset=[TARGET_FWD_COL])\n"
            "\n"
            "# Select requested features\n"
            "ha_cols = [c for c in df.columns if 'ha_ha_' in c] # ha_ha_open, ha_ha_high, etc\n"
            "trend_cols = ['bos', 'choch', 'mss']\n"
            "sma_cols = ['sma_20', 'sma_50']\n"
            "ind_cols = ['rsi_14', 'macd_macd_hist']\n"
            "\n"
            "feature_cols = ha_cols + trend_cols + sma_cols + ind_cols\n"
            "print(f'Selected {len(feature_cols)} features: {feature_cols}')\n"
            "\n"
            "# Split (75/15/10 assets)\n"
            "assets = sorted(df['Asset_ID'].unique())\n"
            "train_assets = assets[:75]\n"
            "test_assets = assets[90:]\n"
            "\n"
            "df_train = df[df['Asset_ID'].isin(train_assets)].copy()\n"
            "df_test = df[df['Asset_ID'].isin(test_assets)].copy()\n"
            "\n"
            "X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)\n"
            "y_train = df_train[TARGET_FWD_COL]\n"
            "X_test = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)\n"
            "y_test = df_test[TARGET_FWD_COL]\n"
        )
    )

    cells.append(_md("## 2) Standard Ridge Model\n"))

    cells.append(
        _code(
            "pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('model', Ridge(alpha=1.0))\n"
            "])\n"
            "\n"
            "pipe.fit(X_train, y_train)\n"
            "df_test['y_pred_std'] = pipe.predict(X_test)\n"
            "\n"
            "print('Standard Ridge Coefficients:')\n"
            "coef_df = pd.DataFrame({'feature': feature_cols, 'coef': pipe.named_steps['model'].coef_})\n"
            "print(coef_df)\n"
        )
    )

    cells.append(_md("## 3) Custom Ridge: Influence Heiken Ashi\n"))

    cells.append(
        _md(
            "To make Heiken Ashi weights more influential (less penalty), we can manually scale the features before feeding them into Ridge. "
            "In Ridge, $\\alpha$ is applied equally to all coefficients of *scaled* features. "
            "If we 'un-scale' HA features by multiplying them by a factor > 1 *after* the standard scaler but *before* the model, "
            "their resulting weights in the original feature space will experience less effective penalty."
        )
    )

    cells.append(
        _code(
            "from sklearn.base import BaseEstimator, TransformerMixin\n"
            "\n"
            "class FeatureScaler(BaseEstimator, TransformerMixin):\n"
            "    def __init__(self, cols, factor=5.0, feature_names=None):\n"
            "        self.cols = cols\n"
            "        self.factor = factor\n"
            "        self.feature_names = feature_names\n"
            "    def fit(self, X, y=None): return self\n"
            "    def transform(self, X):\n"
            "        X_new = X.copy()\n"
            "        # Find indices of HA columns\n"
            "        indices = [i for i, name in enumerate(self.feature_names) if any(h in name for h in self.cols)]\n"
            "        X_new[:, indices] *= self.factor\n"
            "        return X_new\n"
            "\n"
            "custom_pipe = Pipeline([\n"
            "    ('imputer', SimpleImputer(strategy='median')),\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('custom_boost', FeatureScaler(cols=ha_cols, factor=10.0, feature_names=feature_cols)),\n"
            "    ('model', Ridge(alpha=1.0))\n"
            "])\n"
            "\n"
            "custom_pipe.fit(X_train, y_train)\n"
            "df_test['y_pred_custom'] = custom_pipe.predict(X_test)\n"
            "\n"
            "print('Custom Ridge Coefficients (HA Boosted):')\n"
            "custom_coef_df = pd.DataFrame({'feature': feature_cols, 'coef': custom_pipe.named_steps['model'].coef_})\n"
            "print(custom_coef_df)\n"
        )
    )

    cells.append(_md("## 4) Backtesting Comparison\n"))

    cells.append(
        _code(
            "from src.backtester.data import load_cleaned_assets\n"
            "\n"
            "test_syms = df_test['Asset_ID'].unique()\n"
            "assets_dict = load_cleaned_assets(symbols=test_syms)\n"
            "close_prices = align_close_prices(assets_dict)\n"
            "\n"
            "def run_ml_backtest(preds_col, name):\n"
            "    # Prepare weights\n"
            "    w = df_test.pivot(columns='Asset_ID', values=preds_col).reindex(close_prices.index).fillna(0)\n"
            "    # Long-only top 5% or simple sign-based?\n"
            "    # Let's use top 5 assets by predicted return if positive\n"
            "    w_rank = w.rank(axis=1, ascending=False)\n"
            "    w_final = ((w_rank <= 5) & (w > 0)).astype(float)\n"
            "    w_final = w_final.div(w_final.sum(axis=1), axis=0).fillna(0)\n"
            "\n"
            "    config = BacktestConfig(rebalance='D', initial_equity=1_000_000)\n"
            "    res = run_backtest(close_prices, w_final, config)\n"
            "    report = compute_backtest_report(result=res, close_prices=close_prices)\n"
            "    return res, report\n"
            "\n"
            "print('Backtesting Standard Ridge...')\n"
            "res_std, rep_std = run_ml_backtest('y_pred_std', 'Standard')\n"
            "print('--- Standard Report ---')\n"
            "display(rep_std)\n"
            "\n"
            "print('\\nBacktesting Custom Ridge (HA Influence)...')\n"
            "res_cust, rep_cust = run_ml_backtest('y_pred_custom', 'Custom')\n"
            "print('--- Custom Report ---')\n"
            "display(rep_cust)\n"
        )
    )

    cells.append(_md("## 5) Equity Curve Comparison\n"))

    cells.append(
        _code(
            "plt.figure(figsize=(12, 6))\n"
            "plt.plot(res_std.equity, label='Standard Ridge')\n"
            "plt.plot(res_cust.equity, label='Custom Ridge (HA Boosted)')\n"
            "plt.title('ML Model Comparison: Standard vs HA-Influenced Ridge')\n"
            "plt.legend()\n"
            "plt.grid(True)\n"
            "plt.show()\n"
        )
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    os.makedirs("notebooks", exist_ok=True)
    out_path = os.path.join("notebooks", "ML_Ridge_Heiken_Ashi_Custom.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    create_notebook()
