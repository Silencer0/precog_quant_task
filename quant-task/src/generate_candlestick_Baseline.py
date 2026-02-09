import json
import os

def generate_candlestick_rf_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 01: Candlestick Pattern Baseline Model\n",
                    "This notebook extracts standard candlestick patterns and trains a Random Forest model to predict subsequent 5-7 day returns."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from sklearn.ensemble import RandomForestRegressor\n",
                    "from bokeh.io import output_notebook, show\n",
                    "output_notebook()\n",
                    "\n",
                    "if os.getcwd().endswith('notebooks/candlestick_analysis'):\n",
                    "    os.chdir('../..')\n",
                    "ROOT = os.getcwd()\n",
                    "if ROOT not in sys.path: sys.path.insert(0, ROOT)\n",
                    "\n",
                    "from src.features.candlestick_patterns import extract_candlestick_patterns\n",
                    "from src.backtester.data import align_close_prices, load_cleaned_assets\n",
                    "from src.backtester.engine import BacktestConfig, run_backtest\n",
                    "from src.backtester.report import compute_backtest_report\n",
                    "from src.backtester.bokeh_plots import build_interactive_portfolio_layout\n",
                    "\n",
                    "DATA_PATH = 'dataset/cleaned/cleaned_stock_data.parquet'\n",
                    "TARGET_HORIZON = 5 # 5-7 days requested\n",
                    "CUTOFF_DATE = '2023-01-01'\n",
                    "SEED = 42"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1) Data Load & Pattern Extraction"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_parquet(DATA_PATH)\n",
                    "if 'Date' in df.columns:\n",
                    "    df['Date'] = pd.to_datetime(df['Date'])\n",
                    "    df = df.set_index('Date')\n",
                    "\n",
                    "def process_asset(group):\n",
                    "    # Get Asset_ID from group name (more robust in pandas apply)\n",
                    "    aid = group.name\n",
                    "    patterns = extract_candlestick_patterns(group)\n",
                    "    patterns['Asset_ID'] = aid\n",
                    "    # Target: Return over the next N days\n",
                    "    patterns['y_target'] = group['Close'].shift(-TARGET_HORIZON) / group['Close'] - 1\n",
                    "    return patterns\n",
                    "\n",
                    "print(\"Extracting patterns... (may take a moment for 100 assets)\")\n",
                    "processed_df = df.groupby('Asset_ID', group_keys=False).apply(process_asset)\n",
                    "processed_df = processed_df.dropna(subset=['y_target'])\n",
                    "feature_cols = [c for c in processed_df.columns if c not in ['Asset_ID', 'y_target']]\n",
                    "print(f\"Features created: {feature_cols}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2) Model Training (Time-Wise Split)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train = processed_df[processed_df.index < CUTOFF_DATE]\n",
                    "test = processed_df[processed_df.index >= CUTOFF_DATE]\n",
                    "\n",
                    "X_train, y_train = train[feature_cols], train['y_target']\n",
                    "X_test, y_test = test[feature_cols], test['y_target']\n",
                    "\n",
                    "rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=SEED)\n",
                    "rf.fit(X_train, y_train)\n",
                    "\n",
                    "preds = rf.predict(X_test)\n",
                    "test_out = test.copy()\n",
                    "test_out['y_pred'] = preds"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3) Backtesting"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "test_syms = test_out['Asset_ID'].unique()\n",
                    "close_prices = align_close_prices(load_cleaned_assets(symbols=test_syms))\n",
                    "close_prices = close_prices[close_prices.index >= CUTOFF_DATE]\n",
                    "\n",
                    "w = test_out.pivot(columns='Asset_ID', values='y_pred').reindex(close_prices.index).fillna(0)\n",
                    "w_rank = w.rank(axis=1, ascending=False)\n",
                    "w_final = ((w_rank <= 5) & (w > 0)).astype(float)\n",
                    "w_final = w_final.div(w_final.sum(axis=1), axis=0).fillna(0)\n",
                    "\n",
                    "res = run_backtest(close_prices, w_final, BacktestConfig(rebalance='D'))\n",
                    "report = compute_backtest_report(result=res, close_prices=close_prices)\n",
                    "print(report)\n",
                    "\n",
                    "# Bokeh Plot\n",
                    "mkt = pd.DataFrame(index=close_prices.index)\n",
                    "mkt['Close'] = close_prices.iloc[:, 0] # Proxy\n",
                    "for c in ['Open', 'High', 'Low']: mkt[c] = mkt['Close']\n",
                    "mkt['Volume'] = 0\n",
                    "\n",
                    "p = build_interactive_portfolio_layout(\n",
                    "    market_ohlcv=mkt, \n",
                    "    equity=res.equity, \n",
                    "    returns=res.returns, \n",
                    "    weights=res.weights, \n",
                    "    turnover=res.turnover, \n",
                    "    costs=res.costs, \n",
                    "    title=\"Candlestick Baseline RF\"\n",
                    ")\n",
                    "show(p)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    with open('notebooks/candlestick_analysis/01_candlestick_regular_rf.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    generate_candlestick_rf_notebook()
