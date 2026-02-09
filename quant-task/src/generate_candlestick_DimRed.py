import json
import os

def generate_candlestick_dimred_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 02: Candlestick Pattern Dimensionality Reduction\n",
                    "This notebook applies PCA, Kernel PCA, and ICA to candlestick features and evaluates their performance in predicting 5-7 day returns."
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
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.decomposition import PCA, FastICA, KernelPCA\n",
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
                    "TARGET_HORIZON = 5\n",
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
                    "    patterns['y_target'] = group['Close'].shift(-TARGET_HORIZON) / group['Close'] - 1\n",
                    "    return patterns\n",
                    "\n",
                    "print(\"Extracting patterns...\")\n",
                    "processed_df = df.groupby('Asset_ID', group_keys=False).apply(process_asset)\n",
                    "processed_df = processed_df.dropna(subset=['y_target'])\n",
                    "feature_cols = [c for c in processed_df.columns if c not in ['Asset_ID', 'y_target']]\n",
                    "\n",
                    "train_df = processed_df[processed_df.index < CUTOFF_DATE]\n",
                    "test_df = processed_df[processed_df.index >= CUTOFF_DATE]\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(train_df[feature_cols])\n",
                    "X_test_scaled = scaler.transform(test_df[feature_cols])"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2) Dimensionality Reduction Pipeline"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def run_dimred_experiment(name, reducer_obj):\n",
                    "    print(f'\\n--- Running {name} ---')\n",
                    "    X_tr_red = reducer_obj.fit_transform(X_train_scaled)\n",
                    "    X_te_red = reducer_obj.transform(X_test_scaled)\n",
                    "    \n",
                    "    rf = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=SEED)\n",
                    "    rf.fit(X_tr_red, train_df['y_target'])\n",
                    "    preds = rf.predict(X_te_red)\n",
                    "    \n",
                    "    test_syms = test_df['Asset_ID'].unique()\n",
                    "    close_prices = align_close_prices(load_cleaned_assets(symbols=test_syms))\n",
                    "    close_prices = close_prices[close_prices.index >= CUTOFF_DATE]\n",
                    "    \n",
                    "    out = test_df.copy(); out['y_pred'] = preds\n",
                    "    w = out.pivot(columns='Asset_ID', values='y_pred').reindex(close_prices.index).fillna(0)\n",
                    "    w_rank = w.rank(axis=1, ascending=False)\n",
                    "    w_final = ((w_rank <= 5) & (w > 0)).astype(float)\n",
                    "    w_final = w_final.div(w_final.sum(axis=1).replace(0, 1), axis=0)\n",
                    "    \n",
                    "    res = run_backtest(close_prices, w_final, BacktestConfig())\n",
                    "    rep = compute_backtest_report(result=res, close_prices=close_prices)\n",
                    "    print(rep)\n",
                    "    \n",
                    "    mkt = pd.DataFrame(index=close_prices.index)\n",
                    "    mkt['Close'] = close_prices.iloc[:, 0]\n",
                    "    for c in ['Open', 'High', 'Low']: mkt[c] = mkt['Close']\n",
                    "    mkt['Volume'] = 0\n",
                    "    p = build_interactive_portfolio_layout(\n",
                    "        market_ohlcv=mkt, \n",
                    "        equity=res.equity, \n",
                    "        returns=res.returns, \n",
                    "        weights=res.weights, \n",
                    "        turnover=res.turnover, \n",
                    "        costs=res.costs, \n",
                    "        title=f\"{name} analysis\"\n",
                    "    )\n",
                    "    show(p)\n",
                    "    return rep\n",
                    "\n",
                    "pca_rep = run_dimred_experiment('PCA', PCA(n_components=5, random_state=SEED))\n",
                    "ica_rep = run_dimred_experiment('ICA', FastICA(n_components=5, random_state=SEED))\n",
                    "kpca_rep = run_dimred_experiment('KernelPCA', KernelPCA(n_components=5, kernel='rbf', random_state=SEED))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    with open('notebooks/candlestick_analysis/02_candlestick_dimred_rf.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    generate_candlestick_dimred_notebook()
