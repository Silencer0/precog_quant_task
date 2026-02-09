import json
import os

def generate_boosted_kpca_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Boosted Algorithms with Approximated Kernel PCA Features\n",
                    "This notebook evaluates all major gradient boosting frameworks using high-dimensional features compressed via **Nystroem Approximated Kernel PCA**.\n",
                    "\n",
                    "### Workflow:\n",
                    "1.  **Bucketing**: Features are grouped into Volume, Volatility, Momentum, and Trend.\n",
                    "2.  **Compression**: Each bucket is reduced to its primary non-linear components using Nystroem + PCA.\n",
                    "3.  **Modeling**: A battery of boosted models are trained and backtested:\n",
                    "    - XGBoost\n",
                    "    - LightGBM\n",
                    "    - CatBoost\n",
                    "    - AdaBoost\n",
                    "    - Gradient Boosting (sklearn)\n",
                    "    - Hist Gradient Boosting (sklearn)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install xgboost lightgbm catboost --quiet\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys, json\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.kernel_approximation import Nystroem\n",
                    "from sklearn.decomposition import PCA\n",
                    "from bokeh.io import output_notebook, show\n",
                    "\n",
                    "import xgboost as xgb\n",
                    "import lightgbm as lgb\n",
                    "from catboost import CatBoostRegressor\n",
                    "from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor\n",
                    "\n",
                    "output_notebook()\n",
                    "\n",
                    "if os.getcwd().endswith('notebooks/boosted algorithms'):\n",
                    "    os.chdir('../..')\n",
                    "ROOT = os.getcwd()\n",
                    "if ROOT not in sys.path: sys.path.insert(0, ROOT)\n",
                    "\n",
                    "from src.backtester.data import align_close_prices, load_cleaned_assets\n",
                    "from src.backtester.engine import BacktestConfig, run_backtest\n",
                    "from src.backtester.report import compute_backtest_report\n",
                    "from src.backtester.bokeh_plots import build_interactive_portfolio_layout\n",
                    "\n",
                    "FEATURES_PATH = 'dataset/features/all_features.parquet'\n",
                    "TARGET_FWD = 'y_ret_1d_fwd'\n",
                    "CUTOFF = '2023-01-01'\n",
                    "SEED = 42\n",
                    "\n",
                    "all_res = {}\n",
                    "equities = {}"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1) Data Prep & Bucket Kernel PCA"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_parquet(FEATURES_PATH)\n",
                    "if 'Date' in df.columns: df = df.set_index('Date')\n",
                    "df[TARGET_FWD] = df.groupby('Asset_ID', sort=False)['ret_1d'].shift(-1)\n",
                    "df = df.dropna(subset=[TARGET_FWD])\n",
                    "\n",
                    "volume_cols = [c for c in df.columns if any(p in c for p in ['volume', 'obv', 'ad_line'])]\n",
                    "volatility_cols = [c for c in df.columns if any(p in c for p in ['roll_std', 'roll_var', 'roll_min', 'roll_max', 'atr', 'realized_vol', 'bb_bandwidth', 'bb_percent_b'])]\n",
                    "momentum_cols = [c for c in df.columns if any(p in c for p in ['ret_lag', 'ret_5d', 'ret_21d', 'cumret', 'rsi', 'macd', 'rmacd', 'roc', 'stoch', 'cci', 'filt_resid'])]\n",
                    "trend_cols = [c for c in df.columns if any(p in c for p in ['sma', 'ema', 'diff_log', 'ha_ha', 'bb_mid', 'bb_upper', 'bb_lower', 'adx', 'aroon', 'ichimoku', 'fib', 'bos', 'choch', 'mss', 'filt_close', 'filt_logret'])]\n",
                    "\n",
                    "buckets = {\n",
                    "    'Volume': volume_cols,\n",
                    "    'Volatility': volatility_cols,\n",
                    "    'Trend': trend_cols,\n",
                    "    'Momentum': momentum_cols\n",
                    "}\n",
                    "\n",
                    "# Remove overlap and ensure valid columns\n",
                    "seen = set()\n",
                    "for b in buckets:\n",
                    "    buckets[b] = [c for c in buckets[b] if c in df.columns and c not in seen]\n",
                    "    seen.update(buckets[b])\n",
                    "\n",
                    "df_train = df[df.index < CUTOFF].copy()\n",
                    "df_test = df[df.index >= CUTOFF].copy()\n",
                    "\n",
                    "tr_feats, te_feats = [], []\n",
                    "for b_name, cols in buckets.items():\n",
                    "    pipe = Pipeline([\n",
                    "        ('imputer', SimpleImputer(strategy='median')),\n",
                    "        ('scaler', StandardScaler()),\n",
                    "        ('kpca_approx', Nystroem(kernel='poly', degree=2, n_components=100, random_state=SEED)),\n",
                    "        ('pca_distill', PCA(n_components=3))\n",
                    "    ])\n",
                    "    X_tr = df_train[cols].replace([np.inf, -np.inf], np.nan).fillna(0)\n",
                    "    X_te = df_test[cols].replace([np.inf, -np.inf], np.nan).fillna(0)\n",
                    "    tr_feats.append(pipe.fit_transform(X_tr))\n",
                    "    te_feats.append(pipe.transform(X_te))\n",
                    "\n",
                    "X_tr_stacked = np.hstack(tr_feats)\n",
                    "X_te_stacked = np.hstack(te_feats)\n",
                    "y_tr = df_train[TARGET_FWD]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2) Model Battery"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "models = {\n",
                    "    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=SEED),\n",
                    "    'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=SEED, verbose=-1),\n",
                    "    'CatBoost': CatBoostRegressor(n_estimators=100, learning_rate=0.05, depth=6, verbose=0, random_state=SEED),\n",
                    "    'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=SEED),\n",
                    "    'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=SEED),\n",
                    "    'HistGBR': HistGradientBoostingRegressor(max_iter=100, max_depth=6, random_state=SEED)\n",
                    "}\n",
                    "\n",
                    "test_syms = df_test['Asset_ID'].unique()\n",
                    "close_prices = align_close_prices(load_cleaned_assets(symbols=test_syms))\n",
                    "close_prices = close_prices[close_prices.index >= CUTOFF]\n",
                    "\n",
                    "for name, model in models.items():\n",
                    "    print(f'\\n--- Testing {name} ---')\n",
                    "    model.fit(X_tr_stacked, y_tr)\n",
                    "    preds = model.predict(X_te_stacked)\n",
                    "    df_out = df_test.copy(); df_out['y_pred'] = preds\n",
                    "    w = df_out.pivot(columns='Asset_ID', values='y_pred').reindex(close_prices.index).fillna(0)\n",
                    "    w_rank = w.rank(axis=1, ascending=False)\n",
                    "    w_final = ((w_rank <= 5) & (w > 0)).astype(float)\n",
                    "    w_final = w_final.div(w_final.sum(axis=1).replace(0, 1), axis=0)\n",
                    "    res = run_backtest(close_prices, w_final, BacktestConfig())\n",
                    "    rep = compute_backtest_report(result=res, close_prices=close_prices)\n",
                    "    print(rep)\n",
                    "    all_res[name], equities[name] = rep, res.equity\n",
                    "    mkt = pd.DataFrame(index=close_prices.index); mkt['Close'] = close_prices.iloc[:, 0]\n",
                    "    for c in ['Open', 'High', 'Low']: mkt[c] = mkt['Close']\n",
                    "    mkt['Volume'] = 0\n",
                    "    show(build_interactive_portfolio_layout(\n        market_ohlcv=mkt, \n        equity=res.equity, \n        returns=res.returns, \n        weights=res.weights, \n        turnover=res.turnover, \n        costs=res.costs, \n        title=f\"{name} (Approximated Kernel PCA)\"\n    ))\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3) Summary Comparison"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "summary = pd.DataFrame(all_res).T[['CAGR [%]', 'Sharpe', 'Max Drawdown [%]']]\n",
                    "display(summary)\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "for name, eq in equities.items(): plt.plot(eq/eq.iloc[0], label=name)\n",
                    "plt.title(\"Boosted Models Performance\"); plt.legend(); plt.grid(True, alpha=0.3); plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    
    # Path handling
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'notebooks', 'boosted algorithms')
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, 'KernelPCA_Boosted_Comparison.ipynb')
    
    with open(target_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook generated at: {target_file}")

if __name__ == "__main__":
    generate_boosted_kpca_notebook()
