import json
import os

def generate_adv_dim_reduction_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Advanced Dimensionality Reduction Comparison (Modular)\n",
                    "This notebook evaluates various feature extraction and dimensionality reduction techniques for alpha generation.\n",
                    "\n",
                    "### Fixed issues:\n",
                    "- **NMF**: Added explicit clipping to prevent tiny floating-point negatives.\n",
                    "- **KernelPCA**: Using **Nystroem Approximation** for large-scale non-linear mapping.\n",
                    "- **Modular Cells**: Each technique is in its own cell for granular control."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install umap-learn --quiet\n"
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
                    "from sklearn.ensemble import RandomForestRegressor\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.kernel_approximation import Nystroem\n",
                    "from sklearn.decomposition import PCA, FastICA, FactorAnalysis, TruncatedSVD, NMF\n",
                    "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
                    "from bokeh.io import output_notebook, show\n",
                    "\n",
                    "try:\n",
                    "    import umap\n",
                    "    HAS_UMAP = True\n",
                    "except ImportError:\n",
                    "    HAS_UMAP = False\n",
                    "\n",
                    "output_notebook()\n",
                    "\n",
                    "if os.getcwd().endswith('notebooks'):\n",
                    "    os.chdir('..')\n",
                    "ROOT = os.getcwd()\n",
                    "if ROOT not in sys.path: sys.path.insert(0, ROOT)\n",
                    "\n",
                    "from src.backtester.data import align_close_prices, load_cleaned_assets\n",
                    "from src.backtester.engine import BacktestConfig, run_backtest\n",
                    "from src.backtester.report import compute_backtest_report\n",
                    "from src.backtester.bokeh_plots import build_interactive_portfolio_layout\n",
                    "\n",
                    "FEATURES_PARQUET_PATH = 'dataset/features/all_features.parquet'\n",
                    "TARGET_COL = 'ret_1d'\n",
                    "TARGET_FWD_COL = 'y_ret_1d_fwd'\n",
                    "SEED = 42\n",
                    "\n",
                    "all_reports = {}\n",
                    "equities = {}"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1) Data Loading & Bucketing"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_parquet(FEATURES_PARQUET_PATH)\n",
                    "if 'Date' in df.columns:\n",
                    "    df['Date'] = pd.to_datetime(df['Date'])\n",
                    "    df = df.set_index('Date')\n",
                    "\n",
                    "df[TARGET_FWD_COL] = df.groupby('Asset_ID', sort=False)[TARGET_COL].shift(-1)\n",
                    "df = df.dropna(subset=[TARGET_FWD_COL])\n",
                    "\n",
                    "buckets = {\n",
                    "    'Volume': [\n",
                    "        'log_volume', 'volume_roll_mean_5', 'volume_roll_mean_20', 'volume_roll_mean_60', \n",
                    "        'volume_roll_std_5', 'volume_roll_std_20', 'volume_roll_std_60', \n",
                    "        'volume_zscore_5', 'volume_zscore_20', 'volume_zscore_60', \n",
                    "        'volume_minmax_20', 'obv', 'obv_roc_10', 'wrobv_20', 'ad_line'\n",
                    "    ],\n",
                    "    'Volatility': [\n",
                    "        'logret_roll_var_5', 'logret_roll_var_10', 'logret_roll_var_20', 'logret_roll_var_60', \n",
                    "        'logret_roll_std_5', 'logret_roll_std_10', 'logret_roll_std_20', 'logret_roll_std_60', \n",
                    "        'atr_14', 'realized_vol_20', 'bb_bb_bandwidth'\n",
                    "    ],\n",
                    "    'Trend': [\n",
                    "        'sma_20', 'sma_50', 'ha_ha_open', 'ha_ha_high', 'ha_ha_low', 'ha_ha_close', \n",
                    "        'sma_ratio_20', 'ema_ratio_20', 'bb_bb_mid', 'bb_bb_upper', 'bb_bb_lower', \n",
                    "        'bb_bb_percent_b', 'adx_plus_di', 'adx_minus_di', 'adx_adx', 'adx_adx_raw', \n",
                    "        'aroon_aroon_up', 'aroon_aroon_down', 'ichimoku_ichimoku_conv', \n",
                    "        'ichimoku_ichimoku_base', 'ichimoku_ichimoku_span_a', 'ichimoku_ichimoku_span_b', \n",
                    "        'ichimoku_ichimoku_lagging', 'bos', 'choch', 'mss'\n",
                    "    ],\n",
                    "    'Momentum': [\n",
                    "        'rsi_14', 'macd_macd', 'macd_macd_signal', 'macd_macd_hist', \n",
                    "        'rmacd_12_26_9', 'stoch_stoch_k', 'stoch_stoch_d', 'roc_10', 'cci_20'\n",
                    "    ]\n",
                    "}\n",
                    "\n",
                    "for b in buckets: buckets[b] = [c for c in buckets[b] if c in df.columns]\n",
                    "\n",
                    "cutoff = '2023-01-01'\n",
                    "df_train = df[df.index < cutoff].copy()\n",
                    "df_test = df[df.index >= cutoff].copy()\n",
                    "\n",
                    "print(f'Train/Test rows: {len(df_train)} / {len(df_test)}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2) Technique Framework"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def run_dim_reduction_pipeline(method_name, reducer_obj):\n",
                    "    print(f'\\n--- Running {method_name} ---')\n",
                    "    tr_feats, te_feats = [], []\n",
                    "    for b_name, cols in buckets.items():\n",
                    "        X_tr = df_train[cols].replace([np.inf, -np.inf], np.nan).fillna(0)\n",
                    "        X_te = df_test[cols].replace([np.inf, -np.inf], np.nan).fillna(0)\n",
                    "        scaler = MinMaxScaler() if method_name == 'NMF' else StandardScaler()\n",
                    "        steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', scaler)]\n",
                    "        if method_name == 'NMF':\n",
                    "            steps.append(('clipper', FunctionTransformer(lambda x: np.clip(x, 1e-9, None))))\n",
                    "        steps.append(('reducer', reducer_obj))\n",
                    "        pipe = Pipeline(steps)\n",
                    "        if method_name == 'LDA':\n",
                    "            pipe.fit(X_tr, np.sign(df_train[TARGET_FWD_COL]))\n",
                    "        else:\n",
                    "            pipe.fit(X_tr)\n",
                    "        tr_feats.append(pipe.transform(X_tr))\n",
                    "        te_feats.append(pipe.transform(X_te))\n",
                    "    X_tr_stacked = np.hstack(tr_feats)\n",
                    "    X_te_stacked = np.hstack(te_feats)\n",
                    "    rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=SEED)\n",
                    "    rf.fit(X_tr_stacked, df_train[TARGET_FWD_COL])\n",
                    "    preds = rf.predict(X_te_stacked)\n",
                    "    test_syms = df_test['Asset_ID'].unique()\n",
                    "    close_prices = align_close_prices(load_cleaned_assets(symbols=test_syms))\n",
                    "    df_out = df_test.copy(); df_out['y_pred'] = preds\n",
                    "    w = df_out.pivot(columns='Asset_ID', values='y_pred').reindex(close_prices.index).fillna(0)\n",
                    "    w_rank = w.rank(axis=1, ascending=False)\n",
                    "    w_final = ((w_rank <= 5) & (w > 0)).astype(float)\n",
                    "    w_final = w_final.div(w_final.sum(axis=1), axis=0).fillna(0)\n",
                    "    res = run_backtest(close_prices, w_final, BacktestConfig(rebalance='D'))\n",
                    "    report = compute_backtest_report(result=res, close_prices=close_prices)\n",
                    "    \n",
                    "    # PRINT FULL REPORT\n",
                    "    print(f'\\n--- {method_name} Detailed Report ---')\n",
                    "    print(report)\n",
                    "    \n",
                    "    # SHOW BOKEH PLOT\n",
                    "    try:\n",
                    "        # Create a proxy market OHLCV for visualization\n",
                    "        mkt = pd.DataFrame(index=close_prices.index)\n",
                    "        mkt['Close'] = close_prices.iloc[:, 0]\n",
                    "        mkt['Open'] = mkt['Close']; mkt['High'] = mkt['Close']; mkt['Low'] = mkt['Close']; mkt['Volume'] = 0\n",
                    "        \n",
                    "        p = build_interactive_portfolio_layout(\n",
                    "            market_ohlcv=mkt, \n",
                    "            equity=res.equity,\n",
                    "            returns=res.returns,\n",
                    "            weights=res.weights,\n",
                    "            turnover=res.turnover,\n",
                    "            costs=res.costs,\n",
                    "            title=f\"{method_name} Backtest Analysis\"\n",
                    "        )\n",
                    "        show(p)\n",
                    "    except Exception as e:\n",
                    "        print(f'Bokeh plot failed for {method_name}: {e}')\n",
                    "        \n",
                    "    return report, res.equity"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3) Methods Execution (Fast Methods)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fast_methods = {\n",
                    "    'ICA': FastICA(n_components=2, random_state=SEED),\n",
                    "    'LDA': LinearDiscriminantAnalysis(n_components=1),\n",
                    "    'SVD': TruncatedSVD(n_components=2, random_state=SEED),\n",
                    "    'NMF': NMF(n_components=2, random_state=SEED, init='nndsvd', max_iter=1000)\n",
                    "}\n",
                    "for name, obj in fast_methods.items():\n",
                    "    rep, eq = run_dim_reduction_pipeline(name, obj)\n",
                    "    all_reports[name], equities[name] = rep, eq\n",
                    "    print(f'{name} CAGR: {rep[\"CAGR [%]\"]:.2f}%')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3.1) Factor Analysis (FA) - Slow"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "name = 'FA'\n",
                    "obj = FactorAnalysis(n_components=2, random_state=SEED)\n",
                    "rep, eq = run_dim_reduction_pipeline(name, obj)\n",
                    "all_reports[name], equities[name] = rep, eq\n",
                    "print(f'{name} CAGR: {rep[\"CAGR [%]\"]:.2f}%')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3.2) Kernel PCA Approximation (Nystroem)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "name = 'KernelPCA_Approx'\n",
                    "obj = Pipeline([\n",
                    "    ('nystroem', Nystroem(kernel='poly', degree=2, n_components=100, random_state=SEED)),\n",
                    "    ('pca', PCA(n_components=2))\n",
                    "])\n",
                    "rep, eq = run_dim_reduction_pipeline(name, obj)\n",
                    "all_reports[name], equities[name] = rep, eq\n",
                    "print(f'{name} CAGR: {rep[\"CAGR [%]\"]:.2f}%')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3.3) UMAP"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if HAS_UMAP:\n",
                    "    name = 'UMAP'\n",
                    "    obj = umap.UMAP(n_components=2, random_state=SEED)\n",
                    "    rep, eq = run_dim_reduction_pipeline(name, obj)\n",
                    "    all_reports[name], equities[name] = rep, eq\n",
                    "    print(f'{name} CAGR: {rep[\"CAGR [%]\"]:.2f}%')\n",
                    "else:\n",
                    "    print('UMAP not found. Skipping...')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3.4) Autoencoder"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch; import torch.nn as nn; import torch.optim as optim\n",
                    "class SimpleAE(nn.Module):\n",
                    "    def __init__(self, input_dim, hidden_dim=2):\n",
                    "        super().__init__()\n",
                    "        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, hidden_dim))\n",
                    "        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, input_dim))\n",
                    "    def forward(self, x): return self.decoder(self.encoder(x))\n",
                    "ae_tr_list, ae_te_list = [], []\n",
                    "for _, cols in buckets.items():\n",
                    "    X_tr_raw = df_train[cols].replace([np.inf, -np.inf], np.nan)\n",
                    "    X_te_raw = df_test[cols].replace([np.inf, -np.inf], np.nan)\n",
                    "    X_tr = SimpleImputer(strategy='median').fit_transform(X_tr_raw)\n",
                    "    scaler = StandardScaler().fit(X_tr)\n",
                    "    X_tr_s = torch.FloatTensor(scaler.transform(X_tr))\n",
                    "    X_te = SimpleImputer(strategy='median').fit_transform(X_te_raw)\n",
                    "    X_te_s = torch.FloatTensor(scaler.transform(X_te))\n",
                    "    model = SimpleAE(X_tr.shape[1])\n",
                    "    opt = optim.Adam(model.parameters(), lr=0.01); crit = nn.MSELoss()\n",
                    "    for _ in range(10): opt.zero_grad(); l = crit(model(X_tr_s), X_tr_s); l.backward(); opt.step()\n",
                    "    model.eval()\n",
                    "    with torch.no_grad(): ae_tr_list.append(model.encoder(X_tr_s).numpy()); ae_te_list.append(model.encoder(X_te_s).numpy())\n",
                    "X_ae_tr, X_ae_te = np.hstack(ae_tr_list), np.hstack(ae_te_list)\n",
                    "rf_ae = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=SEED)\n",
                    "rf_ae.fit(X_ae_tr, df_train[TARGET_FWD_COL]); preds = rf_ae.predict(X_ae_te)\n",
                    "df_ae = df_test.copy(); df_ae['y_pred'] = preds\n",
                    "cp = align_close_prices(load_cleaned_assets(symbols=df_test['Asset_ID'].unique()))\n",
                    "w = df_ae.pivot(columns='Asset_ID', values='y_pred').reindex(cp.index).fillna(0)\n",
                    "w_final = (w.rank(axis=1, ascending=False) <= 5) & (w > 0)\n",
                    "w_final = w_final.astype(float).div(w_final.sum(axis=1), axis=0).fillna(0)\n",
                    "res = run_backtest(cp, w_final, BacktestConfig())\n",
                    "rep = compute_backtest_report(result=res, close_prices=cp)\n",
                    "all_reports['Autoencoder'], equities['Autoencoder'] = rep, res.equity\n",
                    "print(f'\\n--- Autoencoder Detailed Report ---')\n",
                    "print(rep)\n",
                    "try:\n",
                    "    mkt = pd.DataFrame(index=cp.index)\n",
                    "    mkt['Close'] = cp.iloc[:, 0]\n",
                    "    mkt['Open'] = mkt['Close']; mkt['High'] = mkt['Close']; mkt['Low'] = mkt['Close']; mkt['Volume'] = 0\n",
                    "    p = build_interactive_portfolio_layout(market_ohlcv=mkt, equity=res.equity, returns=res.returns, weights=res.weights, turnover=res.turnover, costs=res.costs, title=\"Autoencoder Backtest Analysis\")\n",
                    "    show(p)\n",
                    "except Exception as e: print(f\"Plot fail: {e}\")\n",
                    "print(f'Autoencoder CAGR: {rep[\"CAGR [%]\"]:.2f}%')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 4) Summary Comparison"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "comp_df = pd.DataFrame(all_reports).T[['CAGR [%]', 'Sharpe', 'Max Drawdown [%]', 'Volatility (ann) [%]']]\n",
                    "display(comp_df)\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "for name, eq in equities.items():\n",
                    "    plt.plot(eq / eq.iloc[0], label=f'{name} (CAGR: {comp_df.loc[name, \"CAGR [%]\"]:.1f}%)')\n",
                    "plt.title('Alpha Comparison: Advanced Dimensionality Reduction'); plt.legend(); plt.grid(True, alpha=0.3); plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    with open('/home/anivarth/college/quant-task/notebooks/ML_Model_Comparison_Advanced_DimRed.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    generate_adv_dim_reduction_notebook()
