import json
import os

def generate_clustering_kpca_lgbm_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Cluster-Wise Dimensionality Reduction with LightGBM\n",
                    "This notebook implements a sophisticated pipeline:\n",
                    "1.  **Clustering**: Use KMeans to partition the feature space into 6 distinct regimes.\n",
                    "2.  **Regime-Specific Compression**: Apply Nystroem-approximated Kernel PCA within each cluster to capture local non-linearities.\n",
                    "3.  **Modeling**: Train a global LightGBM model on these regime-distilled features.\n",
                    "4.  **Backtesting**: Evaluate performance using the interactive Bokeh dashboard."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "!pip install lightgbm --quiet\n"
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
                    "from sklearn.cluster import KMeans\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.kernel_approximation import Nystroem\n",
                    "from sklearn.decomposition import PCA\n",
                    "import lightgbm as lgb\n",
                    "from bokeh.io import output_notebook, show\n",
                    "\n",
                    "output_notebook()\n",
                    "\n",
                    "if os.getcwd().endswith('notebooks/clustering_analysis'):\n",
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
                    "TARGET_COL = 'ret_1d'\n",
                    "TARGET_FWD = 'y_ret_1d_fwd'\n",
                    "CUTOFF = '2023-01-01'\n",
                    "N_CLUSTERS = 6\n",
                    "SEED = 42\n",
                    "COMPONENTS_PER_CLUSTER = 3"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1) Data Preparation"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Loading features...\")\n",
                    "df = pd.read_parquet(FEATURES_PATH)\n",
                    "if 'Date' in df.columns: df = df.set_index('Date')\n",
                    "\n",
                    "# Create target\n",
                    "df[TARGET_FWD] = df.groupby('Asset_ID', sort=False)[TARGET_COL].shift(-1)\n",
                    "df = df.dropna(subset=[TARGET_FWD])\n",
                    "\n",
                    "feature_cols = [c for c in df.columns if c not in ['Asset_ID', TARGET_FWD, TARGET_COL, 'close', 'volume']]\n",
                    "\n",
                    "# Clean and Scale\n",
                    "imputer = SimpleImputer(strategy='median')\n",
                    "scaler = StandardScaler()\n",
                    "\n",
                    "X = df[feature_cols].replace([np.inf, -np.inf], np.nan)\n",
                    "X_clean = imputer.fit_transform(X)\n",
                    "X_scaled = scaler.fit_transform(X_clean)\n",
                    "\n",
                    "df_train = df[df.index < CUTOFF]\n",
                    "df_test = df[df.index >= CUTOFF]\n",
                    "\n",
                    "X_train = X_scaled[:len(df_train)]\n",
                    "X_test = X_scaled[len(df_train):]\n",
                    "y_train = df_train[TARGET_FWD]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2) KMeans Clustering & Cluster-wise Kernel PCA"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Clustering into {N_CLUSTERS} regimes...\")\n",
                    "kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)\n",
                    "clusters_train = kmeans.fit_predict(X_train)\n",
                    "clusters_test = kmeans.predict(X_test)\n",
                    "\n",
                    "X_train_reduced = np.zeros((len(X_train), N_CLUSTERS * COMPONENTS_PER_CLUSTER))\n",
                    "X_test_reduced = np.zeros((len(X_test), N_CLUSTERS * COMPONENTS_PER_CLUSTER))\n",
                    "\n",
                    "for i in range(N_CLUSTERS):\n",
                    "    print(f\"  Processing Cluster {i} features...\")\n",
                    "    mask_tr = (clusters_train == i)\n",
                    "    \n",
                    "    # Using Nystroem for non-linear mapping\n",
                    "    nystroem = Nystroem(kernel='rbf', n_components=100, random_state=SEED)\n",
                    "    pca = PCA(n_components=COMPONENTS_PER_CLUSTER)\n",
                    "    \n",
                    "    if mask_tr.any():\n",
                    "        # Fit on cluster data\n",
                    "        X_cluster = X_train[mask_tr]\n",
                    "        nystroem.fit(X_cluster)\n",
                    "        X_mapped = nystroem.transform(X_cluster)\n",
                    "        pca.fit(X_mapped)\n",
                    "        \n",
                    "        # Transform ALL data (we want all samples to have values for cluster-i features)\n",
                    "        col_start = i * COMPONENTS_PER_CLUSTER\n",
                    "        col_end = (i + 1) * COMPONENTS_PER_CLUSTER\n",
                    "        \n",
                    "        X_train_reduced[:, col_start:col_end] = pca.transform(nystroem.transform(X_train))\n",
                    "        X_test_reduced[:, col_start:col_end] = pca.transform(nystroem.transform(X_test))\n",
                    "\n",
                    "print(f\"Reduced feature shape: {X_train_reduced.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3) LightGBM Training & Backtesting"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Training LightGBM on cluster-distilled features...\")\n",
                    "model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=SEED, verbose=-1)\n",
                    "model.fit(X_train_reduced, y_train)\n",
                    "\n",
                    "preds = model.predict(X_test_reduced)\n",
                    "\n",
                    "test_syms = df_test['Asset_ID'].unique()\n",
                    "close_prices = align_close_prices(load_cleaned_assets(symbols=test_syms))\n",
                    "close_prices = close_prices[close_prices.index >= CUTOFF]\n",
                    "\n",
                    "df_out = df_test.copy(); df_out['y_pred'] = preds\n",
                    "w = df_out.pivot(columns='Asset_ID', values='y_pred').reindex(close_prices.index).fillna(0)\n",
                    "w_rank = w.rank(axis=1, ascending=False)\n",
                    "w_final = ((w_rank <= 5) & (w > 0)).astype(float)\n",
                    "w_final = w_final.div(w_final.sum(axis=1).replace(0, 1), axis=0)\n",
                    "\n",
                    "res = run_backtest(close_prices, w_final, BacktestConfig())\n",
                    "report = compute_backtest_report(result=res, close_prices=close_prices)\n",
                    "print(report)\n",
                    "\n",
                    "mkt = pd.DataFrame(index=close_prices.index); mkt['Close'] = close_prices.iloc[:, 0]\n",
                    "for c in ['Open', 'High', 'Low']: mkt[c] = mkt['Close']\n",
                    "mkt['Volume'] = 0\n",
                    "\n",
                    "show(build_interactive_portfolio_layout(\n",
                    "    market_ohlcv=mkt, \n",
                    "    equity=res.equity, \n",
                    "    returns=res.returns, \n",
                    "    weights=res.weights, \n",
                    "    turnover=res.turnover, \n",
                    "    costs=res.costs, \n",
                    "    title=\"KMeans + Kernel PCA + LightGBM\"\n",
                    "))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
    
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root, 'notebooks', 'clustering_analysis')
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, 'KMeans_KPCA_LightGBM.ipynb')
    
    with open(target_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook generated at: {target_file}")

if __name__ == "__main__":
    generate_clustering_kpca_lgbm_notebook()
