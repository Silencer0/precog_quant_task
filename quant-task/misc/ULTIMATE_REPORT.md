# ULTIMATE REPORT

---

Last updated: [PLACEHOLDER — update when saving]

Summary
- This document summarizes: (1) Data cleaning & feature extraction, (2) Backtesting engine & metrics, (3) Implemented models and results, and (4) Cointegration experiments and intuition.
- All items described below refer only to code and notebooks that were implemented and saved in the repository. Where numeric results or graphs are needed, placeholders are included; populate from the CSV `ultimate_report_model_metrics.csv`, per-notebook `stats.csv` files, or exported Bokeh/PNG outputs.

---

1) Data cleaning & feature extraction
====================================

Overview
- Goal: produce a clean, consistent feature dataset from raw OHLCV for model training and backtesting.
- Constraints: dataset does NOT contain sector/industry labels — no sector/industry-based features were added.

Pipeline steps (implemented)
1. Data ingestion
   - Read OHLCV time series (per-ticker) from cleaned dataset (parquet/csv).
   - Ensure timestamp is timezone-aware or normalized to UTC; set index to pandas.DatetimeIndex.

2. Alignment & reindexing
   - Union of timestamps across tickers; missing timestamps forward-filled where appropriate for features requiring aligned panels (prices left as NaN where appropriate so returns are not artificially filled).
   - Resampling for multi-period features: monthly/3-month/6-month windows use month-end frequency (use 'ME' / month-end) to avoid ambiguous month boundaries.

3. Missing value handling / Imputation
   - Forward-fill (ffill) short runs of missing values for price series where appropriate; longer gaps left as NaN and removed from training windows.
   - Imputation for indicator components: median or rolling median imputation for indicator denominators; optional clip to avoid division-by-zero.

4. Outlier detection & filtering
   - Winsorization / clipping of extreme percentiles for return-based features prior to normalization (e.g., clip to [1st, 99th] percentiles).
   - Z-score filtering for intraday anomaly removal in indicator computation.

5. Smoothing & noise reduction
   - Savitzky-Golay smoothing applied where required (e.g., for Smoothed filter snapshots).
   - Kalman filter and Wiener filter implemented for signal smoothing; snapshots are extracted at feature timestamps.

6. Feature extraction
   - Price-derived features:
     - Returns: simple return r_t = (P_t / P_{t-1}) - 1
     - Log returns: lr_t = log(P_t) - log(P_{t-1})
     - Cumulative returns and rolling returns over windows
   - Momentum & trend:
     - Rolling mean (SMA) over windows: SMA_t = mean(P_{t-w+1: t})
     - Exponential moving average (EMA): EMA_t = alpha * P_t + (1-alpha) * EMA_{t-1}, alpha = 2/(w+1)
     - MACD = EMA_short - EMA_long; Signal line = EMA(MACD)
     - Momentum: P_t - P_{t-w}
   - Volatility:
     - Rolling standard deviation of returns: sigma_t = std(r_{t-w+1: t})
     - ATR (Average True Range) computed from H, L, previous close
   - Oscillators & technical indicators:
     - RSI (Relative Strength Index): RSI_t = 100 - 100/(1 + RS_t); RS_t = avg_up / avg_down
     - Bollinger Bands: Middle = SMA; Band width = k * rolling_std
     - Stochastic oscillator, On-Balance Volume approximations (if volume present)
   - Cross-sectional normalization:
     - Indneutralize: cross-sectional demeaning per date (x - mean_across_tickers) to remove market-wide signals for factor alpha tests.
   - Heiken Ashi features (special cointegration experiments):
     - Heiken Ashi candles computed for monthly / 3-month / 6-month aggregation (using month-end / custom period).
     - For KMeans, features derived from slope (direction / normalized slope) rather than absolute magnitude.
   - Filter snapshots (Kalman, Wiener, SavGol, SMURF)
     - For each asset at timestamp t, snapshot features include the current filtered signal value, short-term slope of the filter, and normalized residuals.
   - Misc:
     - VWAP approximation when not available: approx_vwap_t = (H_t + L_t + C_t) / 3 used as typical price in place of exact VWAP.
     - Rolling higher-moments: skewness and kurtosis on returns windows.

Math reference (concise)
- Returns: r_t = P_t / P_{t-1} - 1
- Log returns: lr_t = ln(P_t) - ln(P_{t-1})
- EMA: EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}; alpha = 2/(N+1)
- RSI: RS_t = avg_gain_t / avg_loss_t; RSI_t = 100 - 100/(1+RS_t)
- ATR: ATR_t = EMA(TrueRange, N), TrueRange = max(H-L, |H-P_prev|, |L-P_prev|)
- Kalman filter: state-space predict-update (written in notebook code); outputs state estimate and variance — snapshots saved per asset/timestamp
- Savitzky-Golay: polynomial smoothing over window with local polynomial fit.

Files / notebooks referenced (implemented)
- dataset/features/all_features.parquet (feature dataset produced by pipeline)
- notebooks/cointegrated movement/*
- notebooks/alphas/101_alphas_backtests.ipynb (uses indneutralize & VWAP approximations)
- notebooks/random forest models/* (use available features; add feature examples when missing)
- notes: see src/features/core.py and src/cleaning/* for exact implemented formulas and code.

Sources
- (Leave blank here — add citations you want to include)
- Suggested to include: original indicator references, ML4T chapter citations, relevant PDFs: 101_alphas.pdf, finmamba.pdf, hierachial_nueral_network.pdf.

---

2) Backtesting engine
=====================

Overview & architecture
- The backtester accepts signals / weights from model notebooks and runs period-by-period portfolio simulation using historical OHLCV.
- Key modules:
  - engine: orchestration of simulation (apply weights, compute PnL)
  - portfolio: weight normalization, transaction cost & slippage application
  - metrics: compute CAGR, Sharpe, Max Drawdown, drawdown series
  - report: assemble summary table + Bokeh visualization outputs
  - bokeh_plots: produces interactive visualizations saved as HTML

Execution flow (per backtest)
1. Inputs
   - Price series for universe subset (close prices), model-produced signals or weight matrices (date x tickers).
2. Preprocessing
   - Align price & weight matrices; trim initial dates where all weights are zero (fix applied so computed "Start" equals first active testing date).
3. Apply weights → returns
   - Daily / periodic portfolio return r_p,t = sum_{i} w_{i,t-1} * r_{i,t} (weights usually lagged by one period to avoid lookahead).
4. Portfolio accounting
   - Transaction costs applied on weight adjustments: cost = turnover * fee_rate
   - Rebalancing schedule: daily or custom (notebooks use time-split rebalancing or per-strategy setting).
5. Risk & performance metrics
   - Cumulative returns, NAV, drawdown series.

Visualization layer
- Bokeh used to produce interactive charts (equity curve, drawdown, per-asset weight heatmaps).
- Notebooks save Bokeh outputs (HTML) or static PNG depending on notebook settings; the report embeds or links to these.

Key metrics (math + intuition)
- CAGR (annualized return)
  - CAGR = (NAV_end / NAV_start)^(1/years) - 1
  - Intuition: per-year constant return that would produce observed cumulative return.
- Sharpe ratio (annualized)
  - Sharpe = (mean(r_p) / std(r_p)) * sqrt(annualization_factor)
  - Where r_p are periodic returns and mean is excess over risk-free (not always subtracted in code — check notebook config). Intuition: risk-adjusted return per unit of volatility.
- Max Drawdown (MDD)
  - drawdown_t = (NAV_t / max_{s<=t}(NAV_s)) - 1; MDD = min(drawdown_t)
  - Intuition: worst peak-to-trough loss; key risk metric.
- Turnover
  - turnover_t = sum(|w_t - w_{t-1}|) / 2 (or similar); captures trading intensity and affects transaction costs.

Portfolio theory & allocation
- Modern Portfolio Theory (MPT)
  - Asset-level mean returns µ and covariance Σ; efficient frontier solves for weights w that optimize expected return for a given risk or minimize variance for target return: min_w w' Σ w subject to w' µ = R_target, sum w=1.
- Neo-Modern Portfolio Theory / 1/N
  - 1/N: equal-weight portfolio as a robust baseline (less estimation error).
  - NMPT: may incorporate robust optimization, shrinkage or hierarchical risk parity variants.
- Stop-loss & risk overlays
  - Stop-loss implemented as conditional cap on drawdown per asset or portfolio-level rule that triggers liquidation / de-risking.
- Agentic backtester behavior (if implemented)
  - Backtester designed to accept modular components:
    - Macro model (allocation overlays, e.g., regime detection)
    - Micro model (alpha model producing weights / signals)
    - Portfolio manager (rebalancing & constraints)
  - Implementation pattern: chain of components applied in engine; notebooks provide examples wiring these together.

Bug fixes & notable changes
- Start-date correction:
  - The report module now trims leading dates where portfolio weights are all zero; the reported "Start" is the first date with non-zero active weights. This corrects previous mismatch where reports showed 2016 for many strategies where testing started later.

Files / code references
- src/backtester/engine.py
- src/backtester/report.py (start-date trimming)
- src/backtester/metrics.py
- src/backtester/bokeh_plots.py
- notebooks/* (each notebook calls run_backtest or run_portfolio wrapper)

---

3) Models — theory, implementation notes & results
==================================================

Notes on evaluation practice
- Time-split convention: training on the first ~7 years and testing on the last ~1.5 years (18 months) unless the notebook specifies otherwise.
- Cross-sectional normalization: many factor/alpha pipelines use indneutralize to ensure factors capture cross-section relative signals.
- Models are trained to produce signals or class labels which are converted to weights per the backtester’s adapter layer (e.g., score → rank → long/short weights, or probabilistic outputs → expected return based weights).

Implemented model groups and short math/theory summary
- Decision trees / Random forests (Chapter 11)
  - Decision tree splits on features to partition feature space; objective: minimize impurity (Gini/Entropy for classification) or MSE (regression).
  - Random Forest: ensemble of trees trained on bootstrap samples with feature subsampling; final prediction is averaged (regression) or majority-vote (classification).
  - Notebooks: notebooks/random forest models/* (includes time-split variants and cointegration pairs).
- Bagging / Boosting (Chapter 12)
  - Bagging: reduce variance by averaging bootstrapped model predictions.
  - AdaBoost: sequential reweighting of training examples; final predictor is weighted sum of weak learners.
  - Gradient boosting (GBDT): sequentially fit models to residuals (negative gradient).
  - HistGradientBoosting: histogram-based gradient boosting for speed.
  - XGBoost / LightGBM / CatBoost: optimized GBDT implementations with system-specific enhancements (regularization, leaf-wise growth, categorical handling).
  - Notebooks: notebooks/boosted algorithms/*
- CNNs (Chapter 18 / image-like feature grids)
  - 1D CNN for time-series: convolution over time axis to extract local temporal patterns.
  - 2D CNNs over feature-grid: treat feature cross-section as spatial grid; use AdaptiveAvgPool2d to avoid hardcoding flatten sizes.
  - LeNet/AlexNet-like architectures adjusted for small channels and adaptive pooling.
  - Notebooks: notebooks/convolutional neural nets/*
- LSTM / ALSTM
  - LSTM: RNN variant capturing long-range dependencies using gated units.
  - ALSTM: attention-augmented LSTMs (attention layer over sequence).
  - Notebooks: notebooks/hierarchical graph neural network/* (LSTM variants included)
- Graph Neural Networks (GCN / GAT / HGNN)
  - GCN: spectral/spatial message passing aggregated from neighbors via learnable transforms.
  - GAT: attention-weighted neighbor aggregation.
  - HGNN: hierarchical graph neural network as per the provided PDF — implemented variants: HGNN_M, HGNN_I, HGNN_full with different hierarchy construction and pruning strategies. Edge pruning uses top-k significant connections; market-aware graphs included in FinMamba-like architectures.
  - Notebooks: notebooks/hierarchical graph neural network/*
- FinMamba (PDF-inspired implementation)
  - Custom PyTorch module combining SSM-like blocks, graph attention aggregation, and multi-loss (MSE + pairwise ranking + GIB term).
  - Trained for EPOCHS=3 in provided notebook; uses repo feature dataset (ensured).
  - Files: src/models/finmamba.py, notebooks/finmamba/FinMamba_Implementation.ipynb
- 101 Alphas
  - Implementation of formulaic alphas from the 101_alphas.pdf; many alphas use cross-sectional rank, lagged returns, price/volume relationships, and pattern detection.
  - Notebook: notebooks/alphas/101_alphas_backtests.ipynb
- Cointegration & clustering models (separate section below).

Model outputs & results
- The repository contains extracted metric CSV(s):
  - ultimate_report_model_metrics.csv — master CSV with rows: model, notebook_path, CAGR [%], Sharpe, Max Drawdown [%], notes (if available).
  - notebooks/finmamba/stats.csv — finmamba-specific metrics.
- Table template below should be populated from ultimate_report_model_metrics.csv. If you want me to embed exact numbers, provide the CSV or allow me to read it; I will then fill this table and, if provided, embed saved plot images or Bokeh HTML links.

Model performance table (template — populate from CSV)
| Notebook / Model path | Model name | Notes | CAGR [%] | Sharpe | Max Drawdown [%] |
|---|---:|---|---:|---:|---:|
| notebooks/random forest models/04_random_forest_regressor.ipynb | RandomForestRegressor | time-split | [CAGR] | [Sharpe] | [MDD] |
| notebooks/random forest models/07_random_forest_classifier_time_split.ipynb | RF Classifier (time-split) | long-short by probability | [CAGR] | [Sharpe] | [MDD] |
| notebooks/boosted algorithms/02_gradient_boosting_classifier_time_split.ipynb | GradientBoosting | time-split | [CAGR] | [Sharpe] | [MDD] |
| notebooks/convolutional neural nets/03_lenet5_like_2d_cnn_time_split.ipynb | 2D CNN (LeNet-like) | adaptive pooling | [CAGR] | [Sharpe] | [MDD] |
| notebooks/hierarchical graph neural network/11_hgnn_full_time_split.ipynb | HGNN (full) | hierarchical edges | [CAGR] | [Sharpe] | [MDD] |
| notebooks/finmamba/FinMamba_Implementation.ipynb | FinMamba | epochs=3 | [CAGR] | [Sharpe] | [MDD] |
| notebooks/alphas/101_alphas_backtests.ipynb | 101 Alphas (multiple) | results per-alpha in folder | [see per-alpha rows] | | |

Notes:
- Replace bracketed placeholders with CSV values. For the 101 alphas, consider a separate table or attach per-alpha CSV.

---

4) Cointegration experiments
============================

Goals
- Find groups of assets that move together (cointegrated or exhibit similar trend/shape) for pair/trade construction, statistical arbitrage, or portfolio construction.

Approaches implemented
1. Heiken Ashi monthly slope clustering
   - Transform daily OHLC to Heiken Ashi monthly candles (month-end / custom period), compute per-period slope normalized by period length (slope = delta(close) / days).
   - KMeans clustering on slopes across assets — clusters group assets with similar slope direction and profile, not magnitude.
   - Notebooks:
     - notebooks/cointegrated movement/01_kmeans_heiken_ashi_monthly_slope_clusters.ipynb
     - 02_ (3-month) and 03_ (6-month) variants

2. Hierarchical clustering (ML4T Chapter-13 ideas)
   - Build distance matrix based on correlation or shape distance (DTW optional if implemented), apply scipy hierarchical clustering (Ward, average).
   - Visualize dendrogram and cut at desired threshold to produce groups.

3. Density-based & mixture models
   - DBSCAN: clusters dense regions in slope/feature space to isolate similar movers.
   - Gaussian Mixture Models (GMM): fit mixtures and assign posterior cluster responsibilities.

4. Kernel PCA on indicators
   - KernelPCA for nonlinear dimensionality reduction of indicator space; cluster in reduced space to find indicators that co-move assets.
   - Notebook: notebooks/cointegrated movement/05_kernel_pca_indicators_comovement.ipynb

5. Kalman/Wiener/SavGol/SMURF snapshots + clustering
   - Snapshot features (signal value, slope, residual) per asset per timestamp; KMeans/DBSCAN/GMM clustering to find potential cointegrated groups.
   - Notebook: notebooks/cointegrated movement/07_filter_snapshot_clustering.ipynb

6. Engle-Granger (pairwise cointegration) + ADF residual test
   - For candidate pairs/groups, run Engle-Granger cointegration: regress series A on B to get residuals, test residuals with Augmented Dickey-Fuller (ADF) — stationarity implies cointegration.
   - Notebook: notebooks/cointegrated movement/08_engle_granger_adf_cointegration.ipynb

Interpretation & intuition
- Slope-based clustering finds assets with similar growth direction over aggregate periods — useful to pick groups for relative-value or hedged trend strategies.
- Statistical cointegration (EG + ADF) is stronger: implies a stable linear relation between series — suitable for mean-reverting pairs trading.
- Filtering-based snapshots capture latent signals (smoothed price processes); clustering snapshots can reveal assets whose smoothed dynamics align.
- Kernel PCA finds nonlinear shared structure in indicator space that may precede co-movement.

Outputs & visualization
- For each cluster or cointegrated group, notebooks produce a multi-asset price chart with each asset labeled and colored in legend (user requested asset names + colors).
- For pairwise cointegration, show residual series and ADF p-value.

---

Graphs, embedding & instructions
================================

Embedding static images or Bokeh HTML
- Preferred workflow:
  1. From each notebook, export or save Bokeh outputs as HTML to a folder: `reports/bokeh/<notebook-name>.html`.
  2. Export PNG snapshots of Bokeh plots (or of Matplotlib charts) to `reports/plots/<notebook-name>.png`.
  3. In the Markdown file, embed static PNG images using:
     - `![Equity curve - Random Forest](reports/plots/04_random_forest_regressor_equity.png)`
  4. For interactive Bokeh HTML, link to the HTML file:
     - `[Bokeh interactive plot - Random Forest](reports/bokeh/04_random_forest_regressor.html)`
- If you provide a ZIP or attach the `reports/plots` and `reports/bokeh` folders, I will embed images and link HTML in the report automatically.
 - If you provide a ZIP or attach the `reports/plots` and `reports/bokeh` folders, I will embed images and link HTML in the report automatically.

How I will embed metrics if given CSVs
- Provide `ultimate_report_model_metrics.csv` (or allow me to read it). I will:
  - Parse file
  - Fill the Model performance table and per-model subsections
  - Insert small PNG thumbnails next to each table row if corresponding plot PNGs exist

---

Next steps & checklist (actions for you)
========================================
1. Run notebooks that have not been executed and export their backtest outputs (do not change code):
   - Save Bokeh HTML to `reports/bokeh/<notebook>.html`
   - Export PNG thumbnails to `reports/plots/<notebook>_equity.png` and `<notebook>_drawdown.png`
   - Save per-notebook stats CSV to `notebooks/<notebook>/stats.csv` or append to `ultimate_report_model_metrics.csv`
2. Provide the updated `ultimate_report_model_metrics.csv` and the `reports/plots` and `reports/bokeh` folders (or allow me to read them in the repo). When available I will:
   - Fill the model performance table with real numbers
   - Embed plot images and link interactive Bokeh output
3. OPTIONAL: If you want me to produce the Markdown file inside the repo, say “write report” and I will create ULTIMATE_REPORT.md at repo root (no code edits).
4. If you want a minimal table of which notebooks lack saved outputs, tell me “list missing outputs” and I will scan the repo for expected stats files and list missing ones.

Checklist (I will do once you provide outputs)
- [ ] Fill model table with exact metrics
- [ ] Embed PNG thumbnails and link Bokeh HTML pages
- [ ] Run lsp_diagnostics on modified/created files (the Markdown file does not produce diagnostics problems)
- [ ] Deliver final ULTIMATE_REPORT.md in repo (if you request writing)

---

Appendix A — quick math cheatsheet
- CAGR = (NAV_end / NAV_start)^(1/years) - 1
- Periodic Sharpe = mean(R_p - rf) / std(R_p), Annualized Sharpe = Periodic Sharpe * sqrt(N_periods_per_year)
- Drawdown_t = 1 - NAV_t / max_{s<=t} NAV_s
- Max Drawdown = max_t (peak_to_trough_loss)
- EMA_alpha = 2 / (N + 1)

---

Appendix B — Where to find code (implemented)
- Feature extraction and cleaning:
  - src/export_features.py
  - src/features/core.py
  - src/cleaning/*.py
- Backtester:
  - src/backtester/engine.py
  - src/backtester/report.py
  - src/backtester/metrics.py
  - src/backtester/bokeh_plots.py
- Models and notebooks:
  - notebooks/random forest models/
  - notebooks/boosted algorithms/
  - notebooks/convolutional neural nets/
  - notebooks/hierarchical graph neural network/
  - notebooks/cointegrated movement/
  - notebooks/finmamba/
  - notebooks/alphas/

---

End of report content.


### Populated model metrics (from ultimate_report_model_metrics.csv)

| Notebook | Run | CAGR [%] | Sharpe | Max Drawdown [%] |
|---|---|---:|---:|---:|
| notebooks/ML_Linear_Models_01_OLS_Ridge_Lasso.ipynb | Linear Model (ridge) - Original Style 1N | 16.959697 | 0.894865 | -38.488 |
| notebooks/ML_Linear_Models_01_OLS_Ridge_Lasso.ipynb | Linear Model (ridge) - Original Style MPT | 17.731467 | 0.938796 | -34.178475 |
| notebooks/ML_Linear_Models_02_ElasticNet.ipynb | Linear Model (elasticnet) - Original Style 1N | 16.710969 | 0.925644 | -35.774485 |
| notebooks/ML_Linear_Models_02_ElasticNet.ipynb | Linear Model (elasticnet) - Original Style MPT | 14.724528 | 0.79301 | -34.308294 |
| notebooks/ML_Linear_Models_03_BayesianRidge.ipynb | Linear Model (bayesianridge) - Original Style 1N | 15.955106 | 0.849282 | -37.144722 |
| notebooks/ML_Linear_Models_03_BayesianRidge.ipynb | Linear Model (bayesianridge) - Original Style MPT | 16.721692 | 0.892932 | -35.516285 |
| notebooks/ML_Linear_Models_04_TimeSplit_OLS_Ridge_Lasso.ipynb | TimeSplit Linear (ridge) - Original Style 1N (Test Window) | 33.362219 | 1.503062 | -22.607705 |
| notebooks/ML_Linear_Models_04_TimeSplit_OLS_Ridge_Lasso.ipynb | TimeSplit Linear (ridge) - Original Style MPT (Test Window) | 24.474231 | 1.168396 | -27.840206 |
| notebooks/ML_Linear_Models_05_SMC_Indicators_TimeSplit.ipynb | SMC Linear (lasso) - 1/N (Test Window) | 26.235639 | 1.396587 | -18.490968 |
| notebooks/ML_Linear_Models_05_SMC_Indicators_TimeSplit.ipynb | SMC Linear (lasso) - MPT (Test Window) | 11.746236 | 0.76154 | -21.849477 |
| notebooks/ML_Linear_Models_05_SMC_Indicators_TimeSplit.ipynb | SMC Linear (ols) - 1/N (Test Window) | 8.187535 | 0.486278 | -22.598123 |
| notebooks/ML_Linear_Models_05_SMC_Indicators_TimeSplit.ipynb | SMC Linear (ridge) - 1/N (Test Window) | 23.50101 | 1.064191 | -21.478081 |
| notebooks/Original_Style_1N.ipynb | A/D Divergence Strategy | 19.247701 | 1.046417 | -32.49443 |
| notebooks/Original_Style_1N.ipynb | Aroon Strategy | 15.51511 | 0.948653 | -29.783701 |
| notebooks/Original_Style_1N.ipynb | Ichimoku Strategy | 15.507275 | 0.989093 | -25.484508 |
| notebooks/Original_Style_1N.ipynb | MACD Strategy | 15.780972 | 0.958961 | -24.866126 |
| notebooks/Original_Style_1N.ipynb | OBV Divergence Strategy | 18.476995 | 1.020018 | -33.348459 |
| notebooks/Original_Style_1N.ipynb | RSI Strategy | 20.048042 | 1.054984 | -35.106321 |
| notebooks/Original_Style_1N.ipynb | SMA Golden Cross Strategy | 16.510455 | 1.00892 | -32.258407 |
| notebooks/Original_Style_MPT.ipynb | OBV Divergence Strategy | 16.109601 | 0.909969 | -31.23273 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 001 | -10.753469 | -2.109452 | -67.575958 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 002 | -7.090334 | -2.222187 | -53.099686 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 003 | -7.715197 | -2.165121 | -55.04658 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 004 | -9.154082 | -1.662835 | -61.840631 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 005 | -12.744258 | -2.467293 | -74.549016 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 006 | -6.867977 | -1.450781 | -51.301245 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 007 | -18.496057 | -2.915052 | -86.130851 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 008 | -7.535819 | -1.427579 | -53.9769 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 009 | -20.469854 | -3.826114 | -89.990318 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 010 | -17.237871 | -4.22625 | -85.123124 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 011 | -10.939085 | -2.600474 | -68.416009 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 012 | -17.805699 | -3.758271 | -86.684842 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 013 | -7.44725 | -2.504522 | -53.75082 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 014 | -11.607774 | -2.341809 | -71.126833 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 015 | -15.345619 | -2.56349 | -81.209105 |
| notebooks/alphas/101_alphas_backtests.ipynb | Alpha 016 | -9.702411 | -3.3256 | -63.810937 |
| notebooks/bayesian models/01_conjugate_priors_beta_binomial_direction.ipynb | Bayes Conjugate Priors (Beta-Binomial) - 1N | 22.053517 | 1.483627 | -12.921751 |
| notebooks/bayesian models/01_conjugate_priors_beta_binomial_direction.ipynb | Bayes Conjugate Priors (Beta-Binomial) - MPT | 10.841123 | 0.750565 | -13.418566 |
| notebooks/bayesian models/03_bayesian_logistic_regression_mcmc_metropolis.ipynb | Bayes Logistic Regression (MH MCMC) - 1N | 25.092596 | 1.371819 | -18.137274 |
| notebooks/bayesian models/03_bayesian_logistic_regression_mcmc_metropolis.ipynb | Bayes Logistic Regression (MH MCMC) - MPT | 12.051576 | 0.771102 | -23.362502 |
| notebooks/bayesian models/04_bayesian_logistic_regression_variational_inference.ipynb | Bayes Logistic Regression (VI) - 1N | 13.095023 | 0.81309 | -17.262441 |
| notebooks/bayesian models/04_bayesian_logistic_regression_variational_inference.ipynb | Bayes Logistic Regression (VI) - MPT | -0.532981 | 0.057541 | -22.022878 |
| notebooks/bayesian models/05_bayesian_sharpe_ratio_asset_selection.ipynb | Bayes Sharpe Ratio (P[SR&gt;0]) - 1N | 21.413185 | 1.315634 | -13.072936 |
| notebooks/bayesian models/05_bayesian_sharpe_ratio_asset_selection.ipynb | Bayes Sharpe Ratio (P[SR&gt;0]) - MPT | 17.129073 | 1.072677 | -13.69582 |
| notebooks/bayesian models/06_bayesian_rolling_regression_pairs_trading.ipynb | Bayes Rolling Regression (Pairs) - Long/Short | -9.86604 | -1.091691 | -17.012709 |
| notebooks/bayesian models/07_stochastic_volatility_particle_filter.ipynb | Stochastic Volatility (Particle Filter) - 1N | 27.727263 | 1.684742 | -12.852631 |
| notebooks/bayesian models/07_stochastic_volatility_particle_filter.ipynb | Stochastic Volatility (Particle Filter) - MPT | 20.139548 | 1.222446 | -15.593907 |
| notebooks/bayesian models/08_bayesian_logistic_regression_mcmc_nuts.ipynb | Bayes Logistic Regression (NUTS) - 1N | 25.433826 | 1.269686 | -19.680988 |
| notebooks/bayesian models/08_bayesian_logistic_regression_mcmc_nuts.ipynb | Bayes Logistic Regression (NUTS) - MPT | 14.223194 | 0.800078 | -25.489476 |
| notebooks/bayesian models/feature_dataset/01_conjugate_priors_beta_binomial_direction_features.ipynb | Conjugate Priors (Feature Store) - 1N | 21.145548 | 1.433001 | -12.921751 |
| notebooks/bayesian models/feature_dataset/01_conjugate_priors_beta_binomial_direction_features.ipynb | Conjugate Priors (Feature Store) - MPT | 8.901471 | 0.637857 | -13.418566 |
| notebooks/boosted algorithms/01_adaboost_classifier_time_split.ipynb | AdaBoostClassifier (Time Split) - Weekly Top-K Long-Only | 12.567775 | 0.769042 | -15.924873 |
| notebooks/boosted algorithms/02_gradient_boosting_classifier_time_split.ipynb | GradientBoostingClassifier (Time Split) - Weekly Top-K Long-Only | 15.845085 | 0.880362 | -14.199231 |
| notebooks/convolutional neural nets/01_autoregressive_1d_cnn_time_split.ipynb | Autoregressive 1D CNN (Conv1d) - Time Split | 4.33095 | 0.318121 | -19.777488 |
| notebooks/convolutional neural nets/02_cnn_ta_small_2d_cnn_time_split.ipynb | CNN-TA Small 2D CNN (Conv2d) - Time Split | 15.466812 | 0.833904 | -17.801731 |
| notebooks/hierarchical graph neural network/05_lstm_time_split.ipynb | LSTM - Time Split | 18.338928 | 0.91287 | -19.959862 |
| notebooks/hierarchical graph neural network/09_hgnn_m_time_split.ipynb | HGNN_M (node+macro) - Time Split | 10.03192 | 0.592807 | -18.986846 |
| notebooks/hierarchical graph neural network/10_hgnn_i_time_split.ipynb | HGNN_I (node+relation) - Time Split | 18.145175 | 0.882607 | -19.594495 |
| notebooks/hierarchical graph neural network/11_hgnn_full_time_split.ipynb | HGNN (node+relation+macro) - Time Split | 15.016678 | 0.74381 | -20.126466 |
| notebooks/random forest models/01_decision_tree_regressor.ipynb | Backtest Report | 16.234722 | 0.90789 | -34.359566 |
| notebooks/random forest models/02_decision_tree_classifier.ipynb | Backtest Report | 17.875911 | 1.018193 | -29.800315 |
| notebooks/random forest models/03_bagged_decision_trees_regressor.ipynb | Backtest Report | 17.691423 | 0.99182 | -29.624449 |
| notebooks/random forest models/04_random_forest_regressor.ipynb | Backtest Report | 17.082073 | 0.963911 | -31.46595 |
| notebooks/random forest models/05_random_forest_classifier.ipynb | Backtest Report | 17.039264 | 0.977184 | -30.686674 |
| notebooks/random forest models/06_decision_tree_classifier_time_split.ipynb | Backtest Report | 12.767687 | 0.723665 | -16.727732 |
| notebooks/random forest models/07_random_forest_classifier_time_split.ipynb | Backtest Report | 19.700468 | 0.987652 | -13.275442 |
| notebooks/random forest models/08_random_forest_cointegration_pairs_time_split.ipynb | RF Cointegration Pairs - Backtest Report | 2.071068 | 0.249845 | -8.610758 |
