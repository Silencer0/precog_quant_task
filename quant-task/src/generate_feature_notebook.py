import nbformat as nbf
import os


def create_feature_notebook() -> None:
    nb = nbf.v4.new_notebook()

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "# Quant Research Notebook: Daily Feature Extraction for OHLCV Stock Data\n"
            "This notebook extracts daily model-ready features from cleaned OHLCV data.\n"
            "Important: the cleaned dataset in `dataset/cleaned/` is *not* smoothed/filtered. Any filtering methods are used only as *features* and are exported separately to `dataset/features/`.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## Feature Catalog (Exact Columns Exported)\n\n"
            "This notebook exports a large set of *parameterized* features (multiple windows/periods).\n"
            "To satisfy the research requirement of documenting *every extracted feature*, the following table programmatically lists the exact exported column names and attaches the matching math definition and parameter values.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "# Build a feature catalog from the exporter logic\n"
            "import re\n"
            "\n"
            "FEATURE_DOCS = []\n"
            "\n"
            "def add_doc(name, family, math, notes=''):\n"
            "    FEATURE_DOCS.append({'feature': name, 'family': family, 'math': math, 'notes': notes})\n"
            "\n"
            "# Returns\n"
            "add_doc('ret_1d', 'returns', 'R_t = C_t/C_{t-1} - 1')\n"
            "add_doc('logret_1d', 'returns', 'r_t = ln(C_t/C_{t-1})')\n"
            "add_doc('excess_ret_1d', 'returns', 'R_t^{excess} = R_t - r_{f,t}')\n"
            "\n"
            "# Lags\n"
            "add_doc('logret_lag_1', 'lag', 'logret_{t-1}')\n"
            "add_doc('logret_lag_5', 'lag', 'logret_{t-5}')\n"
            "add_doc('ret_lag_1', 'lag', 'ret_{t-1}')\n"
            "add_doc('ret_lag_5', 'lag', 'ret_{t-5}')\n"
            "\n"
            "# Multi-horizon returns\n"
            "add_doc('ret_5d', 'returns', 'C_t/C_{t-5} - 1')\n"
            "add_doc('ret_21d', 'returns', 'C_t/C_{t-21} - 1')\n"
            "add_doc('logret_5d', 'returns', 'ln(C_t) - ln(C_{t-5})')\n"
            "add_doc('logret_21d', 'returns', 'ln(C_t) - ln(C_{t-21})')\n"
            "\n"
            "# Cumulative returns\n"
            "add_doc('cumret_5d', 'cumulative return', 'prod_{i=0..4}(1+R_{t-i}) - 1')\n"
            "add_doc('cumret_21d', 'cumulative return', 'prod_{i=0..20}(1+R_{t-i}) - 1')\n"
            "\n"
            "# Differencing on log price\n"
            "add_doc('diff_log_close_1', 'differencing', 'Delta ln(C_t) = ln(C_t) - ln(C_{t-1})')\n"
            "add_doc('diff_log_close_2', 'differencing', 'Delta^2 ln(C_t) = Delta(Delta ln(C_t))')\n"
            "\n"
            "# Rolling statistics on log returns\n"
            "for w in (5, 10, 20, 60):\n"
            "    add_doc(f'logret_roll_mean_{w}', 'rolling mean', f'mu_t^(w) = (1/w) * sum_{i=0..w-1} r_{t-i}')\n"
            "    add_doc(f'logret_roll_var_{w}', 'rolling variance', f'sigma_t^2(w) = (1/w) * sum (r - mu)^2')\n"
            "    add_doc(f'logret_roll_std_{w}', 'rolling std', f'sigma_t(w) = sqrt(sigma_t^2(w))')\n"
            "    add_doc(f'logret_roll_min_{w}', 'rolling min', f'min(r_{t-w+1..t})')\n"
            "    add_doc(f'logret_roll_max_{w}', 'rolling max', f'max(r_{t-w+1..t})')\n"
            "\n"
            "# Volume rolling stats + z-scores\n"
            "add_doc('log_volume', 'transform', 'ln(V_t)')\n"
            "for w in (5, 20, 60):\n"
            "    add_doc(f'volume_roll_mean_{w}', 'rolling mean', f'mu_V,t^(w) = (1/w) * sum V_{t-i}')\n"
            "    add_doc(f'volume_roll_std_{w}', 'rolling std', f'sigma_V,t^(w)')\n"
            "    add_doc(f'volume_zscore_{w}', 'z-score', f'z_t = (V_t - mu_V,t^(w)) / sigma_V,t^(w)')\n"
            "\n"
            "# Scaling / normalization\n"
            "add_doc('logret_zscore_20', 'z-score', 'z_t = (r_t - mu_t^(20)) / sigma_t^(20)')\n"
            "add_doc('close_minmax_20', 'min-max scaling', '(C_t - min_t^(20)) / (max_t^(20) - min_t^(20))')\n"
            "add_doc('volume_minmax_20', 'min-max scaling', '(V_t - min_V,t^(20)) / (max_V,t^(20) - min_V,t^(20))')\n"
            "\n"
            "# Volatility\n"
            "add_doc('atr_14', 'ATR', 'TR_t = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|); ATR = SMA_14(TR)')\n"
            "add_doc('realized_vol_20', 'realized volatility', 'sqrt(252) * Std_{20}(logret_1d)')\n"
            "\n"
            "# Technical indicators\n"
            "add_doc('rsi_14', 'RSI', 'RSI = 100 - 100/(1+RS), RS=AvgGain/AvgLoss (Wilder)')\n"
            "add_doc('sma_ratio_20', 'SMA (paper)', 'SMAhat = C_t / SMA_20(C)_t (arXiv:2412.15448 Eq. 5)')\n"
            "add_doc('ema_ratio_20', 'EMA (paper)', 'EMAhat = C_t / EMA_20(C)_t (arXiv:2412.15448 Eq. 7)')\n"
            "for c in ('macd', 'macd_signal', 'macd_hist'):\n"
            "    add_doc(f'macd_{c}', 'MACD', 'MACD=EMA12-EMA26; Signal=EMA9(MACD); Hist=MACD-Signal')\n"
            "add_doc('rmacd_12_26_9', 'MACD (paper)', 'rMACD=(MACD-SIG)/(0.5*(|MACD|+|SIG|)) (arXiv:2412.15448 Eq. 9)')\n"
            "for c in ('bb_mid', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'bb_percent_b'):\n"
            "    add_doc(f'bb_{c}', 'Bollinger', 'Mid=SMA20; Upper/Lower=Mid +/- 2*Std20; %B=(C-L)/(U-L)')\n"
            "add_doc('roc_10', 'ROC', 'ROC_t = C_t/C_{t-10} - 1')\n"
            "for c in ('stoch_k', 'stoch_d'):\n"
            "    add_doc(f'stoch_{c}', 'stochastic', '%K = 100*(C-LL14)/(HH14-LL14); %D=SMA3(%K)')\n"
            "add_doc('cci_20', 'CCI (paper)', 'CCI=(p-SMA(p))/(0.015*MAD) with p=(H+L+C)/3 (arXiv:2412.15448 Eq. 19)')\n"
            "\n"
            "# Volume indicators\n"
            "add_doc('obv', 'OBV', 'OBV_t = OBV_{t-1} +/- V_t based on sign(C_t-C_{t-1})')\n"
            "add_doc('obv_roc_10', 'OBV ROC', 'ROC_t = OBV_t/OBV_{t-10} - 1')\n"
            "add_doc('wrobv_20', 'WROBV (paper)', 'WROBV=sum_{i=0..19}OBV_{t-i}/sum_{i=0..19}V_{t-i} (arXiv:2412.15448 Eq. 17)')\n"
            "add_doc('ad_line', 'Accumulation/Distribution', 'CLV=((C-L)-(H-C))/(H-L); AD_t=AD_{t-1}+CLV*V')\n"
            "\n"
            "# Trend indicators\n"
            "for c in ('plus_di', 'minus_di', 'adx', 'adx_raw'):\n"
            "    add_doc(f'adx_{c}', 'ADX', '+DI=100*Sm(+DM)/ATR; -DI=100*Sm(-DM)/ATR; ADX=Sm(DX)')\n"
            "for c in ('aroon_up', 'aroon_down'):\n"
            "    add_doc(f'aroon_{c}', 'Aroon', 'AroonUp=100*(N-days_since_HH)/N; AroonDown=100*(N-days_since_LL)/N')\n"
            "\n"
            "# Ichimoku\n"
            "for c in ('ichimoku_conv', 'ichimoku_base', 'ichimoku_span_a', 'ichimoku_span_b', 'ichimoku_lagging'):\n"
            "    add_doc(f'ichimoku_{c}', 'Ichimoku', 'Conv=(HH9+LL9)/2; Base=(HH26+LL26)/2; spans shifted +/-26')\n"
            "\n"
            "# Fibonacci levels\n"
            "for c in ('fib_swing_high', 'fib_swing_low', 'fib_23_6', 'fib_38_2', 'fib_50_0', 'fib_61_8', 'fib_76_4'):\n"
            "    add_doc(f'fib_{c}', 'Fibonacci', 'Levels derived from rolling swing high/low: H - p*(H-L)')\n"
            "add_doc('fib_retr_60', 'Fibonacci (paper)', 'R(t)=(H_60(t)-C_t)/(H_60(t)-L_60(t)) (arXiv:2412.15448 Eq. 14)')\n"
            "\n"
            "# Filter-based features (cleaning-pipeline filters used as features)\n"
            "FILTERS = [\n"
            "    'sma_5', 'ema_12', 'wma_5', 'savgol_11_2', 'kalman_r0p1_q0p01', 'wiener_5',\n"
            "    'spectral_0p1', 'lms_mu1e-4_taps5', 'lattice_demo'\n"
            "]\n"
            "for f in FILTERS:\n"
            "    add_doc(f'filt_close_{f}', 'filter feature', 'Filtered close: C~_t')\n"
            "    add_doc(f'filt_resid_{f}', 'filter feature', 'Residual: e_t = C_t - C~_t')\n"
            "    add_doc(f'filt_logret_{f}', 'filter feature', 'Filtered log return: ln(C~_t/C~_{t-1})')\n"
            "\n"
            "# Raw context columns\n"
            "add_doc('close', 'raw context', 'C_t')\n"
            "add_doc('volume', 'raw context', 'V_t')\n"
            "\n"
            "catalog = pd.DataFrame(FEATURE_DOCS).sort_values(['family', 'feature']).reset_index(drop=True)\n"
            "catalog\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "import sys\nimport os\nimport pandas as pd\nimport numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "sys.path.append(os.path.abspath('..'))\n"
            "# If you re-run this notebook after editing feature code, reload modules\n"
            "import importlib\n"
            "import src.cleaning.ts_models as ts_models\n"
            "import src.export_features as export_features\n"
            "importlib.reload(ts_models)\n"
            "importlib.reload(export_features)\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## Input Data Assumptions (Daily OHLCV)\n"
            "We assume each asset has daily rows with columns: `Open`, `High`, `Low`, `Close`, `Volume`, indexed by `Date`.\n"
            "The cleaned dataset in `dataset/cleaned/` is produced via imputation-only steps (no smoothing filters).\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "sample = pd.read_csv('../dataset/cleaned/Asset_001.csv', parse_dates=['Date']).set_index('Date')\n"
            "sample.head()\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## Feature Families\n"
            "Each feature below includes: (1) intuition, (2) math definition, (3) implementation notes for daily data.\n"
            "All features are computed per asset per day, and exported to `dataset/features/`.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## Convention: Notation and Data\n\n"
            "We use daily OHLCV notation:\n"
            "- $O_t, H_t, L_t, C_t$ = open, high, low, close on day $t$\n"
            "- $V_t$ = traded volume on day $t$\n"
            "- Returns are derived from $C_t$ unless stated otherwise.\n\n"
            "Most rolling features use a trailing window of length $w$ and are undefined (NaN) until $w$ observations exist.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 1) Return Features (Simple, Log, Excess)\n\n"
            "### Intuition\n"
            "Raw prices are not comparable across assets. Returns normalize by the price level: a $5$ move on a $10$ stock is large, but on a $1000$ stock is small.\n\n"
            "### Math\n"
            "- Simple return: $R_t = \\frac{C_t}{C_{t-1}} - 1$\n"
            "- Log return: $r_t = \\ln\\left(\\frac{C_t}{C_{t-1}}\\right) = \\ln(C_t) - \\ln(C_{t-1})$\n"
            "- Excess return: $R_t^{excess} = R_t - r_{f,t}$ where $r_{f,t}$ is the per-day risk-free rate.\n\n"
            "### Notes\n"
            "- Log returns are additive over time, which helps with modeling and aggregation.\n"
            "- Excess returns isolate compensation over the risk-free baseline.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 2) Lag Features (Momentum)\n\n"
            "### Intuition\n"
            "Momentum effects suggest recent returns can contain predictive information. Lagged returns give the model direct access to recent history.\n\n"
            "### Math\n"
            "For a series $x_t$, the $k$-lag feature is: $x_{t-k}$.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 3) Rolling Statistics (Mean/Var/Min/Max/Std)\n\n"
            "### Intuition\n"
            "Rolling statistics summarize local behavior: trend (mean), dispersion (variance/std), and extremes (min/max).\n\n"
            "### Math\n"
            "For a trailing window of length $w$:\n"
            "- Rolling mean: $\\mu_t^{(w)} = \\frac{1}{w}\\sum_{i=0}^{w-1} x_{t-i}$\n"
            "- Rolling variance: $\\sigma_t^{2,(w)} = \\frac{1}{w}\\sum_{i=0}^{w-1}(x_{t-i}-\\mu_t^{(w)})^2$\n"
            "- Rolling min/max: $\\min_{i=0..w-1} x_{t-i}$, $\\max_{i=0..w-1} x_{t-i}$\n\n"
            "### Notes\n"
            "Initial periods have NaNs until enough history accumulates.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 4) Differencing (Trend Removal on Semilog Scale)\n\n"
            "### Intuition\n"
            "Differencing removes slow-moving trends and makes a series more stationary. On a semilog plot, differencing log-price corresponds to log returns.\n\n"
            "### Math\n"
            "- First difference: $\\Delta x_t = x_t - x_{t-1}$\n"
            "- Second difference: $\\Delta^2 x_t = (x_t-x_{t-1}) - (x_{t-1}-x_{t-2})$\n"
            "Applied to log price: $\\Delta \\ln(C_t) = \\ln(C_t/C_{t-1})$\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 5) Rolling Z-Scores for Volume\n\n"
            "### Intuition\n"
            "A volume z-score tells how unusual today's activity is compared to the recent past (liquidity and attention proxy).\n\n"
            "### Math\n"
            "$z_t = \\frac{V_t - \\mu_t^{(w)}}{\\sigma_t^{(w)}}$ where $\\mu_t^{(w)}$ and $\\sigma_t^{(w)}$ are rolling mean/std of volume over window $w$.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 6) Volatility Indicators (Rolling Vol, ATR)\n\n"
            "### Intuition\n"
            "Volatility measures dispersion of price moves and is central to risk-based strategies. ATR uses OHLC to quantify daily range volatility.\n\n"
            "### Math\n"
            "- Realized volatility (annualized): $\\sigma_{t}^{(w)} = \\sqrt{252}\\cdot \\text{Std}(r_{t-w+1:t})$\n"
            "- True Range: $TR_t = \\max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)$\n"
            "- ATR: $ATR_t^{(w)} = \\frac{1}{w}\\sum_{i=0}^{w-1} TR_{t-i}$\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 7) Technical + Momentum Indicators\n\n"
            "This section covers common technical indicators used as features: MACD, RSI, Bollinger Bands, ROC, and Stochastic Oscillator.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "### MACD\n"
            "- MACD line: $MACD_t = EMA_{12}(C)_t - EMA_{26}(C)_t$\n"
            "- Signal line: $Signal_t = EMA_{9}(MACD)_t$\n"
            "- Histogram: $Hist_t = MACD_t - Signal_t$\n\n"
            "EMA recursion: $EMA_t = \\alpha C_t + (1-\\alpha)EMA_{t-1}$ with $\\alpha=2/(n+1)$.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "### RSI (14)\n"
            "RSI measures the balance of recent gains vs losses.\n\n"
            "- $RS = \\frac{AvgGain}{AvgLoss}$\n"
            "- $RSI = 100 - \\frac{100}{1+RS}$\n\n"
            "Wilder smoothing is implemented via exponential smoothing with $\\alpha=1/14$.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "### Bollinger Bands (20, 2)\n"
            "- Mid: $Mid_t = SMA_{20}(C)_t$\n"
            "- Upper/Lower: $Mid_t \\pm 2\\cdot Std_{20}(C)_t$\n"
            "Also exported: Bandwidth and %B position within bands.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "### ROC (Rate of Change)\n$ROC_t^{(n)} = \\frac{C_t}{C_{t-n}} - 1$\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "### Stochastic Oscillator (14,3)\n"
            "$\\%K_t = 100\\cdot \\frac{C_t - L^{(14)}_t}{H^{(14)}_t - L^{(14)}_t}$, where $H^{(14)}_t$ is rolling highest-high and $L^{(14)}_t$ is rolling lowest-low.\n"
            "$\\%D_t = SMA_3(\\%K)_t$.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 8) Volume-Based Indicators (OBV, A/D Line)\n\n"
            "### OBV Intuition\n"
            "Volume can precede price. OBV accumulates volume conditioned on price direction.\n\n"
            "### OBV Math\n"
            "Let $OBV_0 = 0$. For $t>0$:\n"
            "- if $C_t > C_{t-1}$: $OBV_t = OBV_{t-1} + V_t$\n"
            "- if $C_t < C_{t-1}$: $OBV_t = OBV_{t-1} - V_t$\n"
            "- else: $OBV_t = OBV_{t-1}$\n\n"
            "We also export OBV rate-of-change because absolute OBV magnitude is less meaningful.\n\n"
            "### Divergence (OBV)\n"
            "- Price rising while OBV falls/plateaus: potential distribution (selling pressure).\n"
            "- Price falling/sideways while OBV rises: potential accumulation (buying pressure).\n\n"
            "### Limitations (OBV)\n"
            "- A single day of extreme volume can dominate the series.\n"
            "- OBV uses direction only; magnitude of price moves is ignored.\n"
            "- Best used alongside other indicators (e.g., RSI/MFI).\n\n"
            "### Accumulation/Distribution (A/D) Line\n"
            "Close Location Value: $CLV_t = \\frac{(C_t-L_t)-(H_t-C_t)}{H_t-L_t}$\n"
            "Money Flow Volume: $MFV_t = CLV_t\\cdot V_t$\n"
            "A/D line: $AD_t = \\sum_{i\\le t} MFV_i$\n"
            "\n"
            "### Trend Confirmation (A/D)\n"
            "- If price rises and A/D rises, buying pressure supports the uptrend.\n"
            "- If price falls but A/D rises, buyers may be stepping in (potential reversal).\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 9) Trend Strength (ADX, Aroon)\n\n"
            "### ADX\n"
            "ADX measures trend strength (non-directional). We export +DI, -DI, and ADX.\n\n"
            "Directional movement: $+DM_t = H_t - H_{t-1}$, $-DM_t = L_{t-1} - L_t$ with standard gating rules.\n"
            "True range and ATR normalize these to +DI/-DI, then $DX_t = 100\\cdot \\frac{|+DI_t - -DI_t|}{|+DI_t + -DI_t|}$, and $ADX$ is a smoothed DX.\n\n"
            "Interpretation (common heuristic):\n"
            "- $ADX < 20$: weak/absent trend\n"
            "- $ADX > 25$: trend strength increasing\n\n"
            "### Aroon (25)\n"
            "AroonUp: $100\\cdot \\frac{N - \\text{periods since highest high}}{N}$\n"
            "AroonDown: $100\\cdot \\frac{N - \\text{periods since lowest low}}{N}$\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 12) Filter-Based Features (From the Cleaning Pipeline)\n\n"
            "### Why include filters as *features*?\n"
            "Filtering can create alternative views of the same price process (trend + residual decomposition). In quant modeling, we often feed both the filtered signal and the residual (original minus filtered) to let the model learn which regime benefits from filtering.\n\n"
            "Important: For stock data, aggressive filtering can blur true jumps/gaps and may degrade predictive features. This notebook keeps filtering strictly in the feature layer; the cleaned dataset is not filtered.\n\n"
            "### Features exported\n"
            "For each filter we export three series:\n"
            "- Filtered close: $\\tilde{C}_t$\n"
            "- Residual: $e_t = C_t - \\tilde{C}_t$\n"
            "- Filtered log return: $\\ln(\\tilde{C}_t/\\tilde{C}_{t-1})$\n\n"
            "### Filters included (and math)\n"
            "1) SMA (5): $\\tilde{C}_t = \\frac{1}{k}\\sum_{i=-(k-1)/2}^{(k-1)/2} C_{t+i}$ (centered window in implementation).\n"
            "2) EMA (12): $\\tilde{C}_t = \\alpha C_t + (1-\\alpha)\\tilde{C}_{t-1}$, $\\alpha=2/(k+1)$.\n"
            "3) WMA (5): $\\tilde{C}_t = \\frac{\\sum_{i=1}^k w_i C_{t-k+i}}{\\sum_{i=1}^k w_i}$ with linear weights $w_i=i$.\n"
            "4) Savitzky-Golay (11,2): fit a degree-2 polynomial over a local window via least squares and evaluate at the center point.\n"
            "5) 1D Kalman filter: state model $x_t = x_{t-1} + w_t$, observation $z_t = x_t + v_t$; update via Kalman gain $K_t$.\n"
            "6) Wiener filter: minimum MSE linear filter; frequency-domain form $H(f)=S_{xx}(f)/(S_{xx}(f)+S_{nn}(f))$.\n"
            "7) Spectral low-pass (FFT): zero high-frequency coefficients and inverse FFT to recover a low-pass signal.\n"
            "8) LMS adaptive filter: $w_{t+1}=w_t + 2\\mu e_t x_t$ (online gradient descent).\n"
            "9) Lattice filter (demo): recursive forward/backward prediction error update using reflection coefficients $k_m$.\n"
            "\n"
            "AR/ARMA model-based filters are excluded because they often emit convergence/non-stationarity warnings on stock series.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 10) Ichimoku Cloud\n\n"
            "We export conversion, base, leading spans A/B (shifted forward), and lagging span (shifted back).\n"
            "- Conversion (9): $(HH_9 + LL_9)/2$\n"
            "- Base (26): $(HH_{26} + LL_{26})/2$\n"
            "- Span A: (Conversion + Base)/2 shifted +26\n"
            "- Span B: $(HH_{52} + LL_{52})/2$ shifted +26\n"
            "- Lagging: Close shifted -26\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## 11) Fibonacci Retracement Levels (Rolling Swing)\n\n"
            "Using a rolling swing high/low over a window, we compute retracement levels: 38.2%, 50%, 61.8%.\n"
            "If $H^{(w)}_t$ is rolling high and $L^{(w)}_t$ rolling low, range $R_t = H^{(w)}_t - L^{(w)}_t$.\n"
            "Then level 61.8%: $H^{(w)}_t - 0.618\\cdot R_t$ (and similarly for 50%, 38.2%).\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_markdown_cell(
            "## Export Features (All Assets)\n"
            "The exporter writes per-asset daily feature CSVs and a combined file:\n"
            "- `dataset/features/Asset_XXX.csv`\n"
            "- `dataset/features/all_features.parquet` (if `pyarrow` or `fastparquet` is installed)\n"
            "- `dataset/features/all_features.csv` (fallback if no Parquet engine is available)\n\n"
            "Risk-free rate is parameterized; default is 0.0 unless you set it explicitly.\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "from src.export_features import export_feature_data\n"
            "export_feature_data(cleaned_dir='../dataset/cleaned', output_dir='../dataset/features', risk_free_rate_annual=0.0)\n"
        )
    )

    nb.cells.append(
        nbf.v4.new_code_cell(
            "feat_sample = pd.read_csv('../dataset/features/Asset_001.csv', parse_dates=['Date']).set_index('Date')\n"
            "feat_sample.head()\n"
        )
    )

    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/Feature_Extraction.ipynb", "w") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    create_feature_notebook()
