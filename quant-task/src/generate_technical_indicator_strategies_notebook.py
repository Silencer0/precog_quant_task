import json
import os
from typing import Any, Dict, List, Tuple


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

    # --- 1. Header ---
    cells.append(
        _md(
            "# Research Notebook: Baseline Performance of Technical Indicator Strategies\n"
            "This notebook evaluates *pure technical indicator* strategies using the local OHLCV dataset under `dataset/cleaned/`.\n"
            "\n"
            "Constraints (as requested):\n"
            "- No ML models\n"
            "- Run the technical indicators with standard (‘as-is’) trading interpretations\n"
            "- Produce a separate equity chart for each indicator strategy\n"
        )
    )

    # --- 2. Setup ---
    cells.append(
        _md(
            "## 0) Setup (Local Data Only)\n"
            "We explicitly load local CSVs using `load_cleaned_assets()`; no live market data is used.\n"
        )
    )

    cells.append(
        _code(
            "import os, sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "from bokeh.io import output_notebook, show\n"
            "\n"
            "def find_project_root(*, max_up: int = 8) -> str:\n"
            "    here = os.getcwd()\n"
            "    for _ in range(max_up + 1):\n"
            "        has_data = os.path.isdir(os.path.join(here, 'dataset', 'cleaned'))\n"
            "        if has_data:\n"
            "            return here\n"
            "        parent = os.path.dirname(here)\n"
            "        if parent == here:\n"
            "            break\n"
            "        here = parent\n"
            "    raise RuntimeError(\n"
            "        'Could not locate project root (expected dataset/cleaned). '\n"
            "        f'cwd={os.getcwd()!r}'\n"
            "    )\n"
            "\n"
            "ROOT = find_project_root()\n"
            "\n"
            "# Ensure our project imports resolve regardless of Jupyter working directory\n"
            "sys.path.insert(0, ROOT)\n"
            "print('Project root:', ROOT)\n"
            "\n"
            "from src.backtester.data import load_cleaned_assets, align_close_prices\n"
            "from src.backtester.engine import BacktestConfig, run_backtest\n"
            "from src.backtester.metrics import compute_performance_stats\n"
            "from src.backtester.bokeh_plots import build_interactive_portfolio_layout\n"
            "from src.backtester.report import compute_backtest_report\n"
            "\n"
            "from src.features.core import (\n"
            "    adx, aroon, atr, bollinger_bands, cci, ema, ema_ratio, fib_retracement_ratio,\n"
            "    ichimoku, macd, obv, accumulation_distribution_line, roc, rsi, sma, sma_ratio,\n"
            "    stochastic_oscillator,\n"
            ")\n"
            "\n"
            "output_notebook()\n"
        )
    )

    cells.append(
        _md(
            "## 0.1) Backtest Parameters\n"
            "Change these to test different assumptions:\n"
            "- `INITIAL_CAPITAL`: Starting equity\n"
            "- `TX_COST_BPS`: Transaction costs in basis points (applied to turnover)\n"
            "- `REBALANCE_MODE`: Rebalancing logic\n"
            "  - `'D'`: Daily (Equal-Weight Target)\n"
            "  - `'W'`: Weekly\n"
            "  - `None`: Buy-and-Hold (Drift) until signal changes\n"
            "- `ENGINE_MODE`: Backtester Engine Type\n"
            "  - `'event_driven'`: Realistic (Unit/Cash tracking, precise costs)\n"
            "  - `'vectorized'`: Fast (Weight-based approximation)\n"
            "- `NO_SELL_LOGIC`: If True, do not sell winners to fund new buys. Only sell if signal exits.\n"
        )
    )

    cells.append(
        _code(
            "INITIAL_CAPITAL = 1_000_000\n"
            "TX_COST_BPS = 5\n"
            "REBALANCE_MODE = None\n"
            "ENGINE_MODE = 'event_driven'\n"
            "STRICT_SIGNALS = True\n"
            "STOP_LOSS_PCT = 0.15  # 15% Stop Loss to reduce noise\n"
            "\n"
            "cfg = BacktestConfig(\n"
            "    initial_equity=INITIAL_CAPITAL,\n"
            "    transaction_cost_bps=TX_COST_BPS,\n"
            "    rebalance=REBALANCE_MODE,\n"
            "    mode=ENGINE_MODE,\n"
            "    strict_signals=STRICT_SIGNALS,\n"
            "    stop_loss_pct=STOP_LOSS_PCT\n"
            ")\n"
        )
    )

    # --- 3. Data Loading ---
    cells.append(
        _md(
            "## 1) Load Data\n"
            "We run multi-asset strategies by computing signals per asset and equally-weighting active long positions.\n"
        )
    )

    cells.append(
        _code(
            "# Load ALL assets available in the dataset directory\n"
            "SYMBOLS = None\n"
            "assets = load_cleaned_assets(symbols=SYMBOLS)\n"
            "close = align_close_prices(assets)\n"
            "\n"
            "# Prepare OHLC for Intraday Logic\n"
            "def _get_col(assets, col):\n"
            "    return pd.concat([df[col].astype(float).rename(s) for s, df in assets.items()], axis=1).sort_index()\n"
            "\n"
            "opens = _get_col(assets, 'Open')\n"
            "highs = _get_col(assets, 'High')\n"
            "lows = _get_col(assets, 'Low')\n"
            "\n"
            "print(f'Loaded {len(assets)} assets with OHLC.')\n"
            "\n"
            "# Calculate Equal-Weight Benchmark (Long Only Buy & Hold)\n"
            "ew_weights = pd.DataFrame(1.0 / len(assets), index=close.index, columns=close.columns)\n"
            "benchmark_res = run_backtest(close, ew_weights, BacktestConfig(initial_equity=INITIAL_CAPITAL, transaction_cost_bps=0))\n"
            "benchmark_stats = compute_performance_stats(equity=benchmark_res.equity, returns=benchmark_res.returns)\n"
            "print(f'Benchmark Return: {benchmark_stats.total_return*100:.2f}%')\n"
            "close.tail()\n"
        )
    )

    # --- 4. Helper Functions ---
    cells.append(
        _md(
            "## 2) Utilities\n"
            "Helper functions for signal conversion and plotting.\n"
        )
    )

    cells.append(
        _code(
            "def signals_to_long_only_weights(signals: pd.DataFrame, max_assets: int = 20) -> pd.DataFrame:\n"
            "    # Fixed fractional allocation: Each active signal gets 1/max_assets allocation.\n"
            "    # This prevents the 'Winner Takes All' problem where the first signal consumes 100% capital.\n"
            "    # It also ensures weights don't fluctuate daily (refactoring) based on executing pairs.\n"
            "    alloc_per_asset = 1.0 / max_assets\n"
            "    \n"
            "    # 1. Capture explicit exits and entries\n"
            "    # Hysteresis signals: 1.0 (Long), -1.0 (Exit), 0.0 (Hold/None)\n"
            "    w = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)\n"
            "    \n"
            "    # 2. Assign Fixed Weights\n"
            "    w[signals > 0.5] = alloc_per_asset\n"
            "    \n"
            "    # 3. Explicit Exits (Signal < -0.5 -> Weight = -1.0 to trigger Close)\n"
            "    w[signals < -0.5] = -1.0\n"
            "    \n"
            "    return w\n"
            "\n"
            "def hysteresis_position(enter: pd.Series, exit: pd.Series) -> pd.Series:\n"
            "    \"\"\"Latches a signal state: 1.0 for Long, -1.0 for Out/Exit.\"\"\"\n"
            "    pos = pd.Series(0.0, index=enter.index)\n"
            "    current = -1.0 # Start out of market\n"
            "    for dt in enter.index:\n"
            "        if bool(exit.loc[dt]):\n"
            "            current = -1.0\n"
            "        elif bool(enter.loc[dt]):\n"
            "            current = 1.0\n"
            "        pos.loc[dt] = current\n"
            "    return pos\n"
        )
    )
    
    # --- 5. Plotting Helpers (Moved Up) ---
    cells.append(
        _code(
            "def build_market_proxy_ohlcv(assets: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:\n"
            "    # Full-market view proxy: cross-sectional average OHLC and summed volume.\n"
            "    def _col(name: str) -> pd.DataFrame:\n"
            "        parts = []\n"
            "        for sym, df in assets.items():\n"
            "            s = df[name].astype(float).reindex(index)\n"
            "            parts.append(s.rename(sym))\n"
            "        return pd.concat(parts, axis=1)\n"
            "\n"
            "    opens = _col('Open')\n"
            "    highs = _col('High')\n"
            "    lows = _col('Low')\n"
            "    closes = _col('Close')\n"
            "    vols = _col('Volume')\n"
            "\n"
            "    out = pd.DataFrame(index=index)\n"
            "    out['Open'] = opens.mean(axis=1)\n"
            "    out['High'] = highs.mean(axis=1)\n"
            "    out['Low'] = lows.mean(axis=1)\n"
            "    out['Close'] = closes.mean(axis=1)\n"
            "    out['Volume'] = vols.sum(axis=1)\n"
            "    return out\n"
            "\n"
            "market_df = build_market_proxy_ohlcv(assets, close.index)\n"
            "\n"
            "def indicator_views(strategy_name: str, df: pd.DataFrame):\n"
            "    # Helper to create indicator overlays for the Market Proxy chart\n"
            "    c = df['Close'].astype(float)\n"
            "    h = df['High'].astype(float)\n"
            "    l = df['Low'].astype(float)\n"
            "    v = df['Volume'].astype(float)\n"
            "    \n"
            "    overlay: dict[str, pd.Series] = {}\n"
            "    panel: dict[str, pd.Series] = {}\n"
            "    \n"
            "    if 'SMA' in strategy_name:\n"
            "        overlay['SMA20'] = sma(c, window=20)\n"
            "    elif 'EMA' in strategy_name:\n"
            "        overlay['EMA20'] = ema(c, span=20)\n"
            "    elif 'MACD' in strategy_name:\n"
            "        m = macd(c, fast=12, slow=26, signal=9)\n"
            "        panel['MACD'] = m['macd']\n"
            "        panel['Signal'] = m['macd_signal']\n"
            "        panel['Hist'] = m['macd_hist']\n"
            "    elif 'RSI' in strategy_name:\n"
            "        panel['RSI14'] = rsi(c, window=14)\n"
            "    elif 'Bollinger' in strategy_name:\n"
            "        bb = bollinger_bands(c, window=20, k=2.0)\n"
            "        overlay['BB Mid'] = bb['bb_mid']\n"
            "        overlay['BB Upper'] = bb['bb_upper']\n"
            "        overlay['BB Lower'] = bb['bb_lower']\n"
            "    elif 'ROC' in strategy_name and 'OBV' not in strategy_name:\n"
            "        panel['ROC10'] = roc(c, window=10)\n"
            "    elif 'Stochastic' in strategy_name:\n"
            "        st = stochastic_oscillator(h, l, c, k_window=14, d_window=3)\n"
            "        panel['%K'] = st['stoch_k']\n"
            "        panel['%D'] = st['stoch_d']\n"
            "    elif 'CCI' in strategy_name:\n"
            "        panel['CCI20'] = cci(h, l, c, window=20)\n"
            "    elif 'OBV' in strategy_name and 'ROC' not in strategy_name and 'WROBV' not in strategy_name:\n"
            "        panel['OBV'] = obv(c, v)\n"
            "    elif 'OBV ROC' in strategy_name:\n"
            "        o = obv(c, v)\n"
            "        panel['OBV ROC10'] = roc(o, window=10)\n"
            "    elif 'WROBV' in strategy_name:\n"
            "        o = obv(c, v)\n"
            "        panel['WROBV20'] = o.rolling(window=20, min_periods=20).sum() / v.rolling(window=20, min_periods=20).sum().replace(0.0, np.nan)\n"
            "    elif 'A/D Line' in strategy_name:\n"
            "        panel['A/D Line'] = accumulation_distribution_line(h, l, c, v)\n"
            "    elif 'ADX' in strategy_name:\n"
            "        a = adx(h, l, c, window=14)\n"
            "        panel['ADX'] = a['adx']\n"
            "        panel['+DI'] = a['plus_di']\n"
            "        panel['-DI'] = a['minus_di']\n"
            "    elif 'Aroon' in strategy_name:\n"
            "        ar = aroon(h, l, window=25)\n"
            "        panel['Aroon Up'] = ar['aroon_up']\n"
            "        panel['Aroon Down'] = ar['aroon_down']\n"
            "    elif 'Ichimoku' in strategy_name:\n"
            "        ic = ichimoku(h, l, c)\n"
            "        overlay['Conv'] = ic['ichimoku_conv']\n"
            "        overlay['Base'] = ic['ichimoku_base']\n"
            "        overlay['Span A'] = ic['ichimoku_span_a']\n"
            "        overlay['Span B'] = ic['ichimoku_span_b']\n"
            "    elif 'Fib' in strategy_name:\n"
            "        if 'Level' in strategy_name:\n"
            "            swing_high = h.rolling(window=60, min_periods=60).max()\n"
            "            swing_low = l.rolling(window=60, min_periods=60).min()\n"
            "            overlay['Fib 50%'] = swing_high - 0.5 * (swing_high - swing_low)\n"
            "        else:\n"
            "            panel['Fib Retr (60)'] = fib_retracement_ratio(h, l, c, window=60)\n"
            "    elif 'ATR' in strategy_name:\n"
            "        overlay['SMA20'] = sma(c, window=20)\n"
            "    \n"
            "    return overlay, panel\n"
        )
    )

    # --- 6. Strategy Defs ---
    cells.append(
        _md(
            "## 3) Indicator Strategies (Standard Interpretations)\n"
            "Each strategy defines a function that returns a DataFrame of signals (1.0 = Long, 0.0 = Cash).\n"
        )
    )

    cells.append(
        _code(
            "def strategy_sma_trend(assets: dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        ratio = sma_ratio(df['Close'], window=window)\n"
            "        enter = (ratio > 1.01)\n"
            "        exit = (ratio < 0.99)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_ema_trend(assets: dict[str, pd.DataFrame], span: int = 20) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        ratio = ema_ratio(df['Close'], span=span)\n"
            "        enter = (ratio > 1.01)\n"
            "        exit = (ratio < 0.99)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_macd_crossover(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        m = macd(df['Close'], fast=12, slow=26, signal=9)\n"
            "        diff = m['macd'] - m['macd_signal']\n"
            "        enter = (diff > 0)\n"
            "        exit = (diff < 0)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_rmacd_sign(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        m = macd(df['Close'], fast=12, slow=26, signal=9)\n"
            "        line = m['macd']\n"
            "        sig = m['macd_signal']\n"
            "        denom = 0.5 * (line.abs() + sig.abs())\n"
            "        rmacd_val = (line - sig) / denom.replace(0.0, np.nan)\n"
            "        enter = (rmacd_val > 0.01)\n"
            "        exit = (rmacd_val < -0.01)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_rsi_mean_reversion(assets: dict[str, pd.DataFrame], window: int = 14, lo: float = 30.0, hi: float = 70.0) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        r = rsi(df['Close'], window=window)\n"
            "        enter = (r < lo).fillna(False)\n"
            "        exit = (r > hi).fillna(False)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_bollinger_mean_reversion(assets: dict[str, pd.DataFrame], window: int = 20, k: float = 2.0) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        bb = bollinger_bands(df['Close'], window=window, k=k)\n"
            "        enter = (df['Close'] < bb['bb_lower']).fillna(False)\n"
            "        exit = (df['Close'] > bb['bb_mid']).fillna(False)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_roc_momentum(assets: dict[str, pd.DataFrame], window: int = 10) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        r = roc(df['Close'], window=window)\n"
            "        enter = (r > 1.0)\n"
            "        exit = (r < -1.0)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_stochastic_mean_reversion(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        c = df['Close'].astype(float)\n"
            "        st = stochastic_oscillator(h, l, c, k_window=14, d_window=3)\n"
            "        k = st['stoch_k']\n"
            "        enter = (k < 20.0).fillna(False)\n"
            "        exit = (k > 80.0).fillna(False)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_cci_mean_reversion(assets: dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        ind = cci(df['High'], df['Low'], df['Close'], window=window)\n"
            "        enter = (ind < -100.0).fillna(False)\n"
            "        exit = (ind > 100.0).fillna(False)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_obv_trend(assets: dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        o = obv(df['Close'], df['Volume'])\n"
            "        o_sma = sma(o, window=window)\n"
            "        enter = (o > o_sma * 1.01)\n"
            "        exit = (o < o_sma * 0.99)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_obv_roc_momentum(assets: dict[str, pd.DataFrame], window: int = 10) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        o = obv(df['Close'], df['Volume'])\n"
            "        r = roc(o, window=window)\n"
            "        enter = (r > 1.0)\n"
            "        exit = (r < -1.0)\n"
            "        out[sym] = hysteresis_position(enter, exit)\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_wrobv_momentum(assets: dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:\n"
            "    # WROBV_t is an OBV/Volume ratio; interpret positive values as bullish accumulation.\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        c = df['Close'].astype(float)\n"
            "        v = df['Volume'].astype(float)\n"
            "        o = obv(c, v)\n"
            "        num = o.rolling(window=window, min_periods=window).sum()\n"
            "        den = v.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)\n"
            "        wrobv = num / den\n"
            "        s = (wrobv > 0.0).astype(float)\n"
            "        out[sym] = s\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_ad_line_trend(assets: dict[str, pd.DataFrame], window: int = 20) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        c = df['Close'].astype(float)\n"
            "        v = df['Volume'].astype(float)\n"
            "        ad = accumulation_distribution_line(h, l, c, v)\n"
            "        s = (ad > sma(ad, window=window)).astype(float)\n"
            "        out[sym] = s\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_adx_directional_trend(assets: dict[str, pd.DataFrame], window: int = 14, adx_threshold: float = 25.0) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        c = df['Close'].astype(float)\n"
            "        a = adx(h, l, c, window=window)\n"
            "        s = ((a['adx'] > adx_threshold) & (a['plus_di'] > a['minus_di'])).astype(float)\n"
            "        out[sym] = s\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_aroon_trend(assets: dict[str, pd.DataFrame], window: int = 25) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        ar = aroon(h, l, window=window)\n"
            "        s = (ar['aroon_up'] > ar['aroon_down']).astype(float)\n"
            "        out[sym] = s\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_ichimoku_cloud(assets: dict[str, pd.DataFrame]) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        c = df['Close'].astype(float)\n"
            "        ic = ichimoku(h, l, c)\n"
            "        cloud_top = pd.concat([ic['ichimoku_span_a'], ic['ichimoku_span_b']], axis=1).max(axis=1)\n"
            "        s = ((c > cloud_top) & (ic['ichimoku_conv'] > ic['ichimoku_base'])).astype(float)\n"
            "        out[sym] = s\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_fibonacci_trend(assets: dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:\n"
            "    # Uses the paper-style retracement ratio R(t) = (H_N - C) / (H_N - L_N).\n"
            "    # Interpreted as bullish when price is near the top of the rolling range.\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        c = df['Close'].astype(float)\n"
            "        r = fib_retracement_ratio(h, l, c, window=window)\n"
            "        s = (r < 0.382).astype(float)\n"
            "        out[sym] = s\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_atr_trailing_stop(assets: dict[str, pd.DataFrame], window: int = 14, entry_sma: int = 20, k: float = 3.0) -> pd.DataFrame:\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        c = df['Close'].astype(float)\n"
            "        a = atr(h, l, c, window=window)\n"
            "        base = sma(c, window=entry_sma)\n"
            "        enter_raw = (c > base).fillna(False)\n"
            "\n"
            "        pos = pd.Series(0.0, index=c.index)\n"
            "        trail = pd.Series(np.nan, index=c.index)\n"
            "        in_pos = 0.0\n"
            "        prev_trail = np.nan\n"
            "        for dt in c.index:\n"
            "            price = float(c.loc[dt])\n"
            "            atr_v = a.loc[dt]\n"
            "            if in_pos == 0.0 and bool(enter_raw.loc[dt]) and pd.notna(atr_v):\n"
            "                in_pos = 1.0\n"
            "                prev_trail = price - k * float(atr_v)\n"
            "            elif in_pos == 1.0 and pd.notna(atr_v):\n"
            "                prev_trail = max(prev_trail, price - k * float(atr_v))\n"
            "                if price < prev_trail:\n"
            "                    in_pos = 0.0\n"
            "                    prev_trail = np.nan\n"
            "            pos.loc[dt] = in_pos\n"
            "            trail.loc[dt] = prev_trail\n"
            "        out[sym] = pos\n"
            "    return pd.DataFrame(out)\n"
            "\n"
            "def strategy_fibonacci_level_50(assets: dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:\n"
            "    # Bullish when price is above the rolling 50% retracement level.\n"
            "    out = {}\n"
            "    for sym, df in assets.items():\n"
            "        h = df['High'].astype(float)\n"
            "        l = df['Low'].astype(float)\n"
            "        c = df['Close'].astype(float)\n"
            "        swing_high = h.rolling(window=window, min_periods=window).max()\n"
            "        swing_low = l.rolling(window=window, min_periods=window).min()\n"
            "        level_50 = swing_high - 0.5 * (swing_high - swing_low)\n"
            "        s = (c > level_50).astype(float)\n"
            "        out[sym] = s\n"
            "    return pd.DataFrame(out)\n"
        )
    )

    # --- 7. Execution Loop (One Cell Per Strategy) ---
    cells.append(
        _md(
            "## 4) Run Individual Strategies\n"
            "Each strategy is run in its own cell to minimize memory usage and compute time per step.\n"
        )
    )

    cells.append(
        _code(
            "results = {} # Container for stats if needed later\n"
        )
    )

    # Strategy Mappings
    strategies_info = [
        ("SMA Trend (20)", "strategy_sma_trend"),
        ("EMA Trend (20)", "strategy_ema_trend"),
        ("MACD Crossover (12/26/9)", "strategy_macd_crossover"),
        ("rMACD Sign (12/26/9)", "strategy_rmacd_sign"),
        ("RSI Mean Reversion (14, 30/70)", "strategy_rsi_mean_reversion"),
        ("Bollinger Mean Reversion (20,2)", "strategy_bollinger_mean_reversion"),
        ("ROC Momentum (10)", "strategy_roc_momentum"),
        ("Stochastic Mean Reversion (14,3)", "strategy_stochastic_mean_reversion"),
        ("CCI Mean Reversion (20)", "strategy_cci_mean_reversion"),
        ("OBV Trend (OBV > SMA20)", "strategy_obv_trend"),
        ("OBV ROC Momentum (10)", "strategy_obv_roc_momentum"),
        ("WROBV Momentum (20)", "strategy_wrobv_momentum"),
        ("A/D Line Trend (AD > SMA20)", "strategy_ad_line_trend"),
        ("ADX Directional Trend (14, >25)", "strategy_adx_directional_trend"),
        ("Aroon Trend (25)", "strategy_aroon_trend"),
        ("Ichimoku Cloud", "strategy_ichimoku_cloud"),
        ("Fibonacci Trend (R<0.382, 60)", "strategy_fibonacci_trend"),
        ("Fibonacci Level (C > 50%, 60)", "strategy_fibonacci_level_50"),
        ("ATR Trailing Stop (14, SMA20 entry)", "strategy_atr_trailing_stop"),
    ]

    for name, func_name in strategies_info:
        cells.append(_md(f"### Strategy: {name}"))
        code_block = (
            f"print('Running {name}...')\n"
            f"signals = {func_name}(assets).reindex(close.index)\n"
            "weights = signals_to_long_only_weights(signals)\n"
            "res = run_backtest(close_prices=close, weights=weights, config=cfg, open_prices=opens, high_prices=highs, low_prices=lows)\n"
            "stats = compute_performance_stats(equity=res.equity, returns=res.returns)\n"
            f"results['{name}'] = (res, stats)\n"
            "\n"
            "print('=' * 80)\n"
            f"print('{name}')\n"
            "report = compute_backtest_report(result=res, close_prices=close, benchmark='equal_weight')\n"
            "display(report)\n"
            "\n"
            f"overlay, panel = indicator_views('{name}', market_df)\n"
            "layout = build_interactive_portfolio_layout(\n"
            "    market_ohlcv=market_df,\n"
            "    equity=res.equity,\n"
            "    returns=res.returns,\n"
            "    weights=res.weights,\n"
            "    turnover=res.turnover,\n"
            "    costs=res.costs,\n"
            "    indicator_overlay=overlay,\n"
            "    indicator_panel=panel,\n"
            f"    title='{name}',\n"
            ")\n"
            "show(layout)\n"
        )
        cells.append(_code(code_block))

    # --- 8. Summary Table ---
    cells.append(
        _md(
            "## 5) Overall Performance Summary\n"
        )
    )

    cells.append(
        _code(
            "rows = []\n"
            "rows.append({\n"
            "    'strategy': 'Equal-Weight Benchmark',\n"
            "    'total_return': benchmark_stats.total_return,\n"
            "    'cagr': benchmark_stats.cagr,\n"
            "    'sharpe': benchmark_stats.sharpe,\n"
            "    'max_drawdown': benchmark_stats.max_drawdown,\n"
            "    'trades': 0, # Benchmark is Buy & Hold\n"
            "})\n"
            "for name, (res, stats) in results.items():\n"
            "    # Count entries: Asset weights going from 0 to > 0\n"
            "    w = res.weights\n"
            "    entries = (w > 1e-6).astype(int).diff() == 1\n"
            "    total_entries = entries.sum().sum()\n"
            "    rows.append({\n"
            "        'strategy': name,\n"
            "        'total_return': stats.total_return,\n"
            "        'cagr': stats.cagr,\n"
            "        'sharpe': stats.sharpe,\n"
            "        'max_drawdown': stats.max_drawdown,\n"
            "        'trades': total_entries,\n"
            "    })\n"
            "\n"
            "pd.DataFrame(rows).set_index('strategy').sort_values('sharpe', ascending=False)\n"
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
    out_path = os.path.join(
        "notebooks", "Technical_Indicator_Strategies_Base_Performance.ipynb"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    create_notebook()
