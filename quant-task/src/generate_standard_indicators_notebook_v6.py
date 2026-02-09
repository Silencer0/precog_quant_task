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

    # --- Header ---
    cells.append(_md("# Standard Technical Indicators: Discrete Signal Entry/Exit"))
    cells.append(_md(
        "**Strategy Logic v6 (Signal Driven) with Intraday Stop Loss**:\n"
        "- **Engine Mode**: `strict_signals=True`.\n"
        "- **Signal Protocol**: \n"
        "    - `> 0`: Buy Entry.\n"
        "    - `< 0`: Sell Exit.\n"
        "    - `0.0`: Hold.\n"
        "- **Stop Loss**: Intraday Simulation using OHLC data.\n"
        "    - If `Low_t < Stop_Price`, trigger sell.\n"
        "    - Sell Price = `Open_t` (if Gap Down) or `Stop_Price` (if Intraday).\n"
    ))

    # --- Setup ---
    cells.append(_code(
        "import os, sys\n"
        "\n"
        "# --- Setup correct working directory (ROOT) ---\n"
        "if os.getcwd().endswith('notebooks'):\n"
        "    os.chdir('..')\n"
        "\n"
        "ROOT = os.getcwd()\n"
        "if ROOT not in sys.path:\n"
        "    sys.path.insert(0, ROOT)\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from IPython.display import display\n"
        "from bokeh.io import output_notebook, show\n"
        "\n"
        "from src.backtester.data import load_cleaned_assets, align_close_prices\n"
        "from src.backtester.engine import BacktestConfig, run_backtest\n"
        "from src.backtester.metrics import compute_performance_stats\n"
        "from src.backtester.bokeh_plots import build_interactive_portfolio_layout\n"
        "from src.backtester.report import compute_backtest_report\n"
        "from src.features.core import (\n"
        "    adx, aroon, atr, bollinger_bands, cci, ema, fib_levels, ichimoku, macd, \n"
        "    obv, accumulation_distribution_line, roc, rsi, sma, stochastic_oscillator\n"
        ")\n"
        "\n"
        "output_notebook()\n"
    ))

    # --- Config ---
    cells.append(_code(
        "cfg = BacktestConfig(\n"
        "    initial_equity=1_000_000,\n"
        "    transaction_cost_bps=5,\n"
        "    rebalance=None,\n"
        "    mode='event_driven',\n"
        "    strict_signals=True,\n"
        "    stop_loss_pct=0.15   # 15% Stop Loss (Reduced exit noise for volatile assets)\n"
        ")\n"
    ))

    # --- Load Data (OHLC) ---
    cells.append(_code(
        "assets = load_cleaned_assets(symbols=None)\n"
        "close = align_close_prices(assets)\n"
        "\n"
        "# Build OHLC Arrays for Backtester\n"
        "def _align_col(assets, col):\n"
        "    parts = []\n"
        "    for sym, df in assets.items():\n"
        "        parts.append(df[col].astype(float).rename(sym))\n"
        "    return pd.concat(parts, axis=1).sort_index()\n"
        "\n"
        "open_prices = _align_col(assets, 'Open')\n"
        "high_prices = _align_col(assets, 'High')\n"
        "low_prices = _align_col(assets, 'Low')\n"
        "\n"
        "print(f'Loaded {len(assets)} assets with full OHLC.')\n"
        "market_df = None\n"
        "\n"
        "# Calculate Equal-Weight Benchmark (Long Only Buy & Hold)\n"
        "ew_weights = pd.DataFrame(1.0 / len(assets), index=close.index, columns=close.columns)\n"
        "benchmark_res = run_backtest(close, ew_weights, BacktestConfig(initial_equity=cfg.initial_equity, transaction_cost_bps=0))\n"
        "benchmark_stats = compute_performance_stats(equity=benchmark_res.equity, returns=benchmark_res.returns)\n"
        "print(f'Benchmark Return: {benchmark_stats.total_return*100:.2f}%')\n"
    ))

    # --- Helper Functions ---
    cells.append(_code(
        "ALLOC = 0.01 \n"
        "\n"
        "def smooth_slope(series: pd.Series, window: int = 20) -> pd.Series:\n"
        "    return series.diff(1).ewm(span=window).mean()\n"
        "\n"
        "def hysteresis_position(enter: pd.Series, exit: pd.Series) -> pd.Series:\n"
        "    \"\"\"Latches a signal state: +1.0 for Long, -1.0 for Out/Exit.\"\"\"\n"
        "    pos = pd.Series(0.0, index=enter.index)\n"
        "    current = -1.0\n"
        "    for dt in enter.index:\n"
        "        if bool(exit.loc[dt]):\n"
        "            current = -1.0\n"
        "        elif bool(enter.loc[dt]):\n"
            "            current = ALLOC\n"
        "        pos.loc[dt] = current\n"
        "    return pos\n"
        "\n"
        "def get_divergence_signals(price: pd.Series, indicator: pd.Series, window: int = 20) -> pd.Series:\n"
        "    p_slope = smooth_slope(price, window)\n"
        "    i_slope = smooth_slope(indicator, window)\n"
        "    signals = pd.Series(0.0, index=price.index)\n"
        "    bull_div = (p_slope < 0) & (i_slope > 0)\n"
        "    bear_div = (p_slope > 0) & (i_slope < 0)\n"
        "    return hysteresis_position(bull_div, bear_div)\n"
        "\n"
        "def build_market_proxy_ohlcv(assets: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:\n"
        "    def _col(name: str) -> pd.DataFrame:\n"
        "        parts = []\n"
        "        for sym, df in assets.items():\n"
        "            s = df[name].astype(float).reindex(index)\n"
        "            parts.append(s.rename(sym))\n"
        "        return pd.concat(parts, axis=1)\n"
        "    opens = _col('Open').mean(axis=1)\n"
        "    highs = _col('High').mean(axis=1)\n"
        "    lows = _col('Low').mean(axis=1)\n"
        "    closes = _col('Close').mean(axis=1)\n"
        "    vols = _col('Volume').sum(axis=1)\n"
        "    return pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': vols})\n"
        "market_df = build_market_proxy_ohlcv(assets, close.index)\n"
    ))

    # --- Strategies ---
    
    # 1. OBV
    cells.append(_md("## 1. On-Balance Volume"))
    cells.append(_code(
        "def strategy_obv_discrete(assets): \n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        out[sym] = get_divergence_signals(df['Close'], obv(df['Close'], df['Volume']))\n"
        "    return pd.DataFrame(out)\n"
    ))
    
    # 2. A/D
    cells.append(_md("## 2. A/D Line"))
    cells.append(_code(
        "def strategy_ad_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        ad_v = accumulation_distribution_line(df['High'], df['Low'], df['Close'], df['Volume'])\n"
        "        out[sym] = get_divergence_signals(df['Close'], ad_v)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # 3. RSI
    cells.append(_md("## 3. RSI"))
    cells.append(_code(
        "def strategy_rsi_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        r = rsi(df['Close'], 14)\n"
        "        enter = (r < 30.0)\n"
        "        exit = (r > 70.0)\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
    ))

    # 4. MACD
    cells.append(_md("## 4. MACD"))
    cells.append(_code(
        "def strategy_macd_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        m = macd(df['Close'])\n"
        "        diff = m['macd'] - m['macd_signal']\n"
        "        # Use hysteresis to prevent daily whipsaws on line crosses\n"
        "        enter = (diff > 0) \n"
        "        exit = (diff < 0)\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # 5. Stochastic
    cells.append(_md("## 5. Stochastic"))
    cells.append(_code(
        "def strategy_stochastic_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        st = stochastic_oscillator(df['High'], df['Low'], df['Close'])\n"
        "        k = st['stoch_k']\n"
        "        enter = (k < 20.0)\n"
        "        exit = (k > 80.0)\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # 6. ADX
    cells.append(_md("## 6. ADX"))
    cells.append(_code(
        "def strategy_adx_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        a = adx(df['High'], df['Low'], df['Close'])\n"
        "        enter = (a['adx'] > 25) & (a['plus_di'] > a['minus_di'])\n"
        "        exit = (a['minus_di'] > a['plus_di']) | (a['adx'] < 20)\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # 7. Aroon
    cells.append(_md("## 7. Aroon"))
    cells.append(_code(
        "def strategy_aroon_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        ar = aroon(df['High'], df['Low'])\n"
        "        enter = (ar['aroon_up'] > 70)\n"
        "        exit = (ar['aroon_up'] < 30)\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # 8. CCI
    cells.append(_md("## 8. CCI"))
    cells.append(_code(
        "def strategy_cci_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        c_val = cci(df['High'], df['Low'], df['Close'])\n"
        "        enter = (c_val > 100.0)\n"
        "        exit = (c_val < -100.0)\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # 9. Bollinger
    cells.append(_md("## 9. Bollinger Bands"))
    cells.append(_code(
        "def strategy_bb_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        bb = bollinger_bands(df['Close'])\n"
        "        c = df['Close']\n"
        "        enter = (c > bb['bb_upper'])\n"
        "        exit = (c < bb['bb_mid'])\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # 10. Ichimoku
    cells.append(_md("## 10. Ichimoku Cloud"))
    cells.append(_code(
        "def strategy_ichimoku_discrete(assets):\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        ich = ichimoku(df['High'], df['Low'], df['Close'])\n"
        "        c = df['Close']\n"
        "        span_a = ich['ichimoku_span_a']\n"
        "        span_b = ich['ichimoku_span_b']\n"
        "        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)\n"
        "        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)\n"
        "        enter = (c > cloud_top)\n"
        "        exit = (c < cloud_bottom)\n"
        "        out[sym] = hysteresis_position(enter, exit)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # --- Execution Loop ---
    strategies = [
        ("OBV Native Discrete", "strategy_obv_discrete"),
        ("A/D Native Discrete", "strategy_ad_discrete"),
        ("RSI Native Discrete", "strategy_rsi_discrete"),
        ("MACD Native Discrete", "strategy_macd_discrete"),
        ("Stochastic Native Discrete", "strategy_stochastic_discrete"),
        ("ADX Native Discrete", "strategy_adx_discrete"),
        ("Aroon Native Discrete", "strategy_aroon_discrete"),
        ("CCI Native Discrete", "strategy_cci_discrete"),
        ("Bollinger Bands Native Discrete", "strategy_bb_discrete"),
        ("Ichimoku Native Discrete", "strategy_ichimoku_discrete"),
    ]
    
    cells.append(_md("## Execution"))
    
    for name, func in strategies:
        cells.append(_md(f"### {name}"))
        cells.append(_code(
            f"print('Running {name}...')\n"
            f"signals = {func}(assets).reindex(close.index)\n"
            "# Pass OHLC Data to Engine for Stop Loss Simulation\n"
            "res = run_backtest(\n"
            "    close_prices=close, \n"
            "    weights=signals, \n"
            "    config=cfg, \n"
            "    open_prices=open_prices,\n"
            "    high_prices=high_prices,\n"
            "    low_prices=low_prices\n"
            ")\n"
            "report = compute_backtest_report(result=res, close_prices=close, benchmark='equal_weight')\n"
            "display(report)\n"
            "layout = build_interactive_portfolio_layout(\n"
            "    market_ohlcv=market_df,\n"
            "    equity=res.equity,\n"
            "    returns=res.returns,\n"
            "    weights=res.weights,\n"
            "    turnover=res.turnover,\n"
            "    costs=res.costs,\n"
            "    close_prices=close, \n"
            f"    title='{name}'\n"
            ")\n"
            "show(layout)\n"
        ))

    # --- Write ---
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    os.makedirs("notebooks", exist_ok=True)
    out_path = os.path.join("notebooks", "Standard_Technical_Indicators_Strategies.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    create_notebook()
