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
    cells.append(_md("# Standard Technical Indicator Strategies (Trend & Divergence)"))
    cells.append(_md(
        "**Fixed Logic**: \n"
        "- **Divergence** is treated as an **Entry Signal**.\n"
        "- **Exits** are determined by **Trend Breakdown** (e.g. Price < SMA).\n"
        "- This prevents selling immediately when the price recovers (which kills the divergence condition).\n"
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
        "    rebalance=None,  # Trade on Signal Change\n"
        "    mode='event_driven',\n"
        "    no_sell=False\n"
        ")\n"
    ))

    # --- Load Data ---
    cells.append(_code(
        "assets = load_cleaned_assets(symbols=None)\n"
        "close = align_close_prices(assets)\n"
        "print(f'Loaded {len(assets)} assets.')\n"
        "market_df = None\n"
    ))

    # --- Helper Functions (Signal Latch & Smooth Slope) ---
    cells.append(_code(
        "def signals_to_long_only_weights(signals: pd.DataFrame) -> pd.DataFrame:\n"
        "    s = signals.fillna(0.0).clip(lower=0.0)\n"
        "    row_sum = s.sum(axis=1).replace(0.0, np.nan)\n"
        "    w = s.div(row_sum, axis=0).fillna(0.0)\n"
        "    return w\n"
        "\n"
        "def smooth_slope(series: pd.Series, window: int = 10) -> pd.Series:\n"
        "    # Smooth slope: EMA of period-1 differences\n"
        "    # This is less noisy than diff(N)\n"
        "    return series.diff(1).ewm(span=window).mean()\n"
        "\n"
        "def divergence_entry(price: pd.Series, indicator: pd.Series, window: int = 10) -> pd.Series:\n"
        "    # Bullish Divergence Entry: Price Slope < 0, Ind Slope > 0\n"
        "    p_slope = smooth_slope(price, window)\n"
        "    i_slope = smooth_slope(indicator, window)\n"
        "    sig = ((p_slope < 0) & (i_slope > 0)).astype(float)\n"
        "    # Require divergence to persist for at least 2 days to filter blips?\n"
        "    # Let's trust the smoothed slope.\n"
        "    return sig\n"
        "\n"
        "def latch_signals(entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:\n"
        "    # State machine: Hold 1 after Entry until Exit triggers.\n"
        "    out = pd.DataFrame(0.0, index=entries.index, columns=entries.columns)\n"
        "    for col in entries.columns:\n"
        "        state = 0.0\n"
        "        ent_v = entries[col].values\n"
        "        ex_v = exits[col].values\n"
        "        res = np.zeros_like(ent_v)\n"
        "        for t in range(len(ent_v)):\n"
        "            if ent_v[t] > 0:\n"
        "                state = 1.0\n"
        "            elif ex_v[t] > 0:\n"
        "                state = 0.0\n"
        "            res[t] = state\n"
        "        out[col] = res\n"
        "    return out\n"
        "\n"
        "# Standard Exit: Close < SMA(20)\n"
        "def get_trend_exit(close: pd.Series, window: int = 20) -> pd.Series:\n"
        "    return (close < sma(close, window)).astype(float)\n"
        "\n"
        "def build_market_proxy_ohlcv(assets: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:\n"
        "    def _col(name: str) -> pd.DataFrame:\n"
        "        parts = []\n"
        "        for sym, df in assets.items():\n"
        "            s = df[name].astype(float).reindex(index)\n"
        "            parts.append(s.rename(sym))\n"
        "        return pd.concat(parts, axis=1)\n"
        "    \n"
        "    opens = _col('Open').mean(axis=1)\n"
        "    highs = _col('High').mean(axis=1)\n"
        "    lows = _col('Low').mean(axis=1)\n"
        "    closes = _col('Close').mean(axis=1)\n"
        "    vols = _col('Volume').sum(axis=1)\n"
        "    return pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': vols})\n"
        "\n"
        "market_df = build_market_proxy_ohlcv(assets, close.index)\n"
    ))

    # --- Strategies ---
    
    # 1. OBV Divergence (With Latch)
    cells.append(_md("## 1. On-Balance Volume (OBV) Divergence"))
    cells.append(_code(
        "def strategy_obv_divergence_latched(assets): \n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        v = df['Volume']\n"
        "        o = obv(c, v)\n"
        "        # Entry: Divergence\n"
        "        entries[sym] = divergence_entry(c, o, window=10)\n"
        "        # Exit: Price drops below SMA(20) (Trend broken)\n"
        "        exits[sym] = get_trend_exit(c, 20)\n"
        "        \n"
        "    ent_df = pd.DataFrame(entries)\n"
        "    ex_df = pd.DataFrame(exits)\n"
        "    return latch_signals(ent_df, ex_df)\n"
    ))

    # 2. A/D Divergence
    cells.append(_md("## 2. A/D Line Divergence"))
    cells.append(_code(
        "def strategy_ad_divergence_latched(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        ad = accumulation_distribution_line(df['High'], df['Low'], c, df['Volume'])\n"
        "        entries[sym] = divergence_entry(c, ad, window=10)\n"
        "        exits[sym] = get_trend_exit(c, 20)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # 3. RSI Divergence
    cells.append(_md("## 3. RSI Divergence"))
    cells.append(_code(
        "def strategy_rsi_divergence_latched(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        r = rsi(c, 14)\n"
        "        # Entry: Divergence\n"
        "        entries[sym] = divergence_entry(c, r, window=10)\n"
        "        # Exit: RSI crosses back below 50? Or Price < SMA20? \n"
        "        # Let's use RSI < 45 as Exit to capture the swing.\n"
        "        exits[sym] = (r < 45).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # 4. Aroon Crossover (Latched)
    cells.append(_md("## 4. Aroon Trend"))
    cells.append(_code(
        "def strategy_aroon_latched(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        ar = aroon(df['High'], df['Low'], window=25)\n"
        "        # Entry: Up > Down\n"
        "        entries[sym] = (ar['aroon_up'] > ar['aroon_down']).astype(float)\n"
        "        # Exit: Down > Up\n"
        "        exits[sym] = (ar['aroon_up'] < ar['aroon_down']).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 5. MACD (Standard) - Already effectively latched by crossover condition > 
    # But let's be explicit
    cells.append(_md("## 5. MACD Trend"))
    cells.append(_code(
        "def strategy_macd_latched(assets):\n"
        "    # Standard: MACD > Signal\n"
        "    # This is naturally latched (stateful based on line positions)\n"
        "    out = {}\n"
        "    for sym, df in assets.items():\n"
        "        m = macd(df['Close'], 12, 26, 9)\n"
        "        out[sym] = (m['macd'] > m['macd_signal']).astype(float)\n"
        "    return pd.DataFrame(out)\n"
    ))

    # --- Execution Loop ---
    strategies = [
        ("OBV Div + Trend Exit", "strategy_obv_divergence_latched"),
        ("A/D Div + Trend Exit", "strategy_ad_divergence_latched"),
        ("RSI Div + RSI Exit", "strategy_rsi_divergence_latched"),
        ("Aroon Trend Latch", "strategy_aroon_latched"),
        ("MACD Trend", "strategy_macd_latched"),
    ]
    
    cells.append(_md("## Execution"))
    
    for name, func in strategies:
        cells.append(_md(f"### {name}"))
        cells.append(_code(
            f"print('Running {name}...')\n"
            f"signals = {func}(assets).reindex(close.index)\n"
            "weights = signals_to_long_only_weights(signals)\n"
            "res = run_backtest(close_prices=close, weights=weights, config=cfg)\n"
            "stats = compute_performance_stats(equity=res.equity, returns=res.returns)\n"
            "report = compute_backtest_report(result=res, close_prices=close, benchmark='equal_weight')\n"
            "display(report)\n"
            "layout = build_interactive_portfolio_layout(\n"
            "    market_ohlcv=market_df,\n"
            "    equity=res.equity,\n"
            "    returns=res.returns,\n"
            "    weights=res.weights,\n"
            "    turnover=res.turnover,\n"
            "    costs=res.costs,\n"
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
        "nbformat_minor": 5,
    }

    os.makedirs("notebooks", exist_ok=True)
    out_path = os.path.join("notebooks", "Standard_Technical_Indicators_Strategies.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    create_notebook()
