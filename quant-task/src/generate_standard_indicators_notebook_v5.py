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
    cells.append(_md("# Standard Technical Indicators: Native Entry & Exit Strategies"))
    cells.append(_md(
        "**Strategy Logic v5 (Native Exits)**:\n"
        "- **Entry**: Indicator-specific signal (e.g. Bullish Divergence, Golden Cross, Breakout).\n"
        "- **Exit**: Indicator-specific exit (e.g. Bearish Divergence, Death Cross, Overbought).\n"
        "- **No External Filters**: We rely solely on the indicator's own logic as requested.\n"
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
        "    no_sell=True,\n"
        "    trade_buffer=0.01\n"
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
        "def smooth_slope(series: pd.Series, window: int = 20) -> pd.Series:\n"
        "    return series.diff(1).ewm(span=window).mean()\n"
        "\n"
        "def get_divergence_signals(price: pd.Series, indicator: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:\n"
        "    # Slopes\n"
        "    p_slope = smooth_slope(price, window)\n"
        "    i_slope = smooth_slope(indicator, window)\n"
        "    \n"
        "    # Bullish Divergence (Entry): Price Trend Down, Ind Trend Up\n"
        "    bull_div = ((p_slope < 0) & (i_slope > 0)).astype(float)\n"
        "    # Bearish Divergence (Exit): Price Trend Up, Ind Trend Down\n"
        "    bear_div = ((p_slope > 0) & (i_slope < 0)).astype(float)\n"
        "    return bull_div, bear_div\n"
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
        "            if ex_v[t] > 0:\n"
        "                state = 0.0\n"
        "            elif ent_v[t] > 0:\n"
        "                state = 1.0\n"
        "            res[t] = state\n"
        "        out[col] = res\n"
        "    return out\n"
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
    
    # 1. OBV (Entry: Bull Div, Exit: Bear Div)
    cells.append(_md("## 1. On-Balance Volume (Native)"))
    cells.append(_code(
        "def strategy_obv_native(assets): \n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        v = df['Volume']\n"
        "        o = obv(c, v)\n"
        "        # Native: Divergence Entry & Exit\n"
        "        bull, bear = get_divergence_signals(c, o, window=20)\n"
        "        entries[sym] = bull\n"
        "        exits[sym] = bear\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # 2. A/D Line (Entry: Bull Div, Exit: Bear Div)
    cells.append(_md("## 2. A/D Line (Native)"))
    cells.append(_code(
        "def strategy_ad_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        ad = accumulation_distribution_line(df['High'], df['Low'], c, df['Volume'])\n"
        "        bull, bear = get_divergence_signals(c, ad, window=20)\n"
        "        entries[sym] = bull\n"
        "        exits[sym] = bear\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # 3. RSI (Entry: Bull Div, Exit: Overbought > 70 OR Bear Div)
    cells.append(_md("## 3. RSI (Native)"))
    cells.append(_code(
        "def strategy_rsi_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        r = rsi(c, 14)\n"
        "        bull, bear = get_divergence_signals(c, r, window=20)\n"
        "        # Entry: Bullish Divergence\n"
        "        entries[sym] = bull\n"
        "        # Exit: Bearish Divergence OR Overbought (Standard RSI Exit)\n"
        "        exits[sym] = (bear > 0) | (r > 70)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 4. Stochastic (Entry: Bull Div, Exit: Overbought > 80 OR Bear Div)
    cells.append(_md("## 4. Stochastic (Native)"))
    cells.append(_code(
        "def strategy_stochastic_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        st = stochastic_oscillator(df['High'], df['Low'], c)\n"
        "        k = st['stoch_k']\n"
        "        bull, bear = get_divergence_signals(c, k, window=14)\n"
        "        # Exit: Overbought > 80 is standard\n"
        "        entries[sym] = bull\n"
        "        exits[sym] = (bear > 0) | (k > 80)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 5. MACD (Entry: Golden Cross, Exit: Death Cross)
    cells.append(_md("## 5. MACD (Native Crossover)"))
    cells.append(_code(
        "def strategy_macd_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        m = macd(df['Close'], 12, 26, 9)\n"
        "        line = m['macd']\n"
        "        sig = m['macd_signal']\n"
        "        # Entry: Line crosses Above Signal\n"
        "        entries[sym] = (line > sig).astype(float)\n"
        "        # Exit: Line crosses Below Signal\n"
        "        exits[sym] = (line < sig).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # 6. Aroon (Entry: Up>Down, Exit: Down>Up)
    cells.append(_md("## 6. Aroon (Native)"))
    cells.append(_code(
        "def strategy_aroon_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        ar = aroon(df['High'], df['Low'], window=25)\n"
        "        up = ar['aroon_up']\n"
        "        down = ar['aroon_down']\n"
        "        entries[sym] = (up > down).astype(float)\n"
        "        exits[sym] = (down > up).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 7. ADX (Entry: Strength w/ Direction, Exit: Direction Flip)
    cells.append(_md("## 7. ADX (Native)"))
    cells.append(_code(
        "def strategy_adx_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        a = adx(df['High'], df['Low'], df['Close'], window=14)\n"
        "        val = a['adx']\n"
        "        plus = a['plus_di']\n"
        "        minus = a['minus_di']\n"
        "        # Entry: Strong Uptrend (+DI > -DI & ADX > 25)\n"
        "        entries[sym] = ((val > 25) & (plus > minus)).astype(float)\n"
        "        # Exit: Trend Weakens (<20) OR Direction Change (-DI > +DI)\n"
        "        exits[sym] = ((val < 20) | (minus > plus)).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 8. CCI (Entry: >100, Exit: <100)
    cells.append(_md("## 8. CCI (Native)"))
    cells.append(_code(
        "def strategy_cci_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        val = cci(df['High'], df['Low'], df['Close'], window=20)\n"
        "        # Standard Trend Usage: Long when CCI > 100\n"
        "        entries[sym] = (val > 100).astype(float)\n"
        "        # Exit (Close Long) when CCI < 100 (Momentum faded)\n"
        "        # Or alternatively CCI < 0? Let's use CCI < 0 to allow some pullback.\n"
        "        exits[sym] = (val < 0).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 9. Bollinger Band (Entry: Breakout, Exit: Mean Reversion)
    cells.append(_md("## 9. Bollinger Band (Native Breakout)"))
    cells.append(_code(
        "def strategy_bb_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        bb = bollinger_bands(df['Close'], window=20)\n"
        "        up = bb['bb_upper']\n"
        "        mid = bb['bb_mid']\n"
        "        c = df['Close']\n"
        "        # Entry: Close > Upper Band (Walking the Band)\n"
        "        entries[sym] = (c > up).astype(float)\n"
        "        # Exit: Close < Mid Band (Trend Failure)\n"
        "        # This is the standard 'Vol Breakout' exit.\n"
        "        exits[sym] = (c < mid).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 10. Ichimoku (Entry: >Cloud, Exit: <Cloud)
    cells.append(_md("## 10. Ichimoku Cloud (Native)"))
    cells.append(_code(
        "def strategy_ichimoku_native(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        ic = ichimoku(df['High'], df['Low'], df['Close'])\n"
        "        c = df['Close']\n"
        "        cloud_top = pd.concat([ic['ichimoku_span_a'], ic['ichimoku_span_b']], axis=1).max(axis=1)\n"
        "        cloud_bottom = pd.concat([ic['ichimoku_span_a'], ic['ichimoku_span_b']], axis=1).min(axis=1)\n"
        "        \n"
        "        # Entry: Price Breakout Above Cloud\n"
        "        entries[sym] = (c > cloud_top).astype(float)\n"
        "        # Exit: Price Breakdown Below Cloud\n"
        "        exits[sym] = (c < cloud_bottom).astype(float)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # --- Execution Loop ---
    strategies = [
        ("OBV Native Divergence", "strategy_obv_native"),
        ("A/D Native Divergence", "strategy_ad_native"),
        ("RSI Native", "strategy_rsi_native"),
        ("Stochastic Native", "strategy_stochastic_native"),
        ("MACD Native Crossover", "strategy_macd_native"),
        ("Aroon Native", "strategy_aroon_native"),
        ("ADX Native", "strategy_adx_native"),
        ("CCI Native", "strategy_cci_native"),
        ("Bollinger Native", "strategy_bb_native"),
        ("Ichimoku Native", "strategy_ichimoku_native"),
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
