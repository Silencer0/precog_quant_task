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
    cells.append(_md("# Standard Technical Indicators: Divergence & Trend Strategies"))
    cells.append(_md(
        "This notebook assumes a **Long-Term Logic**:\n"
        "- **Entry**: Triggered by **Bullish Divergence** (Price Lower, Indicator Higher) or **Golden Cross**.\n"
        "- **Exit**: Triggered by **Bearish Divergence** (Price Higher, Indicator Lower) or **Death Cross**.\n"
        "- **Holding**: Positions are held between Entry and Exit signals (Latched).\n"
        "\n"
        "This approach ensures trades run to their full extent, filtering out intraday noise."
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
        "    no_sell=True,    # Prefer holding cash/assets unless Exit Signal triggers\n"
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
        "    # Smooth slope: EMA of period-1 differences\n"
        "    # Using a slightly longer window (20) to capture 'Trend Direction' securely\n"
        "    return series.diff(1).ewm(span=window).mean()\n"
        "\n"
        "def get_divergence_signals(price: pd.Series, indicator: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:\n"
        "    # Slopes\n"
        "    p_slope = smooth_slope(price, window)\n"
        "    i_slope = smooth_slope(indicator, window)\n"
        "    \n"
        "    # Bullish Divergence (Entry): Price Trend Down (<0), but Indicator Trend Up (>0)\n"
        "    # Logic: Momentum is shifting up while price is still lagging.\n"
        "    bull_div = ((p_slope < 0) & (i_slope > 0)).astype(float)\n"
        "    \n"
        "    # Bearish Divergence (Exit): Price Trend Up (>0), but Indicator Trend Down (<0)\n"
        "    # Logic: Price is rising but momentum is fading.\n"
        "    bear_div = ((p_slope > 0) & (i_slope < 0)).astype(float)\n"
        "    \n"
        "    return bull_div, bear_div\n"
        "\n"
        "def latch_signals(entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:\n"
        "    # State machine: Hold 1 after Entry until Exit triggers.\n"
        "    # If both Entry and Exit trigger same day? We prioritize Exit (Safety).\n"
        "    out = pd.DataFrame(0.0, index=entries.index, columns=entries.columns)\n"
        "    for col in entries.columns:\n"
        "        state = 0.0\n"
        "        ent_v = entries[col].values\n"
        "        ex_v = exits[col].values\n"
        "        res = np.zeros_like(ent_v)\n"
        "        for t in range(len(ent_v)):\n"
        "            e = ent_v[t]\n"
        "            x = ex_v[t]\n"
        "            \n"
        "            if x > 0:\n"
        "                state = 0.0\n"
        "            elif e > 0:\n"
        "                state = 1.0\n"
        "            # Else maintain state\n"
        "            \n"
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
    
    # 1. OBV Divergence (Bull Entry / Bear Exit)
    cells.append(_md("## 1. On-Balance Volume (OBV)"))
    cells.append(_code(
        "def strategy_obv_full_divergence(assets): \n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        v = df['Volume']\n"
        "        o = obv(c, v)\n"
        "        \n"
        "        bull, bear = get_divergence_signals(c, o, window=20)\n"
        "        entries[sym] = bull\n"
        "        exits[sym] = bear\n"
        "        \n"
        "    ent_df = pd.DataFrame(entries)\n"
        "    ex_df = pd.DataFrame(exits)\n"
        "    return latch_signals(ent_df, ex_df)\n"
    ))

    # 2. A/D Divergence
    cells.append(_md("## 2. A/D Line"))
    cells.append(_code(
        "def strategy_ad_full_divergence(assets):\n"
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

    # 3. RSI Divergence
    cells.append(_md("## 3. RSI Divergence"))
    cells.append(_code(
        "def strategy_rsi_full_divergence(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        r = rsi(c, 14)\n"
        "        # Entry: Bullish Div\n"
        "        # Exit: Bearish Div OR RSI > 75 (Overbought Safety)\n"
        "        bull, bear = get_divergence_signals(c, r, window=20)\n"
        "        entries[sym] = bull\n"
        "        exits[sym] = (bear > 0) | (r > 75)\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 4. Stochastic Divergence
    cells.append(_md("## 4. Stochastic Divergence"))
    cells.append(_code(
        "def strategy_stochastic_full_divergence(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        st = stochastic_oscillator(df['High'], df['Low'], c)\n"
        "        k = st['stoch_k']\n"
        "        \n"
        "        # Use K line for divergence\n"
        "        bull, bear = get_divergence_signals(c, k, window=14)\n"
        "        entries[sym] = bull\n"
        "        exits[sym] = bear\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 5. MACD (Histogram Divergence)
    cells.append(_md("## 5. MACD Histogram Divergence"))
    cells.append(_code(
        "def strategy_macd_hist_divergence(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        c = df['Close']\n"
        "        m = macd(c)\n"
        "        hist = m['macd_hist']\n"
        "        \n"
        "        # Histogram Divergence is a strong signal\n"
        "        bull, bear = get_divergence_signals(c, hist, window=20)\n"
        "        entries[sym] = bull\n"
        "        exits[sym] = bear\n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # 6. Aroon (Trend Reversal)
    cells.append(_md("## 6. Aroon Trend Reversal"))
    cells.append(_code(
        "def strategy_aroon_reversal(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        ar = aroon(df['High'], df['Low'], window=25)\n"
        "        up = ar['aroon_up']\n"
        "        down = ar['aroon_down']\n"
        "        \n"
        "        # Entry: Up crosses Down (Golden)\n"
        "        entries[sym] = (up > down).astype(float)\n"
        "        # Exit: Down crosses Up (Death)\n"
        "        exits[sym] = (down > up).astype(float)\n"
        "        \n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 7. ADX (Trend Strength)
    cells.append(_md("## 7. ADX Trend Following"))
    cells.append(_code(
        "def strategy_adx_bipolar(assets):\n"
        "    # Enter when Trend Strong (>20) AND DI+ > DI-\n"
        "    # Exit when Trend Weakens (<20) OR DI- > DI+\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        a = adx(df['High'], df['Low'], df['Close'], window=14)\n"
        "        val = a['adx']\n"
        "        plus = a['plus_di']\n"
        "        minus = a['minus_di']\n"
        "        \n"
        "        entries[sym] = ((val > 20) & (plus > minus)).astype(float)\n"
        "        exits[sym] = ((val < 20) | (minus > plus)).astype(float)\n"
        "        \n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 8. CCI Trend/Zero Cross
    cells.append(_md("## 8. CCI Trend"))
    cells.append(_code(
        "def strategy_cci_trend_latched(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        val = cci(df['High'], df['Low'], df['Close'], window=20)\n"
        "        # Differs from mean reversion. Trend: CCI > 100 is strong up.\n"
        "        entries[sym] = (val > 100).astype(float)\n"
        "        # Exit: CCI < 0 (Trend lost momentum)\n"
        "        exits[sym] = (val < 0).astype(float)\n"
        "        \n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 9. Bollinger Band Breakout
    cells.append(_md("## 9. Bollinger Band Breakout"))
    cells.append(_code(
        "def strategy_bb_trend_latched(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        bb = bollinger_bands(df['Close'], window=20)\n"
        "        mid = bb['bb_mid']\n"
        "        up = bb['bb_upper']\n"
        "        low = bb['bb_lower']\n"
        "        c = df['Close']\n"
        "        \n"
        "        # Entry: Close > Upper (Breakout)\n"
        "        entries[sym] = (c > up).astype(float)\n"
        "        # Exit: Close < Mid (Mean Reversion / Trend Lost)\n"
        "        exits[sym] = (c < mid).astype(float)\n"
        "        \n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))
    
    # 10. Ichimoku
    cells.append(_md("## 10. Ichimoku Cloud Follower"))
    cells.append(_code(
        "def strategy_ichimoku_latched(assets):\n"
        "    entries = {}\n"
        "    exits = {}\n"
        "    for sym, df in assets.items():\n"
        "        ic = ichimoku(df['High'], df['Low'], df['Close'])\n"
        "        c = df['Close']\n"
        "        cloud_top = pd.concat([ic['ichimoku_span_a'], ic['ichimoku_span_b']], axis=1).max(axis=1)\n"
        "        cloud_bottom = pd.concat([ic['ichimoku_span_a'], ic['ichimoku_span_b']], axis=1).min(axis=1)\n"
        "        \n"
        "        # Entry: Price > Cloud (Bullish Territory)\n"
        "        entries[sym] = (c > cloud_top).astype(float)\n"
        "        # Exit: Price < Cloud (Bearish Territory)\n"
        "        exits[sym] = (c < cloud_bottom).astype(float)\n"
        "        \n"
        "    return latch_signals(pd.DataFrame(entries), pd.DataFrame(exits))\n"
    ))

    # --- Execution Loop ---
    strategies = [
        ("OBV Full Divergence", "strategy_obv_full_divergence"),
        ("A/D Full Divergence", "strategy_ad_full_divergence"),
        ("RSI Full Divergence", "strategy_rsi_full_divergence"),
        ("Stochastic Divergence", "strategy_stochastic_full_divergence"),
        ("MACD Hist Divergence", "strategy_macd_hist_divergence"),
        ("Aroon Trend", "strategy_aroon_reversal"),
        ("ADX Trend", "strategy_adx_bipolar"),
        ("CCI Trend", "strategy_cci_trend_latched"),
        ("Bollinger Breakout", "strategy_bb_trend_latched"),
        ("Ichimoku Cloud", "strategy_ichimoku_latched"),
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
