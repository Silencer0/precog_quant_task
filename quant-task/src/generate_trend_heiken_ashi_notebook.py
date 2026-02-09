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

    cells.append(
        _md(
            "# Advanced Heiken Ashi Strategy: Trend Confirmation & Fib Exits\n"
            "This notebook implements a sophisticated trading strategy combining:\n"
            "1.  **Trend Confirmation**: Using Market Structure Shift (MSS/BOS) to ensure trades only occur in confirmed upward trends.\n"
            "2.  **Entry Signal**: First Green Heiken Ashi candle after trend confirmation.\n"
            "3.  **Exit Strategy**: Sell only if the candle is Red **AND** price has broken below a key Fibonacci retracement level (61.8%).\n"
        )
    )

    cells.append(_md("## 0) Setup\n"))

    cells.append(
        _code(
            "import os, sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from bokeh.io import output_notebook, show\n"
            "\n"
            "# --- Setup correct working directory (ROOT) ---\n"
            "if os.getcwd().endswith('notebooks'):\n"
            "    os.chdir('..')\n"
            "\n"
            "ROOT = os.getcwd()\n"
            "if ROOT not in sys.path:\n"
            "    sys.path.insert(0, ROOT)\n"
            "\n"
            "from src.backtester.data import load_cleaned_assets, align_close_prices\n"
            "from src.backtester.engine import BacktestConfig, run_backtest\n"
            "from src.backtester.metrics import compute_performance_stats\n"
            "from src.backtester.models import TrendHeikenAshiFibModel\n"
            "from src.backtester.bokeh_plots import build_interactive_portfolio_layout\n"
            "from src.backtester.report import compute_backtest_report\n"
            "\n"
            "output_notebook()\n"
        )
    )

    cells.append(_md("## 1) Data Loading\n"))

    cells.append(
        _code(
            "assets = load_cleaned_assets(symbols=None)\n"
            "close = align_close_prices(assets)\n"
            "\n"
            "cfg = BacktestConfig(\n"
            "    initial_equity=1_000_000,\n"
            "    transaction_cost_bps=5,\n"
            "    mode='event_driven',\n"
            "    strict_signals=True, # Discrete entry/exit\n"
            "    stop_loss_pct=0.0\n"
            ")\n"
            "\n"
            "model = TrendHeikenAshiFibModel(fib_level='fib_61_8', lookback_window=60)\n"
            "print(f'Loaded {len(assets)} assets.')\n"
        )
    )

    cells.append(_md("## 2) Backtest Execution\n"))

    cells.append(
        _code(
            "print('Computing Signals...')\n"
            "signals = model.compute_signals(assets)\n"
            "\n"
            "print('Running Backtest...')\n"
            "res = run_backtest(close_prices=close, weights=signals, config=cfg)\n"
            "report = compute_backtest_report(result=res, close_prices=close)\n"
            "display(report)\n"
        )
    )

    cells.append(_md("## 3) Interactive Visualization\n"))

    cells.append(
        _code(
            "def build_market_proxy_ohlcv(assets, index):\n"
            "    opens = pd.concat([df['Open'].reindex(index).astype(float) for df in assets.values()], axis=1).mean(axis=1)\n"
            "    highs = pd.concat([df['High'].reindex(index).astype(float) for df in assets.values()], axis=1).mean(axis=1)\n"
            "    lows = pd.concat([df['Low'].reindex(index).astype(float) for df in assets.values()], axis=1).mean(axis=1)\n"
            "    closes = pd.concat([df['Close'].reindex(index).astype(float) for df in assets.values()], axis=1).mean(axis=1)\n"
            "    return pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})\n"
            "\n"
            "market_df = build_market_proxy_ohlcv(assets, close.index)\n"
            "\n"
            "layout = build_interactive_portfolio_layout(\n"
            "    market_ohlcv=market_df,\n"
            "    equity=res.equity,\n"
            "    returns=res.returns,\n"
            "    weights=res.weights,\n"
            "    turnover=res.turnover,\n"
            "    costs=res.costs,\n"
            "    close_prices=close, \n"
            "    title='Heiken Ashi Trend + Fib Strategy'\n"
            ")\n"
            "show(layout)\n"
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
    out_path = os.path.join("notebooks", "Heiken_Ashi_Trend_Fib_Strategy.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    create_notebook()
