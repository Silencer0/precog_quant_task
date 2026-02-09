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
            "# Heiken Ashi Trading Strategy: 1/N vs MPT\n"
            "This notebook explores the performance of a Heiken Ashi-based trading strategy using two distinct allocation models:\n"
            "1.  **1/N Allocation (Equal Weight)**: Active assets receive equal portions of the portfolio.\n"
            "2.  **MPT Allocation (Modern Portfolio Theory)**: Active assets are weighted using mean-variance optimization to maximize the Sharpe ratio.\n"
            "\n"
            "### Signals:\n"
            "- **Entry**: First Green Heiken Ashi candle.\n"
            "- **Exit**: First Red Heiken Ashi candle.\n"
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
            "from src.backtester.models import HeikenAshiMicroModel, EqualWeightAllocator, MPTAllocator, combine_models_to_weights\n"
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
            "    rebalance='D', # Daily rebalance to track optimal weights\n"
            "    mode='event_driven',\n"
            "    strict_signals=False # Weight-based tracking\n"
            ")\n"
            "\n"
            "model = HeikenAshiMicroModel()\n"
            "print(f'Loaded {len(assets)} assets.')\n"
        )
    )

    cells.append(_md("## 2) 1/N Backtest\n"))

    cells.append(
        _code(
            "print('Computing 1/N Weights...')\n"
            "w_ew = combine_models_to_weights(assets=assets, micro_model=model, allocator=EqualWeightAllocator())\n"
            "\n"
            "print('Running 1/N Backtest...')\n"
            "res_ew = run_backtest(close_prices=close, weights=w_ew, config=cfg)\n"
            "report_ew = compute_backtest_report(result=res_ew, close_prices=close)\n"
            "display(report_ew)\n"
        )
    )

    cells.append(_md("## 3) MPT Backtest\n"))

    cells.append(
        _code(
            "print('Computing MPT Weights (this may take a moment)...')\n"
            "w_mpt = combine_models_to_weights(assets=assets, micro_model=model, allocator=MPTAllocator(lookback=252))\n"
            "\n"
            "print('Running MPT Backtest...')\n"
            "res_mpt = run_backtest(close_prices=close, weights=w_mpt, config=cfg)\n"
            "report_mpt = compute_backtest_report(result=res_mpt, close_prices=close)\n"
            "display(report_mpt)\n"
        )
    )

    cells.append(_md("## 4) Comparison\n"))

    cells.append(
        _code(
            "plt.figure(figsize=(14, 7))\n"
            "plt.plot(res_ew.equity, label='1/N Allocation', color='#1f77b4', linewidth=2)\n"
            "plt.plot(res_mpt.equity, label='MPT Allocation', color='#2ca02c', linewidth=2)\n"
            "plt.title('Equity Curve Comparison: Heiken Ashi Strategy', fontsize=14)\n"
            "plt.xlabel('Date')\n"
            "plt.ylabel('Portfolio Value ($)')\n"
            "plt.legend()\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()\n"
        )
    )

    cells.append(_md("## 5) Detailed Interactive Analysis (MPT)\n"))

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
            "    equity=res_mpt.equity,\n"
            "    returns=res_mpt.returns,\n"
            "    weights=res_mpt.weights,\n"
            "    turnover=res_mpt.turnover,\n"
            "    costs=res_mpt.costs,\n"
            "    close_prices=close,\n"
            "    title='Heiken Ashi Strategy (MPT Allocation)'\n"
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
    out_path = os.path.join("notebooks", "Heiken_Ashi_Strategy_Comparison.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    create_notebook()
