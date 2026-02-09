from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd

from .engine import BacktestResult
from .metrics import PerformanceStats


def plot_backtest_result(
    *,
    result: BacktestResult,
    stats: PerformanceStats | None = None,
    title: str = "Backtest",
) -> None:
    eq = result.equity
    rets = result.returns
    dd = eq / eq.cummax() - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title)
    axes[0].plot(eq.index, eq.values, label="Equity")
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dd.index, dd.values, label="Drawdown")
    axes[1].set_title("Drawdown (Equity / Peak - 1)")
    axes[1].set_ylabel("Drawdown")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rets.index, rets.values, label="Daily Return")
    axes[2].set_title("Daily Returns")
    axes[2].set_ylabel("Return")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].grid(True, alpha=0.3)

    if stats is not None:
        s = asdict(stats)
        lines = [
            f"CAGR: {s['cagr']:.2%}",
            f"Vol (ann): {s['vol_ann']:.2%}",
            f"Sharpe: {s['sharpe']:.2f}",
            f"Max DD: {s['max_drawdown']:.2%}",
            f"Total: {s['total_return']:.2%}",
        ]
        axes[0].text(
            0.01,
            0.98,
            "\n".join(lines),
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.show()


def plot_weights_heatmap(weights: pd.DataFrame, *, title: str = "Weights") -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(weights.T.values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_yticks(range(weights.shape[1]))
    ax.set_yticklabels(list(weights.columns))
    ax.set_xlabel("Time")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.show()
