from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult
from .metrics import compute_performance_stats


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _annualization_factor(index: pd.Index, *, default: int = 252) -> int:
    # Keep it simple and deterministic: default daily bars.
    # (The project dataset is daily; strategies are computed daily.)
    return default


def compute_backtest_report(
    *,
    result: BacktestResult,
    close_prices: Optional[pd.DataFrame] = None,
    benchmark: str = "equal_weight",
) -> pd.Series:
    """Return a rich statistics Series for a backtest.

    This is weights-based (not trade-ledger-based), so the report focuses on
    portfolio/equity/return + turnover/cost diagnostics.
    """
    eq = result.equity.dropna()
    rets = result.returns.reindex(eq.index).fillna(0.0)
    weights = result.weights.reindex(eq.index).fillna(0.0)
    turnover = result.turnover.reindex(eq.index).fillna(0.0)
    costs = result.costs.reindex(eq.index).fillna(0.0)

    if len(eq) == 0:
        raise ValueError("Empty equity series")

    # If the strategy is inactive at the beginning (all-zero weights), exclude that
    # flat prefix from performance statistics. This avoids reporting a misleading
    # start date and distorting CAGR/Sharpe with long stretches of 0 returns.
    gross_exposure = weights.abs().sum(axis=1)
    active_mask = gross_exposure > 0
    if bool(active_mask.any()):
        first_active = pd.Timestamp(active_mask[active_mask].index[0])
        eq = eq.loc[first_active:]
        rets = rets.loc[first_active:]
        weights = weights.loc[first_active:]
        turnover = turnover.loc[first_active:]
        costs = costs.loc[first_active:]
        gross_exposure = gross_exposure.loc[first_active:]

    start = pd.Timestamp(eq.index[0])
    end = pd.Timestamp(eq.index[-1])
    duration = end - start

    ppy = _annualization_factor(eq.index)
    perf = compute_performance_stats(equity=eq, returns=rets, periods_per_year=ppy)

    net_exposure = weights.sum(axis=1)
    exposure_time_pct = float((gross_exposure > 0).mean() * 100.0)

    peak = float(eq.max())
    final = float(eq.iloc[-1])
    initial = float(eq.iloc[0])

    best_day = float(rets.max())
    worst_day = float(rets.min())

    # Rebalance count: number of days with non-zero turnover.
    rebalance_days = int((turnover > 0).sum())

    total_cost = float(costs.sum())
    total_turnover = float(turnover.sum())

    bench_total_return = float("nan")
    bench_cagr = float("nan")
    if close_prices is not None and benchmark == "equal_weight":
        prices = close_prices.reindex(eq.index).sort_index()
        asset_rets = prices.pct_change().fillna(0.0)
        bench_rets = asset_rets.mean(axis=1)
        bench_eq = (1.0 + bench_rets).cumprod() * initial
        bench_total_return = float(bench_eq.iloc[-1] / bench_eq.iloc[0] - 1.0)
        n = max(1, len(bench_eq) - 1)
        bench_cagr = float((1.0 + bench_total_return) ** (ppy / n) - 1.0)

    out: Dict[str, Any] = {
        "Start": start,
        "End": end,
        "Duration": duration,
        "Initial Equity": initial,
        "Final Equity": final,
        "Equity Peak": peak,
        # Return/risk metrics are reported in percent points for readability.
        # Example: 3.90 (decimal) => 390 (%).
        "Total Return [%]": 100.0 * perf.total_return,
        "CAGR [%]": 100.0 * perf.cagr,
        "Volatility (ann) [%]": 100.0 * perf.vol_ann,
        "Sharpe": perf.sharpe,
        "Sortino": perf.sortino,
        "Max Drawdown [%]": 100.0 * perf.max_drawdown,
        "Calmar": perf.calmar,
        "Best Day [%]": 100.0 * best_day,
        "Worst Day [%]": 100.0 * worst_day,
        "Avg Gross Exposure": float(gross_exposure.mean()),
        "Avg Net Exposure": float(net_exposure.mean()),
        "Exposure Time [%]": exposure_time_pct,
        "Rebalance Days": rebalance_days,
        "Total Turnover": total_turnover,
        "Avg Daily Turnover": float(turnover.mean()),
        "Transaction Cost (bps)": float(result.config.transaction_cost_bps),
        "Total Costs": total_cost,
        "Costs / Initial Equity [%]": (100.0 * float(total_cost / initial))
        if initial != 0
        else float("nan"),
        "Benchmark Total Return [%]": 100.0 * bench_total_return,
        "Benchmark CAGR [%]": 100.0 * bench_cagr,
    }

    # Keep a stable ordering similar to popular backtest reports
    order = [
        "Start",
        "End",
        "Duration",
        "Initial Equity",
        "Final Equity",
        "Equity Peak",
        "Total Return [%]",
        "CAGR [%]",
        "Volatility (ann) [%]",
        "Sharpe",
        "Sortino",
        "Max Drawdown [%]",
        "Calmar",
        "Best Day [%]",
        "Worst Day [%]",
        "Avg Gross Exposure",
        "Avg Net Exposure",
        "Exposure Time [%]",
        "Rebalance Days",
        "Total Turnover",
        "Avg Daily Turnover",
        "Transaction Cost (bps)",
        "Total Costs",
        "Costs / Initial Equity [%]",
        "Benchmark Total Return [%]",
        "Benchmark CAGR [%]",
    ]

    return pd.Series({k: out[k] for k in order}, name="Backtest Report")
