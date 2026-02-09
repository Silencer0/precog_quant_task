from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceStats:
    cagr: float
    vol_ann: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    total_return: float


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def compute_performance_stats(
    *,
    equity: pd.Series,
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> PerformanceStats:
    eq = equity.dropna()
    rets = returns.reindex(eq.index).fillna(0.0)

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    n = max(1, len(eq) - 1)
    cagr = float((1.0 + total_return) ** (periods_per_year / n) - 1.0)

    vol_ann = (
        float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if len(rets) > 1 else 0.0
    )
    excess = rets - (risk_free_rate / periods_per_year)
    sharpe = (
        float(excess.mean() / (rets.std(ddof=1) + 1e-12) * np.sqrt(periods_per_year))
        if len(rets) > 1
        else 0.0
    )

    downside = rets.clip(upper=0.0)
    downside_std = float(downside.std(ddof=1) + 1e-12) if len(rets) > 1 else 0.0
    sortino = (
        float(excess.mean() / downside_std * np.sqrt(periods_per_year))
        if len(rets) > 1
        else 0.0
    )

    max_dd = _max_drawdown(eq)
    calmar = float(cagr / (-max_dd + 1e-12))

    return PerformanceStats(
        cagr=cagr,
        vol_ann=vol_ann,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        total_return=total_return,
    )
