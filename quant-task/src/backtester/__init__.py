"""A small, modular backtesting toolkit.

This package is intentionally simple and notebook-friendly:
- models produce either alpha scores or target portfolio weights
- an optional allocator turns alphas into weights (with optional sector constraints)
- an optional macro model can scale risk up/down through time
- the engine turns weights + prices into a portfolio equity curve and stats
"""

from .data import load_cleaned_assets
from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import compute_performance_stats
from .models import (
    RiskOnOffMacroModel,
    SMACrossoverMicroModel,
    HeikenAshiMicroModel,
    TrendHeikenAshiFibModel,
    EqualWeightAllocator,
    MPTAllocator,
    SectorAllocationPostProcessor,
    TopKLongShortAllocator,
    WeightsFromSignalsModel,
)
from .plots import plot_backtest_result
from .stat_arb import PairTradingModel, compute_pair_diagnostics, plot_pair_diagnostics

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "PairTradingModel",
    "RiskOnOffMacroModel",
    "SMACrossoverMicroModel",
    "HeikenAshiMicroModel",
    "TrendHeikenAshiFibModel",
    "EqualWeightAllocator",
    "MPTAllocator",
    "SectorAllocationPostProcessor",
    "TopKLongShortAllocator",
    "WeightsFromSignalsModel",
    "compute_pair_diagnostics",
    "compute_performance_stats",
    "load_cleaned_assets",
    "plot_backtest_result",
    "plot_pair_diagnostics",
    "run_backtest",
]
