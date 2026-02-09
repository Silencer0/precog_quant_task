from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from .data import align_close_prices


@dataclass(frozen=True)
class SMACrossoverMicroModel:
    """Per-asset SMA crossover -> alpha scores in {-1, +1}.

    This is a "micro" model: it uses each asset's own history only.
    """

    fast: int = 20
    slow: int = 100

    def compute_alpha(self, assets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        close = align_close_prices(assets)
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        alpha = np.sign(fast_ma - slow_ma)
        alpha = alpha.replace(0.0, np.nan).ffill().fillna(0.0)
        return alpha


@dataclass(frozen=True)
class RiskOnOffMacroModel:
    """A simple macro model: scale risk based on a market proxy trend.

    Market proxy is an equal-weighted average of asset closes.

    Output is a scalar in {0, 1} per date:
    - 1 if proxy close above its SMA
    - 0 otherwise
    """

    lookback: int = 200

    def compute_risk_scale(self, assets: Mapping[str, pd.DataFrame]) -> pd.Series:
        close = align_close_prices(assets)
        proxy = close.mean(axis=1)
        sma = proxy.rolling(self.lookback).mean()
        scale = (proxy > sma).astype(float)
        scale = scale.fillna(0.0)
        scale.name = "risk_scale"
        return scale


@dataclass(frozen=True)
class TopKLongShortAllocator:
    """Allocator: turn alpha scores into dollar-neutral Top-K weights.

    At each date:
    - long the top K alphas equally
    - short the bottom K alphas equally
    - weights sum to 0, gross exposure sums to 1
    """

    k: int = 5

    def allocate(self, alpha: pd.DataFrame) -> pd.DataFrame:
        alpha = alpha.copy().fillna(0.0)
        w = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
        for dt, row in alpha.iterrows():
            ranks = row.sort_values(ascending=False)
            longs_idx = ranks.head(self.k).index
            shorts_idx = ranks.tail(self.k).index
            if isinstance(longs_idx, pd.Index):
                longs = [str(v) for v in longs_idx]
            else:
                longs = [str(longs_idx)]
            if isinstance(shorts_idx, pd.Index):
                shorts = [str(v) for v in shorts_idx]
            else:
                shorts = [str(shorts_idx)]
            if len(longs) > 0:
                w.loc[dt, longs] = 1.0 / len(longs)
            if len(shorts) > 0:
                w.loc[dt, shorts] = -1.0 / len(shorts)
        gross = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(gross, axis=0).fillna(0.0)
        return w


@dataclass(frozen=True)
class SectorAllocationPostProcessor:
    """Optional portfolio-management hook: enforce sector gross exposures.

    This does not require sectors to exist; if provided, it can be used to map a
    fund's sector allocation outputs into per-asset weights.

    Inputs:
    - sector_map: asset -> sector
    - sector_gross_targets: sector -> target gross exposure (non-negative)

    Behavior (per date):
    - compute current gross exposure per sector: sum(|w_i|)
    - scale all weights within each sector so gross matches target
    - renormalize overall gross exposure back to 1
    """

    sector_map: Dict[str, str]
    sector_gross_targets: Dict[str, float]

    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        w = weights.copy().fillna(0.0)
        sectors = sorted(set(self.sector_map.values()))
        for dt in w.index:
            row = w.loc[dt]
            for sec in sectors:
                target = float(self.sector_gross_targets.get(sec, np.nan))
                if not np.isfinite(target):
                    continue
                members = [
                    a for a, s in self.sector_map.items() if s == sec and a in w.columns
                ]
                if not members:
                    continue
                gross = float(row[members].abs().sum())
                if gross <= 0:
                    continue
                row[members] = row[members] * (target / gross)
            w.loc[dt] = row

        gross_total = w.abs().sum(axis=1).replace(0.0, np.nan)
        w = w.div(gross_total, axis=0).fillna(0.0)
        return w


@dataclass(frozen=True)
class WeightsFromSignalsModel:
    """ "Regular" model wrapper: return weights directly.

    `signals` should be a DataFrame aligned to Close prices:
    - index: dates
    - columns: assets
    - values: desired weights (will be normalized by the engine unless allow_leverage=True)
    """

    signals: pd.DataFrame

    def compute_weights(self) -> pd.DataFrame:
        return self.signals.copy()


@dataclass(frozen=True)
class HeikenAshiMicroModel:
    """Heiken Ashi strategy macro/micro model.

    Logic:
    - 1.0 on FIRST green candle (HA_Close > HA_Open) -> Buy Entry
    - -1.0 on FIRST red candle (HA_Close < HA_Open) -> Sell Exit
    - 0.0 otherwise -> Hold
    """

    def compute_signals(self, assets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from src.features.core import heiken_ashi

        out = {}
        for sym, df in assets.items():
            ha = heiken_ashi(df["Open"], df["High"], df["Low"], df["Close"])
            green = ha["ha_close"] > ha["ha_open"]
            red = ha["ha_close"] < ha["ha_open"]

            first_green = green & (~green.shift(1).fillna(False))
            first_red = red & (~red.shift(1).fillna(False))

            sig = pd.Series(0.0, index=df.index)
            sig.loc[first_green] = 1.0
            sig.loc[first_red] = -1.0
            out[sym] = sig
        return pd.DataFrame(out)

    def compute_states(self, assets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from src.features.core import heiken_ashi

        out = {}
        for sym, df in assets.items():
            ha = heiken_ashi(df["Open"], df["High"], df["Low"], df["Close"])
            green = ha["ha_close"] > ha["ha_open"]
            red = ha["ha_close"] < ha["ha_open"]

            state = pd.Series(0.0, index=df.index)
            is_long = False
            for i in range(len(df)):
                g = bool(green.iat[i])
                r = bool(red.iat[i])
                if g and not is_long:
                    is_long = True
                elif r and is_long:
                    is_long = False
                if is_long:
                    state.iat[i] = 1.0
            out[sym] = state
        return pd.DataFrame(out)


@dataclass(frozen=True)
class TrendHeikenAshiFibModel:
    """Trend-following Heiken Ashi + Fibonacci strategy.

    Logic:
    - Trend Confirmation: Confirmed Uptrend if last non-zero MSS (Market Structure Shift) was Bullish (+1).
    - Entry: First Green Heiken Ashi candle while in a Confirmed Uptrend.
    - Exit: Heiken Ashi Red candle AND price crosses below the 61.8% Fibonacci retracement level.
    """

    fib_level: str = "fib_61_8"
    lookback_window: int = 60

    def compute_signals(self, assets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        from src.features.core import fib_levels, heiken_ashi, market_structure_signals

        out = {}
        for sym, df in assets.items():
            # 1. Heiken Ashi
            ha = heiken_ashi(df["Open"], df["High"], df["Low"], df["Close"])
            green = ha["ha_close"] > ha["ha_open"]
            red = ha["ha_close"] < ha["ha_open"]

            # 2. Trend Confirmation (MSS/CHoCH/BOS)
            try:
                mss_df = market_structure_signals(df["High"], df["Low"], df["Close"])
                # Uptrend if the last shift was bullish (+1)
                trend = mss_df["mss"].replace(0, np.nan).ffill().fillna(0)
                uptrend = trend == 1
            except Exception:
                uptrend = pd.Series(False, index=df.index)

            # 3. Fibonacci Levels
            fibs = fib_levels(df["High"], df["Low"], window=self.lookback_window)
            fib_support = fibs[self.fib_level]

            # 4. Entry: First Green while Uptrend is Active
            valid_green = green & uptrend
            entry = valid_green & (~valid_green.shift(1).fillna(False))

            # 5. Exit: Red & Break below Fib support
            exit_signal = red & (ha["ha_close"] < fib_support)

            sig = pd.Series(0.0, index=df.index)
            sig.loc[entry] = 1.0
            sig.loc[exit_signal] = -1.0
            out[sym] = sig

        return pd.DataFrame(out)


@dataclass(frozen=True)
class EqualWeightAllocator:
    """Allocator: 1/N allocation across assets with non-zero alpha/state."""

    def allocate(self, alpha: pd.DataFrame) -> pd.DataFrame:
        mask = (alpha > 0).astype(float)
        row_sums = mask.sum(axis=1).replace(0.0, np.nan)
        return mask.div(row_sums, axis=0).fillna(0.0)


@dataclass(frozen=True)
class MPTAllocator:
    """Allocator: Mean-Variance Optimization across active assets."""

    lookback: int = 126

    def allocate(
        self, state: pd.DataFrame, assets: Mapping[str, pd.DataFrame]
    ) -> pd.DataFrame:
        from .portfolio import optimize_mpt

        close = align_close_prices(assets)
        returns = close.pct_change()

        w_df = pd.DataFrame(0.0, index=state.index, columns=state.columns)
        prev_cand_set = set()
        last_weights = pd.Series(0.0, index=state.columns)

        for dt, row in state.iterrows():
            candidates = list(row[row > 0].index)
            cand_set = set(candidates)

            if not cand_set:
                last_weights = pd.Series(0.0, index=state.columns)
                prev_cand_set = set()
            elif cand_set != prev_cand_set:
                weights_dict = optimize_mpt(
                    returns, candidates, dt, lookback_days=self.lookback
                )
                last_weights = pd.Series(0.0, index=state.columns)
                for sym, w in weights_dict.items():
                    last_weights[sym] = w
                prev_cand_set = cand_set

            w_df.loc[dt] = last_weights

        return w_df


def combine_models_to_weights(
    *,
    assets: Mapping[str, pd.DataFrame],
    micro_model: SMACrossoverMicroModel
    | HeikenAshiMicroModel
    | TrendHeikenAshiFibModel
    | None = None,
    macro_model: RiskOnOffMacroModel | None = None,
    allocator: TopKLongShortAllocator
    | EqualWeightAllocator
    | MPTAllocator
    | None = None,
    regular_weights_model: WeightsFromSignalsModel | None = None,
    sector_post: SectorAllocationPostProcessor | None = None,
) -> pd.DataFrame:
    """Combine optional components to produce a final weights time series.

    Rules (simple, deterministic):
    - If `regular_weights_model` is provided, its weights are the baseline.
      Otherwise, micro->allocator produces weights.
    - If `macro_model` is provided, multiply all weights by risk_scale(t).
    """
    close = align_close_prices(assets)
    index = close.index

    if regular_weights_model is not None:
        w = regular_weights_model.compute_weights().reindex(index).fillna(0.0)
    else:
        if micro_model is None:
            raise ValueError("Provide either regular_weights_model or micro_model.")

        if isinstance(micro_model, (HeikenAshiMicroModel, TrendHeikenAshiFibModel)):
            if isinstance(allocator, (EqualWeightAllocator, MPTAllocator)):
                if isinstance(micro_model, TrendHeikenAshiFibModel):
                    # For trend model, we still need states to allocate correctly
                    # but signals are discrete. We'll derive states from signals for allocation
                    # OR we could just use compute_signals directly.
                    # For simplicity, let's just use compute_signals for now as weights
                    w = micro_model.compute_signals(assets).reindex(index).fillna(0.0)
                else:
                    state = micro_model.compute_states(assets)
                    if isinstance(allocator, MPTAllocator):
                        w = allocator.allocate(state, assets)
                    else:
                        w = allocator.allocate(state)
            else:
                # Default to discrete signals if no allocator or standard allocator
                w = micro_model.compute_signals(assets).reindex(index).fillna(0.0)
        else:
            if allocator is None:
                raise ValueError("SMACrossoverMicroModel requires an allocator.")
            alpha = micro_model.compute_alpha(assets)
            w = allocator.allocate(alpha).reindex(index).fillna(0.0)

    if macro_model is not None:
        scale = macro_model.compute_risk_scale(assets).reindex(index).fillna(0.0)
        w = w.mul(scale, axis=0)

    if sector_post is not None:
        w = sector_post.apply(w)

    return w
