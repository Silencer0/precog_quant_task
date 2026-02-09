from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 1_000_000.0
    transaction_cost_bps: float = 5.0
    rebalance: Optional[str] = "D"
    allow_leverage: bool = False
    mode: Literal["event_driven", "vectorized"] = "event_driven"
    trade_buffer: float = 0.0
    no_sell: bool = False

    # If True, the engine operates in 'Signal-Driven' mode:
    # Signals > 0: Buy Entry (Allocation relative to portfolio value)
    # Signals < 0: Sell Exit (Sell all units of that asset)
    # Signals == 0: Hold (Ignore current position, don't rebalance)
    strict_signals: bool = False

    # Stop Loss Percentage (e.g., 0.05 for 5% loss).
    # If price drops below Average Entry Price * (1 - stop_loss_pct), sell immediately.
    stop_loss_pct: float = 0.0          # e.g. 0.05 for 5%
    stop_loss_type: Literal["hard", "trailing"] = "trailing"


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    config: BacktestConfig


def _normalize_weights(weights: pd.DataFrame, *, allow_leverage: bool) -> pd.DataFrame:
    w = weights.copy()
    w = w.fillna(0.0)
    if allow_leverage:
        return w
    gross = w.abs().sum(axis=1)
    gross = gross.replace(0.0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)
    return w


def _run_vectorized(
    close_prices: pd.DataFrame, 
    weights: pd.DataFrame, 
    cfg: BacktestConfig
) -> BacktestResult:
    prices = close_prices.sort_index()
    w = weights.reindex(prices.index).ffill().fillna(0.0)
    w = w.reindex(columns=prices.columns, fill_value=0.0)
    
    if cfg.rebalance and cfg.rebalance != "D":
        anchor = w.resample(cfg.rebalance).last()
        w = anchor.reindex(w.index, method="ffill").fillna(0.0)

    w = _normalize_weights(w, allow_leverage=cfg.allow_leverage)
    asset_rets = prices.pct_change().fillna(0.0)
    w_prev = w.shift(1).fillna(0.0)
    gross_ret = (w_prev * asset_rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    costs = (cfg.transaction_cost_bps / 10_000.0) * turnover
    net_ret = gross_ret - costs
    equity = (1.0 + net_ret).cumprod() * float(cfg.initial_equity)

    return BacktestResult(
        equity=equity.rename("Equity"),
        returns=net_ret.rename("Return"),
        weights=w,
        turnover=turnover.rename("Turnover"),
        costs=costs.rename("Costs"),
        config=cfg,
    )


def _run_event_driven(
    close_prices: pd.DataFrame, 
    weights: pd.DataFrame, 
    cfg: BacktestConfig,
    open_prices: pd.DataFrame | None = None,
    high_prices: pd.DataFrame | None = None,
    low_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    prices = close_prices.sort_index()
    target_weights_df = weights.reindex(prices.index).ffill().fillna(0.0)
    target_weights_df = target_weights_df.reindex(columns=prices.columns, fill_value=0.0)
    
    # Normalization only applies to standard mode. 
    # In strict mode, we use raw signal values for allocation.
    if cfg.strict_signals:
        target_weights_vals = target_weights_df.values
    else:
        target_weights_vals = _normalize_weights(target_weights_df, allow_leverage=cfg.allow_leverage).values
        
    price_vals = prices.values
    n_days, n_assets = price_vals.shape
    
    # Pre-fetch OHLC arrays if available for Intraday Logic
    open_vals = open_prices.values if open_prices is not None else price_vals
    high_vals = high_prices.values if high_prices is not None else price_vals
    low_vals = low_prices.values if low_prices is not None else price_vals
    
    if cfg.rebalance:
        rebal_idx = pd.Series(index=prices.index, data=np.arange(n_days))
        rebal_days_idx = rebal_idx.resample(cfg.rebalance).last().dropna().astype(int).values
        if 0 not in rebal_days_idx:
            rebal_days_idx = np.insert(rebal_days_idx, 0, 0)
        is_rebal_day = np.zeros(n_days, dtype=bool)
        is_rebal_day[rebal_days_idx] = True
    else:
        is_rebal_day = np.zeros(n_days, dtype=bool)

    cash = float(cfg.initial_equity)
    units = np.zeros(n_assets)
    equity_curve = np.zeros(n_days)
    turnover_curve = np.zeros(n_days)
    cost_curve = np.zeros(n_days)
    weight_history = np.zeros((n_days, n_assets))
    prev_target = np.zeros(n_assets)
    avg_entry_price = np.zeros(n_assets) # Track average entry price for Stop Loss
    high_water_mark = np.zeros(n_assets)
    
    # State tracking for strict signals
    stopped_out = np.zeros(n_assets, dtype=bool)
    
    for t in range(n_days):
        p_t = price_vals[t]
        o_t = open_vals[t]
        h_t = high_vals[t] # Added h_t
        l_t = low_vals[t]
        
        valid_mask = np.isfinite(p_t) & (p_t > 0)
        
        # Reset 'stopped_out' when signal goes to 0 or negative
        tw_t = target_weights_vals[t]
        stopped_out[tw_t <= 1e-6] = False

        # --- STOP LOSS LOGIC (INTRADAY) ---
        # Check if Low of the day hit the Stop Price
        if cfg.stop_loss_pct > 0:
            # Calculate Stop Prices
            # Condition: Has Position
            has_pos = (units > 0)
            if np.any(has_pos):
                if cfg.stop_loss_type == "trailing":
                    # Trailing Stop: Update high water mark
                    high_water_mark = np.maximum(high_water_mark, h_t)
                    # Reset for new positions if needed
                    high_water_mark[units <= 0] = p_t[units <= 0]
                    sl_thresholds = high_water_mark * (1.0 - cfg.stop_loss_pct)
                else:
                    # Hard Stop: relative to avg_entry_price
                    sl_thresholds = avg_entry_price * (1.0 - cfg.stop_loss_pct)
                
                # Hit if Low < SL Threshold
                sl_hit_mask = has_pos & (l_t < sl_thresholds) & valid_mask
                
                if np.any(sl_hit_mask):
                    sl_indices = np.where(sl_hit_mask)[0]
                    
                    # Execute Sells
                    # Price Logic:
                    # If Open < SL (Gap Down), we sell at Open (Simulate panic sell at open)
                    # Else, we sell exacty at SL (Simulate Stop Order trigger)
                    # Note: real slippage would mean slightly below SL, but SL is fair approx.
                    
                    exec_prices = np.where(
                        o_t[sl_indices] < sl_thresholds[sl_indices],
                        o_t[sl_indices],       # Gap Down case
                        sl_thresholds[sl_indices] # Intraday trigger case
                    )
                    
                    sl_units = units[sl_indices]
                    sl_val = np.sum(sl_units * exec_prices)
                    sl_cost = sl_val * (cfg.transaction_cost_bps / 10_000.0)
                    
                    cash += sl_val - sl_cost
                    turnover_curve[t] += sl_val 
                    cost_curve[t] += sl_cost
                    
                    units[sl_indices] = 0.0
                    avg_entry_price[sl_indices] = 0.0
                    stopped_out[sl_indices] = True # Mark as stopped out for this signal cycle
        
        current_holding_val = np.sum(units[valid_mask] * p_t[valid_mask])
        portfolio_val = cash + current_holding_val
        target_w = target_weights_vals[t]
        
        should_trade = False
        if cfg.rebalance:
            if is_rebal_day[t]: should_trade = True
        else:
            if not np.allclose(target_w, prev_target, atol=1e-6): should_trade = True
        
        if should_trade and cfg.trade_buffer > 0:
            if portfolio_val > 1e-6:
                current_w = np.zeros(n_assets)
                current_w[valid_mask] = (units[valid_mask] * p_t[valid_mask]) / portfolio_val
                deviation = np.sum(np.abs(current_w - target_w))
                if deviation < cfg.trade_buffer: should_trade = False

        turnover_t = 0.0
        cost_t = 0.0
        
        if should_trade and np.any(valid_mask):
            # --- EXECUTION LOGIC ---
            if cfg.strict_signals:
                # ENTRY (target > 0) / EXIT (target < 0) / HOLD (target == 0)
                # 1. Sells First
                sell_mask = (target_w < -1e-6) & (units > 0)
                if np.any(sell_mask):
                    v_sell = units[sell_mask]
                    v_price = p_t[sell_mask]
                    total_sell_val = np.sum(v_sell * v_price)
                    sell_fee = total_sell_val * (cfg.transaction_cost_bps / 10_000.0)
                    cash += total_sell_val - sell_fee
                    turnover_t += total_sell_val
                    cost_t += sell_fee
                    
                    if cfg.mode == 'debug':
                        print(f"DEBUG {t}: Selling {np.where(sell_mask)[0]} - Signal {target_w[sell_mask]}")
                    
                    units[sell_mask] = 0
                    avg_entry_price[sell_mask] = 0.0 # Reset entry price
                
                # 2. Buys: Only buy if signal > 0 AND not already in position AND NOT stopped out
                buy_indices = np.where((target_w > 1e-6) & (units <= 0) & (~stopped_out) & valid_mask)[0]
                for i in buy_indices:
                    alloc = target_w[i]
                    desired_cost = portfolio_val * alloc
                    if cash > 1e-2:
                        actual_cost = min(desired_cost, cash * 0.999)
                        buy_fee = actual_cost * (cfg.transaction_cost_bps / 10_000.0)
                        if cash >= (actual_cost + buy_fee):
                            new_units = actual_cost / p_t[i]
                            
                            if cfg.mode == 'debug':
                                print(f"DEBUG {t}: Buying {i} - Signal {alloc}")
                            
                            old_units = units[i]
                            total_new_units = old_units + new_units
                            if total_new_units > 0:
                                avg_entry_price[i] = (old_units * avg_entry_price[i] + new_units * p_t[i]) / total_new_units
                                
                            units[i] += new_units
                            cash -= (actual_cost + buy_fee)
                            turnover_t += actual_cost
                            cost_t += buy_fee
            else:
                # Standard Weight-Based
                desired_val = portfolio_val * target_w
                ideal_units = np.zeros(n_assets)
                ideal_units[valid_mask] = desired_val[valid_mask] / p_t[valid_mask]
                delta_units = ideal_units - units
                
                if cfg.no_sell:
                    # ... [No Sell Logic remains same, omitted for brevity if unchanged logic is compatible] ...
                    allowed_sell_mask = (delta_units < 0) & (target_w < (prev_target - 1e-6))
                    new_delta = np.zeros_like(delta_units)
                    new_delta[allowed_sell_mask] = delta_units[allowed_sell_mask]
                    
                    sell_v = np.sum(np.abs(new_delta[new_delta < 0] * p_t[new_delta < 0]))
                    sell_c = sell_v * (cfg.transaction_cost_bps / 10_000.0)
                    cash_plus = np.sum(-new_delta[new_delta < 0] * p_t[new_delta < 0]) - sell_c
                    
                    temp_cash = cash + cash_plus
                    buy_m = (delta_units > 0)
                    if np.any(buy_m):
                        cost_f = 1.0 + (cfg.transaction_cost_bps / 10_000.0)
                        need_c = np.sum(delta_units[buy_m] * p_t[buy_m]) * cost_f
                        ratio = min(1.0, temp_cash / need_c) if need_c > 0 else 1.0
                        new_delta[buy_m] = delta_units[buy_m] * ratio
                    delta_units = new_delta

                # Execute Standard Trades
                trade_v_abs = np.abs(delta_units * p_t)
                trade_c = np.sum(trade_v_abs) * (cfg.transaction_cost_bps / 10_000.0)
                
                # Update Entry Prices for Buys
                buy_indices = np.where((delta_units > 0) & valid_mask)[0]
                for i in buy_indices:
                    d_u = delta_units[i]
                    old_u = units[i]
                    # Only update if we are adding to position
                    if old_u >= 0:
                        new_total = old_u + d_u
                        if new_total > 0:
                             avg_entry_price[i] = (old_u * avg_entry_price[i] + d_u * p_t[i]) / new_total
                    else:
                        # Covering short (not fully supported by simple logic, but treat as reset if flipping)
                        if d_u > abs(old_u): # Net long
                            avg_entry_price[i] = p_t[i]

                units = units + delta_units
                cash = cash - np.sum(delta_units * p_t) - trade_c
                turnover_t = np.sum(trade_v_abs)
                cost_t = trade_c
                
                # Reset Entry Price for Sells (Closed positions)
                # If units drop to essentially zero
                closed_mask = (np.abs(units) < 1e-6)
                avg_entry_price[closed_mask] = 0.0

            prev_target = target_w
        
        # Accumulate turnover/cost if Stop Loss triggered earlier
        turnover_curve[t] += turnover_t
        cost_curve[t] += cost_t
        
        final_holding_val = np.sum(units[valid_mask] * p_t[valid_mask])
        final_equity = cash + final_holding_val
        
        equity_curve[t] = final_equity
        if final_equity > 1e-6:
            weight_history[t, valid_mask] = (units[valid_mask] * p_t[valid_mask]) / final_equity
        
    res_equity = pd.Series(equity_curve, index=prices.index, name="Equity")
    res_equity.iloc[0] = cfg.initial_equity
    return BacktestResult(
        equity=res_equity,
        returns=res_equity.pct_change().fillna(0.0),
        weights=pd.DataFrame(weight_history, index=prices.index, columns=prices.columns),
        turnover=pd.Series(turnover_curve, index=prices.index),
        costs=pd.Series(cost_curve, index=prices.index),
        config=cfg,
    )


def run_backtest(
    close_prices: pd.DataFrame,
    weights: pd.DataFrame,
    config: BacktestConfig | None = None,
    open_prices: pd.DataFrame | None = None,
    high_prices: pd.DataFrame | None = None,
    low_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    if cfg.mode == "vectorized":
        return _run_vectorized(close_prices, weights, cfg)

    # Standardize OHLC indexes if provided
    o = open_prices.sort_index().reindex(close_prices.index).ffill() if open_prices is not None else None
    h = high_prices.sort_index().reindex(close_prices.index).ffill() if high_prices is not None else None
    l = low_prices.sort_index().reindex(close_prices.index).ffill() if low_prices is not None else None

    return _run_event_driven(close_prices, weights, cfg, open_prices=o, high_prices=h, low_prices=l)
