from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import itertools

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure
from bokeh.layouts import column


@dataclass(frozen=True)
class TradeEvent:
    time: pd.Timestamp
    symbol: str
    side: str  # 'long' or 'short'
    action: str  # 'entry' or 'exit'
    weight: float
    price: float
    equity: float | None
    turnover: float | None
    cost: float | None


@dataclass(frozen=True)
class _OpenTrade:
    entry_time: pd.Timestamp
    entry_price: float
    entry_weight: float
    side: str


def _as_series(x: pd.Series) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        raise ValueError("Expected a Series")
    return x


def _compute_symbol_trade_events(
    *,
    symbol: str,
    close: pd.Series,
    weights: pd.Series,
    equity: pd.Series | None = None,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derive entry/exit events from weight sign changes.

    This backtester is weights-based (no explicit orders). For visualization,
    we interpret a "trade" as contiguous periods where weight != 0.
    """
    c = _as_series(close).astype(float)
    w = _as_series(weights).reindex(c.index).fillna(0.0).astype(float)

    eq = equity.reindex(c.index).astype(float) if equity is not None else None
    to = turnover.reindex(c.index).astype(float) if turnover is not None else None
    co = costs.reindex(c.index).astype(float) if costs is not None else None

    w_prev = w.shift(1).fillna(0.0)

    # A position exists if weight != 0.
    pos = np.sign(w)
    pos_prev = np.sign(w_prev)

    # Entry: previously flat, now non-flat
    is_entry = (pos_prev == 0) & (pos != 0)
    # Exit: previously non-flat, now flat
    is_exit = (pos_prev != 0) & (pos == 0)
    # Flip: sign changes without going flat -> treat as exit+entry
    is_flip = (pos_prev != 0) & (pos != 0) & (pos_prev != pos)

    events: list[TradeEvent] = []
    for t in c.index:
        if bool(is_flip.loc[t]):
            prev_side = "long" if float(pos_prev.loc[t]) > 0 else "short"
            next_side = "long" if float(pos.loc[t]) > 0 else "short"
            events.append(
                TradeEvent(
                    time=pd.Timestamp(t),
                    symbol=symbol,
                    side=prev_side,
                    action="exit",
                    weight=float(w_prev.loc[t]),
                    price=float(c.loc[t]),
                    equity=float(eq.loc[t]) if eq is not None else None,
                    turnover=float(to.loc[t]) if to is not None else None,
                    cost=float(co.loc[t]) if co is not None else None,
                )
            )
            events.append(
                TradeEvent(
                    time=pd.Timestamp(t),
                    symbol=symbol,
                    side=next_side,
                    action="entry",
                    weight=float(w.loc[t]),
                    price=float(c.loc[t]),
                    equity=float(eq.loc[t]) if eq is not None else None,
                    turnover=float(to.loc[t]) if to is not None else None,
                    cost=float(co.loc[t]) if co is not None else None,
                )
            )
            continue

        if bool(is_entry.loc[t]):
            side = "long" if float(pos.loc[t]) > 0 else "short"
            events.append(
                TradeEvent(
                    time=pd.Timestamp(t),
                    symbol=symbol,
                    side=side,
                    action="entry",
                    weight=float(w.loc[t]),
                    price=float(c.loc[t]),
                    equity=float(eq.loc[t]) if eq is not None else None,
                    turnover=float(to.loc[t]) if to is not None else None,
                    cost=float(co.loc[t]) if co is not None else None,
                )
            )

        if bool(is_exit.loc[t]):
            side = "long" if float(pos_prev.loc[t]) > 0 else "short"
            events.append(
                TradeEvent(
                    time=pd.Timestamp(t),
                    symbol=symbol,
                    side=side,
                    action="exit",
                    weight=float(w_prev.loc[t]),
                    price=float(c.loc[t]),
                    equity=float(eq.loc[t]) if eq is not None else None,
                    turnover=float(to.loc[t]) if to is not None else None,
                    cost=float(co.loc[t]) if co is not None else None,
                )
            )

    events_df = pd.DataFrame(
        [
            {
                "time": e.time,
                "symbol": e.symbol,
                "side": e.side,
                "action": e.action,
                "weight": e.weight,
                "price": e.price,
                "equity": e.equity,
                "turnover": e.turnover,
                "cost": e.cost,
            }
            for e in events
        ]
    )

    # Build per-trade P/L at exits by pairing consecutive entry->exit.
    trades: list[dict[str, object]] = []
    open_trade: _OpenTrade | None = None
    for _, row in events_df.sort_values("time").iterrows():
        if row["action"] == "entry":
            open_trade = _OpenTrade(
                entry_time=pd.Timestamp(row["time"]),
                entry_price=float(row["price"]),
                entry_weight=float(row["weight"]),
                side=str(row["side"]),
            )
        elif row["action"] == "exit" and open_trade is not None:
            entry_price = float(open_trade.entry_price)
            exit_price = float(row["price"])
            side = str(open_trade.side)
            direction = 1.0 if side == "long" else -1.0
            ret = direction * (exit_price / entry_price - 1.0)
            trades.append(
                {
                    "entry_time": open_trade.entry_time,
                    "exit_time": row["time"],
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return_pct": float(ret),
                    "size": float(abs(float(open_trade.entry_weight))),
                }
            )
            open_trade = None

    trades_df = pd.DataFrame(trades)
    return events_df, trades_df


def build_interactive_backtest_layout(
    *,
    ohlcv: pd.DataFrame,
    symbol: str,
    equity: pd.Series,
    returns: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
    costs: pd.Series,
    indicator_overlay: Optional[Dict[str, pd.Series]] = None,
    indicator_panel: Optional[Dict[str, pd.Series]] = None,
    title: str = "Backtest",
):
    """Build a Bokeh layout with price+trades+indicators and equity/drawdown/P&L panels."""
    from bokeh.layouts import column
    from bokeh.models import ColumnDataSource, HoverTool, Span
    from bokeh.plotting import figure

    df = ohlcv.copy()
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("ohlcv.index must be a DatetimeIndex")

    close = df["Close"].astype(float)
    w_sym = (
        weights[symbol] if symbol in weights.columns else pd.Series(0.0, index=df.index)
    )
    events_df, trades_df = _compute_symbol_trade_events(
        symbol=symbol,
        close=close,
        weights=w_sym,
        equity=equity,
        turnover=turnover,
        costs=costs,
    )

    # Price chart
    price_src = ColumnDataSource(
        {
            "time": df.index,
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": close,
        }
    )

    p_price = figure(
        x_axis_type="datetime",
        height=350,
        width=1100,
        title=f"{title} - {symbol} (Price, Indicators, Trades)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p_price.line("time", "close", source=price_src, line_width=2, legend_label="Close")

    # Overlay indicators on price axis
    if indicator_overlay:
        for name, series in indicator_overlay.items():
            s = series.reindex(df.index)
            src = ColumnDataSource({"time": df.index, "value": s.astype(float)})
            p_price.line(
                "time", "value", source=src, line_width=2, alpha=0.8, legend_label=name, color="orange"
            )

    # Trade markers
    if not events_df.empty:
        marker_src = ColumnDataSource(
            {
                "time": pd.to_datetime(events_df["time"]),
                "price": events_df["price"].astype(float),
                "action": events_df["action"].astype(str),
                "side": events_df["side"].astype(str),
                "weight": events_df["weight"].astype(float),
                "equity": events_df["equity"].astype(float)
                if "equity" in events_df
                else np.nan,
                "turnover": events_df["turnover"].astype(float)
                if "turnover" in events_df
                else np.nan,
                "cost": events_df["cost"].astype(float)
                if "cost" in events_df
                else np.nan,
            }
        )
        p_price.scatter(
            x="time",
            y="price",
            source=marker_src,
            marker="triangle",
            size=10,
            color="#1f77b4",
            legend_label="Trade event",
        )
        p_price.add_tools(
            HoverTool(
                tooltips=[
                    ("time", "@time{%F}"),
                    ("action", "@action"),
                    ("side", "@side"),
                    ("price", "@price{0.00}"),
                    ("weight", "@weight{0.000}"),
                    ("equity", "@equity{0,0.00}"),
                    ("turnover", "@turnover{0.000}"),
                    ("cost", "@cost{0.0000}"),
                ],
                formatters={"@time": "datetime"},
                mode="vline",
                renderers=p_price.renderers[-1:],
            )
        )

    p_price.legend.click_policy = "hide"

    figs = [p_price]

    # Optional indicator panel (separate axis)
    if indicator_panel:
        p_ind = figure(
            x_axis_type="datetime",
            height=200,
            width=1100,
            title="Indicator Panel",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            x_range=p_price.x_range,
        )
        p_ind.add_layout(
            Span(location=0, dimension="width", line_color="#888", line_dash="dashed")
        )
        for name, series in indicator_panel.items():
            s = series.reindex(df.index)
            src = ColumnDataSource({"time": df.index, "value": s.astype(float)})
            p_ind.line(
                "time", "value", source=src, line_width=2, alpha=0.9, legend_label=name, color="orange"
            )
        p_ind.legend.click_policy = "hide"
        figs.append(p_ind)

    # Equity
    eq = equity.reindex(df.index).astype(float)
    dd = eq / eq.cummax() - 1.0
    eq_src = ColumnDataSource({"time": df.index, "equity": eq, "drawdown": dd})

    p_eq = figure(
        x_axis_type="datetime",
        height=220,
        width=1100,
        title="Equity",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=p_price.x_range,
    )
    p_eq.line("time", "equity", source=eq_src, line_width=2)
    p_eq.add_tools(
        HoverTool(
            tooltips=[("time", "@time{%F}"), ("equity", "@equity{0,0.00}")],
            formatters={"@time": "datetime"},
            mode="vline",
        )
    )

    # Drawdown
    p_dd = figure(
        x_axis_type="datetime",
        height=140,
        width=1100,
        title="Drawdown (Equity / Peak - 1)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=p_price.x_range,
    )
    p_dd.line("time", "drawdown", source=eq_src, line_width=2, color="#d62728")
    p_dd.add_layout(
        Span(location=0, dimension="width", line_color="#888", line_dash="dashed")
    )

    # Profit / Loss markers (per-symbol trade returns at exit)
    p_pl = figure(
        x_axis_type="datetime",
        height=140,
        width=1100,
        title="Profit / Loss (per-trade return at exit)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=p_price.x_range,
    )
    p_pl.add_layout(
        Span(location=0, dimension="width", line_color="#888", line_dash="dashed")
    )
    if not trades_df.empty:
        # Determine color: green for profit, red for loss (or zero)
        trades_df["color"] = np.where(trades_df["return_pct"] >= 0, "#2ca02c", "#d62728")
        
        size = trades_df["size"].astype(float)
        if float(size.max()) > float(size.min()):
            marker_size = np.interp(
                size, (float(size.min()), float(size.max())), (8.0, 20.0)
            )
        else:
            marker_size = np.full(len(size), 12.0)
            
        pl_src = ColumnDataSource(
            {
                "time": pd.to_datetime(trades_df["exit_time"]),
                "ret": trades_df["return_pct"].astype(float),
                "side": trades_df["side"].astype(str),
                "entry_time": pd.to_datetime(trades_df["entry_time"]),
                "entry_price": trades_df["entry_price"].astype(float),
                "exit_price": trades_df["exit_price"].astype(float),
                "size": trades_df["size"].astype(float),
                "marker_size": marker_size,
                "color": trades_df["color"],
            }
        )
        p_pl.segment(
            x0="time", y0=0.0, x1="time", y1="ret", source=pl_src, line_color="color", line_alpha=0.6, line_width=2
        )
        r = p_pl.scatter(
            x="time",
            y="ret",
            source=pl_src,
            size="marker_size",
            marker="triangle",
            color="color",
            line_color="black"
        )
        p_pl.add_tools(
            HoverTool(
                tooltips=[
                    ("entry", "@entry_time{%F}"),
                    ("exit", "@time{%F}"),
                    ("return", "@ret{+0.00%}"),
                    ("side", "@side"),
                    ("entry_px", "@entry_price{0.00}"),
                    ("exit_px", "@exit_price{0.00}"),
                    ("size(|w|)", "@size{0.000}"),
                ],
                formatters={"@time": "datetime", "@entry_time": "datetime"},
                renderers=[r],
            )
        )

    figs.extend([p_eq, p_dd, p_pl])
    return column(*figs)


def build_interactive_portfolio_layout(
    *,
    market_ohlcv: pd.DataFrame,
    equity: pd.Series,
    returns: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
    costs: pd.Series,
    close_prices: Optional[pd.DataFrame] = None,
    indicator_overlay: Optional[Dict[str, pd.Series]] = None,
    indicator_panel: Optional[Dict[str, pd.Series]] = None,
    title: str = "Backtest",
):
    """Interactive Bokeh layout for a multi-asset portfolio.

    The "market" view uses a proxy OHLCV series (e.g. cross-sectional average)
    for context, while markers represent portfolio rebalance events.
    """
    from bokeh.layouts import column
    from bokeh.models import ColumnDataSource, HoverTool, Span
    from bokeh.plotting import figure

    df = market_ohlcv.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("market_ohlcv.index must be a DatetimeIndex")

    idx = df.index
    eq = equity.reindex(idx).astype(float)
    rets = returns.reindex(idx).fillna(0.0).astype(float)
    w = weights.reindex(idx).fillna(0.0)
    to = turnover.reindex(idx).fillna(0.0).astype(float)
    co = costs.reindex(idx).fillna(0.0).astype(float)

    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1)
    active = (w.abs() > 0).sum(axis=1).astype(int)

    close = df["Close"].reindex(idx).astype(float)

    # Portfolio rebalance events are times with turnover > 0
    is_event = to > 0
    events = pd.DataFrame(
        {
            "time": idx[is_event],
            "price": close[is_event],
            "turnover": to[is_event],
            "cost": co[is_event],
            "gross": gross[is_event],
            "net": net[is_event],
            "active": active[is_event],
            "equity": eq[is_event],
        }
    )

    # Per-Asset Round-Trip Trade P/L markers
    pl_rows: list[dict[str, object]] = []
    
    if close_prices is not None:
        # We calculate discrete trades per asset
        cp = close_prices.reindex(idx).ffill()
        for sym in w.columns:
            asset_w = w[sym].values
            asset_p = cp[sym].values
            
            # Find Entry (0 -> >0)
            entries = np.where((asset_w[1:] > 0) & (asset_w[:-1] == 0))[0] + 1
            # Find Exit (>0 -> 0)
            exits = np.where((asset_w[1:] == 0) & (asset_w[:-1] > 0))[0] + 1
            
            # Match entries to next exit
            for start_idx in entries:
                # Find first exit after start_idx
                future_exits = exits[exits >= start_idx]
                if len(future_exits) > 0:
                    end_idx = future_exits[0]
                    p0 = asset_p[start_idx]
                    p1 = asset_p[end_idx]
                    if p0 > 0:
                        pl_rows.append({
                            "time": idx[end_idx],
                            "entry_time": idx[start_idx],
                            "symbol": sym,
                            "ret": float(p1 / p0 - 1.0),
                            "entry_price": float(p0),
                            "exit_price": float(p1),
                        })
    else:
        # Fallback to portfolio rebalance periods
        event_times = list(events["time"]) if not events.empty else []
        for t0, t1 in zip(event_times, event_times[1:]):
            e0 = float(eq.loc[pd.Timestamp(t0)])
            e1 = float(eq.loc[pd.Timestamp(t1)])
            if e0 != 0:
                pl_rows.append({
                    "time": pd.Timestamp(t1),
                    "entry_time": pd.Timestamp(t0),
                    "symbol": "Portfolio",
                    "ret": float(e1 / e0 - 1.0),
                })
    
    pl_df = pd.DataFrame(pl_rows)

    # Market proxy price chart
    price_src = ColumnDataSource(
        {
            "time": idx,
            "open": df["Open"].reindex(idx).astype(float),
            "high": df["High"].reindex(idx).astype(float),
            "low": df["Low"].reindex(idx).astype(float),
            "close": close,
        }
    )

    p_price = figure(
        x_axis_type="datetime",
        height=350,
        width=1100,
        title=f"{title} - Market View (Proxy Price, Indicators, Rebalances)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p_price.line(
        "time", "close", source=price_src, line_width=2, legend_label="Proxy Close"
    )

    if indicator_overlay:
        for name, series in indicator_overlay.items():
            s = series.reindex(idx)
            src = ColumnDataSource({"time": idx, "value": s.astype(float)})
            p_price.line(
                "time", "value", source=src, line_width=2, alpha=0.8, legend_label=name, color="orange"
            )

    if not events.empty:
        ev_src = ColumnDataSource(
            {
                "time": pd.to_datetime(events["time"]),
                "price": events["price"].astype(float),
                "turnover": events["turnover"].astype(float),
                "cost": events["cost"].astype(float),
                "gross": events["gross"].astype(float),
                "net": events["net"].astype(float),
                "active": events["active"].astype(int),
                "equity": events["equity"].astype(float),
            }
        )
        r_ev = p_price.scatter(
            x="time",
            y="price",
            source=ev_src,
            marker="triangle",
            size=10,
            color="#1f77b4",
            legend_label="Rebalance",
        )
        p_price.add_tools(
            HoverTool(
                tooltips=[
                    ("time", "@time{%F}"),
                    ("proxy_px", "@price{0.00}"),
                    ("turnover", "@turnover{0.000}"),
                    ("cost", "@cost{0.0000}"),
                    ("gross", "@gross{0.000}"),
                    ("net", "@net{0.000}"),
                    ("active_assets", "@active{0}"),
                    ("equity", "@equity{0,0.00}"),
                ],
                formatters={"@time": "datetime"},
                renderers=[r_ev],
            )
        )

    p_price.legend.click_policy = "hide"

    figs = [p_price]

    if indicator_panel:
        p_ind = figure(
            x_axis_type="datetime",
            height=200,
            width=1100,
            title="Indicator Panel (Market Proxy)",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            x_range=p_price.x_range,
        )
        p_ind.add_layout(
            Span(location=0, dimension="width", line_color="#888", line_dash="dashed")
        )
        for name, series in indicator_panel.items():
            s = series.reindex(idx)
            src = ColumnDataSource({"time": idx, "value": s.astype(float)})
            p_ind.line(
                "time", "value", source=src, line_width=2, alpha=0.9, legend_label=name, color="orange"
            )
        p_ind.legend.click_policy = "hide"
        figs.append(p_ind)

    # Equity
    dd = eq / eq.cummax() - 1.0
    eq_src = ColumnDataSource({"time": idx, "equity": eq, "drawdown": dd, "ret": rets})

    p_eq = figure(
        x_axis_type="datetime",
        height=220,
        width=1100,
        title="Equity (Portfolio)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=p_price.x_range,
    )
    p_eq.line("time", "equity", source=eq_src, line_width=2)
    p_eq.add_tools(
        HoverTool(
            tooltips=[("time", "@time{%F}"), ("equity", "@equity{0,0.00}")],
            formatters={"@time": "datetime"},
            mode="vline",
        )
    )

    p_dd = figure(
        x_axis_type="datetime",
        height=140,
        width=1100,
        title="Drawdown (Portfolio)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=p_price.x_range,
    )
    p_dd.line("time", "drawdown", source=eq_src, line_width=2, color="#d62728")
    p_dd.add_layout(
        Span(location=0, dimension="width", line_color="#888", line_dash="dashed")
    )

    # Profit / Loss markers (per-rebalance period return)
    p_pl = figure(
        x_axis_type="datetime",
        height=140,
        width=1100,
        title="Profit / Loss (equity return between rebalances)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=p_price.x_range,
    )
    p_pl.add_layout(
        Span(location=0, dimension="width", line_color="#888", line_dash="dashed")
    )
    if not pl_df.empty:
        # Determine color based on return sign: Green for profit, Red for Loss
        pl_df["color"] = np.where(pl_df["ret"] >= 0, "#2ca02c", "#d62728")

        pl_src = ColumnDataSource(
            {
                "time": pd.to_datetime(pl_df["time"]),
                "ret": pl_df["ret"].astype(float),
                "entry_time": pd.to_datetime(pl_df["entry_time"]),
                "symbol": pl_df["symbol"],
                "color": pl_df["color"],
            }
        )
        p_pl.segment(
            x0="time", y0=0.0, x1="time", y1="ret", source=pl_src, line_color="color", line_alpha=0.6, line_width=2
        )
        r = p_pl.scatter(
            x="time", 
            y="ret", 
            source=pl_src, 
            size=12, 
            marker="circle", 
            color="color",
            line_color="black"
        )
        p_pl.add_tools(
            HoverTool(
                tooltips=[
                    ("Asset", "@symbol"),
                    ("Entry", "@entry_time{%F}"),
                    ("Exit", "@time{%F}"),
                    ("Return", "@ret{+0.00%}"),
                ],
                formatters={"@time": "datetime", "@entry_time": "datetime"},
                renderers=[r],
            )
        )

    # --- Final Portfolio Allocation (Bar Chart) ---
    # Calculate Final Asset Values: Weight * Equity at the last timestamp
    final_idx = idx[-1]
    last_w = weights.iloc[-1].fillna(0.0)
    last_eq = float(eq.iloc[-1])
    
    # Value per asset
    final_values = last_w * last_eq
    # Filter for non-zero (or very small epsilon)
    # Be careful to handle series alignment
    final_alloc = final_values[final_values.abs() > 1e-2].sort_values(ascending=False)
    
    if final_alloc.empty:
        assets = ["Cash"]
        values = [last_eq]
    else:
        # Convert index to strings for FactorRange
        assets = [str(x) for x in final_alloc.index.tolist()]
        values = final_alloc.values.tolist()

    source_bar = ColumnDataSource({"original_assets": assets, "values": values})
    
    p_alloc = figure(
        x_range=assets,
        height=300,
        width=1100,
        title=f"Final Portfolio Allocation (at {final_idx.date()})",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        tooltips=[("Asset", "@original_assets"), ("Value", "@values{$0,0.00}")]
    )
    
    p_alloc.vbar(x="original_assets", top="values", width=0.8, source=source_bar, line_color="white", fill_color="#1f77b4")
    
    p_alloc.xgrid.grid_line_color = None
    p_alloc.y_range.start = 0
    p_alloc.xaxis.major_label_orientation = 1.2  # Rotate labels if many assets
    p_alloc.yaxis.axis_label = "Value ($)"

    figs.extend([p_alloc, p_eq, p_dd, p_pl])
    return column(*figs)
