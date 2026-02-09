import pandas as pd
import numpy as np
import os
from src.backtester.engine import BacktestConfig, run_backtest
from src.features.core import cci

# Load one asset for debugging
assets = {}
files = [f for f in os.listdir('dataset/cleaned') if f.endswith('.csv')][:5]
for f in files:
    sym = f.replace('.csv', '')
    df = pd.read_csv(os.path.join('dataset/cleaned', f), index_col=0, parse_dates=True)
    assets[sym] = df

close = pd.concat([df['Close'].rename(s) for s, df in assets.items()], axis=1).sort_index()
highs = pd.concat([df['High'].rename(s) for s, df in assets.items()], axis=1).sort_index()
lows = pd.concat([df['Low'].rename(s) for s, df in assets.items()], axis=1).sort_index()
opens = pd.concat([df['Open'].rename(s) for s, df in assets.items()], axis=1).sort_index()

# Hysteresis Position Helper
def hysteresis_position(enter, exit, alloc=0.1):
    pos = pd.Series(-1.0, index=enter.index)
    current = -1.0
    for dt in enter.index:
        if exit.loc[dt]:
            current = -1.0
        elif enter.loc[dt]:
            current = alloc
        pos.loc[dt] = current
    return pos

# CCI Strategy with Buffer
def strategy_cci_debug(assets):
    out = {}
    for sym, df in assets.items():
        c_val = cci(df['High'], df['Low'], df['Close'])
        enter = (c_val > 100)
        exit = (c_val < -100)
        out[sym] = hysteresis_position(enter, exit)
    return pd.DataFrame(out)

signals = strategy_cci_debug(assets).reindex(close.index)

cfg = BacktestConfig(
    initial_equity=1_000_000,
    strict_signals=True,
    stop_loss_pct=0.05,
    mode='debug'
)

res = run_backtest(close, signals, cfg, open_prices=opens, high_prices=highs, low_prices=lows)

# Analyze trades
print(f"Total Equity: {res.equity.iloc[-1]:.2f}")
trades = res.turnover[res.turnover > 0]
print(f"Number of trade days: {len(trades)}")

# Per asset hold time
hold_times = []
for sym in close.columns:
    w = res.weights[sym]
    # Find transitions
    pos = (w > 1e-6).astype(int)
    enters = (pos.diff() == 1)
    exits = (pos.diff() == -1)
    
    ent_idx = np.where(enters)[0]
    ex_idx = np.where(exits)[0]
    
    for en in ent_idx:
        # Find first exit after this entry
        after_ex = ex_idx[ex_idx > en]
        if len(after_ex) > 0:
            hold_times.append(after_ex[0] - en)
            
if hold_times:
    print(f"Average Hold Time: {np.mean(hold_times):.2f} days")
    print(f"Median Hold Time: {np.median(hold_times):.2f} days")
else:
    print("No complete trades found.")

# Check CCI values and signals for an asset
sym = res.weights.columns[0]
asset_debug = pd.DataFrame({
    'Close': close[sym],
    'Signal': signals[sym],
    'Weight': res.weights[sym]
})

print(f"\nDebug log for {sym} (Days with signals or weight change):")
# Filter where signal != 0 or weight changed from previous
weight_changed = asset_debug['Weight'].diff().abs() > 1e-6
print(asset_debug[(asset_debug['Signal'] != 0) | weight_changed].head(30))
