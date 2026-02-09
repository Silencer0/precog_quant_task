import pandas as pd
import numpy as np
import os
from src.backtester.data import load_cleaned_assets
from src.features.core import rsi, sma

def smooth_slope(series: pd.Series, window: int = 20) -> pd.Series:
    return series.diff(1).ewm(span=window).mean()

def analyze_signal_durations(symbol="Asset_001"):
    data_dir = "dataset/cleaned"
    path = os.path.join(data_dir, f"{symbol}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    c = df['Close']
    
    # 1. Divergence Signals
    r = rsi(c, 14)
    p_slope = smooth_slope(c, 20)
    i_slope = smooth_slope(r, 20)
    
    bull_div = ((p_slope < 0) & (i_slope > 0)).astype(int)
    bear_div = ((p_slope > 0) & (i_slope < 0)).astype(int)
    
    # Analyze signal "Blocks"
    def get_max_duration(sig):
        if sig.sum() == 0: return 0
        blocks = (sig != sig.shift()).cumsum()
        counts = sig.groupby(blocks).transform('count')
        active_counts = counts[sig == 1]
        return active_counts.max() if not active_counts.empty else 0

    def get_avg_duration(sig):
        if sig.sum() == 0: return 0
        diff = sig.diff()
        starts = (diff == 1).sum()
        if starts == 0: return sig.sum() # Always on
        return sig.sum() / starts

    print(f"Analysis for {symbol}:")
    print(f"  Bullish Divergence: Total Days={bull_div.sum()}, Avg Duration={get_avg_duration(bull_div):.2f}, Max={get_max_duration(bull_div)}")
    print(f"  Bearish Divergence: Total Days={bear_div.sum()}, Avg Duration={get_avg_duration(bear_div):.2f}, Max={get_max_duration(bear_div)}")
    
    # 2. RSI Crossings
    overbought = (r > 70).astype(int)
    print(f"  RSI Overbought: Total Days={overbought.sum()}, Avg Duration={get_avg_duration(overbought):.2f}")

    # 3. Latch Logic Check
    # If we latch: Enter on Bull, Exit on Bear.
    state = 0
    durations = []
    current_dur = 0
    for t in range(len(bull_div)):
        if bear_div.iloc[t] == 1:
            if state == 1:
                durations.append(current_dur)
            state = 0
            current_dur = 0
        elif bull_div.iloc[t] == 1:
            state = 1
            current_dur += 1
        else:
            if state == 1:
                current_dur += 1
    
    if durations:
        print(f"  Latched Strategy (Entry Bull, Exit Bear): Avg Trade Life={np.mean(durations):.2f}, Max={np.max(durations)}")
    else:
        print("  Latched Strategy: No completed trades detected.")

if __name__ == "__main__":
    analyze_signal_durations("Asset_001")
    analyze_signal_durations("Asset_002")
    analyze_signal_durations("Asset_003")
