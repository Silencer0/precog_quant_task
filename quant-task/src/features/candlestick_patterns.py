import pandas as pd
import numpy as np

def extract_candlestick_patterns(df):
    """
    Extracts various candlestick patterns from OHLC data using mathematical operators.
    df must have columns: Open, High, Low, Close
    """
    res = pd.DataFrame(index=df.index)
    
    O = df['Open']
    H = df['High']
    L = df['Low']
    C = df['Close']
    
    # Pre-calculate body and shadows
    body = (C - O).abs()
    body_dir = np.sign(C - O) # 1 for green, -1 for red, 0 for dojiish
    upper_shadow = H - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - L
    range_total = H - L
    range_total = range_total.replace(0, 1e-9) # Avoid div by zero
    
    # 1. Bullish Engulfing
    # Prev is red, current is green, current body engulfs prev body
    res['bullish_engulfing'] = (
        (body_dir.shift(1) == -1) & 
        (body_dir == 1) & 
        (C > O.shift(1)) & 
        (O < C.shift(1))
    ).astype(int)
    
    # 2. Bearish Engulfing
    res['bearish_engulfing'] = (
        (body_dir.shift(1) == 1) & 
        (body_dir == -1) & 
        (C < O.shift(1)) & 
        (O > C.shift(1))
    ).astype(int)
    
    # 3. Morning Star (3-candle)
    # 1: Long Bearish, 2: Short body (can be doji), 3: Long Bullish closing into 1st body
    res['morning_star'] = (
        (body_dir.shift(2) == -1) & 
        (body.shift(1) < (body.shift(2) * 0.3)) & 
        (body_dir == 1) & 
        (C > (O.shift(2) + C.shift(2))/2)
    ).astype(int)
    
    # 4. Evening Star (3-candle)
    res['evening_star'] = (
        (body_dir.shift(2) == 1) & 
        (body.shift(1) < (body.shift(2) * 0.3)) & 
        (body_dir == -1) & 
        (C < (O.shift(2) + C.shift(2))/2)
    ).astype(int)
    
    # 5. Three White Soldiers
    res['three_white_soldiers'] = (
        (body_dir == 1) & (body_dir.shift(1) == 1) & (body_dir.shift(2) == 1) &
        (C > C.shift(1)) & (C.shift(1) > C.shift(2)) &
        (O > O.shift(1)) & (O.shift(1) > O.shift(2))
    ).astype(int)
    
    # 6. Three Black Crows
    res['three_black_crows'] = (
        (body_dir == -1) & (body_dir.shift(1) == -1) & (body_dir.shift(2) == -1) &
        (C < C.shift(1)) & (C.shift(1) < C.shift(2)) &
        (O < O.shift(1)) & (O.shift(1) < O.shift(2))
    ).astype(int)
    
    # 7. Piercing Line
    # Prev is red, current is green, current opens below prev low and closes above mid of prev body
    res['piercing_line'] = (
        (body_dir.shift(1) == -1) &
        (body_dir == 1) &
        (O < L.shift(1)) &
        (C > (O.shift(1) + C.shift(1))/2) &
        (C < O.shift(1))
    ).astype(int)
    
    # 8. Hanging Man
    # Small body at top of range, long lower shadow, occurs in uptrend (here just pattern)
    res['hanging_man'] = (
        (lower_shadow > 2 * body) &
        (upper_shadow < 0.1 * body) &
        (body < 0.3 * range_total)
    ).astype(int)
    
    # 9. Hammer
    # Same logic as hanging man, usually in downtrend
    res['hammer'] = (
        (lower_shadow > 2 * body) &
        (upper_shadow < 0.1 * body) &
        (body < 0.3 * range_total)
    ).astype(int)
    
    # 10. Inverse Hammer
    res['inverse_hammer'] = (
        (upper_shadow > 2 * body) &
        (lower_shadow < 0.1 * body) &
        (body < 0.3 * range_total)
    ).astype(int)
    
    # 11. Tweezer Tops
    # Two candles with similar highs
    res['tweezer_tops'] = (
        (np.abs(H - H.shift(1)) / range_total < 0.01) &
        (upper_shadow > 0.4 * range_total)
    ).astype(int)
    
    # 12. Doji
    res['doji'] = (body < 0.1 * range_total).astype(int)
    
    # 13. Spinning Tops
    res['spinning_tops'] = (
        (body < 0.3 * range_total) & 
        (upper_shadow > 0.3 * range_total) & 
        (lower_shadow > 0.3 * range_total)
    ).astype(int)
    
    return res
