from typing import cast

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    out = series.ewm(span=span, adjust=False).mean()
    return cast(pd.Series, out)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (recursive form, matching the paper's definition)."""
    return _ema(series, span=span)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over a trailing window."""
    return rolling_mean(series, window)


def sma_ratio(close: pd.Series, window: int) -> pd.Series:
    """Normalized SMA feature used in arXiv:2412.15448: C_t / SMA_N(t)."""
    denom = sma(close, window).replace(0.0, np.nan)
    return cast(pd.Series, close / denom)


def ema_ratio(close: pd.Series, span: int) -> pd.Series:
    """Normalized EMA feature used in arXiv:2412.15448: C_t / EMA_t."""
    denom = ema(close, span).replace(0.0, np.nan)
    return cast(pd.Series, close / denom)


def log_return(close: pd.Series) -> pd.Series:
    """Compute log returns: r_t = log(C_t / C_{t-1})."""
    ratio = close / close.shift(1)
    return cast(pd.Series, ratio.apply(np.log))


def simple_return(close: pd.Series) -> pd.Series:
    """Compute simple returns: R_t = (C_t / C_{t-1}) - 1."""
    return cast(pd.Series, close.pct_change())


def excess_return(
    returns: pd.Series,
    risk_free_rate_annual: float,
    periods_per_year: int = 252,
    *,
    compounding: str = "discrete",
) -> pd.Series:
    """Compute excess return by subtracting a per-period risk-free rate.

    compounding:
      - "discrete": r_f,period = (1 + r_f,annual)^(1/periods_per_year) - 1
      - "simple":   r_f,period = r_f,annual / periods_per_year
    """
    if compounding == "discrete":
        rf_period = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    elif compounding == "simple":
        rf_period = risk_free_rate_annual / float(periods_per_year)
    else:
        raise ValueError("compounding must be 'discrete' or 'simple'")

    return returns - rf_period


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: z_t = (x_t - mean_t) / std_t over a trailing window."""
    mean = cast(pd.Series, series.rolling(window=window, min_periods=window).mean())
    std = cast(pd.Series, series.rolling(window=window, min_periods=window).std(ddof=0))
    return cast(pd.Series, (series - mean) / std.replace(0.0, np.nan))


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    return cast(pd.Series, series.rolling(window=window, min_periods=window).min())


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return cast(pd.Series, series.rolling(window=window, min_periods=window).max())


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return cast(pd.Series, series.rolling(window=window, min_periods=window).mean())


def rolling_var(series: pd.Series, window: int) -> pd.Series:
    return cast(
        pd.Series, series.rolling(window=window, min_periods=window).var(ddof=0)
    )


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    return cast(
        pd.Series, series.rolling(window=window, min_periods=window).std(ddof=0)
    )


def rolling_minmax_scale(series: pd.Series, window: int) -> pd.Series:
    """Rolling min-max scaling: (x - min) / (max - min) over a trailing window."""
    lo = rolling_min(series, window)
    hi = rolling_max(series, window)
    denom = (hi - lo).replace(0.0, np.nan)
    return cast(pd.Series, (series - lo) / denom)


def cumulative_return(returns: pd.Series, window: int) -> pd.Series:
    """Cumulative simple return over a window: prod(1+R) - 1."""
    out = (
        (1.0 + returns)
        .rolling(window=window, min_periods=window)
        .apply(lambda x: float(np.prod(x) - 1.0), raw=True)
    )
    return cast(pd.Series, out)


def lag(series: pd.Series, k: int) -> pd.Series:
    return cast(pd.Series, series.shift(k))


def difference(series: pd.Series, k: int = 1) -> pd.Series:
    """k-th order differencing: Δ^k x_t."""
    out: pd.Series = series
    for _ in range(k):
        out = cast(pd.Series, out.diff())
    return cast(pd.Series, out)


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    """True Range: max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)."""
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    out = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return cast(pd.Series, out)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average True Range using a simple moving average of True Range."""
    prev_close = cast(pd.Series, close.shift(1))
    tr = true_range(high, low, prev_close)
    return cast(pd.Series, tr.rolling(window=window, min_periods=window).mean())


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing).

    RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss.
    """
    delta = cast(pd.Series, close.diff())
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = cast(pd.Series, gain.ewm(alpha=1.0 / window, adjust=False).mean())
    avg_loss = cast(pd.Series, loss.ewm(alpha=1.0 / window, adjust=False).mean())

    rs = cast(pd.Series, avg_gain / avg_loss.replace(0.0, np.nan))
    return cast(pd.Series, 100.0 - (100.0 / (1.0 + rs)))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = ema(close, span=fast)
    ema_slow = ema(close, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def rmacd(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    """Normalized MACD ratio from arXiv:2412.15448.

    rMACD = (MACD_t - SIG_t) / (0.5 * (|MACD_t| + |SIG_t|)).
    """
    denom = cast(pd.Series, 0.5 * (macd_line.abs() + signal_line.abs()))
    return cast(pd.Series, (macd_line - signal_line) / denom.replace(0.0, np.nan))


def bollinger_bands(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: mid=SMA, upper/lower=mid +/- k*std."""
    mid = rolling_mean(close, window)
    std = rolling_std(close, window)
    upper = mid + k * std
    lower = mid - k * std
    bandwidth = (upper - lower) / mid
    percent_b = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_percent_b": percent_b,
        }
    )


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of Change: (C_t / C_{t-n}) - 1."""
    return cast(pd.Series, close / close.shift(window) - 1.0)


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator %K and %D."""
    lowest_low = rolling_min(low, k_window)
    highest_high = rolling_max(high, k_window)
    percent_k = cast(
        pd.Series, 100.0 * (close - lowest_low) / (highest_high - lowest_low)
    )
    percent_d = cast(
        pd.Series, percent_k.rolling(window=d_window, min_periods=d_window).mean()
    )
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV)."""
    direction = cast(pd.Series, close.diff()).apply(np.sign)
    signed_volume = cast(pd.Series, volume * direction.fillna(0.0))
    return cast(pd.Series, signed_volume.cumsum())


def wrobv(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Windowed Relative OBV (WROBV) from arXiv:2412.15448.

    WROBV_t = sum_{i=0..N-1} OBV_{t-i} / sum_{i=0..N-1} V_{t-i}.
    """
    obv_series = obv(close, volume)
    num = cast(pd.Series, obv_series.rolling(window=window, min_periods=window).sum())
    den = cast(pd.Series, volume.rolling(window=window, min_periods=window).sum())
    return cast(pd.Series, num / den.replace(0.0, np.nan))


def accumulation_distribution_line(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Accumulation/Distribution (A/D) Line using Close Location Value (CLV)."""
    denom = (high - low).replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / denom
    money_flow_volume = clv * volume
    return money_flow_volume.cumsum()


def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.DataFrame:
    """Aroon Up/Down: time since highest high / lowest low."""

    def _periods_since_extreme(x: np.ndarray, extreme: str) -> float:
        if extreme == "max":
            idx = int(np.argmax(x))
        else:
            idx = int(np.argmin(x))
        return float(len(x) - 1 - idx)

    periods_since_high = cast(
        pd.Series,
        high.rolling(window=window, min_periods=window).apply(
            lambda x: _periods_since_extreme(np.asarray(x), "max"), raw=False
        ),
    )
    periods_since_low = cast(
        pd.Series,
        low.rolling(window=window, min_periods=window).apply(
            lambda x: _periods_since_extreme(np.asarray(x), "min"), raw=False
        ),
    )

    aroon_up = cast(pd.Series, 100.0 * (window - periods_since_high) / window)
    aroon_down = cast(pd.Series, 100.0 * (window - periods_since_low) / window)
    return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down})


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.DataFrame:
    """Average Directional Index (ADX) with +DI and -DI (Wilder style)."""
    up_move = cast(pd.Series, high.diff())
    down_move = cast(pd.Series, -low.diff())

    plus_dm_arr = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm_arr = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = cast(pd.Series, close.shift(1))
    tr = true_range(high, low, prev_close)

    atr_wilder = cast(pd.Series, tr.ewm(alpha=1.0 / window, adjust=False).mean())
    plus_dm_sm = (
        pd.Series(plus_dm_arr, index=high.index)
        .ewm(alpha=1.0 / window, adjust=False)
        .mean()
    )
    minus_dm_sm = (
        pd.Series(minus_dm_arr, index=high.index)
        .ewm(alpha=1.0 / window, adjust=False)
        .mean()
    )

    plus_di = cast(pd.Series, 100.0 * plus_dm_sm / atr_wilder.replace(0.0, np.nan))
    minus_di = cast(pd.Series, 100.0 * minus_dm_sm / atr_wilder.replace(0.0, np.nan))
    dx = cast(
        pd.Series,
        100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan),
    )
    adx_val = cast(pd.Series, dx.ewm(alpha=1.0 / window, adjust=False).mean())

    # Paper (arXiv:2412.15448) uses the raw ratio (scaled to [0, 1]) as the ADX feature.
    adx_raw = cast(
        pd.Series,
        (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan),
    )

    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val, "adx_raw": adx_raw}
    )


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3.0
    sma_tp = rolling_mean(tp, window)
    mad = tp.rolling(window=window, min_periods=window).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0.0, np.nan))


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    conv_window: int = 9,
    base_window: int = 26,
    span_b_window: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku Cloud components (daily)."""
    conversion = (rolling_max(high, conv_window) + rolling_min(low, conv_window)) / 2.0
    base = (rolling_max(high, base_window) + rolling_min(low, base_window)) / 2.0
    span_a = ((conversion + base) / 2.0).shift(displacement)
    span_b = (
        (rolling_max(high, span_b_window) + rolling_min(low, span_b_window)) / 2.0
    ).shift(displacement)
    lagging = close.shift(displacement)
    return pd.DataFrame(
        {
            "ichimoku_conv": conversion,
            "ichimoku_base": base,
            "ichimoku_span_a": span_a,
            "ichimoku_span_b": span_b,
            "ichimoku_lagging": lagging,
        }
    )


def fib_levels(high: pd.Series, low: pd.Series, window: int = 60) -> pd.DataFrame:
    """Rolling Fibonacci retracement levels derived from rolling swing high/low."""
    swing_high = rolling_max(high, window)
    swing_low = rolling_min(low, window)
    rng = swing_high - swing_low
    level_236 = swing_high - 0.236 * rng
    level_382 = swing_high - 0.382 * rng
    level_500 = swing_high - 0.5 * rng
    level_618 = swing_high - 0.618 * rng
    level_764 = swing_high - 0.764 * rng
    return pd.DataFrame(
        {
            "fib_swing_high": swing_high,
            "fib_swing_low": swing_low,
            "fib_23_6": level_236,
            "fib_38_2": level_382,
            "fib_50_0": level_500,
            "fib_61_8": level_618,
            "fib_76_4": level_764,
        }
    )


def fib_retracement_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Fibonacci retracement ratio from arXiv:2412.15448.

    R(t) = (H_N(t) - C_t) / (H_N(t) - L_N(t)).
    """
    hi = rolling_max(high, window)
    lo = rolling_min(low, window)
    denom = (hi - lo).replace(0.0, np.nan)
    return cast(pd.Series, (hi - close) / denom)


def market_structure_signals(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    swing_window: int = 3,
) -> pd.DataFrame:
    """Market structure event indicators for daily OHLC data.

    Returns three integer-valued series (in {-1, 0, 1}):
      - bos: Break of structure (bullish=+1, bearish=-1)
      - choch: Change of character (bullish=+1, bearish=-1)
      - mss: Market structure shift confirmation (bullish=+1, bearish=-1)

    Implementation notes:
      - Swing highs/lows are detected as local extrema over a centered window of
        size (2*swing_window + 1).
      - BOS triggers when Close breaks above the most recent prior swing high
        (bullish) or below the most recent prior swing low (bearish).
      - CHoCH is defined as a BOS event opposite to the prevailing structure
        direction inferred from the last non-zero BOS.
      - MSS is a simple confirmation of CHoCH: after a CHoCH, the next BOS in the
        new direction marks the shift as confirmed.
    """
    if swing_window < 1:
        raise ValueError("swing_window must be >= 1")

    win = 2 * swing_window + 1

    # Local extrema (trailing). A peak is confirmed at the end of the window.
    # We use a window of size (2*swing_window + 1). At time t, we check if
    # high[t-swing_window] was the maximum in [t-2*swing_window, t].
    roll_hi = cast(
        pd.Series, high.rolling(window=win, center=False, min_periods=win).max()
    )
    roll_lo = cast(
        pd.Series, low.rolling(window=win, center=False, min_periods=win).min()
    )

    is_swing_high = high.shift(swing_window) == roll_hi
    is_swing_low = low.shift(swing_window) == roll_lo

    # Map the detected swings to their confirmation time t.
    swing_high = high.shift(swing_window).where(is_swing_high)
    swing_low = low.shift(swing_window).where(is_swing_low)

    last_swing_high = cast(pd.Series, swing_high.ffill()).shift(0)
    last_swing_low = cast(pd.Series, swing_low.ffill()).shift(0)

    bos = pd.Series(0, index=close.index, dtype="int64")
    bos_up = close > last_swing_high
    bos_down = close < last_swing_low
    bos.loc[bos_up.fillna(False)] = 1
    bos.loc[bos_down.fillna(False)] = -1

    structure_dir = cast(pd.Series, bos.replace(0, np.nan).ffill().fillna(0)).astype(
        "int64"
    )
    prev_dir = cast(pd.Series, structure_dir.shift(1).fillna(0)).astype("int64")

    choch = pd.Series(0, index=close.index, dtype="int64")
    choch.loc[(prev_dir == -1) & (bos == 1)] = 1
    choch.loc[(prev_dir == 1) & (bos == -1)] = -1

    # MSS confirmation: after a CHoCH, mark the next BOS in the same direction.
    mss = pd.Series(0, index=close.index, dtype="int64")
    pending: int = 0
    for i in range(len(close)):
        c = int(choch.iat[i])
        b = int(bos.iat[i])
        if pending == 0:
            if c != 0:
                pending = c
            continue

        if b == pending:
            mss.iat[i] = pending
            pending = 0
        elif c != 0 and c != pending:
            pending = c

    return pd.DataFrame({"bos": bos, "choch": choch, "mss": mss})


def heiken_ashi(
    open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.DataFrame:
    """Compute Heiken Ashi OHLC.

    HA_Close = (Open + High + Low + Close) / 4
    HA_Open_t = (HA_Open_{t-1} + HA_Close_{t-1}) / 2
    HA_High = max(High, HA_Open, HA_Close)
    HA_Low = min(Low, HA_Open, HA_Close)
    """
    ha_close = (open + high + low + close) / 4.0

    # HA_Open is recursive: ha_open[t] = 0.5 * ha_open[t-1] + 0.5 * ha_close[t-1]
    # This matches the form: y[t] = (1-alpha)*y[t-1] + alpha*x[t]
    # with y = ha_open, x = ha_close.shift(1), alpha = 0.5
    # Initial value y[0] = (open[0] + close[0]) / 2

    # We use ewm with adjust=False for the recursive formula
    # To handle the initial value correctly, we prepend it.
    ha_open_start = (open.iloc[0] + close.iloc[0]) / 2.0
    ha_close_shifted = ha_close.shift(1)
    ha_close_shifted.iloc[0] = ha_open_start

    ha_open = ha_close_shifted.ewm(alpha=0.5, adjust=False).mean()

    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)

    return pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        }
    )
