import numpy as np
import pandas as pd

def ema(series, period):
    """
    Calculate the Exponential Moving Average (EMA).
    """
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    """
    Calculate the Simple Moving Average (SMA).
    """
    return series.rolling(window=period).mean()

def rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) using Wilder's smoothing.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's smoothing uses alpha = 1 / period, equivalent to com = period - 1
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def wavetrend(high, low, close, n1=10, n2=11):
    """
    Calculate the WaveTrend Oscillator (by LazyBear).
    Returns (WT1, WT2) where WT1 is the main line and WT2 is the signal line.
    """
    ap = (high + low + close) / 3
    esa = ema(ap, n1)
    d = ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d + 1e-10)
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)
    return wt1, wt2

def cci(high, low, close, period=20):
    """
    Calculate the Commodity Channel Index (CCI).
    """
    tp = (high + low + close) / 3
    tp_sma = sma(tp, period)
    # Mean Absolute Deviation (MAD)
    mad = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - tp_sma) / (0.015 * mad + 1e-10)

def adx(high, low, close, period=14):
    """
    Calculate the Average Directional Index (ADX) using Wilder's smoothing.
    """
    # True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement (+DM and -DM)
    up_move = high.diff()
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Smooth TR and DMs
    atr = tr.ewm(com=period-1, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(com=period-1, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(com=period-1, adjust=False).mean() / (atr + 1e-10)
    
    # Calculate DX and smooth it to get ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_series = dx.ewm(com=period-1, adjust=False).mean()
    return adx_series

def nadaraya_watson_rational_quadratic(close, h=8, r=8.0, lookback=25):
    """
    Calculate a causal (non-repainting) Nadaraya-Watson Kernel Regression
    using the Rational Quadratic Kernel.
    
    h: lookback window (bandwidth)
    r: relative weighting parameter (alpha)
    lookback: sliding window size for local regression
    """
    # Compute kernel weights for historical offsets (0 to lookback-1)
    weights = np.zeros(lookback)
    for i in range(lookback):
        weights[i] = (1 + (i**2) / (2 * r * (h**2)))**(-r)
        
    weights_sum = np.sum(weights)
    
    result = np.zeros(len(close))
    close_arr = close.values
    
    # Compute the weighted average for each bar in a sliding window
    for t in range(len(close)):
        if t < lookback - 1:
            result[t] = close_arr[t]  # fallback for early bars
        else:
            # Take the window of closing prices leading up to t, reversed
            window = close_arr[t - lookback + 1 : t + 1][::-1]
            result[t] = np.sum(window * weights) / weights_sum
            
    return pd.Series(result, index=close.index)
