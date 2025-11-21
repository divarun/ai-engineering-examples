import pandas as pd
import numpy as np

def bullish_engulfing(prev, curr):
    """
    Detects Bullish Engulfing candlestick pattern (two-candle).
    prev, curr: pd.Series for previous and current candle.
    Returns: True/False
    """
    try:
        prev_open, prev_close = float(prev["Open"]), float(prev["Close"])
        curr_open, curr_close = float(curr["Open"]), float(curr["Close"])
    except (KeyError, ValueError, TypeError):
        return False

    # 1. Previous candle is Bearish (Red)
    # 2. Current candle is Bullish (Green)
    # 3. Current Open is below Previous Close
    # 4. Current Close is above Previous Open (Engulfing)
    return (
            prev_close < prev_open and
            curr_close > curr_open and
            curr_open < prev_close and
            curr_close > prev_open
    )

def hammer(curr, upper_wick_max_ratio=0.25, lower_wick_min_ratio=2.0):
    """
    Detects Hammer candlestick pattern (single-candle).
    curr: pd.Series for current candle.
    upper_wick_max_ratio: Max ratio of upper wick to body size (e.g., 0.25).
    lower_wick_min_ratio: Min ratio of lower wick to body size (e.g., 2.0).
    Returns: True/False
    """
    try:
        open_, close = float(curr["Open"]), float(curr["Close"])
        high, low = float(curr["High"]), float(curr["Low"])
    except (KeyError, ValueError, TypeError):
        return False

    body = abs(close - open_)

    # Avoid division by zero if body is zero (e.g., Doji).
    if body == 0:
        return False

    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)

    # 1. Lower wick must be at least twice the body size (Hammer/Hanging Man rule).
    # 2. Upper wick must be very small (e.g., <= 25% of the body size).
    return (
            lower_wick >= lower_wick_min_ratio * body and
            upper_wick <= upper_wick_max_ratio * body
    )

def detect_patterns(df):
    """
    Detect candlestick patterns in OHLCV dataframe.
    Returns list of dicts with 'pattern' and 'date'.
    """
    if df is None or df.empty:
        return []

    df = df.copy()

    # Handle index resetting carefully to preserve original date/time info
    if "Date" not in df.columns:
        # Use 'index' as 'date' if 'Date' column is missing
        df.reset_index(inplace=True)
        date_col_name = df.columns[0] # The new column holding the index values
    else:
        date_col_name = "Date"

    patterns = []

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        date = df.iloc[i][date_col_name]

        if bullish_engulfing(prev, curr):
            patterns.append({"pattern": "Bullish Engulfing", "date": date})

        # Note: This checks for the *shape* of a Hammer/Hanging Man.
        # A true Hammer requires a prior downtrend (context not checked here).
        if hammer(curr):
            patterns.append({"pattern": "Hammer", "date": date})

    return patterns