import numpy as np
import pandas as pd


# -------------------------
# 1️⃣ RSI (Relative Strength Index)
# -------------------------
def rsi(df, window=14):
    """
    Calculate RSI (Relative Strength Index) and add it to the DataFrame.
    Uses Wilder's Smoothing (RMA) for the standard calculation.
    """
    if "Close" not in df.columns or df["Close"].empty:
        raise ValueError("DataFrame must have a non-empty 'Close' column.")

    close = df["Close"].to_numpy()
    if len(close) < 2:
        df["RSI"] = 50
        return df

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    #  Explicitly pass index=df.index
    avg_gain = pd.Series(gain, index=df.index).ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = pd.Series(loss, index=df.index).ewm(com=window - 1, adjust=False, min_periods=window).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.divide(avg_gain, avg_loss)
        rs[np.isinf(rs)] = 100

    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi.fillna(50.0).clip(0, 100)
    return df


# -------------------------
# 2️⃣ MACD (Moving Average Convergence Divergence)
# -------------------------
def macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD, Signal line, and Histogram.
    """
    if "Close" not in df.columns or df["Close"].empty:
        raise ValueError("DataFrame must have a non-empty 'Close' column.")

    close = df["Close"].to_numpy()
    if len(close) < slow:
        df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = 0
        return df

    #  Add index=df.index to Series creation to retain DatetimeIndex
    exp_fast = pd.Series(close, index=df.index).ewm(span=fast, adjust=False).mean()
    exp_slow = pd.Series(close, index=df.index).ewm(span=slow, adjust=False).mean()

    df["MACD"] = exp_fast - exp_slow

    # Signal line is the EWM of MACD
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Fill NaNs
    for col in ["MACD", "MACD_Signal", "MACD_Hist"]:
        df[col].fillna(method="bfill", inplace=True)
        df[col].fillna(method="ffill", inplace=True)
        df[col].fillna(0, inplace=True)
    return df


# -------------------------
# 3️⃣ Bollinger Bands
# -------------------------
def bollinger(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands (Mid, Upper, Lower) and add to DataFrame.
    """
    if "Close" not in df.columns or df["Close"].empty:
        raise ValueError("DataFrame must have a non-empty 'Close' column.")

    close = df["Close"].to_numpy()

    # Add index=df.index to Series creation
    rolling_mean = pd.Series(close, index=df.index).rolling(window=window, min_periods=1).mean()

    # Rolling std will be NaN for min_periods < 2. Fill with 0 for safety.
    rolling_std = pd.Series(close, index=df.index).rolling(window=window, min_periods=1).std(ddof=0).fillna(0)

    # Calculate Bands
    bb_mid = rolling_mean
    bb_upper = rolling_mean + rolling_std * num_std
    bb_lower = rolling_mean - rolling_std * num_std

    # Fill NaNs
    mean_close = close.mean()
    df["BB_MID"] = bb_mid.fillna(method="bfill").fillna(method="ffill").fillna(mean_close)
    df["BB_UPPER"] = bb_upper.fillna(method="bfill").fillna(method="ffill").fillna(mean_close)
    df["BB_LOWER"] = bb_lower.fillna(method="bfill").fillna(method="ffill").fillna(mean_close)
    return df


# -------------------------
# 4️⃣ ATR (Average True Range)
# -------------------------
def atr(df, window=14):
    """
    Calculate ATR (Average True Range) and add it to the DataFrame.
    """
    required_cols = ["High", "Low", "Close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")

    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    close = df["Close"].to_numpy()

    if len(close) < 2:
        df["ATR"] = 0
        return df

    # Calculate TR
    h_l = high - low
    h_c_prev = np.abs(high - np.roll(close, 1))
    l_c_prev = np.abs(low - np.roll(close, 1))
    tr = np.maximum(h_l, np.maximum(h_c_prev, l_c_prev))
    tr[0] = h_l[0]

    # Create ATR series using correct index
    atr_series = pd.Series(tr, index=df.index).ewm(
        com=window - 1, adjust=False, min_periods=window
    ).mean()

    # FIX: must use .loc not .iloc
    first_valid_idx = atr_series.first_valid_index()
    first_valid_atr = atr_series.loc[first_valid_idx] if first_valid_idx is not None else 0

    df["ATR"] = (
        atr_series
        .fillna(method="bfill")
        .fillna(method="ffill")
        .fillna(first_valid_atr)
    )

    df["ATR"] = df["ATR"].clip(lower=0)
    return df



# -------------------------
# 5️⃣ SMA (Simple Moving Average)
# -------------------------
def calculate_smas(df):
    """
    Calculate SMA50 and SMA200 for trend analysis.
    """
    if "Close" not in df.columns or df["Close"].empty:
        return df

    # Rolling functions automatically retain the correct index
    df["SMA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["SMA200"] = df["Close"].rolling(200, min_periods=1).mean()
    return df