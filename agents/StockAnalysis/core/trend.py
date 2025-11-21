import pandas as pd

def classify_trend(df):
    """
    Classify market trend using SMA50 vs SMA200.

    Returns:
        "Bullish", "Bearish", or "Neutral"
    """
    if "Close" not in df or df["Close"].empty:
        return "Neutral"

    sma50 = df["Close"].rolling(50, min_periods=1).mean()
    sma200 = df["Close"].rolling(200, min_periods=1).mean()

    last_sma50 = sma50.iloc[-1]
    last_sma200 = sma200.iloc[-1]

    if pd.isna(last_sma50) or pd.isna(last_sma200):
        return "Neutral"
    elif last_sma50 > last_sma200:
        return "Bullish"
    elif last_sma50 < last_sma200:
        return "Bearish"
    else:
        return "Neutral"
