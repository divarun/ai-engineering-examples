import pandas as pd
import numpy as np

def _safe_float(series, default=None):
    """Safely extracts the last float value from a Pandas Series."""
    if series is not None and not series.empty:
        val = series.iloc[-1]
        if pd.notna(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return default

def trend_fn(df):
    """
    Trend classifier using SMA50 and SMA200.
    Returns:
        dict with trend label and confidence level:
        {
            "trend": "Bullish"|"Bearish"|"Neutral",
            "strength": "Strong"|"Moderate"|"Weak",
            "sma50": float or None,
            "sma200": float or None
        }
    """

    # 1. Data Extraction (using the robust helper)
    sma50 = _safe_float(df.get("SMA50"))
    sma200 = _safe_float(df.get("SMA200"))

    if sma50 is None or sma200 is None:
        return {"trend": "Unknown", "strength": "Unknown", "sma50": None, "sma200": None}

    # Check for near-zero SMA200 to prevent division by zero
    if sma200 == 0:
        return {"trend": "Unknown", "strength": "Unknown", "sma50": sma50, "sma200": sma200}

    # 2. Trend Classification
    if sma50 > sma200:
        trend = "Bullish"
    elif sma50 < sma200:
        trend = "Bearish"
    else: # sma50 == sma200
        trend = "Neutral"

    # 3. Strength Assignment based on normalized difference (SMA Gap)
    diff_abs = abs(sma50 - sma200) / sma200

    # Grounding: Setting a tighter band (e.g., 0.5%) for true Neutrality
    # when SMAs are crossing or converging.
    if diff_abs < 0.005: # < 0.5% difference
        trend = "Neutral"
        strength = "Weak" # Can be considered weak regardless of direction

    elif diff_abs >= 0.03: # >= 3% difference (Strong separation)
        strength = "Strong"
    elif diff_abs >= 0.015: # >= 1.5% difference (Moderate separation)
        strength = "Moderate"
    else:
        # 0.5% <= diff_abs < 1.5%
        strength = "Weak"

    return {"trend": trend, "strength": strength, "sma50": sma50, "sma200": sma200}


def build_summary(ticker, df, mtf=None, patterns=None, s_r=None, trade_plan=None):
    """
    Build a structured summary of market data for LLM interpretation.
    Handles missing or insufficient columns safely.
    """

    # Helper: multi-timeframe trend
    def safe_mtf_trend(mtf_dict):
        trends = {}
        if not mtf_dict:
            return trends
        # Apply trend_fn to each DataFrame in the multi-timeframe dict
        for tf, mdf in mtf_dict.items():
            # Ensure mdf is a valid DataFrame before passing it
            if isinstance(mdf, pd.DataFrame):
                trends[tf] = trend_fn(mdf)
            else:
                trends[tf] = {"trend": "Unknown", "strength": "Unknown", "sma50": None, "sma200": None}
        return trends

    # Build summary
    summary = {
        "ticker": ticker,
        "indicators": {
            "rsi": _safe_float(df.get("RSI")),
            "macd": {
                "macd": _safe_float(df.get("MACD")),
                "signal": _safe_float(df.get("MACD_Signal")),
                "hist": _safe_float(df.get("MACD_Hist"))
            },
            "bollinger": {
                "upper": _safe_float(df.get("BB_UPPER")),
                "mid": _safe_float(df.get("BB_MID")),
                "lower": _safe_float(df.get("BB_LOWER"))
            },
            "atr": _safe_float(df.get("ATR")),
        },
        "trend": safe_mtf_trend(mtf),
        "patterns": patterns if patterns else [],
        "support_resistance": s_r if s_r else {"support_zones": [], "resistance_zones": []},
        "trade_plan": trade_plan if trade_plan else {}
    }

    return summary