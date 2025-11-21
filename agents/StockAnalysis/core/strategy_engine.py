import pandas as pd
import numpy as np


def build_trade_plan(df, s_r):
    if df is None or df.empty:
        return {}

    last_close = float(df["Close"].iloc[-1])

    support_zones = s_r.get("support_zones", [])
    resistance_zones = s_r.get("resistance_zones", [])

    # Normalize zones: dict → float(level)
    support_levels = [
        z["level"] if isinstance(z, dict) else float(z)
        for z in support_zones
    ]
    resistance_levels = [
        z["level"] if isinstance(z, dict) else float(z)
        for z in resistance_zones
    ]

    # Find nearest levels
    nearest_support = max([s for s in support_levels if s < last_close] or [0])
    nearest_resistance = min([r for r in resistance_levels if r > last_close] or [float('inf')])

    # Impossible structure (no real R above price)
    if nearest_resistance == float("inf"):
        return {"note": "No valid resistance zone above price."}

    rr = (nearest_resistance - last_close) / (last_close - nearest_support) if nearest_support > 0 else None

    return {
        "direction": "LONG" if nearest_support < last_close < nearest_resistance else None,
        "entry": last_close,
        "stop_loss": nearest_support,
        "take_profit": nearest_resistance,
        "risk_reward": round(rr, 2) if rr else None,
        "note": "Trade plan generated based on support/resistance levels."
    }
