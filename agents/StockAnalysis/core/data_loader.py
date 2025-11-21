import yfinance as yf
import pandas as pd


# -------------------------
# 1️⃣ Single timeframe data
# -------------------------
def load_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Load historical OHLCV data for a single timeframe.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., "AAPL").
        period (str): Data period (default "1y").
        interval (str): Data interval (default "1d").

    Returns:
        pd.DataFrame: OHLCV data with index as datetime, or empty DataFrame if failed.
    """
    try:
        # Use a longer history (3 years) to ensure better indicator calculation
        df = yf.download(ticker, period="3y", interval=interval, progress=False)
        df.dropna(inplace=True)
        if df.empty:
            print(f"Warning: No data returned for {ticker} ({period}, {interval})")
        return df
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return pd.DataFrame()


# -------------------------
# 2️⃣ Multi-timeframe data
# -------------------------
def load_multi_timeframe(ticker: str) -> dict:
    """
    Load OHLCV data across multiple common timeframes: 1D, 4H, 1H.

    Returns:
        dict: Keys = timeframe strings, Values = OHLCV DataFrames.
    """
    timeframes = {
        "1D": {"period": "3y", "interval": "1d"},
        "4H": {"period": "1y", "interval": "4h"},
        "1H": {"period": "180d", "interval": "1h"}
    }

    mtf_data = {}
    for tf, params in timeframes.items():
        # Adjusting load_data to use the correct period/interval
        df = load_data(ticker, period=params["period"], interval=params["interval"])

        # Ensure only the required columns are in the MTF data
        if not df.empty:
            mtf_data[tf] = df[['Close', 'Volume']].tail(1)  # Only need last row for context
        else:
            mtf_data[tf] = pd.DataFrame()

    return mtf_data