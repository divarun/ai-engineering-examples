import streamlit as st
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from crewai import LLM

from core.data_loader import load_data, load_multi_timeframe
from core.indicators import rsi, macd, bollinger, atr,calculate_smas
from core.patterns import detect_patterns
from core.support_resistance import support_resistance
from core.strategy_engine import build_trade_plan
from core.summarizer import build_summary
from ai.llm_agent import build_interpretation_agent, interpret

# -------------------------
# 0. ENVIRONMENT SETUP
# -------------------------
load_dotenv()

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

if not OLLAMA_MODEL_NAME or not OLLAMA_BASE_URL:
    st.error("❌ CRITICAL: OLLAMA_MODEL_NAME and OLLAMA_BASE_URL must be set in your .env file.")
    st.stop()


# -------------------------
# 1. LLM Initialization
# -------------------------
@st.cache_resource
def initialize_llm():
    """Initializes and caches the CrewAI LLM connection."""
    try:
        llm = LLM(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        st.sidebar.success("✅ LLM Connection Established")
        return llm
    except Exception:
        st.sidebar.error("❌ LLM Error: Could not connect to Ollama.")
        st.stop()
        return None


# -------------------------
# 2. DATA & INDICATOR CACHING
# -------------------------
@st.cache_data(show_spinner=False)
def get_data_and_indicators(ticker: str):
    """Load data, compute indicators, and multi-timeframe data with caching."""
    df = load_data(ticker)
    if df.empty:
        return None, None

    #  Standardize columns and drop the MultiIndex Ticker level
    if isinstance(df.columns, pd.MultiIndex):
        # Drop Level 1 (the Ticker, e.g., 'TSLA') to leave only the metric (e.g., 'Close')
        df.columns = df.columns.droplevel(level=1)

    # Ensure all column names are simple strings and capitalized (e.g., 'close' -> 'Close')
    df.columns = [str(col).capitalize() for col in df.columns]

    # Fallback for 'Adj Close'
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    required_cols = ["High", "Low", "Close"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing core columns after standardization: {required_cols}")
        return None, None

    # Safe indicator calculations
    try:
        df = rsi(df)
        df = macd(df)
        df = bollinger(df)
        df = atr(df)
        df = calculate_smas(df)  # Correctly integrated SMA calculation
    except Exception as e:
        # It's better to log the exception, not just pass silently
        st.error(f"Error during indicator calculation: {e}")
        pass

    # Multi-timeframe
    try:
        mtf = load_multi_timeframe(ticker)
    except Exception as e:
        st.error(f"Error loading multi-timeframe data: {e}")
        mtf = {}

    return df, mtf

# -------------------------
# 3. STREAMLIT UI
# -------------------------
def main():
    st.set_page_config(page_title="Advanced Trading Analysis", layout="wide")
    st.title("🕯️📊 Advanced Trading Analysis")
    st.markdown("Enter a stock ticker to get AI-powered multi-indicator trading analysis.")

    # Initialize local LLM
    llm = initialize_llm()

    ticker_input = st.text_input(
        "Stock Ticker Symbol (e.g., AAPL, GOOGL, TSLA, MSFT)",
        key="ticker_input"
    ).upper().strip()

    if st.button("Run Advanced Analysis", type="primary"):
        if not ticker_input:
            st.warning("Please enter a valid stock ticker symbol.")
            return

        # -------------------------
        # DATA LOADING & INDICATORS
        # -------------------------
        df, mtf = get_data_and_indicators(ticker_input)
        if df is None or df.empty:
            st.error(f"❌ Could not load data for {ticker_input}. Please check the ticker.")
            return

        # -------------------------
        # DISPLAY OHLCV TABLE & CHART
        # -------------------------
        st.subheader(f"📈 Last 20 Trading Days for {ticker_input}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(20))
        with col2:
            st.line_chart(df['Close'].tail(20))

        # -------------------------
        # PATTERNS & SUPPORT/RESISTANCE
        # -------------------------
        try:
            analysis_df = df.tail(50).copy()
            patterns = detect_patterns(analysis_df)

        except Exception:
            patterns = []

        try:
            s_r = support_resistance(analysis_df)
        except Exception:
            s_r = {"support_zones": [], "resistance_zones": []}

        # -------------------------
        # TRADE PLAN
        # -------------------------
        trade_plan = build_trade_plan(df, s_r) or {}

        # -------------------------
        # SUMMARY STRUCTURE
        # -------------------------
        summary = build_summary(ticker_input, df, mtf, patterns, s_r, trade_plan)

        # -------------------------
        # LLM INTERPRETATION
        # -------------------------
        with st.spinner("⏳ Running AI interpretation in background..."):
            try:
                agent = build_interpretation_agent(llm)
                interpretation = interpret(agent, summary, ohlcv_df=df)
            except Exception as e:
                interpretation = f"Error generating AI interpretation: {e}"

        # -------------------------
        # DISPLAY AI RESULTS
        # -------------------------
        st.subheader("📊 Analysis Results")

        # --- AI INTERPRETATION  ---

        st.markdown("### AI Interpretation")
        st.markdown(interpretation, unsafe_allow_html=False)

        st.markdown("---")  # Optional: Add a separator line for clarity

        # --- DETERMINISTIC TRADE PLAN  ---
        st.markdown("### Deterministic Trade Plan")

        # Display trade plan clearly using Streamlit metrics
        if trade_plan.get('direction'):
            st.metric(label="Direction", value=trade_plan['direction'], delta=f"R/R: {trade_plan['risk_reward']}")
            st.metric(label="Entry Price", value=f"${trade_plan['entry']:.2f}")
            st.metric(label="Stop Loss", value=f"${trade_plan['stop_loss']:.2f}")
            st.metric(label="Take Profit", value=f"${trade_plan['take_profit']:.2f}")
        else:
            st.warning("No clear trade plan could be established.")

        st.markdown("---")
        st.markdown(f"**Note:** {trade_plan.get('note', 'No note available.')}")

        st.success("✅ Analysis complete!")

        st.markdown("""
        ---
        > **⚠️ DISCLAIMER:** AI-generated analysis is for educational purposes only  
        > and is **NOT** financial advice.
        """)


# -------------------------
# 4. START APP
# -------------------------
if __name__ == "__main__":
    main()
