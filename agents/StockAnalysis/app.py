import streamlit as st
import os
import yfinance as yf
import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent
from dotenv import load_dotenv

# --- 0. Environment and Config Setup ---
load_dotenv()

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

if not OLLAMA_MODEL_NAME or not OLLAMA_BASE_URL:
    st.error("❌ CRITICAL: OLLAMA_MODEL_NAME and OLLAMA_BASE_URL must be set in your .env file.")
    st.stop()

# --- 1. LLM Initialization (Cached) ---
@st.cache_resource
def initialize_llm():
    """Initializes and caches the CrewAI LLM connection."""
    try:
        llm = LLM(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )
        st.sidebar.success("✅ LLM Connection Established")
        return llm
    except Exception as e:
        # It's better to catch the exception here and let the user know what's wrong.
        st.sidebar.error(f"❌ LLM Error: Could not connect to Ollama.")
        st.stop()
        return None

# --- 2. Data Fetching and Indicator Calculation ---

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    # Requires 'Close' column
    delta = data['Close'].diff()
    # Ensure gain/loss calculation handles NaN values initially from diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

@st.cache_data(show_spinner="Fetching stock data and calculating indicators...")
def fetch_stock_data_with_indicators(ticker, period="3mo"):
    """
    Fetches stock data, calculates key indicators (RSI, MAs),
    and prepares a summary for the agents.

    Note: Using "3mo" means SMA200 will likely be NaN. A longer period (e.g., "1y") is recommended for a valid SMA200.
    """
    try:
        # Fetch data. Using "1y" would ensure sufficient data for SMA200
        data = yf.download(ticker, period=period, interval="1d", progress=False)

        if data.empty:
            return None, f"Error: Could not retrieve data for {ticker}. Ticker may be invalid or data unavailable.", None

        # Calculate Indicators
        data = calculate_rsi(data, window=14)
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()

        # FIX: The original code dropped the indicator columns here, causing the 'RSI' error.
        # We now slice the last 15 days while keeping ALL calculated columns.
        ohlcv_data = data.tail(15).copy() # Use .copy() to ensure a clean DataFrame

        if ohlcv_data.empty:
            return None, f"Error: No trading days found in the last {period}.", None

        # Generate Indicator Summary for Agent Context
        last_day = ohlcv_data.index[-1]

        # Check for NaN and handle gracefully before rounding
        rsi_val = ohlcv_data['RSI'].iloc[-1]
        sma50_val = ohlcv_data['SMA50'].iloc[-1]
        sma200_val = ohlcv_data['SMA200'].iloc[-1]

        summary = {
            # Use a check to display 'N/A' if indicators are NaN (due to insufficient history)
            "RSI_14": round(rsi_val, 2) if pd.notna(rsi_val) else 'N/A',
            "SMA50": round(sma50_val, 2) if pd.notna(sma50_val) else 'N/A',
            "SMA200": round(sma200_val, 2) if pd.notna(sma200_val) else 'N/A',
            "Current_Price": round(ohlcv_data['Close'].iloc[-1], 2),
            "Average_Volume_10D": round(ohlcv_data['Volume'].iloc[-10:-1].mean(), 0),
            "Last_Volume": ohlcv_data['Volume'].iloc[-1],
        }

        # Return the 15-day DataFrame (including indicators), the 10-day OHLCV string for the agent, and the summary
        return ohlcv_data, ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).to_string(), summary

    except Exception as e:
        # A more generic error catch for data-related issues
        return None, f"An error occurred while fetching data: {e}", None

# --- 3. Crew Execution Function ---
def run_candlestick_analysis(llm, ticker, ohlcv_data_string, summary):
    """
    Sets up and executes the CrewAI task based on the user's ticker, data, and indicators.
    """

    # --- Indicator Summary String for Agent Context ---
    # The summary now safely handles 'N/A' values
    summary_string = dedent(f"""
        ---
        KEY INDICATOR SUMMARY for {ticker}:
        - Current Price: {summary['Current_Price']}
        - 14-Day RSI: {summary['RSI_14']} (Context: <30=Oversold, >70=Overbought)
        - 50-Day SMA: {summary['SMA50']}
        - 200-Day SMA: {summary['SMA200']}
        - Average 10-Day Volume: {summary['Average_Volume_10D']}
        - Last Trading Day Volume: {summary['Last_Volume']}
        ---
    """)

    # --- Agent Definitions (No change needed) ---
    pattern_detector = Agent(
        role='Candlestick and Volume Expert',
        goal=dedent(f"""
            Analyze the provided real historical OHLCV data for {ticker}. Identify known candlestick patterns
            and assess their reliability based on the last day's volume relative to the average 10-day volume.
        """),
        backstory=dedent("""
            You are a highly specialized technical analyst who identifies price patterns and confirms their
            validity using trading volume metrics.
        """),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    market_analyst = Agent(
        role='Multi-Indicator Market Strategist',
        goal=dedent(f"""
            Based on the identified candlestick patterns, the 10-day trend, and the provided technical
            indicators (RSI, SMAs), interpret the current market sentiment and forecast the likely short-term
            price movement (Bullish, Bearish, or Neutral).
        """),
        backstory=dedent("""
            You are a professional financial market strategist. You synthesize signals from multiple
            indicators to provide a clear, professional, and cautionary trading conclusion.
        """),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    risk_architect = Agent(
        role='Trading Strategy Architect',
        goal=dedent(f"""
            Based on the final market sentiment for {ticker}, formulate a concise, actionable trade plan.
            This plan MUST include a recommended Entry price, a Stop-Loss price, and a Take-Profit (Target)
            price. The plan should aim for a minimum 1:1 Risk/Reward ratio. Use the last 10 days of prices
            to define these levels around the detected pattern.
        """),
        backstory=dedent("""
            You are an expert in risk management, converting complex market analysis into simple,
            quantifiable trading instructions.
        """),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    # --- Task Definitions ---
    pattern_task = Task(
        description=dedent(f"""
            Analyze the real historical data for {ticker}.
            **IMPORTANT:** The most recent closing price and indicator values are explicitly provided in the 'KEY INDICATOR SUMMARY'. Always use this summary for the current market state and for the last day's price.
            
            Data (last 10 trading days):
            ---
            {ohlcv_data_string}
            ---
            Indicator Summary:
            {summary_string}
            
            Identify all significant patterns and explicitly state the volume context (above/below average)
            for the most recent candle.
        """),
        agent=pattern_detector,
        expected_output="A list of identified candlestick patterns, their dates, volume confirmation, and short-term implications."
    )

    report_task = Task(
        description=dedent(f"""
            Synthesize the pattern analysis and the provided RSI and SMA values to create a comprehensive
            market forecast for {ticker}. The report must include:
            1. A summary of the 10-day trend, confirmed by the RSI status (Oversold/Overbought).
            2. How the current price relates to the 50-day and 200-day SMAs.
            3. The final market sentiment (e.g., "Strong Bullish Reversal Signal").
        """),
        agent=market_analyst,
        context=[pattern_task],
        expected_output=f"A professional, multi-indicator technical analysis report for {ticker} with a clear market sentiment."
    )

    strategy_task = Task(
        description=dedent(f"""
            Based *only* on the market forecast provided by the Market Strategist, formulate the final
            actionable trade plan for {ticker}.
        """),
        agent=risk_architect,
        context=[report_task],
        expected_output=f"A concise trading plan for {ticker} with a clear Entry Price, Stop-Loss Price, and Take-Profit Price, formatted clearly."
    )

    # --- Crew Setup and Execution (No change needed) ---
    candlestick_crew = Crew(
        agents=[pattern_detector, market_analyst, risk_architect],
        tasks=[pattern_task, report_task, strategy_task],
        process=Process.sequential,
        verbose=True,
        manager_llm=llm
    )

    with st.spinner(f"🚀 Running advanced analysis for {ticker}..."):
        try:
            result = candlestick_crew.kickoff()
            # Return both the report and the strategy
            return result
        except Exception as e:
            return f"🚨 CRITICAL EXECUTION ERROR: An error occurred during crew execution. {e}"

# --- 4. Streamlit UI Layout (No change needed) ---
def main():
    st.set_page_config(page_title="Advanced Trading Analysis", layout="wide")
    st.title("🕯️📊 Advanced Trading Analysis")
    st.markdown("Enter a stock ticker to get a **multi-indicator technical analysis** and a **quantified trading plan**.")

    # Sidebar for configuration and status
    st.sidebar.header("Configuration")
    llm = initialize_llm()

    # Main input
    ticker_input = st.text_input(
        "Stock Ticker Symbol (e.g., AAPL, GOOGL, TSLA)",
        key="ticker_input",
        value="",
        max_chars=10
    ).upper().strip()

    if st.button("Run Advanced Analysis", type="primary", use_container_width=True):
        if not ticker_input:
            st.warning("Please enter a valid stock ticker symbol.")
            return

        # --- Data Fetching ---
        ohlcv_df, ohlcv_data_string, summary = fetch_stock_data_with_indicators(ticker_input)

        if ohlcv_df is None:
            st.error(f"❌ Data Error: {ohlcv_data_string}")
            return

        # --- Display Raw Data and Chart ---
        st.subheader(f"📈 Price & Volume Data for {ticker_input} (Last 15 Trading Days)")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))

        with col2:
            st.line_chart(ohlcv_df['Close'].tail(30))

        st.markdown("---")

        # --- Crew Execution ---
        final_report = run_candlestick_analysis(llm, ticker_input, ohlcv_data_string, summary)

        # --- Display Results ---
        st.header(f"✨ Final Trading Report & Strategy: {ticker_input}")

        if "CRITICAL EXECUTION ERROR" in final_report:
            st.error(final_report)
        else:
            st.info(f"**Actionable Trading Plan from Risk Architect:**\n\n{final_report}")

            with st.expander("📝 View Detailed Market Analysis"):
                st.markdown(final_report)

        st.success("✅ Advanced Analysis Complete!")

        # --- DISCLAIMER SECTION ADDED ---
        st.markdown("---")
        st.markdown(
            """
            > **⚠️ DISCLAIMER:** This analysis is generated by an experimental AI model (CrewAI agents) for educational and informational purposes only. It is based *only* on historical price, volume, and technical indicators and does **NOT** constitute financial advice, investment recommendation, or an offer to buy or sell any security. **Trading stocks involves substantial risk of loss.** You should consult with a qualified professional before making any investment decisions. **Use this only as a recommendation or starting point for your own due diligence.**
            """
        )

if __name__ == "__main__":
    main()