# 🕯️📊 Advanced Trading Insights Analyzer

This project is a sophisticated Streamlit application that utilizes a CrewAI multi-agent system and a local Ollama-hosted Large Language Model (LLM) to perform in-depth technical analysis and generate actionable trading plans for any stock ticker. It moves beyond simple charting by integrating crucial financial indicators and risk management into the analysis pipeline.

## 🚀 Features

* **Multi-Indicator Analysis**: Automatically calculates and integrates the 14-day RSI, 50-day SMA, and 200-day SMA for comprehensive market context.

* **Volume-Confirmed Patterns**: Identifies significant candlestick patterns (e.g., Hammer, Doji, Engulfing) and assesses their reliability based on recent trading volume.

* **Three Specialized AI Agents (CrewAI Pipeline)**: A robust, sequential pipeline ensures specialized and grounded analysis:

    1. Candlestick and Volume Expert: Focuses on short-term price action and volume confirmation.

    2. Multi-Indicator Market Strategist: Synthesizes all data (Patterns, RSI, SMAs) into a market forecast.

    3. Trading Strategy Architect: Creates a quantifiable trade plan with Entry, Stop-Loss, and Take-Profit levels.

* **Local LLM Integration**: Uses the CrewAI Ollama wrapper to connect to a local model (e.g., Mistral, Llama 3), ensuring data privacy and cost efficiency.

* **Interactive UI**: A Streamlit interface for easy ticker input, visualization of the last 30 days of price data, and clear display of the final trade plan.

* **Output**: Generates a professional-grade analysis report with a clear, actionable trading conclusion.

## 📋 Installation and Setup

1. Get Ollama and Model

This project requires the Ollama server to run the LLM locally.

* Install Ollama: Download and install the application from the official Ollama website.

* Pull an LLM: Pull a model (like mistral or llama3) using the Ollama CLI. We recommend Mistral for its performance and speed on analysis tasks.

```ollama pull mistral ```


* Keep Ollama Running: Ensure the Ollama server is active and running in the background before running the Python script.

2. Install Python Dependencies

Set up your Python environment and install the required packages.
```
# It's recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install all necessary packages
pip install streamlit yfinance pandas crewai crewai-tools python-dotenv numpy
```


3. Environment Variables

Create a file named .env in the root directory of your project to configure the LLM connection details. These details are crucial for the crewai-tools.Ollama connection.
```
# .env file content
# Ollama LLM Configuration
# The default base URL for Ollama
OLLAMA_BASE_URL="http://localhost:11434"
# MUST match the model you pulled (e.g., mistral, llama3, etc.)
OLLAMA_MODEL_NAME="mistral"
```

4. Run the Application

Start the Streamlit application from your terminal:
```
streamlit run trading_insights.py
```

The application will launch in your web browser, typically at http://localhost:8501.

## 🧠 CrewAI Agent Pipeline

The analysis flows sequentially through the three specialized agents, ensuring that the final trading plan is fully grounded in the technical analysis.

| Stage | Agent Role |Key Task | Output Consumed By |
|-------|:----------:|:--------:|:-------------------|
| 1     | Candlestick Expert  |Identify candlestick patterns and confirm their reliability using volume data.| Market Strategist  |
| 2     | Market Strategist  | Synthesize patterns with RSI and SMA indicators to determine the short-term market forecast (Bullish/Bearish/Neutral).| Strategy Architect |
| 3     |Strategy Architect  |Use the forecast to create a clear, actionable trading plan including Entry, Stop-Loss, and Take-Profit prices. | Final Report       |

