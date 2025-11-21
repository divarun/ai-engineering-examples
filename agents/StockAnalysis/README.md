# 🕯️📊 Advanced Trading Insights Analyzer

This project is a sophisticated Streamlit application that utilizes a CrewAI multi-agent system and a local Ollama-hosted Large Language Model (LLM). 
It performs an in-depth technical analysis for any stock ticker, synthesizing complex market data to generate a detailed, actionable trading plan focused on risk management.

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
python -m streamlit run app.py
```

The application will launch in your web browser, typically at http://localhost:8501.

