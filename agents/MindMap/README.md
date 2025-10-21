# AI MindMap Generator 🧠

This project is a **Streamlit** application that uses **CrewAI** agents to automatically generate a visual **Mermaid.js Mind Map** from any public, long-form article URL. It leverages an **Ollama**-hosted Large Language Model (LLM) to scrape the content, analyze its core hierarchy, and format the structure into a clean, hierarchical graph.

## Features

* **URL-Based Analysis**: Automatically ingests and processes content from any public web page or PDF link.
* **Multi-Agent AI Pipeline**: Uses a sophisticated four-agent CrewAI pipeline:
    1.  **Web Content Scraper**: Extracts clean, full-text main article content from URLs.
    2.  **Content Analysis Specialist**: Analyzes text content and breaks down articles into structured mind map hierarchies.
    3.  **Content Analysis Specialist** (Validation): Validates the hierarchy to ensure all items are grounded in the original article.
    4.  **Mermaid Code Generator**: Converts the validated hierarchy into precise Mermaid.js diagram code.
* **Local LLM Integration**: Uses the **`langchain_ollama`** wrapper, allowing you to run the heavy-lifting with a local model (e.g., Llama 3) via the Ollama server.
* **Interactive UI**: A Streamlit interface for easy input, control over generation parameters, and viewing/downloading the final code.
* **Robust Error Handling**: Comprehensive validation and fallback mechanisms to ensure valid Mermaid code generation.
* **Timestamped Output**: Generated mind maps are saved with datetime timestamps for easy organization.


---

## Installation and Setup

### 1. Get Ollama and Model

This project relies on the **Ollama** server to run the LLM locally.

1.  **Install Ollama**: Download and install [Ollama](https://ollama.com/download) for your operating system.
2.  **Pull an LLM**: Use the Ollama CLI to pull a powerful model you want to use. We recommend **Llama 3** for strong reasoning.
    ```bash
    ollama pull llama3
    ```

### 2. Install Python Dependencies

You'll need a Python environment (e.g., using `conda` or `venv`). Install the required packages using `pip`.

```bash
# It's highly recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install all necessary packages
pip install crewai crewai-tools streamlit python-dotenv

```

### 3. Environment Variables

Create a file named **`.env`** in the root directory of your project and populate it with your Ollama configuration.

```dotenv
# Ollama LLM Configuration
# MUST match the model you pulled with `ollama pull` (e.g., llama3, mistral, etc.)
OLLAMA_MODEL_NAME="ollama/granite4"
# The default base URL for Ollama is usually this.
OLLAMA_BASE_URL="http://localhost:11434"
```

### 4. Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

---

## How It Works

### Workflow

1. **Input**: Paste a URL from a public article
2. **Scraping**: The Web Content Scraper extracts clean article content
3. **Analysis**: The Content Analysis Specialist creates a structured hierarchy
4. **Validation**: The validation agent ensures all items are grounded in the article
5. **Mapping**: The Mermaid Code Generator converts the hierarchy to Mermaid.js code
6. **Output**: View and download the generated mind map code or render it directly

### Agent Configuration

- **Web Content Scraper**: Uses ScrapeWebsiteTool to extract main article content
- **Content Analysis Specialist**: Creates indented bullet lists with 1 main topic, 3-5 subtopics, and 2-3 key ideas per subtopic
- **Mermaid Code Generator**: Converts hierarchies to valid Mermaid.js syntax with proper node formatting

### Output Format

The generated mind maps follow this structure:
- **Main Topic**: `A((Topic Name))` - Double parentheses for main topics
- **Subtopics**: `A --> B1(Subtopic Name)` - Single parentheses for subtopics  
- **Key Ideas**: `B1 --> C1(Key Idea Name)` - Single parentheses for key ideas

---

## File Structure

```
agents/MindMapAI/
├── app.py              # Streamlit application
├── pipeline.py         # CrewAI agent pipeline
├── config.py          # LLM configuration
├── utils.py           # Utility functions
├── .env.example       # Environment variables template
└── README.md          # This file
```

---

## Dependencies

- **crewai**: Multi-agent AI framework
- **crewai-tools**: Tools for web scraping
- **streamlit**: Web application framework
- **python-dotenv**: Environment variable management
- **langchain-ollama**: Ollama LLM integration

---

## License

This project is part of the AI Engineering Examples collection.