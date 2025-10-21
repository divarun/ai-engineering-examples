# AI Job Application & Match Consultant 💼

This project is a  **Streamlit** application built using **LangGraph** to create a multi-step, stateful workflow for automating and optimizing job application materials. It analyzes a job description against an existing resume, provides a detailed match analysis, generates an **ATS-friendly, tailored resume**, and optionally creates a personalized **cover letter**.

It leverages an **Ollama**-hosted Large Language Model (LLM) to perform the deep contextual analysis and generation tasks locally.

## Features

* **Holistic Application Workflow**: A stateful LangGraph pipeline manages the entire process from analysis to final document generation.

* **Job Match Analysis**: Provides a detailed breakdown of the candidate's fit against the job description, including a confidence score and key gaps.

* **ATS Optimization**: The core agent adjusts the resume to maximize its keyword match and compliance with Applicant Tracking Systems (ATS) for higher screening success.

* **Conditional Routing**: The workflow is flexible; a routing function determines whether to proceed to the optional Cover Letter Generation step based on user input.

* **Local LLM Integration**: Uses the langchain_ollama wrapper, enabling all sensitive resume-tailoring and analysis to be run on a local model (e.g., Llama 3 or Granite) via the Ollama server.

* **Interactive UI**: A Streamlit interface for easy input of the Job Description and Original Resume, control over the optional steps, and clear display of final tailored documents.

* **Typed State Management**: Utilizes Python TypedDict for clear, explicit state management across the LangGraph nodes.

* **Flexible Document Options**: The output section offers three practical ways to save the tailored documents: Direct PDF Download, and Copy/Paste with instructions for conversion to Microsoft Word (DOCX) or Google Docs

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
pip install langgraph langchain-ollama streamlit python-dotenv markdown-pdf

```

### 3. Environment Variables

Create a file named **`.env`** in the root directory of your project and populate it with your Ollama configuration.

```dotenv
# Ollama LLM Configuration
# MUST match the model you pulled with `ollama pull` (e.g., llama3, mistral, etc.)
OLLAMA_MODEL_NAME="granite4"
```

### 4. Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

---


## License

This project is part of the AI Engineering Examples collection.