import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

if not OLLAMA_MODEL_NAME or not OLLAMA_BASE_URL:
    raise EnvironmentError("Missing OLLAMA_MODEL_NAME or OLLAMA_BASE_URL in .env")

def get_llm() -> LLM:
    return LLM(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.0)


