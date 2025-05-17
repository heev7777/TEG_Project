import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pathlib import Path
import logging

# Define the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file from the project root
load_dotenv(dotenv_path=BASE_DIR / ".env")

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Product Feature Comparison Assistant"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Compare product features using RAG and Multi-Agent Systems."

    # LLM and LangSmith
    OPENAI_API_KEY: str
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https_api.smith.langchain.com")
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "Product-Comparison-Assistant")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini") # Cheaper and faster for development

    # MCP Server
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "127.0.0.1") # Use 127.0.0.1 for local
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", 8001)) # Changed port to avoid conflict with potential FastAPI backend

    # Backend Server (if you create a separate one for agent logic, distinct from Streamlit's implicit backend)
    BACKEND_SERVER_HOST: str = os.getenv("BACKEND_SERVER_HOST", "127.0.0.1")
    BACKEND_SERVER_PORT: int = int(os.getenv("BACKEND_SERVER_PORT", 8000))

    # RAG Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    RAG_TOP_K_RESULTS: int = int(os.getenv("RAG_TOP_K_RESULTS", 3))

    # Temp file storage for uploads
    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    class Config:
        case_sensitive = True
        # If you were using .env file directly with pydantic_settings v2:
        # env_file = ".env"
        # env_file_encoding = "utf-8"

try:
    settings = Settings()
except Exception as e:
    logger.error(f"Configuration error: {e}. Please check your .env file for required keys.")
    raise

# Ensure upload directory exists
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# LangSmith Configuration (explicitly set environment variables for LangChain)
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = str(settings.LANGCHAIN_ENDPOINT) # Ensure it's a string
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY