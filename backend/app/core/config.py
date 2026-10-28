import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "Universal Document Intelligence Platform"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Environment
    ENV: str = "development"
    DEBUG: bool = True
    
    # PostgreSQL Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "doc_intelligence"
    
    # Storage settings
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "storage", "uploads")
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "jpg", "jpeg", "png"]
    
    # OCR Engine settings
    TESSERACT_CMD: str = Field(default="tesseract", description="Path to tesseract executable")
    OCR_DPI: int = 300
    
    # LLM Provider settings (Open-weight models: Qwen / Llama via OpenAI-compatible endpoints)
    LLM_PROVIDER: str = Field(default="open_weight", description="open_weight, ollama, or none")
    LLM_BASE_URL: Optional[str] = Field(default=None, description="OpenAI-compatible base URL (e.g., Groq, OpenRouter, Ollama, vLLM)")
    LLM_API_KEY: Optional[str] = Field(default=None, description="API Key for hosted inference")
    LLM_MODEL: str = Field(default="qwen-2.5-72b-instruct", description="Model name to use for extraction")
    LLM_TIMEOUT: int = 180
    
    # RAG Settings
    ENABLE_RAG: bool = True
    
    # Security
    API_KEY_AUTH_ENABLED: bool = False
    API_KEY: Optional[str] = None
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
