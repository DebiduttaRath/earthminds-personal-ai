"""
Configuration settings for the Business Intelligence Platform
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import os
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    llm_backend: str = Field(default=os.getenv("LLM_BACKEND", "groq"))
    
    # Groq Configuration
    groq_api_key: str = Field(default=os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    
    # DeepSeek Configuration
    deepseek_api_key: str = Field(default=os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = Field(default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    
    # Ollama Configuration (local fallback)
    ollama_base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_chat_model: str = Field(default=os.getenv("OLLAMA_CHAT_MODEL", "deepseek-v3"))
    ollama_reason_model: str = Field(default=os.getenv("OLLAMA_REASON_MODEL", "deepseek-r1:7b"))
    
    # Database Configuration
    database_url: Optional[str] = Field(default=os.getenv("DATABASE_URL"))
    checkpoint_db: str = Field(default=os.getenv("CHECKPOINT_DB", "bi_checkpoints.sqlite"))
    
    # Runtime Configuration
    thread_id: str = Field(default=os.getenv("THREAD_ID", "bi-session-1"))
    user_agent: str = Field(default=os.getenv("USER_AGENT", "BusinessIntelligencePlatform/1.0 (+contact: admin@company.com)"))
    http_timeout: int = Field(default=int(os.getenv("HTTP_TIMEOUT", "30")))
    http_retries: int = Field(default=int(os.getenv("HTTP_RETRIES", "3")))
    
    # Model Parameters
    max_context_tokens: int = Field(default=32768)  # Increased for business intelligence
    max_output_tokens: int = Field(default=4096)    # Increased for detailed analysis
    default_temperature: float = Field(default=0.6)
    
    # Business Intelligence Specific
    enable_reasoning_traces: bool = Field(default=True)
    max_sources_per_analysis: int = Field(default=10)
    enable_financial_data: bool = Field(default=True)
    enable_competitor_analysis: bool = Field(default=True)
    
    # OpenTelemetry Configuration
    otel_endpoint: Optional[str] = Field(default=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
    otel_service_name: str = Field(default=os.getenv("OTEL_SERVICE_NAME", "business-intelligence-platform"))
    
    # Security Settings
    rate_limit_requests_per_minute: int = Field(default=60)
    enable_content_filtering: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validate critical settings
if not settings.groq_api_key and settings.llm_backend == "groq":
    print("Warning: GROQ_API_KEY not set. Groq backend will not be available.")

if not settings.deepseek_api_key and settings.llm_backend == "deepseek":
    print("Warning: DEEPSEEK_API_KEY not set. DeepSeek backend will not be available.")
