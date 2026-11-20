"""Typed application settings.

Values come from environment variables (prefix RAG_) or a .env file at the repo root.
The same code runs locally, on Lambda and on ECS; only the environment changes.
Provider API keys (OPENAI_API_KEY, OPENROUTER_API_KEY) are read directly by the SDKs
and are deliberately NOT stored here.
"""
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Put .env values (incl. OPENAI_API_KEY / OPENROUTER_API_KEY) into the process environment,
# where the model SDKs look for them. Real environment variables win over .env.
load_dotenv(REPO_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Models
    chat_model: str = "openrouter/openai/gpt-oss-120b"
    embedding_model: str = "text-embedding-3-large"

    # Vector store (local Chroma for now; S3 Vectors in task 1.2.6)
    db_path: Path = REPO_ROOT / "preprocessed_db"
    collection_name: str = "docs"

    # Retrieval
    retrieval_k: int = Field(20, ge=1, le=100)
    final_k: int = Field(10, ge=1, le=50)

    # Reliability
    llm_timeout_s: int = Field(30, ge=1, le=120)
    retry_attempts: int = Field(3, ge=1, le=5)


@lru_cache
def get_settings() -> Settings:
    """Load and validate settings once per process."""
    return Settings()
