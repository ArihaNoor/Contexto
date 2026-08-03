from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings, overridable via environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    huggingfacehub_api_token: str = ""

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_temperature: float = 0.5
    llm_max_tokens: int = 512

    # Ingestion / retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    max_file_size_mb: int = 20

    # Storage
    chroma_dir: str = str(BASE_DIR / "chroma_db")

    # Observability
    log_level: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


settings = Settings()
