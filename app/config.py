from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Every supported provider speaks the OpenAI chat-completions protocol, so a
# preset is just a base URL plus a sensible default model.
PROVIDER_PRESETS: dict[str, tuple[str, str]] = {
    "huggingface": (
        "https://router.huggingface.co/v1",
        "meta-llama/Llama-3.1-8B-Instruct",
    ),
    "ollama": ("http://localhost:11434/v1", "llama3.2"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.0-flash",
    ),
}


class Settings(BaseSettings):
    """Application settings, overridable via environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM provider. Pick a preset, or set llm_base_url for anything else that
    # exposes an OpenAI-compatible /chat/completions endpoint (vLLM, LM Studio,
    # OpenRouter, Together, ...).
    llm_provider: str = "huggingface"
    llm_base_url: str = ""  # overrides the preset base URL
    llm_api_key: str = ""  # overrides the provider token
    llm_model: str = ""  # empty -> the preset's default model
    llm_temperature: float = 0.5
    llm_max_tokens: int = 512
    llm_timeout_seconds: int = 60

    # Convenience alias kept so existing Hugging Face setups keep working.
    huggingfacehub_api_token: str = ""

    # Embeddings run locally on CPU via fastembed — no API key, no torch.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

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

    def _preset(self) -> tuple[str, str]:
        try:
            return PROVIDER_PRESETS[self.llm_provider.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. "
                f"Choose one of {sorted(PROVIDER_PRESETS)} or set LLM_BASE_URL."
            ) from None

    @property
    def resolved_base_url(self) -> str:
        return (self.llm_base_url or self._preset()[0]).rstrip("/")

    @property
    def resolved_model(self) -> str:
        return self.llm_model or self._preset()[1]

    @property
    def resolved_api_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_provider.lower() == "huggingface":
            return self.huggingfacehub_api_token
        return ""

    @property
    def requires_api_key(self) -> bool:
        """Local runtimes (Ollama, LM Studio, vLLM) need no credentials."""
        return not self.resolved_base_url.startswith(
            ("http://localhost", "http://127.0.0.1", "http://host.docker.internal")
        )


settings = Settings()
