"""Runtime configuration, read once from the environment."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the ingestion pipeline and the answering service.

    ``edgar_user_agent`` has no default on purpose. The SEC rejects automated
    requests that do not identify their sender, and a shared placeholder would
    get this project rate limited for everyone using it.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    edgar_user_agent: str = Field(min_length=5)
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "qwen3:32b"
    min_retrieval_score: float = Field(default=0.55, ge=0.0, le=1.0)
    index_dir: str = "data/index"

    # Leave brier_url unset to judge relevance by cosine similarity alone. Set it
    # to a running brier service to judge it with a calibrated model instead; the
    # cosine gate stays on as the fallback if that service is unreachable.
    brier_url: str = ""
    brier_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
