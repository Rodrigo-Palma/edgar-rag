"""Runtime configuration, read from the environment by each entry point.

The service, the ingestion and the evaluation each read their own settings,
so running one never demands what only another needs: the service never calls
EDGAR and does not ask for a User-Agent. Every variable carries the
``EDGAR_RAG_`` prefix, so ``EDGAR_RAG_INDEX_DIR`` sets ``index_dir``.
"""

from pathlib import Path

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from edgar_rag.edgar.user_agent import validate_user_agent

LOCAL_OLLAMA = HttpUrl("http://localhost:11434")


class _SharedSettings(BaseSettings):
    """Where the index lives and which model embeds: all three must agree on it."""

    model_config = SettingsConfigDict(
        env_prefix="EDGAR_RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_base_url: HttpUrl = LOCAL_OLLAMA
    embedding_model: str = "nomic-embed-text"
    index_dir: Path = Path("data/index")


class ServiceSettings(_SharedSettings):
    """Settings for the answering service.

    The service is meant to run on the machine that asks it, so it binds to
    127.0.0.1 unless told otherwise: it has no authentication and no rate
    limit, which is acceptable only there.

    Leave ``brier_url`` unset to judge relevance by cosine similarity alone.
    Set it to a running brier service to judge it with a calibrated model
    instead; the cosine gate stays on as the fallback if that service is
    unreachable.
    """

    generation_model: str = "qwen3:32b"
    min_retrieval_score: float = Field(default=0.55, ge=0.0, le=1.0)
    brier_url: HttpUrl | None = None
    brier_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # A generation holds the GPU for about twenty seconds, so a third one at
    # the same time only makes all of them slower. Past this many, /ask
    # answers 503 at once instead of queueing.
    max_concurrent_generations: int = Field(default=2, ge=1)
    request_timeout_seconds: float = Field(default=90.0, gt=0)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class IngestSettings(_SharedSettings):
    """Settings for downloading and indexing a filing.

    ``edgar_user_agent`` has no default on purpose. The SEC rejects automated
    requests that do not identify their sender, and a shared placeholder would
    get this project rate limited for everyone using it, so the placeholder in
    ``.env.example`` is refused here rather than by the SEC.
    """

    edgar_user_agent: str

    @field_validator("edgar_user_agent")
    @classmethod
    def _declares_a_sender(cls, user_agent: str) -> str:
        return validate_user_agent(user_agent)


class EvalSettings(_SharedSettings):
    """Settings for comparing gates over an existing index."""

    min_retrieval_score: float = Field(default=0.55, ge=0.0, le=1.0)
