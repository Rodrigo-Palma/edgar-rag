"""Runtime configuration, read from the environment by each entry point.

The service, the ingestion and the evaluation each read their own settings,
so running one never demands what only another needs: the service never calls
EDGAR and does not ask for a User-Agent. Every variable carries the
``EDGAR_RAG_`` prefix, so ``EDGAR_RAG_INDEX_DIR`` sets ``index_dir``.
"""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from edgar_rag.edgar.user_agent import validate_user_agent

LOCAL_OLLAMA = HttpUrl("http://localhost:11434")

GateChoice = Literal["none", "cosine", "period+cosine", "brier", "period+brier"]
"""Which checks run before the model is asked: none, a relevance gate, or the
period guard in front of one."""

ServiceMode = Literal["live", "replay"]
"""Where the models' replies come from: Ollama, or a tape recorded from it."""

CI_TAPE = Path("eval/ci/tape")
GOLDEN_SET = Path("eval/golden/v1.jsonl")


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

    ``gate`` picks the checks that run before the model is asked. The default,
    ``period+cosine``, declines a question about a year the filing does not
    report and then one whose closest passage is not close enough; it needs
    nothing but the index, so anyone can run it. ``brier`` and
    ``period+brier`` judge relevance with a calibrated model at ``brier_url``
    instead, with cosine as the fallback when that service is unreachable.
    ``none`` asks the model every time and leaves the refusal to it.

    A ``brier_url`` is required by the brier gates and refused by the others,
    so a URL that would be silently ignored stops the service instead.

    ``mode=replay`` answers without a model: the question embeddings and the
    generations come from the tape at ``replay_tape``, recorded by the
    evaluation over the golden set at ``replay_questions``, and only those
    questions can be asked. The index, the search, the gate and the citation
    check run as they do live. Brier is not on the CI tape, so replay refuses
    the brier gates.
    """

    generation_model: str = "qwen3:32b"
    gate: GateChoice = "period+cosine"
    mode: ServiceMode = "live"
    replay_tape: Path = CI_TAPE
    replay_questions: Path = GOLDEN_SET
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

    @model_validator(mode="after")
    def _brier_url_matches_the_gate(self) -> Self:
        uses_brier = self.gate.endswith("brier")
        if uses_brier and self.brier_url is None:
            raise ValueError(f"EDGAR_RAG_GATE={self.gate} needs EDGAR_RAG_BRIER_URL")
        if not uses_brier and self.brier_url is not None:
            raise ValueError(
                f"EDGAR_RAG_BRIER_URL is set but EDGAR_RAG_GATE={self.gate} does not use it; "
                "choose brier or period+brier, or unset the URL"
            )
        if uses_brier and self.mode == "replay":
            raise ValueError(f"EDGAR_RAG_MODE=replay has no brier on its tape: {self.gate}")
        return self


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


class SnapshotIngestSettings(_SharedSettings):
    """Settings for indexing the golden set's pinned filings from their snapshots.

    Nothing is downloaded, so no SEC contact is asked for.
    """


class EvalSettings(_SharedSettings):
    """Settings for running the evaluation over an existing index.

    No threshold is configured here: the evaluation scores every gate and
    fits each threshold from the scores (see ``eval.arms``). ``brier_url`` is
    optional, because brier is a private plugin; without it the arms that need
    it are reported as not run. ``brier_sha`` is the commit the plugin was
    built from, stated by whoever runs it, since the service does not report
    one.
    """

    generation_model: str = "qwen3:32b"
    brier_url: HttpUrl | None = None
    brier_sha: str | None = None
