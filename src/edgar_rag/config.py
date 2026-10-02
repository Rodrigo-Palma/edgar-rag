"""Runtime configuration, read from the environment by each entry point.

The service, the ingestion and the evaluation each read their own settings,
so running one never demands what only another needs: the service never calls
EDGAR and does not ask for a User-Agent. Every variable carries the
``EDGAR_RAG_`` prefix, so ``EDGAR_RAG_INDEX_DIR`` sets ``index_dir``.
"""

import ipaddress
import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from edgar_rag.edgar.user_agent import validate_user_agent

logger = logging.getLogger(__name__)

LOCAL_OLLAMA = HttpUrl("http://localhost:11434")

GateChoice = Literal["none", "cosine", "period+cosine"]
"""Which checks run before the model is asked: none, a relevance gate, or the
period guard in front of one."""

ServiceMode = Literal["live", "replay"]
"""Where the models' replies come from: Ollama, or a tape recorded from it."""

COSINE_R90 = 0.7329
"""The cosine score that admits 90% of the answerable golden questions of all
24 companies, fitted after the headline run; ``make cosine-threshold`` prints
it (ADR-0014)."""

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
    ``none``, asks the model every time and leaves the refusal to it: on the
    headline run the model alone answered 13 of 300 unanswerable questions,
    too few for a gate to pay for itself in false answers (ADR-0014).
    ``period+cosine`` declines a question about a year the filing does not
    report and then one whose closest passage scores under
    ``min_retrieval_score``. It needs nothing but the index and saves
    generator time (6.59 s per question against 13.21 s) at a recall cost of
    0.6 p.p.; ``cosine`` is the relevance check alone.

    ``mode=replay`` answers without a model: the question embeddings and the
    generations come from the tape at ``replay_tape``, recorded by the
    evaluation over the golden set at ``replay_questions``, and only those
    questions can be asked. The index, the search, the gate and the citation
    check run as they do live.
    """

    generation_model: str = "qwen3:32b"
    gate: GateChoice = "none"
    mode: ServiceMode = "live"
    replay_tape: Path = CI_TAPE
    replay_questions: Path = GOLDEN_SET
    min_retrieval_score: float = Field(default=COSINE_R90, ge=0.0, le=1.0)

    # A generation holds the GPU for about twenty seconds, so a third one at
    # the same time only makes all of them slower. Past this many, /ask
    # answers 503 at once instead of queueing.
    max_concurrent_generations: int = Field(default=2, ge=1)
    request_timeout_seconds: float = Field(default=90.0, gt=0)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def _warn_beyond_loopback(cls, host: str) -> str:
        """Allowed, because a container has to listen on every interface, but never silent."""
        if not _is_loopback(host):
            logger.warning(
                "EDGAR_RAG_HOST=%s is not a loopback address: the service has no "
                "authentication and no rate limit, so anyone who can reach it can use it",
                host,
            )
        return host


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
    fits each threshold from the scores (see ``eval.arms``).
    """

    generation_model: str = "qwen3:32b"
