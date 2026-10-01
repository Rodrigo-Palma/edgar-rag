"""The service: one endpoint that answers, and one that reports readiness."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from edgar_rag import __version__
from edgar_rag.answer import AbstentionReason, Answer, answer_question
from edgar_rag.config import Settings, get_settings
from edgar_rag.embeddings import (
    Embedder,
    Generator,
    ModelError,
    OllamaEmbedder,
    OllamaGenerator,
)
from edgar_rag.gate import BrierGate, CosineGate, GateError, RelevanceGate
from edgar_rag.index import FilingIndex

logger = logging.getLogger(__name__)

app = FastAPI(title="edgar-rag", version=__version__)


class AskRequest(BaseModel):
    # Trimmed before the length check, so a question of only whitespace is
    # rejected here instead of reaching the core.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    top_k: int = Field(default=4, ge=1, le=10)


class CitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker: int = Field(description="The [n] in the answer text that points here.")
    item: str
    title: str
    quote: str = Field(description="The window of the passage around what the question asks.")
    score: float = Field(description="Cosine similarity of the passage to the question.")


class AskResponse(BaseModel):
    """What ``/ask`` returns, declared so the OpenAPI schema is the contract.

    ``extra="forbid"`` makes a field added to ``Answer`` and not here fail the
    request in tests, instead of reaching clients undocumented.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    text: str | None = Field(description="The answer, or null when the service abstained.")
    citations: list[CitationResponse]
    abstained: bool
    reason: AbstentionReason | None = Field(
        description="Which check withheld the answer; null when there is an answer."
    )
    detail: str = Field(description="The decision in words, for a person reading it.")
    retrieval_score: float = Field(description="Cosine similarity of the closest passage.")
    gate_score: float = Field(description="The relevance gate's confidence in its decision.")
    degraded: bool = Field(
        description="The gate decided on part of its evidence or on its fallback."
    )
    source: dict[str, str] = Field(description="The filing the passages come from.")

    @classmethod
    def of(cls, answer: Answer, source: dict[str, str]) -> "AskResponse":
        return cls.model_validate({**asdict(answer), "source": source})


class IndexUnavailable(RuntimeError):
    """Raised when there is no index to answer from."""


# Failures are reported to the client in these words only. What actually
# failed (a URL, a local path, an exception) is logged, because the client is
# not the operator and has no use for the topology behind the service.
@app.exception_handler(ValueError)
def reject_invalid_input(request: Request, error: ValueError) -> JSONResponse:
    """The core signals a question it cannot use with ``ValueError``: a client error."""
    logger.warning("rejected %s %s: %s", request.method, request.url.path, error)
    return JSONResponse(status_code=422, content={"detail": "the question could not be used"})


@app.exception_handler(ModelError)
def report_model_failure(request: Request, error: ModelError) -> JSONResponse:
    logger.error("model backend failed on %s", request.url.path, exc_info=error)
    return JSONResponse(status_code=502, content={"detail": "model backend unavailable"})


@app.exception_handler(GateError)
def report_gate_failure(request: Request, error: GateError) -> JSONResponse:
    logger.error("relevance gate failed on %s", request.url.path, exc_info=error)
    return JSONResponse(status_code=502, content={"detail": "relevance model unavailable"})


@app.exception_handler(IndexUnavailable)
def report_missing_index(request: Request, error: IndexUnavailable) -> JSONResponse:
    logger.error("no index to answer from on %s", request.url.path, exc_info=error)
    return JSONResponse(
        status_code=503, content={"detail": "no index loaded; run the ingest first"}
    )


def provide_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(provide_settings)]


def provide_index(settings: SettingsDep) -> FilingIndex:
    """Load the index, or tell the caller the service has nothing to answer from."""
    try:
        return FilingIndex.load(Path(settings.index_dir))
    except FileNotFoundError as error:
        raise IndexUnavailable(str(error)) from error


def provide_embedder(settings: SettingsDep) -> Embedder:
    return OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)


def provide_generator(settings: SettingsDep) -> Generator:
    return OllamaGenerator(settings.ollama_base_url, settings.generation_model)


@app.get("/health")
def health(settings: SettingsDep) -> dict[str, object]:
    try:
        index = FilingIndex.load(Path(settings.index_dir))
    except FileNotFoundError:
        return {"status": "no index", "indexed_filing": None, "chunks": 0}
    return {"status": "ready", "indexed_filing": index.source, "chunks": len(index.chunks)}


def provide_gate(settings: SettingsDep) -> RelevanceGate:
    """Cosine on its own, or brier with cosine behind it."""
    cosine = CosineGate(min_score=settings.min_retrieval_score)
    if not settings.brier_url:
        return cosine
    return BrierGate(
        url=settings.brier_url,
        min_confidence=settings.brier_min_confidence,
        fallback=cosine,
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    settings: SettingsDep,
    index: Annotated[FilingIndex, Depends(provide_index)],
    embedder: Annotated[Embedder, Depends(provide_embedder)],
    generator: Annotated[Generator, Depends(provide_generator)],
    gate: Annotated[RelevanceGate, Depends(provide_gate)],
) -> AskResponse:
    answer = answer_question(
        question=request.question,
        index=index,
        embedder=embedder,
        generator=generator,
        gate=gate,
        top_k=request.top_k,
    )
    return AskResponse.of(answer, index.source)
