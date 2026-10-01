"""The service: one endpoint that answers, and one that reports readiness.

``create_app`` is the composition root. Its lifespan reads the index once,
opens one HTTP client for every outbound call and closes it at shutdown;
the endpoints only read what it put in ``app.state``. Run it with
``python -m edgar_rag.api``.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

import httpx
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from edgar_rag import __version__
from edgar_rag.answer import AbstentionReason, Answer, Answerer
from edgar_rag.config import ServiceSettings
from edgar_rag.embeddings import ModelError, OllamaEmbedder, OllamaGenerator
from edgar_rag.gate import BrierGate, CosineGate, GateError, RelevanceGate
from edgar_rag.index import FilingIndex

logger = logging.getLogger(__name__)


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


def build_gate(settings: ServiceSettings, client: httpx.Client) -> RelevanceGate:
    """Cosine on its own, or brier with cosine behind it."""
    cosine = CosineGate(min_score=settings.min_retrieval_score)
    if settings.brier_url is None:
        return cosine
    return BrierGate(
        url=str(settings.brier_url),
        min_confidence=settings.brier_min_confidence,
        fallback=cosine,
        client=client,
    )


def build_answerer(settings: ServiceSettings, client: httpx.Client) -> Answerer | None:
    """Read the index and wire the models to ``client``; ``None`` without an index.

    A missing index does not stop the service from starting: ``/health`` says
    so and ``/ask`` answers 503 until the ingest has run and it is restarted.
    """
    try:
        index = FilingIndex.load(settings.index_dir)
    except FileNotFoundError as error:
        logger.warning("starting without an index: %s", error)
        return None

    ollama = str(settings.ollama_base_url)
    return Answerer(
        index=index,
        embedder=OllamaEmbedder(ollama, settings.embedding_model, client=client),
        generator=OllamaGenerator(ollama, settings.generation_model, client=client),
        gate=build_gate(settings, client),
    )


def create_app(
    settings: ServiceSettings | None = None,
    answerer: Answerer | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    """Build the service.

    ``answerer`` replaces the one built from ``settings``, so a test can
    serve fakes through the real endpoints. ``transport`` is handed to the
    HTTP client the lifespan opens, so a test can stand in for Ollama and
    brier without a network.
    """
    config = settings if settings is not None else ServiceSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        with httpx.Client(transport=transport) as client:
            app.state.settings = config
            app.state.http = client
            built = answerer if answerer is not None else build_answerer(config, client)
            app.state.answerer = built
            yield

    app = FastAPI(title="edgar-rag", version=__version__, lifespan=lifespan)
    _report_failures(app)
    app.include_router(router)
    return app


# Failures are reported to the client in these words only. What actually
# failed (a URL, a local path, an exception) is logged, because the client is
# not the operator and has no use for the topology behind the service.
def reject_invalid_input(request: Request, error: Exception) -> JSONResponse:
    """The core signals a question it cannot use with ``ValueError``: a client error."""
    logger.warning("rejected %s %s: %s", request.method, request.url.path, error)
    return JSONResponse(status_code=422, content={"detail": "the question could not be used"})


def report_model_failure(request: Request, error: Exception) -> JSONResponse:
    logger.error("model backend failed on %s", request.url.path, exc_info=error)
    return JSONResponse(status_code=502, content={"detail": "model backend unavailable"})


def report_gate_failure(request: Request, error: Exception) -> JSONResponse:
    logger.error("relevance gate failed on %s", request.url.path, exc_info=error)
    return JSONResponse(status_code=502, content={"detail": "relevance model unavailable"})


def report_missing_index(request: Request, error: Exception) -> JSONResponse:
    logger.error("no index to answer from on %s", request.url.path, exc_info=error)
    return JSONResponse(
        status_code=503, content={"detail": "no index loaded; run the ingest first"}
    )


def _report_failures(app: FastAPI) -> None:
    app.add_exception_handler(ValueError, reject_invalid_input)
    app.add_exception_handler(ModelError, report_model_failure)
    app.add_exception_handler(GateError, report_gate_failure)
    app.add_exception_handler(IndexUnavailable, report_missing_index)


def loaded_answerer(request: Request) -> Answerer:
    """The answerer the lifespan built, or 503 when there was no index."""
    answerer: Answerer | None = request.app.state.answerer
    if answerer is None:
        raise IndexUnavailable(f"no index in {request.app.state.settings.index_dir}")
    return answerer


router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    answerer: Answerer | None = request.app.state.answerer
    if answerer is None:
        return {"status": "no index", "indexed_filing": None, "chunks": 0}
    return {
        "status": "ready",
        "indexed_filing": answerer.source,
        "chunks": len(answerer.index.chunks),
    }


@router.post("/ask", response_model=AskResponse)
def ask(
    question: AskRequest, answerer: Annotated[Answerer, Depends(loaded_answerer)]
) -> AskResponse:
    answer = answerer.ask(question.question, top_k=question.top_k)
    return AskResponse.of(answer, answerer.source)


def serve(
    settings: ServiceSettings | None = None, *, run: Callable[..., object] = uvicorn.run
) -> None:
    """Run the service on the configured address, 127.0.0.1:8000 by default."""
    config = settings if settings is not None else ServiceSettings()
    run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    serve()
