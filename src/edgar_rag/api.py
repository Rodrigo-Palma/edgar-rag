"""The service: one endpoint that answers, and one that reports readiness.

``create_app`` is the composition root. Its lifespan reads the index once,
opens one HTTP client for every outbound call and closes it at shutdown;
the endpoints only read what it put in ``app.state``. Run it with
``python -m edgar_rag.api``.
"""

import dataclasses
import hashlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import partial
from typing import Annotated

import anyio
import httpx
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from edgar_rag import __version__
from edgar_rag.answer import AbstentionReason, Answer, Answerer
from edgar_rag.config import ServiceSettings
from edgar_rag.embeddings import (
    Generator,
    ModelError,
    OllamaEmbedder,
    OllamaGenerator,
    OllamaProbe,
)
from edgar_rag.gate import BrierGate, CosineGate, GateError, RelevanceGate
from edgar_rag.index import FilingIndex
from edgar_rag.telemetry import StageTimer, log_request, write_to_stderr

logger = logging.getLogger(__name__)

# What a client turned away is told to wait before asking again: a fraction of
# the twenty seconds a generation takes, so it does not come back to a full
# service, nor wait for a slot that freed long ago.
RETRY_AFTER_SECONDS = 5


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


class ServiceBusy(RuntimeError):
    """Raised when every generation slot is taken."""


class RequestTimedOut(RuntimeError):
    """Raised when a request outlives its time budget."""


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationSlots:
    """A generator that runs at most as many generations as ``slots`` allows.

    A generation holds the GPU for about twenty seconds, and requests waiting
    for one each hold a worker thread, so a queue would let a handful of slow
    questions stall the service. A request finding no free slot fails at once
    with ``ServiceBusy``. Questions the gate rejects never get here, so they
    are answered whatever the load.
    """

    generator: Generator
    slots: threading.BoundedSemaphore

    def generate(self, prompt: str) -> str:
        if not self.slots.acquire(blocking=False):
            raise ServiceBusy("every generation slot is taken")
        try:
            return self.generator.generate(prompt)
        finally:
            self.slots.release()


def with_generation_limit(answerer: Answerer, limit: int) -> Answerer:
    slots = GenerationSlots(answerer.generator, threading.BoundedSemaphore(limit))
    return dataclasses.replace(answerer, generator=slots)


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


def index_fingerprint(index: FilingIndex, embedding_model: str) -> dict[str, object]:
    """What an operator needs to tell which index is being served.

    The digest covers the vectors and the chunk ids, so a re-ingest that
    changed either shows up as a different value.
    """
    digest = hashlib.sha256(index.vectors.tobytes())
    for chunk in index.chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
    return {
        "embedding_model": embedding_model,
        "dimensions": int(index.vectors.shape[1]),
        "digest": digest.hexdigest()[:16],
    }


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
            app.state.answerer = (
                None
                if built is None
                else with_generation_limit(built, config.max_concurrent_generations)
            )
            app.state.fingerprint = (
                None if built is None else index_fingerprint(built.index, config.embedding_model)
            )
            app.state.ollama = OllamaProbe(str(config.ollama_base_url), client)
            yield

    app = FastAPI(title="edgar-rag", version=__version__, lifespan=lifespan)
    app.middleware("http")(log_each_request)
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


def report_busy(request: Request, error: Exception) -> JSONResponse:
    logger.warning("turned away %s: %s", request.url.path, error)
    return JSONResponse(
        status_code=503,
        content={"detail": "busy answering other questions; retry shortly"},
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


def report_timeout(request: Request, error: Exception) -> JSONResponse:
    logger.error("gave up on %s: %s", request.url.path, error)
    return JSONResponse(status_code=504, content={"detail": "the answer took too long"})


def _report_failures(app: FastAPI) -> None:
    app.add_exception_handler(ValueError, reject_invalid_input)
    app.add_exception_handler(ModelError, report_model_failure)
    app.add_exception_handler(GateError, report_gate_failure)
    app.add_exception_handler(IndexUnavailable, report_missing_index)
    app.add_exception_handler(ServiceBusy, report_busy)
    app.add_exception_handler(RequestTimedOut, report_timeout)


async def log_each_request(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """One line of JSON per request, whatever its outcome, the 500 included.

    The stages and the answer are what the endpoint left in ``request.state``;
    a request rejected before reaching it is logged with no stages. The
    question is not logged: it is whatever a client typed.
    """
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        log_request(_request_fields(request, status, time.perf_counter() - started))


def _request_fields(request: Request, status: int, seconds: float) -> dict[str, object]:
    stages: StageTimer | None = getattr(request.state, "stages", None)
    answer: Answer | None = getattr(request.state, "answer", None)
    return {
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "seconds": round(seconds, 4),
        "stages": stages.seconds() if stages is not None else {},
        "top_k": getattr(request.state, "top_k", None),
        "abstained": answer.abstained if answer else None,
        "reason": answer.reason if answer else None,
        "degraded": answer.degraded if answer else None,
        "retrieval_score": answer.retrieval_score if answer else None,
        "gate_score": answer.gate_score if answer else None,
        # The generator does not report its token counts yet.
        "prompt_tokens": None,
        "completion_tokens": None,
    }


def loaded_answerer(request: Request) -> Answerer:
    """The answerer the lifespan built, or 503 when there was no index."""
    answerer: Answerer | None = request.app.state.answerer
    if answerer is None:
        raise IndexUnavailable(f"no index in {request.app.state.settings.index_dir}")
    return answerer


router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """The index loaded at startup, its fingerprint, and whether Ollama answers."""
    state = request.app.state
    answerer: Answerer | None = state.answerer
    ollama: OllamaProbe = state.ollama
    return {
        "status": "ready" if answerer is not None else "no index",
        "indexed_filing": answerer.source if answerer is not None else None,
        "chunks": len(answerer.index.chunks) if answerer is not None else 0,
        "fingerprint": state.fingerprint,
        "ollama_reachable": ollama.reachable(),
    }


@router.post("/ask", response_model=AskResponse)
async def ask(
    question: AskRequest,
    request: Request,
    answerer: Annotated[Answerer, Depends(loaded_answerer)],
) -> AskResponse:
    """Answer in a worker thread, giving up on it after the request's time budget.

    The pipeline is synchronous, and a thread cannot be stopped from outside,
    so a request past its budget is answered with 504 while its worker runs
    on to the models' own timeouts. The worker keeps its generation slot
    until then, which is what keeps abandoned work from piling up on the GPU.
    """
    budget = request.app.state.settings.request_timeout_seconds
    stages = StageTimer()
    request.state.stages, request.state.top_k = stages, question.top_k
    work = partial(answerer.ask, question.question, top_k=question.top_k, stages=stages)
    try:
        with anyio.fail_after(budget):
            answer = await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
    except TimeoutError as error:
        raise RequestTimedOut(f"no answer within {budget} s") from error
    request.state.answer = answer
    return AskResponse.of(answer, answerer.source)


def serve(
    settings: ServiceSettings | None = None, *, run: Callable[..., object] = uvicorn.run
) -> None:
    """Run the service on the configured address, 127.0.0.1:8000 by default."""
    config = settings if settings is not None else ServiceSettings()
    write_to_stderr()
    run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    serve()
