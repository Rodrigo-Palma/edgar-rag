"""The service: one endpoint that answers, and one that reports readiness.

``create_app`` is the composition root. Its lifespan reads the index once,
opens one HTTP client for every outbound call and closes it at shutdown;
the endpoints only read what it put in ``app.state``. Run it with
``python -m edgar_rag.service.app``.
"""

import dataclasses
import hashlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

import anyio
import httpx
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import Response

from edgar_rag import __version__
from edgar_rag.answer import Answerer
from edgar_rag.config import ServiceSettings
from edgar_rag.domain import Answer, Generator, RelevanceGate
from edgar_rag.gate import BrierGate, CosineGate
from edgar_rag.index import FilingIndex
from edgar_rag.models import OllamaEmbedder, OllamaGenerator, OllamaProbe
from edgar_rag.service.failures import (
    IndexUnavailable,
    RequestTimedOut,
    ServiceBusy,
    report_failures,
)
from edgar_rag.service.schemas import AskRequest, AskResponse
from edgar_rag.telemetry import StageTimer, log_request, write_to_stderr

logger = logging.getLogger(__name__)

# The service answers from an index it loaded itself, and reports its size and
# fingerprint, so it needs the concrete index behind the answerer.
ServedAnswerer = Answerer[FilingIndex]


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


def with_generation_limit(answerer: ServedAnswerer, limit: int) -> ServedAnswerer:
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


def build_answerer(settings: ServiceSettings, client: httpx.Client) -> ServedAnswerer | None:
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
    answerer: ServedAnswerer | None = None,
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
    report_failures(app)
    app.include_router(router)
    return app


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


def loaded_answerer(request: Request) -> ServedAnswerer:
    """The answerer the lifespan built, or 503 when there was no index."""
    answerer: ServedAnswerer | None = request.app.state.answerer
    if answerer is None:
        raise IndexUnavailable(f"no index in {request.app.state.settings.index_dir}")
    return answerer


router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    """The index loaded at startup, its fingerprint, and whether Ollama answers."""
    state = request.app.state
    answerer: ServedAnswerer | None = state.answerer
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
    answerer: Annotated[ServedAnswerer, Depends(loaded_answerer)],
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
