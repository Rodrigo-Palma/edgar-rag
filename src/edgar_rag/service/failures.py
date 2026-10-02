"""What the service tells a client when it cannot answer, and what it logs.

Every failure the core or an adapter raises is mapped here to a status code
and a fixed sentence; the cause goes to the log only.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from edgar_rag.gate import GateError
from edgar_rag.models import ModelError

logger = logging.getLogger(__name__)

# What a client turned away is told to wait before asking again: a fraction of
# the twenty seconds a generation takes, so it does not come back to a full
# service, nor wait for a slot that freed long ago.
RETRY_AFTER_SECONDS = 5


class IndexUnavailable(RuntimeError):
    """Raised when there is no index to answer from."""


class ServiceBusy(RuntimeError):
    """Raised when every generation slot is taken."""


class RequestTimedOut(RuntimeError):
    """Raised when a request outlives its time budget."""


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


def report_failures(app: FastAPI) -> None:
    app.add_exception_handler(ValueError, reject_invalid_input)
    app.add_exception_handler(ModelError, report_model_failure)
    app.add_exception_handler(GateError, report_gate_failure)
    app.add_exception_handler(IndexUnavailable, report_missing_index)
    app.add_exception_handler(ServiceBusy, report_busy)
    app.add_exception_handler(RequestTimedOut, report_timeout)
