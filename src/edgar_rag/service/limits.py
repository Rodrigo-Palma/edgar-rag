"""What the service refuses before an endpoint runs: a body too big to be a question.

A question is at most 500 characters, so a whole ``/ask`` body fits in about
1 KiB. FastAPI reads and parses the entire body before the schema's
``max_length`` is checked, so without this a single request could make the
process hold gigabytes, and the allocator keeps them after the request ends.
A page on another site can send such a body too: without a content type it is
a "simple request" and needs no preflight.
"""

import json
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 8 * 1024
"""Eight times the largest valid ``/ask`` body: room to spare, nothing to hoard."""

_TOO_LARGE = json.dumps({"detail": "request body too large"}).encode()


class BodySizeLimit:
    """Answer 413 to a body over ``limit`` bytes, before the app reads any of it.

    A declared ``Content-Length`` over the limit is refused without reading.
    Otherwise the body is read here, counting every chunk, so a chunked body or
    one longer than it declared is refused as soon as it crosses the limit;
    one that fits is handed to the app as a single message.
    """

    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _declared_length(scope)
        if declared is not None and declared > self.limit:
            await _refuse(send, f"declared {declared} bytes")
            return
        body = await self._read_within_limit(receive)
        if body is None:
            await _refuse(send, f"streamed more than {self.limit} bytes")
            return
        await self.app(scope, _replaying(body, receive), send)

    async def _read_within_limit(self, receive: Receive) -> Message | None:
        """The whole body as one message; ``None`` once it crosses the limit.

        A message that is not body (a disconnect) is returned as it came.
        """
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return message
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}


def _declared_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def _replaying(first: Message, receive: Receive) -> Receive:
    """``first``, then whatever the server sends next (a disconnect, while streaming)."""
    pending = [first]

    async def replay() -> Message:
        return pending.pop() if pending else await receive()

    return replay


async def _refuse(send: Send, why: str) -> None:
    logger.warning("refused a request body: %s", why)
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_TOO_LARGE)).encode()),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _TOO_LARGE})


def report_invalid_request(request: Request, error: Exception) -> JSONResponse:
    """422 with where and why each field failed, never the value that failed.

    FastAPI's own handler echoes each failing ``input``, which for a body sent
    as text is the whole body, so the response would be as big as the request.
    """
    if not isinstance(error, RequestValidationError):
        raise error
    detail = [
        {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


def limit_requests(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, report_invalid_request)
    app.add_middleware(BodySizeLimit)
