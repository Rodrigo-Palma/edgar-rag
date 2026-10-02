"""Embedding and generation, served by a local Ollama."""

import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

import httpx
import numpy as np
from numpy.typing import NDArray

from edgar_rag.domain import Generation

REQUEST_TIMEOUT_SECONDS = 120.0
# A health check has to answer fast whatever Ollama is doing.
PROBE_TIMEOUT_SECONDS = 2.0
PROBE_TTL_SECONDS = 10.0

# Fixed so the same prompt should get the same answer: replay keys recorded
# generations on the prompt, and the protocol measures determinism on
# repeated cases. num_ctx is set rather than left to the server, whose
# default would silently truncate four passages and the instructions.
GENERATION_OPTIONS: Mapping[str, int] = MappingProxyType(
    {"temperature": 0, "seed": 0, "num_ctx": 8192}
)


class ModelError(RuntimeError):
    """Raised when the model server cannot answer."""


def _post(url: str, payload: dict[str, Any], client: httpx.Client | None) -> dict[str, Any]:
    """Post and return the JSON object the server replied with.

    The service passes the one ``client`` it opened at startup, so calls share
    its connection pool. Without one, a connection is opened for the call.

    Raises:
        ModelError: when the server is unreachable, fails, or replies with
            anything other than a JSON object.
    """
    try:
        post = client.post if client is not None else httpx.post
        response = post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        reply = response.raise_for_status().json()
    except (httpx.HTTPError, ValueError) as error:
        raise ModelError(f"{url} did not answer: {error}") from error
    if not isinstance(reply, dict):
        raise ModelError(f"{url} replied with a JSON {type(reply).__name__}, not an object")
    return cast(dict[str, Any], reply)


class OllamaEmbedder:
    """Embed passages with a model running on the machine.

    Text is lower-cased before it is sent. On Ollama 0.18.0 with
    nomic-embed-text, every capitalised token collapses onto a single vector:
    "Apple", "Cat" and "Zebra" come back identical to eight decimal places, and
    two sentences differing only in a company name come back byte for byte the
    same. A filing is full of proper nouns, so leaving that in place throws away
    exactly the words that identify what a passage is about. Lower-casing
    restores the distinction; measured on one such pair, cosine went from 1.000
    to 0.813. Pass ``lowercase=False`` to send text as written, once the upstream
    tokenizer stops doing this.

    Both the index and the queries have to agree, so this is set at
    construction and applies to every call.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        lowercase: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._lowercase = lowercase
        self._client = client

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("nothing to embed")

        sent = [text.lower() for text in texts] if self._lowercase else list(texts)
        payload = _post(self._url, {"model": self._model, "input": sent}, self._client)
        vectors = payload.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise ModelError(f"{self._model} returned {len(vectors or [])} of {len(texts)} vectors")
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ModelError(f"{self._model} returned vectors that are not numbers") from error
        if matrix.ndim != 2:
            raise ModelError(f"{self._model} returned vectors of unequal or no length")
        return matrix


class OllamaGenerator:
    """Generate an answer with a model running on the machine.

    Sampling is pinned by ``GENERATION_OPTIONS`` and thinking is off, so the
    reply depends on the prompt alone. ``clock`` measures how long the caller
    waited; a test passes its own.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/generate"
        self._model = model
        self._client = client
        self._clock = clock

    def generate(self, prompt: str) -> Generation:
        started = self._clock()
        payload = _post(
            self._url,
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": dict(GENERATION_OPTIONS),
            },
            self._client,
        )
        seconds = self._clock() - started
        response = payload.get("response") or ""
        if not isinstance(response, str):
            raise ModelError(f"{self._model} returned a {type(response).__name__}, not text")
        answer = response.strip()
        if not answer:
            raise ModelError(f"{self._model} returned an empty answer")
        return Generation(
            text=answer,
            prompt_tokens=self._count(payload, "prompt_eval_count"),
            completion_tokens=self._count(payload, "eval_count"),
            seconds=seconds,
        )

    def _count(self, payload: dict[str, Any], field: str) -> int | None:
        """A token count from the reply; absent is unknown, not zero."""
        count = payload.get(field)
        if count is None:
            return None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ModelError(f"{self._model} returned {field}={count!r}, not a token count")
        return count


class OllamaProbe:
    """Whether Ollama answers, asked at most once per ``ttl`` seconds.

    ``/health`` is polled, and a check that called Ollama on every poll would
    put load on the machine it is meant to observe.
    """

    def __init__(
        self,
        base_url: str,
        client: httpx.Client,
        *,
        ttl: float = PROBE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/version"
        self._client = client
        self._ttl = ttl
        self._clock = clock
        self._last: tuple[float, bool] | None = None

    def reachable(self) -> bool:
        now = self._clock()
        if self._last is not None and now - self._last[0] < self._ttl:
            return self._last[1]
        answered = self._ask()
        self._last = (now, answered)
        return answered

    def _ask(self) -> bool:
        try:
            response = self._client.get(self._url, timeout=PROBE_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            return False
        return response.is_success
