"""Embedding and generation, served by a local Ollama."""

from collections.abc import Sequence
from typing import Any, Protocol, cast

import httpx
import numpy as np
from numpy.typing import NDArray

REQUEST_TIMEOUT_SECONDS = 120.0


class ModelError(RuntimeError):
    """Raised when the model server cannot answer."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        return cast(dict[str, Any], response.raise_for_status().json())
    except httpx.HTTPError as error:
        raise ModelError(f"{url} did not answer: {error}") from error


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

    def __init__(self, base_url: str, model: str, *, lowercase: bool = True) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._lowercase = lowercase

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("nothing to embed")

        sent = [text.lower() for text in texts] if self._lowercase else list(texts)
        payload = _post(self._url, {"model": self._model, "input": sent})
        vectors = payload.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise ModelError(f"{self._model} returned {len(vectors or [])} of {len(texts)} vectors")
        return np.asarray(vectors, dtype=np.float32)


class OllamaGenerator:
    """Generate an answer with a model running on the machine."""

    def __init__(self, base_url: str, model: str) -> None:
        self._url = f"{base_url.rstrip('/')}/api/generate"
        self._model = model

    def generate(self, prompt: str) -> str:
        payload = _post(
            self._url,
            {"model": self._model, "prompt": prompt, "stream": False, "think": False},
        )
        answer = (payload.get("response") or "").strip()
        if not answer:
            raise ModelError(f"{self._model} returned an empty answer")
        return answer
