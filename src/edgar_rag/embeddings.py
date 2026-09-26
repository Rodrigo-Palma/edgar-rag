"""Embedding and generation, served by a local Ollama."""

from typing import Protocol

import httpx
import numpy as np

REQUEST_TIMEOUT_SECONDS = 120.0


class ModelError(RuntimeError):
    """Raised when the model server cannot answer."""


class Embedder(Protocol):
    def embed(self, texts: tuple[str, ...]) -> np.ndarray: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


def _post(url: str, payload: dict) -> dict:
    try:
        response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        return response.raise_for_status().json()
    except httpx.HTTPError as error:
        raise ModelError(f"{url} did not answer: {error}") from error


class OllamaEmbedder:
    """Embed passages with a model running on the machine."""

    def __init__(self, base_url: str, model: str) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model

    def embed(self, texts: tuple[str, ...]) -> np.ndarray:
        if not texts:
            raise ValueError("nothing to embed")

        payload = _post(self._url, {"model": self._model, "input": list(texts)})
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
