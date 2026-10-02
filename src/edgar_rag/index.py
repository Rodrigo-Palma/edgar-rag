"""A vector index small enough to keep in two files on disk."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from edgar_rag.domain import DEFAULT_TOP_K, Chunk, ScoredChunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"


@dataclass(frozen=True, slots=True)
class FilingIndex:
    """Chunks and their unit-length vectors, in the same order."""

    source: dict[str, str]
    chunks: tuple[Chunk, ...]
    vectors: NDArray[np.float32]

    def search(
        self, query: NDArray[np.float32], top_k: int = DEFAULT_TOP_K
    ) -> tuple[ScoredChunk, ...]:
        """Return the ``top_k`` closest chunks, best first."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores = self.vectors @ _unit(query.reshape(-1))
        best = np.argsort(scores)[::-1][:top_k]
        return tuple(ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in best)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self.vectors)
        payload = {"source": self.source, "chunks": [asdict(chunk) for chunk in self.chunks]}
        (directory / CHUNKS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "FilingIndex":
        """Read an index from disk.

        Raises:
            FileNotFoundError: when the directory holds no index.
        """
        vectors_path = directory / VECTORS_FILE
        chunks_path = directory / CHUNKS_FILE
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"no index in {directory}; run `edgar-rag ingest` first")

        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = tuple(Chunk(**chunk) for chunk in payload["chunks"])
        vectors = np.load(vectors_path).astype(np.float32, copy=False)
        return cls(source=payload["source"], chunks=chunks, vectors=vectors)


def _unit(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale rows to unit length so a dot product is a cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scaled = vectors / np.where(norms == 0, 1.0, norms)
    return scaled.astype(np.float32, copy=False)


def build_index(
    source: dict[str, str], chunks: tuple[Chunk, ...], vectors: NDArray[np.floating[Any]]
) -> FilingIndex:
    """Pair chunks with their vectors.

    Raises:
        ValueError: when the two do not line up.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks against {len(vectors)} vectors")
    if not chunks:
        raise ValueError("an index needs at least one chunk")
    return FilingIndex(
        source=source, chunks=chunks, vectors=_unit(np.asarray(vectors, dtype=np.float32))
    )
