"""The vector index: one shard per filing, under a manifest that pins the embedder.

On disk (format 2)::

    <root>/manifest.json            format_version, fingerprint, one entry per filing
    <root>/<accession>/vectors.npy  float32, one unit-length row per chunk
    <root>/<accession>/chunks.json  the chunks, in the same order

The fingerprint is the embedding model, whether its input was lower-cased and
the vector size. Vectors from another model, or from text cased differently,
would still load and search and return passages that look plausible and are
not the closest, so an index is only loaded for the embedder it was built
with. The manifest also records the SHA-256 of both files of every shard, so a
shard that changed after it was written fails the load instead of being
searched.

A question is scoped to one filing (``Scope``) and searched against that
filing's shard alone: no passage of another filing can be returned, whatever
its score. Search is exact, a dot product over a few hundred rows.

The manifest is written last, through a temporary file, so it never lists a
shard that was not fully written; a directory it does not list is ignored.
"""

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from edgar_rag.domain import (
    DEFAULT_TOP_K,
    Chunk,
    EmbedderSpec,
    EmbeddingFingerprint,
    IndexedFiling,
    Scope,
    ScoredChunk,
)

FORMAT_VERSION = 2
MANIFEST_FILE = "manifest.json"
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
# The accession names a directory, so it is checked before any path is built from it.
ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}", re.ASCII)
SHA256 = r"^[0-9a-f]{64}$"


class IndexFormatError(Exception):
    """Raised when an index on disk cannot be trusted: wrong format, embedder or content."""


@dataclass(frozen=True, slots=True)
class Shard:
    """One filing's chunks and their unit-length vectors, in the same order."""

    filing: IndexedFiling
    chunks: tuple[Chunk, ...]
    vectors: NDArray[np.float32]

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])


def build_shard(
    filing: IndexedFiling, chunks: Iterable[Chunk], vectors: NDArray[np.floating[Any]]
) -> Shard:
    """Pair a filing's chunks with their vectors, namespacing every chunk id.

    ``Item 7#3`` exists in every 10-K, so a chunk id is prefixed with the
    accession (``0000320193-24-000123:Item 7#3``) to stay unique across filings.

    Raises:
        ValueError: when the accession is malformed, there are no chunks, the
            chunks and vectors do not line up, or a vector is not finite.
    """
    if ACCESSION.fullmatch(filing.accession) is None:
        raise ValueError(f"{filing.accession!r} is not an accession in the dashed form")
    listed = tuple(chunks)
    matrix = np.asarray(vectors, dtype=np.float32)
    if not listed:
        raise ValueError(f"{filing.accession}: a shard needs at least one chunk")
    if matrix.ndim != 2 or len(listed) != len(matrix):
        raise ValueError(f"{filing.accession}: {len(listed)} chunks against {len(matrix)} vectors")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{filing.accession}: a vector holds a value that is not finite")
    prefix = f"{filing.accession}:"
    namespaced = tuple(replace(chunk, chunk_id=prefix + chunk.chunk_id) for chunk in listed)
    return Shard(filing=filing, chunks=namespaced, vectors=_unit(matrix))


@dataclass(frozen=True, slots=True)
class CorpusIndex:
    """Every indexed filing, searched one filing at a time.

    Build one with ``CorpusIndex.of`` or ``CorpusIndex.load``, which check that
    the shards agree with each other and with the fingerprint.
    """

    fingerprint: EmbeddingFingerprint
    shards: tuple[Shard, ...]

    @classmethod
    def of(cls, spec: EmbedderSpec, shards: Iterable[Shard]) -> "CorpusIndex":
        """An index of ``shards``, embedded as ``spec`` says.

        Raises:
            ValueError: when there is no shard, two share an accession or a
                CIK and fiscal year, or their vectors differ in size.
        """
        ordered = tuple(sorted(shards, key=lambda shard: _scope_key(shard.filing)))
        if not ordered:
            raise ValueError("an index needs at least one filing")
        _check_unique(ordered)
        sizes = {shard.dimensions for shard in ordered}
        if len(sizes) != 1:
            raise ValueError(f"the shards hold vectors of different sizes: {sorted(sizes)}")
        fingerprint = EmbeddingFingerprint(spec.model, spec.lowercase, sizes.pop())
        return cls(fingerprint=fingerprint, shards=ordered)

    @property
    def filings(self) -> tuple[IndexedFiling, ...]:
        return tuple(shard.filing for shard in self.shards)

    @property
    def chunk_count(self) -> int:
        return sum(len(shard.chunks) for shard in self.shards)

    def resolve(self, scope: Scope) -> IndexedFiling | None:
        """The filing ``scope`` names: its fiscal year, or the company's latest."""
        shard = self._shard_for(scope)
        return shard.filing if shard is not None else None

    def search(
        self, query: NDArray[np.float32], scope: Scope, top_k: int = DEFAULT_TOP_K
    ) -> tuple[ScoredChunk, ...]:
        """The ``top_k`` chunks of the scoped filing closest to ``query``, best first.

        Returns no chunk when no filing matches ``scope``.

        Raises:
            ValueError: when ``top_k`` is not positive, or ``query`` is not the
                size of the indexed vectors.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = query.reshape(-1)
        if vector.shape[0] != self.fingerprint.dimensions:
            raise ValueError(
                f"the query has {vector.shape[0]} dimensions, "
                f"the index {self.fingerprint.dimensions}"
            )
        shard = self._shard_for(scope)
        if shard is None:
            return ()
        scores = shard.vectors @ _unit(vector)
        best = np.argsort(scores)[::-1][:top_k]
        return tuple(ScoredChunk(chunk=shard.chunks[i], score=float(scores[i])) for i in best)

    def digest(self) -> str:
        """A short hash of the fingerprint, every vector and every chunk id."""
        digest = hashlib.sha256(repr(self.fingerprint).encode("utf-8"))
        for shard in self.shards:
            digest.update(shard.filing.accession.encode("ascii"))
            digest.update(shard.vectors.tobytes())
            for chunk in shard.chunks:
                digest.update(chunk.chunk_id.encode("utf-8"))
        return digest.hexdigest()[:16]

    def save(self, root: Path) -> None:
        """Write every shard and the manifest under ``root``.

        Raises:
            IndexFormatError: when ``root`` holds an index these shards cannot join.
            ValueError: when ``root`` already holds another filing for one of
                these scopes.
        """
        for shard in self.shards:
            write_shard(root, shard, self.fingerprint.spec)

    @classmethod
    def load(cls, root: Path, expect: EmbedderSpec) -> "CorpusIndex":
        """Read the index under ``root``, refusing one built by another embedder.

        Raises:
            FileNotFoundError: when ``root`` holds no index at all.
            IndexFormatError: when it holds one in another format, built with
                another embedder, or that does not match its manifest.
        """
        manifest = _read_manifest(root)
        fingerprint = manifest.fingerprint.as_domain()
        if fingerprint.spec != expect:
            raise IndexFormatError(
                f"the index in {root} was built with {_describe(fingerprint.spec)}, "
                f"and questions would be embedded with {_describe(expect)}; "
                "re-ingest with the same embedder or point EDGAR_RAG_INDEX_DIR elsewhere"
            )
        shards = [_read_shard(root, entry, fingerprint.dimensions) for entry in manifest.filings]
        try:
            return cls.of(expect, shards)
        except ValueError as error:
            raise IndexFormatError(f"the manifest in {root}: {error}") from error

    def _shard_for(self, scope: Scope) -> Shard | None:
        mine = [shard for shard in self.shards if shard.filing.cik == scope.cik]
        if scope.fiscal_year is None:
            return max(mine, key=lambda shard: shard.filing.fiscal_year, default=None)
        return next(
            (shard for shard in mine if shard.filing.fiscal_year == scope.fiscal_year), None
        )


def write_shard(root: Path, shard: Shard, spec: EmbedderSpec) -> None:
    """Add ``shard`` to the index under ``root``, creating the index if there is none.

    A filing already indexed under the same accession is replaced. The shard
    is written first and the manifest last, each through a temporary path.

    Raises:
        IndexFormatError: when ``root`` holds an index in another format, built
            with another embedder, or with vectors of another size.
        ValueError: when another accession is already indexed for the same
            company and fiscal year, which would make that scope ambiguous.
    """
    fingerprint = EmbeddingFingerprint(spec.model, spec.lowercase, shard.dimensions)
    existing = _existing_entries(root, fingerprint)
    clash = [
        entry.accession
        for entry in existing
        if (entry.cik, entry.fiscal_year) == _scope_key(shard.filing)
        and entry.accession != shard.filing.accession
    ]
    if clash:
        raise ValueError(
            f"CIK {shard.filing.cik} fiscal {shard.filing.fiscal_year} is already indexed "
            f"as {clash[0]}; adding {shard.filing.accession} would make that scope ambiguous"
        )

    root.mkdir(parents=True, exist_ok=True)
    entry = _write_shard_files(root, shard)
    kept = [listed for listed in existing if listed.accession != shard.filing.accession]
    manifest = _Manifest(
        format_version=FORMAT_VERSION,
        fingerprint=_Fingerprint.of(fingerprint),
        filings=tuple(sorted([*kept, entry], key=lambda listed: (listed.cik, listed.fiscal_year))),
    )
    _write_atomically(root / MANIFEST_FILE, _manifest_bytes(manifest))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Fingerprint(_Frozen):
    model: str = Field(min_length=1)
    lowercase: bool
    dimensions: int = Field(gt=0)

    @classmethod
    def of(cls, fingerprint: EmbeddingFingerprint) -> "_Fingerprint":
        return cls(
            model=fingerprint.model,
            lowercase=fingerprint.lowercase,
            dimensions=fingerprint.dimensions,
        )

    def as_domain(self) -> EmbeddingFingerprint:
        return EmbeddingFingerprint(self.model, self.lowercase, self.dimensions)


class _Entry(_Frozen):
    accession: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    cik: int = Field(gt=0)
    fiscal_year: int = Field(gt=0)
    form: str = Field(min_length=1)
    company: str
    filing_date: date
    url: str
    chunks: int = Field(gt=0)
    vectors_sha256: str = Field(pattern=SHA256)
    chunks_sha256: str = Field(pattern=SHA256)

    def as_filing(self) -> IndexedFiling:
        return IndexedFiling(
            accession=self.accession,
            cik=self.cik,
            fiscal_year=self.fiscal_year,
            form=self.form,
            company=self.company,
            filing_date=self.filing_date,
            url=self.url,
        )


class _Manifest(_Frozen):
    format_version: int
    fingerprint: _Fingerprint
    filings: tuple[_Entry, ...] = Field(min_length=1)


class _StoredChunk(_Frozen):
    chunk_id: str
    item: str
    title: str
    text: str


def _existing_entries(root: Path, adding: EmbeddingFingerprint) -> tuple[_Entry, ...]:
    """The filings already under ``root``, once it is known the new one can join them."""
    if not (root / MANIFEST_FILE).exists():
        if _holds_format_one(root):
            raise _format_one_error(root)
        return ()
    manifest = _read_manifest(root)
    indexed = manifest.fingerprint.as_domain()
    if indexed.spec != adding.spec:
        raise IndexFormatError(
            f"the index in {root} was built with {_describe(indexed.spec)}; "
            f"this filing was embedded with {_describe(adding.spec)}"
        )
    if indexed.dimensions != adding.dimensions:
        raise IndexFormatError(
            f"the index in {root} holds {indexed.dimensions}-dimension vectors; "
            f"this filing's have {adding.dimensions}"
        )
    return manifest.filings


def _read_manifest(root: Path) -> _Manifest:
    path = root / MANIFEST_FILE
    if not path.is_file():
        if _holds_format_one(root):
            raise _format_one_error(root)
        raise FileNotFoundError(f"no index in {root}; run `edgar-rag ingest` first")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise IndexFormatError(f"{path} cannot be read as JSON: {error}") from error
    version = payload.get("format_version") if isinstance(payload, dict) else None
    if version != FORMAT_VERSION:
        raise IndexFormatError(
            f"{path} is index format {version!r} and this version reads {FORMAT_VERSION}; "
            "re-ingest into an empty directory"
        )
    try:
        return _Manifest.model_validate(payload)
    except ValidationError as error:
        raise IndexFormatError(f"{path} is not a valid manifest: {error}") from error


def _read_shard(root: Path, entry: _Entry, dimensions: int) -> Shard:
    """Read one shard, checking it against its manifest entry before trusting it."""
    directory = root / entry.accession
    vectors_bytes = _read_checked(directory / VECTORS_FILE, entry.vectors_sha256)
    chunks_bytes = _read_checked(directory / CHUNKS_FILE, entry.chunks_sha256)
    try:
        # A .npy file can carry pickled objects; refusing them keeps loading an
        # index from running code, whatever wrote the file.
        vectors = np.load(io.BytesIO(vectors_bytes), allow_pickle=False)
        stored = [_StoredChunk.model_validate(item) for item in json.loads(chunks_bytes)]
    except (ValueError, TypeError) as error:
        raise IndexFormatError(f"{directory} cannot be read: {error}") from error
    expected = (entry.chunks, dimensions)
    if vectors.dtype != np.float32 or vectors.shape != expected:
        raise IndexFormatError(
            f"{directory / VECTORS_FILE} is {vectors.dtype} {vectors.shape}, "
            f"the manifest says float32 {expected}"
        )
    if len(stored) != entry.chunks:
        raise IndexFormatError(
            f"{directory / CHUNKS_FILE} holds {len(stored)} chunks, "
            f"the manifest says {entry.chunks}"
        )
    if not all(chunk.chunk_id.startswith(f"{entry.accession}:") for chunk in stored):
        raise IndexFormatError(f"{directory / CHUNKS_FILE} holds a chunk of another filing")
    if not np.isfinite(vectors).all():
        raise IndexFormatError(f"{directory / VECTORS_FILE} holds a value that is not finite")
    chunks = tuple(Chunk(**chunk.model_dump()) for chunk in stored)
    return Shard(filing=entry.as_filing(), chunks=chunks, vectors=vectors)


def _read_checked(path: Path, sha256: str) -> bytes:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise IndexFormatError(f"{path} is listed in the manifest but cannot be read") from error
    if hashlib.sha256(content).hexdigest() != sha256:
        raise IndexFormatError(f"{path} does not match its SHA-256 in the manifest")
    return content


def _write_shard_files(root: Path, shard: Shard) -> _Entry:
    """Write the shard's directory with one rename, and return its manifest entry."""
    buffer = io.BytesIO()
    np.save(buffer, shard.vectors, allow_pickle=False)
    vectors_bytes = buffer.getvalue()
    chunks_bytes = (
        json.dumps([asdict(chunk) for chunk in shard.chunks], ensure_ascii=False) + "\n"
    ).encode("utf-8")

    target = root / shard.filing.accession
    staging = Path(tempfile.mkdtemp(dir=root, prefix=".partial-"))
    try:
        (staging / VECTORS_FILE).write_bytes(vectors_bytes)
        (staging / CHUNKS_FILE).write_bytes(chunks_bytes)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    filing = shard.filing
    return _Entry(
        accession=filing.accession,
        cik=filing.cik,
        fiscal_year=filing.fiscal_year,
        form=filing.form,
        company=filing.company,
        filing_date=filing.filing_date,
        url=filing.url,
        chunks=len(shard.chunks),
        vectors_sha256=hashlib.sha256(vectors_bytes).hexdigest(),
        chunks_sha256=hashlib.sha256(chunks_bytes).hexdigest(),
    )


def _manifest_bytes(manifest: _Manifest) -> bytes:
    return (json.dumps(manifest.model_dump(mode="json"), indent=1) + "\n").encode("utf-8")


def _write_atomically(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _check_unique(shards: tuple[Shard, ...]) -> None:
    accessions = [shard.filing.accession for shard in shards]
    if len(set(accessions)) != len(accessions):
        raise ValueError("two shards share an accession")
    scopes = [_scope_key(shard.filing) for shard in shards]
    if len(set(scopes)) != len(scopes):
        raise ValueError("two shards share a CIK and fiscal year, so a scope would be ambiguous")


def _scope_key(filing: IndexedFiling) -> tuple[int, int]:
    return (filing.cik, filing.fiscal_year)


def _holds_format_one(root: Path) -> bool:
    return (root / VECTORS_FILE).exists() or (root / CHUNKS_FILE).exists()


def _format_one_error(root: Path) -> IndexFormatError:
    return IndexFormatError(
        f"{root} holds a single-filing index (format 1), which is not migrated; "
        "move it away and re-ingest"
    )


def _describe(spec: EmbedderSpec) -> str:
    casing = "lower-cased" if spec.lowercase else "as written"
    return f"{spec.model} on text {casing}"


def _unit(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale rows to unit length so a dot product is a cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scaled = vectors / np.where(norms == 0, 1.0, norms)
    return scaled.astype(np.float32, copy=False)
