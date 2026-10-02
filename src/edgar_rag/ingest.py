"""Turn one downloaded filing into an index: parse, chunk, embed.

Downloading and saving are the caller's (``edgar-rag ingest``), so this runs
the same against a fake embedder in a test as against Ollama.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from edgar_rag.chunking import chunk_sections
from edgar_rag.domain import Embedder
from edgar_rag.edgar.client import Filing
from edgar_rag.edgar.parse import html_to_text, split_into_sections
from edgar_rag.index import FilingIndex, build_index

# Large enough to keep Ollama busy, small enough that one slow batch does not
# hit the request timeout.
EMBED_BATCH_SIZE = 32


def index_filing(
    filing: Filing, embedder: Embedder, *, batch_size: int = EMBED_BATCH_SIZE
) -> FilingIndex:
    """Split ``filing`` into passages and embed them.

    Raises:
        ValueError: when the filing has no text, ``batch_size`` is below one,
            or the embedder returns a different number of vectors than texts.
        ModelError: when the embedder cannot answer.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, not {batch_size}")
    text = html_to_text(filing.html)
    if not text.strip():
        raise ValueError(f"{filing.ref.accession} has no text to index")

    chunks = chunk_sections(split_into_sections(text))
    vectors = embed_all(embedder, tuple(chunk.text for chunk in chunks), batch_size)
    return build_index(source=source_of(filing), chunks=chunks, vectors=vectors)


def embed_all(
    embedder: Embedder, texts: Sequence[str], batch_size: int = EMBED_BATCH_SIZE
) -> NDArray[np.float32]:
    """Embed ``texts`` in batches of at most ``batch_size``, in order."""
    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    return np.vstack([embedder.embed(batch) for batch in batches])


def source_of(filing: Filing) -> dict[str, str]:
    """What the service reports as the filing an answer comes from."""
    return {
        "company": filing.company,
        "form": filing.form,
        "filing_date": filing.filing_date,
        "url": filing.document_url,
    }
