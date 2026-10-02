"""Turn one filing into a shard of the index: parse, chunk, embed.

Downloading and writing are the caller's (``edgar-rag ingest``), so this runs
the same against a fake embedder in a test as against Ollama, and the same on
a filing fresh from EDGAR as on the text snapshot the golden set was built
from.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from edgar_rag.chunking import chunk_sections
from edgar_rag.domain import Embedder, IndexedFiling
from edgar_rag.edgar.client import Filing
from edgar_rag.edgar.parse import html_to_text, split_into_sections
from edgar_rag.index import Shard, build_shard

# Large enough to keep Ollama busy, small enough that one slow batch does not
# hit the request timeout.
EMBED_BATCH_SIZE = 32


def index_filing(
    filing: Filing, embedder: Embedder, *, batch_size: int = EMBED_BATCH_SIZE
) -> Shard:
    """Split a downloaded ``filing`` into passages and embed them.

    Raises:
        ValueError: when the filing has no text or no report date, ``batch_size``
            is below one, or the embedder returns a different number of vectors
            than texts.
        ModelError: when the embedder cannot answer.
    """
    return index_text(
        indexed_filing(filing), html_to_text(filing.html), embedder, batch_size=batch_size
    )


def index_text(
    filing: IndexedFiling, text: str, embedder: Embedder, *, batch_size: int = EMBED_BATCH_SIZE
) -> Shard:
    """Split the parsed ``text`` of ``filing`` into passages and embed them.

    The passages are cut exactly as the golden set's builder cuts them, so a
    value it found in the indexed text is in a passage of this shard.

    Raises:
        ValueError: when the text is empty, ``batch_size`` is below one, or the
            embedder returns a different number of vectors than texts.
        ModelError: when the embedder cannot answer.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, not {batch_size}")
    if not text.strip():
        raise ValueError(f"{filing.accession} has no text to index")

    chunks = chunk_sections(split_into_sections(text))
    vectors = embed_all(embedder, tuple(chunk.text for chunk in chunks), batch_size)
    return build_shard(filing, chunks, vectors)


def embed_all(
    embedder: Embedder, texts: Sequence[str], batch_size: int = EMBED_BATCH_SIZE
) -> NDArray[np.float32]:
    """Embed ``texts`` in batches of at most ``batch_size``, in order."""
    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    return np.vstack([embedder.embed(batch) for batch in batches])


def indexed_filing(filing: Filing) -> IndexedFiling:
    """What the index records about a downloaded filing.

    The fiscal year is the calendar year the reporting period ends in, which is
    how the golden set pins its filings.

    Raises:
        ValueError: when EDGAR gave the filing no report date to take it from.
    """
    ref = filing.ref
    if ref.report_date is None:
        raise ValueError(f"{ref.accession} has no report date, so no fiscal year")
    return IndexedFiling(
        accession=ref.accession,
        cik=ref.cik,
        fiscal_year=ref.report_date.year,
        form=ref.form,
        company=filing.company,
        filing_date=ref.filing_date,
        url=ref.document_url,
    )
