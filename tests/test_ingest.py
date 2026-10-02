from collections.abc import Sequence
from datetime import date

import numpy as np
import pytest
from numpy.typing import NDArray

from edgar_rag.edgar.client import Filing
from edgar_rag.edgar.submissions import FilingRef
from edgar_rag.index import FilingIndex
from edgar_rag.ingest import index_filing

REF = FilingRef(
    cik=320193,
    accession="0000320193-24-000123",
    form="10-K",
    filing_date=date(2024, 11, 1),
    report_date=date(2024, 9, 28),
    primary_document="aapl-20240928.htm",
)
HTML = """
<html><body>
<p>Item 1. Business</p>
<p>{business}</p>
<p>Item 1A. Risk Factors</p>
<p>{risks}</p>
</body></html>
""".format(business="The Company designs phones. " * 60, risks="Supply chains may fail. " * 60)


class CountingEmbedder:
    """One distinct vector per text, recording the size of every batch."""

    def __init__(self, *, short_by: int = 0) -> None:
        self.batches: list[int] = []
        self._short_by = short_by
        self._seen = 0

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.batches.append(len(texts))
        rows = []
        for _ in texts:
            self._seen += 1
            rows.append([float(self._seen), 1.0])
        return np.asarray(rows[: len(rows) - self._short_by], dtype=np.float32)


def _filing(html: str = HTML) -> Filing:
    return Filing(ref=REF, company="Apple Inc.", html=html)


def test_a_filing_becomes_an_index_of_its_passages_and_their_vectors():
    index = index_filing(_filing(), CountingEmbedder())

    assert isinstance(index, FilingIndex)
    assert len(index.chunks) == len(index.vectors) > 2
    assert {chunk.item for chunk in index.chunks} == {"Item 1", "Item 1A"}
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)


def test_the_index_names_the_filing_it_was_built_from():
    index = index_filing(_filing(), CountingEmbedder())

    assert index.source == {
        "company": "Apple Inc.",
        "form": "10-K",
        "filing_date": "2024-11-01",
        "url": REF.document_url,
    }


def test_passages_are_embedded_in_batches_of_the_given_size():
    embedder = CountingEmbedder()

    index = index_filing(_filing(), embedder, batch_size=3)

    assert sum(embedder.batches) == len(index.chunks)
    assert all(size <= 3 for size in embedder.batches)
    assert len(embedder.batches) == -(-len(index.chunks) // 3)


def test_an_embedder_that_drops_vectors_fails_the_ingest_instead_of_misaligning():
    with pytest.raises(ValueError, match="chunks against"):
        index_filing(_filing(), CountingEmbedder(short_by=1))


def test_a_filing_with_no_text_is_refused():
    with pytest.raises(ValueError, match="no text"):
        index_filing(_filing("<html><body><script>x()</script></body></html>"), CountingEmbedder())


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_batch_size_below_one_is_refused(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        index_filing(_filing(), CountingEmbedder(), batch_size=batch_size)
