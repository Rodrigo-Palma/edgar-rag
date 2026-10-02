from collections.abc import Sequence
from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from numpy.typing import NDArray

from edgar_rag.domain import IndexedFiling
from edgar_rag.edgar.client import Filing
from edgar_rag.edgar.submissions import FilingRef
from edgar_rag.index import Shard
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

    assert isinstance(index, Shard)
    assert len(index.chunks) == len(index.vectors) > 2
    assert {chunk.item for chunk in index.chunks} == {"Item 1", "Item 1A"}
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)


def test_the_shard_names_the_filing_it_was_built_from():
    shard = index_filing(_filing(), CountingEmbedder())

    assert shard.filing == IndexedFiling(
        accession="0000320193-24-000123",
        cik=320193,
        fiscal_year=2024,
        form="10-K",
        company="Apple Inc.",
        filing_date=date(2024, 11, 1),
        url=REF.document_url,
    )


def test_every_chunk_id_carries_the_accession():
    shard = index_filing(_filing(), CountingEmbedder())

    assert all(chunk.chunk_id.startswith("0000320193-24-000123:") for chunk in shard.chunks)


def test_the_fiscal_year_is_the_year_the_reporting_period_ends():
    """Walmart's fiscal 2025 ends in January 2025: the year of the end date."""
    ref = replace(REF, report_date=date(2025, 1, 31), filing_date=date(2025, 3, 14))

    shard = index_filing(Filing(ref=ref, company="Walmart Inc.", html=HTML), CountingEmbedder())

    assert shard.filing.fiscal_year == 2025


def test_a_filing_without_a_report_date_is_refused_for_having_no_fiscal_year():
    filing = Filing(ref=replace(REF, report_date=None), company="Apple Inc.", html=HTML)

    with pytest.raises(ValueError, match="no report date"):
        index_filing(filing, CountingEmbedder())


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
