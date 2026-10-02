"""The golden set's filings are indexed from the very text the builder checked."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from edgar_rag.domain import EmbedderSpec
from edgar_rag.eval.build import EvalPaths, indexed_text
from edgar_rag.eval.corpus import index_pinned, indexed_filing, pinned_filings
from edgar_rag.eval.snapshot import read_lock, read_text_snapshot
from edgar_rag.index import CorpusIndex

REPO_EVAL = Path(__file__).parents[2] / "eval"
LOCK = read_lock(REPO_EVAL / "filings.lock.json")
SNAPSHOTS = EvalPaths(REPO_EVAL).snapshots
# What ServiceSettings and OllamaEmbedder default to.
SERVICE_EMBEDDER = EmbedderSpec(model="nomic-embed-text", lowercase=True)


class OneVectorPerText:
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)


def test_the_dev_split_is_eight_filings_of_four_companies():
    filings = pinned_filings(LOCK, split="dev")

    assert len(filings) == 8
    assert {filing.ticker for filing in filings} == {"AAPL", "KO", "CAT", "PFE"}


def test_every_pinned_filing_is_chosen_when_nothing_narrows_the_choice():
    assert len(pinned_filings(LOCK)) == 48


def test_tickers_narrow_the_choice_within_the_split():
    filings = pinned_filings(LOCK, split="dev", tickers=["AAPL"])

    assert [(filing.ticker, filing.fiscal_year) for filing in filings] == [
        ("AAPL", 2024),
        ("AAPL", 2025),
    ]


def test_a_ticker_the_lock_does_not_pin_is_refused():
    with pytest.raises(ValueError, match="no filing of XOM"):
        pinned_filings(LOCK, tickers=["XOM"])


def test_a_choice_that_matches_nothing_is_refused():
    with pytest.raises(ValueError, match="no pinned filing"):
        pinned_filings(LOCK, split="eval", tickers=["AAPL"])


def test_a_pinned_filing_is_recorded_as_the_lock_pins_it():
    filing = LOCK.filing("AAPL", 2024)

    indexed = indexed_filing(LOCK, filing)

    assert (indexed.accession, indexed.cik, indexed.fiscal_year) == (
        "0000320193-24-000123",
        320193,
        2024,
    )
    assert indexed.company == "Apple Inc."
    assert indexed.url == filing.url


def test_the_shard_holds_exactly_the_text_the_golden_set_was_checked_against():
    """A positive case is one whose value the builder found in this text."""
    filing = LOCK.filing("AAPL", 2024)

    shard = index_pinned(SNAPSHOTS, LOCK, filing, OneVectorPerText())

    checked = indexed_text(read_text_snapshot(SNAPSHOTS, filing))
    assert "\n\n".join(chunk.text for chunk in shard.chunks) == checked
    assert shard.filing == indexed_filing(LOCK, filing)


def test_a_snapshot_that_is_not_the_locked_text_is_refused(tmp_path):
    filing = LOCK.filing("AAPL", 2024)
    tampered = EvalPaths(tmp_path).snapshots
    tampered.text(filing).parent.mkdir(parents=True)
    tampered.text(filing).write_bytes(SNAPSHOTS.text(LOCK.filing("AAPL", 2025)).read_bytes())

    with pytest.raises(ValueError, match="SHA-256"):
        index_pinned(tampered, LOCK, filing, OneVectorPerText())


def test_the_committed_ci_index_is_the_dev_split_embedded_as_the_service_expects():
    index = CorpusIndex.load(REPO_EVAL / "ci" / "index", SERVICE_EMBEDDER)

    assert {f.accession for f in index.filings} == {
        f.accession for f in pinned_filings(LOCK, split="dev")
    }
    assert index.fingerprint.dimensions == 768


def test_the_committed_ci_index_holds_the_text_the_golden_set_was_checked_against():
    """Fails when chunking or the snapshots change and the CI index was not rebuilt."""
    index = CorpusIndex.load(REPO_EVAL / "ci" / "index", SERVICE_EMBEDDER)
    pinned = {filing.accession: filing for filing in LOCK.filings}

    for shard in index.shards:
        filing = pinned[shard.filing.accession]
        checked = indexed_text(read_text_snapshot(SNAPSHOTS, filing))
        assert "\n\n".join(chunk.text for chunk in shard.chunks) == checked, filing.ticker
        assert shard.filing == indexed_filing(LOCK, filing)
