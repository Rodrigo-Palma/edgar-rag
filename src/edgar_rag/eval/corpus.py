"""Index the golden set's pinned filings from the committed snapshots, offline.

A filing is indexed from its text snapshot in ``eval/snapshots/``, the text
the golden-set builder checked every answerable value against, read through
the same SHA-256 check against ``filings.lock.json`` and cut into the same
passages. A value the builder found in the indexed text is therefore in a
passage of the index, so no positive case becomes unanswerable because the
filing was downloaded and parsed again. Nothing is downloaded.
"""

from collections.abc import Iterable

from edgar_rag.domain import Embedder, IndexedFiling
from edgar_rag.eval.golden import Split
from edgar_rag.eval.snapshot import FilingsLock, LockedFiling, SnapshotPaths, read_text_snapshot
from edgar_rag.index import Shard
from edgar_rag.ingest import index_text


def pinned_filings(
    lock: FilingsLock, *, split: Split | None = None, tickers: Iterable[str] = ()
) -> tuple[LockedFiling, ...]:
    """The locked filings of ``split`` and of ``tickers``; all of them when neither is given.

    Raises:
        ValueError: for a ticker the lock does not pin, or when nothing matches.
    """
    wanted = frozenset(tickers)
    locked = {company.ticker: company.split for company in lock.companies}
    unknown = sorted(wanted - locked.keys())
    if unknown:
        raise ValueError(f"the lock pins no filing of {', '.join(unknown)}")
    chosen = tuple(
        filing
        for filing in lock.filings
        if (split is None or locked[filing.ticker] == split)
        and (not wanted or filing.ticker in wanted)
    )
    if not chosen:
        raise ValueError("no pinned filing matches that split and those tickers")
    return chosen


def indexed_filing(lock: FilingsLock, filing: LockedFiling) -> IndexedFiling:
    """What the index records about a pinned filing, named as EDGAR names the company."""
    return IndexedFiling(
        accession=filing.accession,
        cik=filing.cik,
        fiscal_year=filing.fiscal_year,
        form=filing.form,
        company=lock.company(filing.ticker).entity_name,
        filing_date=filing.filing_date,
        url=filing.url,
    )


def index_pinned(
    snapshots: SnapshotPaths, lock: FilingsLock, filing: LockedFiling, embedder: Embedder
) -> Shard:
    """Embed a pinned filing from its snapshot into a shard.

    Raises:
        ValueError: when the snapshot is missing, corrupt or not the text the
            lock recorded, or the embedder returns the wrong number of vectors.
        ModelError: when the embedder cannot answer.
    """
    text = read_text_snapshot(snapshots, filing)
    return index_text(indexed_filing(lock, filing), text, embedder)
