"""Download what the golden set is built from, once, and pin it.

For every company of the roster, in order: read its ``companyfacts``, find the
10-K whose facts carry each fiscal year of the roster (EDGAR's ``fy`` labels the
filing, which is exactly what is wanted here), look the accession up in the
submissions (in an older page when it has left the recent list), and download
the document through ``EdgarClient``, which paces and caps every GET.

A company is excluded, and the next reserve takes its place in the same split,
when either 10-K is missing under its CIK or when a filing's report date falls
in another calendar year than its fiscal year: "fiscal 2024" has to name the
year the filing says it covers, or the period guard and the labels disagree.

Network failures are not exclusions: they stop the run.
"""

import collections
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from edgar_rag.edgar.client import EdgarClient, decode_document
from edgar_rag.edgar.errors import EdgarError
from edgar_rag.edgar.fetch import MIN_REQUEST_INTERVAL_SECONDS, Clock, Fetcher
from edgar_rag.edgar.parse import html_to_text
from edgar_rag.edgar.submissions import FilingRef, Submissions, SubmissionsPage
from edgar_rag.edgar.xbrl import CompanyFacts, parse_companyfacts
from edgar_rag.eval.golden import Split
from edgar_rag.eval.snapshot import (
    FORM,
    CompanySpec,
    Exclusion,
    FilingsLock,
    LockedCompany,
    LockedFiling,
    Roster,
    SnapshotPaths,
    canonical_json,
    check_roster_ciks,
    gzip_bytes,
    read_lock,
    sha256,
    trim_companyfacts,
    xz_bytes,
)
from edgar_rag.eval.templates import POSITIVE_CONCEPTS, SECTOR_CONCEPTS

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
PAGE_SEARCH_DAYS = 7
SNAPSHOT_TAGS = tuple(
    sorted(
        {tag for concept in POSITIVE_CONCEPTS for tag in concept.tags}
        | {concept.tag for concept in SECTOR_CONCEPTS}
    )
)

Log = Callable[[str], None]


class Excluded(Exception):
    """A company the roster rule leaves out; the message is the recorded reason."""


@dataclass(frozen=True, slots=True)
class Pinned:
    company: LockedCompany
    filings: tuple[LockedFiling, ...]
    snapshot: bytes
    texts: tuple[str, ...]


def fetch_tickers(fetcher: Fetcher, clock: Clock) -> dict[str, dict[str, object]]:
    """The SEC's ticker map, keyed by ticker.

    Raises:
        EdgarError: when EDGAR fails or the payload is not the documented shape.
    """
    content = fetcher.get(COMPANY_TICKERS_URL)
    # The client paces its own requests; this one went through a separate fetcher
    clock.sleep(MIN_REQUEST_INTERVAL_SECONDS)
    try:
        rows = json.loads(content).values()
        return {
            str(row["ticker"]): {"cik": int(row["cik_str"]), "title": str(row["title"])}
            for row in rows
        }
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise EdgarError(f"company_tickers.json is malformed: {error}") from error


def accessions_by_fiscal_year(facts: CompanyFacts) -> dict[int, list[str]]:
    """Each fiscal year to the 10-K accessions whose facts mostly carry it."""
    labels: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for fact in facts.facts:
        if fact.form == FORM and fact.fiscal_year is not None:
            labels[fact.accession][fact.fiscal_year] += 1
    by_year: dict[int, list[str]] = collections.defaultdict(list)
    for accession in sorted(labels):
        year, _ = labels[accession].most_common(1)[0]
        by_year[year].append(accession)
    return by_year


def first_filed(facts: CompanyFacts, accession: str) -> date:
    return min(fact.filed for fact in facts.facts if fact.accession == accession)


def find_ref(
    client: EdgarClient, submissions: Submissions, accession: str, filed: date
) -> FilingRef:
    """The filing's row, from the recent list or the older pages nearest ``filed``.

    EDGAR's page ranges leave gaps of a day or two between pages (Bank of
    America's 10-K filed on 2025-02-25 sits between a page ending on the 24th
    and one starting on the 26th), so the pages are tried by distance from the
    filing date, up to ``PAGE_SEARCH_DAYS`` away.

    Raises:
        EdgarError: when no page near that date lists the accession.
    """
    try:
        return submissions.filing(accession)
    except EdgarError:
        pass
    for page in _pages_near(submissions, filed):
        try:
            return client.submissions_page(submissions, page).filing(accession)
        except EdgarError:
            continue
    raise EdgarError(f"no submissions page of CIK {submissions.cik} lists {accession}")


def _pages_near(submissions: Submissions, filed: date) -> list[SubmissionsPage]:
    def distance(page: SubmissionsPage) -> int:
        if page.filing_from <= filed <= page.filing_to:
            return 0
        return min(abs((filed - page.filing_from).days), abs((filed - page.filing_to).days))

    near = [page for page in submissions.older_pages if distance(page) <= PAGE_SEARCH_DAYS]
    return sorted(near, key=lambda page: (distance(page), page.name))


def pin_refs(
    client: EdgarClient, spec: CompanySpec, facts: CompanyFacts, years: tuple[int, ...]
) -> tuple[FilingRef, ...]:
    """The 10-K of each fiscal year, or ``Excluded`` with the reason.

    Raises:
        Excluded: when the roster rule leaves the company out.
        EdgarError: when EDGAR fails.
    """
    by_year = accessions_by_fiscal_year(facts)
    submissions = client.submissions(spec.cik)
    refs = []
    for year in years:
        found = by_year.get(year, [])
        if len(found) != 1:
            raise Excluded(f"{len(found)} {FORM} filings carry fy {year} under CIK {spec.cik}")
        ref = find_ref(client, submissions, found[0], first_filed(facts, found[0]))
        if ref.form != FORM:
            raise Excluded(f"{ref.accession} is a {ref.form}, not a {FORM}")
        if ref.report_date is None or ref.report_date.year != year:
            raise Excluded(f"{ref.accession} has fy {year} but report date {ref.report_date}")
        refs.append(ref)
    return tuple(refs)


def pin_company(
    client: EdgarClient, spec: CompanySpec, split: Split, years: tuple[int, ...]
) -> Pinned:
    """Pin one company: its filings, their texts and its trimmed facts.

    Raises:
        Excluded: when the roster rule leaves the company out.
        EdgarError: when EDGAR fails.
    """
    raw_facts = client.companyfacts_bytes(spec.cik)
    facts = parse_companyfacts(raw_facts)
    refs = pin_refs(client, spec, facts, years)
    filings, texts = [], []
    for year, ref in zip(years, refs, strict=True):
        document = client.fetch_filing(ref)
        text = html_to_text(decode_document(document))
        texts.append(text)
        filings.append(_locked_filing(spec, year, ref, document, text))
    snapshot = trim_companyfacts(raw_facts, SNAPSHOT_TAGS)
    company = LockedCompany(
        ticker=spec.ticker,
        cik=spec.cik,
        name=spec.name,
        split=split,
        entity_name=facts.entity_name,
        companyfacts_sha256=sha256(raw_facts),
        snapshot_sha256=sha256(snapshot),
    )
    return Pinned(company, tuple(filings), snapshot, tuple(texts))


def _locked_filing(
    spec: CompanySpec, year: int, ref: FilingRef, document: bytes, text: str
) -> LockedFiling:
    assert ref.report_date is not None  # pin_refs excluded filings without one
    encoded = text.encode("utf-8")
    return LockedFiling(
        ticker=spec.ticker,
        cik=spec.cik,
        fiscal_year=year,
        accession=ref.accession,
        form=ref.form,
        filing_date=ref.filing_date,
        report_date=ref.report_date,
        primary_document=ref.primary_document,
        url=ref.document_url,
        document_sha256=sha256(document),
        document_bytes=len(document),
        text_sha256=sha256(encoded),
        text_chars=len(text),
    )


def pin_roster(
    client: EdgarClient, roster: Roster, log: Log
) -> tuple[tuple[Pinned, ...], tuple[Exclusion, ...]]:
    """Pin every company in roster order, filling exclusions from the reserves.

    Raises:
        EdgarError: when EDGAR fails, or the reserves run out.
    """
    reserves = list(roster.reserves)
    pinned: list[Pinned] = []
    excluded: list[Exclusion] = []
    for spec in roster.companies:
        assert spec.split is not None  # load_roster gives every listed company a split
        candidate = spec
        while True:
            try:
                pinned.append(pin_company(client, candidate, spec.split, roster.fiscal_years))
                log(f"pinned {candidate.ticker} ({spec.split})")
                break
            except Excluded as reason:
                replacement = reserves.pop(0) if reserves else None
                excluded.append(
                    Exclusion(
                        ticker=candidate.ticker,
                        cik=candidate.cik,
                        split=spec.split,
                        reason=str(reason),
                        replaced_by=replacement.ticker if replacement else None,
                    )
                )
                log(f"excluded {candidate.ticker}: {reason}")
                if replacement is None:
                    raise EdgarError(f"no reserve left to replace {candidate.ticker}") from None
                candidate = replacement
    return tuple(pinned), tuple(excluded)


def acquire(
    client: EdgarClient,
    fetcher: Fetcher,
    clock: Clock,
    roster: Roster,
    *,
    snapshots: Path,
    lock_path: Path,
    log: Log,
) -> FilingsLock:
    """Download, check, and write the snapshots and the lock.

    Raises:
        EdgarError: when EDGAR fails or the reserves run out.
        ValueError: when a listed CIK differs from the SEC's ticker map.
    """
    tickers = fetch_tickers(fetcher, clock)
    roster_tickers = {spec.ticker: tickers.get(spec.ticker) for spec in roster.everyone()}
    check_roster_ciks(roster, {t: int(str(e["cik"])) for t, e in tickers.items()})
    tickers_snapshot = canonical_json(roster_tickers)
    pinned, excluded = pin_roster(client, roster, log)
    if lock_path.is_file():
        check_unchanged(read_lock(lock_path), pinned)
    paths = SnapshotPaths(snapshots)
    _write(paths.tickers, tickers_snapshot)
    for company in pinned:
        _write(paths.companyfacts(company.company.ticker), gzip_bytes(company.snapshot))
        for filing, text in zip(company.filings, company.texts, strict=True):
            _write(paths.text(filing), xz_bytes(text.encode("utf-8")))
    lock = FilingsLock(
        fiscal_years=roster.fiscal_years,
        company_tickers_sha256=sha256(tickers_snapshot),
        companies=tuple(company.company for company in pinned),
        filings=tuple(filing for company in pinned for filing in company.filings),
        excluded=excluded,
    )
    lock_path.write_bytes(canonical_json(lock.model_dump(mode="json")))
    return lock


def check_unchanged(previous: FilingsLock, pinned: tuple[Pinned, ...]) -> None:
    """Refuse a download whose bytes differ from what the lock recorded.

    An accepted filing never changes on EDGAR, so a different hash for the same
    accession means the download is not the document the lock pinned.

    Raises:
        EdgarError: naming the accession whose bytes changed.
    """
    recorded = {filing.accession: filing.document_sha256 for filing in previous.filings}
    for company in pinned:
        for filing in company.filings:
            expected = recorded.get(filing.accession)
            if expected is not None and expected != filing.document_sha256:
                raise EdgarError(
                    f"{filing.accession} downloaded with SHA-256 {filing.document_sha256}, "
                    f"but the lock pinned {expected}"
                )


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
