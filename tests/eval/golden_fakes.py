"""A small synthetic EDGAR for the golden-set builder: companies, facts, filings.

Every company reports three concepts (revenue, total assets, diluted EPS) in a
10-K per year from 2017 to 2025, each 10-K carrying its year and the one before.
Values are derived from the company's ``base`` so no two companies share a
figure. ``filing_html`` prints the values of its own two years, in millions, the
way a statement does; ``omit`` leaves concepts out to exercise the drop rules.

A plain module, imported by name, for the same reason as ``tests/fakes.py``.
"""

import json
import lzma
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import httpx

from edgar_rag.edgar.client import decode_document
from edgar_rag.edgar.parse import html_to_text
from edgar_rag.eval.acquire import SNAPSHOT_TAGS
from edgar_rag.eval.snapshot import (
    FilingsLock,
    LockedCompany,
    LockedFiling,
    SnapshotPaths,
    canonical_json,
    gzip_bytes,
    read_lock,
    sha256,
    trim_companyfacts,
    write_lock,
    xz_bytes,
)

YEARS = tuple(range(2017, 2026))
FISCAL_YEARS = (2024, 2025)
MILLION = Decimal(10) ** 6


@dataclass(frozen=True)
class FakeCompany:
    ticker: str
    cik: int
    name: str
    split: str
    base: int
    report_month: int = 12
    report_day: int = 31
    # 1 for a retailer whose "fiscal 2024" ends in early 2025, as Home Depot's does
    report_year_offset: int = 0

    def revenue(self, year: int) -> Decimal:
        return Decimal(self.base * 1000 + (year - 2000) * 7) * MILLION

    def assets(self, year: int) -> Decimal:
        return Decimal(self.base * 2000 + (year - 2000) * 13) * MILLION

    def eps(self, year: int) -> Decimal:
        return Decimal(self.base) + Decimal(year - 2000) / 100

    def accession(self, year: int) -> str:
        return f"{self.cik:010d}-{(year + 1) % 100:02d}-{year:06d}"

    def report_date(self, year: int) -> date:
        return date(year + self.report_year_offset, self.report_month, self.report_day)

    def document(self, year: int) -> str:
        return f"{self.ticker.lower()}-{year}1231.htm"


DEV = (
    FakeCompany("AAA", 1001, "Alder Works", "dev", base=11),
    FakeCompany("BBB", 1002, "Birch Mills", "dev", base=23),
)
EVAL = (
    FakeCompany("CCC", 1003, "Cedar Freight", "eval", base=37),
    FakeCompany("DDD", 1004, "Dogwood Foods", "eval", base=41),
)


def _flow(value: Decimal, year: int, filing_year: int, company: FakeCompany) -> dict[str, object]:
    return {
        "start": (company.report_date(year) - timedelta(days=364)).isoformat(),
        "end": company.report_date(year).isoformat(),
        "val": float(value) if value != value.to_integral_value() else int(value),
        "accn": company.accession(filing_year),
        "fy": filing_year,
        "fp": "FY",
        "form": "10-K",
        "filed": f"{filing_year + 1}-02-15",
    }


def companyfacts(company: FakeCompany, *, omit: frozenset[str] = frozenset()) -> bytes:
    """The raw ``companyfacts`` JSON of ``company``, minus the ``omit`` tags."""
    revenue, assets, eps = [], [], []
    for filing_year in YEARS:
        for year in (filing_year, filing_year - 1):
            revenue.append(_flow(company.revenue(year), year, filing_year, company))
            eps.append(_flow(company.eps(year), year, filing_year, company))
            instant = _flow(company.assets(year), year, filing_year, company)
            del instant["start"]
            assets.append(instant)
    concepts = {
        "Revenues": {"units": {"USD": revenue}},
        "Assets": {"units": {"USD": assets}},
        "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
        "AccountsPayableCurrent": {"units": {"USD": []}},
    }
    payload = {
        "cik": company.cik,
        "entityName": company.name.upper(),
        "facts": {"us-gaap": {k: v for k, v in concepts.items() if k not in omit}},
    }
    return json.dumps(payload).encode()


def _millions(value: Decimal) -> str:
    return f"{int(value / MILLION):,}"


def filing_html(company: FakeCompany, year: int, *, omit: frozenset[str] = frozenset()) -> bytes:
    """A 10-K whose Item 8 prints the year's and the prior year's figures."""
    prose = (
        f"{company.name} designs, makes and sells durable goods to customers worldwide. "
        "The discussion below covers the results of operations for the two fiscal years "
        "presented, the liquidity of the company and its capital resources. "
    )
    rows = [f"<p>Item 7. Management's Discussion and Analysis</p><p>{prose * 2}</p>"]
    rows.append(f"<p>Item 8. Financial Statements</p><p>{prose}</p><p>(in millions)</p>")
    if "Revenues" not in omit:
        rows.append(
            f"<p>Total revenues</p><p>$</p><p>{_millions(company.revenue(year))}</p>"
            f"<p>{_millions(company.revenue(year - 1))}</p>"
        )
    if "Assets" not in omit:
        rows.append(
            f"<p>Total assets</p><p>{_millions(company.assets(year))}</p>"
            f"<p>{_millions(company.assets(year - 1))}</p>"
        )
    if "EarningsPerShareDiluted" not in omit:
        rows.append(
            f"<p>Diluted earnings per share</p><p>$</p><p>{company.eps(year):.2f}</p>"
            f"<p>{company.eps(year - 1):.2f}</p>"
        )
    rows.append(f"<p>Item 9A. Controls and Procedures</p><p>{prose * 2}</p>")
    return f"<html><body>{''.join(rows)}</body></html>".encode()


def submissions(
    company: FakeCompany, *, recent_years: tuple[int, ...] = YEARS
) -> dict[str, object]:
    years = sorted(recent_years, reverse=True)
    return {
        "name": company.name.upper(),
        "filings": {
            "recent": {
                "form": ["10-K"] * len(years),
                "accessionNumber": [company.accession(year) for year in years],
                "primaryDocument": [company.document(year) for year in years],
                "filingDate": [f"{year + 1}-02-15" for year in years],
                "reportDate": [company.report_date(year).isoformat() for year in years],
            },
            "files": [],
        },
    }


def tickers_payload(companies: tuple[FakeCompany, ...]) -> dict[str, object]:
    return {
        str(position): {"cik_str": company.cik, "ticker": company.ticker, "title": company.name}
        for position, company in enumerate(companies)
    }


def roster_toml(companies: tuple[FakeCompany, ...], reserves: tuple[FakeCompany, ...] = ()) -> str:
    lines = ["fiscal_years = [2024, 2025]", "fold_seed = 7", "sample_seed = 11", ""]
    for company in companies:
        lines += [
            "[[company]]",
            f'ticker = "{company.ticker}"',
            f"cik = {company.cik}",
            f'name = "{company.name}"',
            f'split = "{company.split}"',
            "",
        ]
    for company in reserves:
        lines += [
            "[[reserve]]",
            f'ticker = "{company.ticker}"',
            f"cik = {company.cik}",
            f'name = "{company.name}"',
            "",
        ]
    return "\n".join(lines)


class FakeSec:
    """Serves the ticker map, company facts, submissions and filing documents."""

    def __init__(self, companies: tuple[FakeCompany, ...], **overrides: object) -> None:
        self.companies = {company.cik: company for company in companies}
        self.overrides = overrides

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in self.overrides:
            override = self.overrides[url]
            if isinstance(override, bytes):
                return httpx.Response(200, content=override)
            return httpx.Response(200, json=override)
        if url.endswith("/files/company_tickers.json"):
            return httpx.Response(200, json=tickers_payload(tuple(self.companies.values())))
        for cik, company in self.companies.items():
            if url.endswith(f"/companyfacts/CIK{cik:010d}.json"):
                return httpx.Response(200, content=companyfacts(company))
            if url.endswith(f"/submissions/CIK{cik:010d}.json"):
                return httpx.Response(200, json=submissions(company))
            for year in YEARS:
                if url.endswith(
                    f"/{company.accession(year).replace('-', '')}/{company.document(year)}"
                ):
                    return httpx.Response(200, content=filing_html(company, year))
        return httpx.Response(404)


def write_eval_root(
    root: Path,
    companies: tuple[FakeCompany, ...] = DEV + EVAL,
    *,
    omit: dict[str, frozenset[str]] | None = None,
) -> FilingsLock:
    """Write the roster, snapshots and lock a build reads, as ``fetch`` would."""
    omit = omit or {}
    root.mkdir(parents=True, exist_ok=True)
    (root / "companies.toml").write_text(roster_toml(companies))
    paths = SnapshotPaths(root / "snapshots")
    tickers = {c.ticker: {"cik": c.cik, "title": c.name} for c in companies}
    tickers_bytes = canonical_json(tickers)
    _write(paths.tickers, tickers_bytes)
    locked_companies, locked_filings = [], []
    for company in companies:
        missing = omit.get(company.ticker, frozenset())
        trimmed = trim_companyfacts(companyfacts(company, omit=missing), SNAPSHOT_TAGS)
        _write(paths.companyfacts(company.ticker), gzip_bytes(trimmed))
        locked_companies.append(
            LockedCompany(
                ticker=company.ticker,
                cik=company.cik,
                name=company.name,
                split=company.split,  # type: ignore[arg-type]
                entity_name=company.name.upper(),
                companyfacts_sha256=sha256(companyfacts(company, omit=missing)),
                snapshot_sha256=sha256(trimmed),
            )
        )
        for year in FISCAL_YEARS:
            html = filing_html(company, year, omit=missing)
            text = html_to_text(decode_document(html))
            filing = LockedFiling(
                ticker=company.ticker,
                cik=company.cik,
                fiscal_year=year,
                accession=company.accession(year),
                form="10-K",
                filing_date=date(year + 1, 2, 15),
                report_date=company.report_date(year),
                primary_document=company.document(year),
                url=f"https://www.sec.gov/Archives/edgar/data/{company.cik}/x/{company.document(year)}",
                document_sha256=sha256(html),
                document_bytes=len(html),
                text_sha256=sha256(text.encode()),
                text_chars=len(text),
            )
            _write(paths.text(filing), xz_bytes(text.encode()))
            locked_filings.append(filing)
    lock = FilingsLock(
        fiscal_years=FISCAL_YEARS,
        company_tickers_sha256=sha256(tickers_bytes),
        companies=tuple(locked_companies),
        filings=tuple(locked_filings),
    )
    write_lock(root / "filings.lock.json", lock)
    return lock


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def rewrite_text(root: Path, ticker: str, year: int, edit: Callable[[str], str]) -> None:
    """Edit one filing's text snapshot and record its new hash in the lock."""
    lock = read_lock(root / "filings.lock.json")
    paths = SnapshotPaths(root / "snapshots")
    filings = []
    for filing in lock.filings:
        if (filing.ticker, filing.fiscal_year) == (ticker, year):
            text = edit(lzma.decompress(paths.text(filing).read_bytes()).decode())
            paths.text(filing).write_bytes(xz_bytes(text.encode()))
            filing = filing.model_copy(update={"text_sha256": sha256(text.encode())})
        filings.append(filing)
    write_lock(root / "filings.lock.json", lock.model_copy(update={"filings": tuple(filings)}))
