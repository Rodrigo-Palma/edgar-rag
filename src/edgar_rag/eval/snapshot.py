"""The roster, the lock and the snapshots the golden set is built from, offline.

``eval/companies.toml`` names the companies; ``acquire`` downloads what they
filed once and writes:

* ``eval/filings.lock.json``: every pinned filing by accession, with the
  SHA-256 of the document as downloaded and of the text kept from it, and every
  company excluded by the roster rule;
* ``eval/snapshots/``: the SEC's ticker map cut to the roster, each company's
  ``companyfacts`` cut to the concepts the questions use (plus the names of
  every concept it ever reported), and each filing's parsed text.

The build reads only these files, so it runs without the network, and it
checks every snapshot against the lock before reading it.

Compression is deterministic (gzip without a timestamp, xz without one by
design), so the same content always gives the same bytes.
"""

import gzip
import hashlib
import io
import json
import lzma
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from edgar_rag.eval.golden import Split

LOCK_FORMAT_VERSION = 1
FORM = "10-K"


@dataclass(frozen=True, slots=True)
class CompanySpec:
    ticker: str
    cik: int
    name: str
    split: Split | None  # None for a reserve, which takes the split it fills


@dataclass(frozen=True, slots=True)
class Roster:
    fiscal_years: tuple[int, ...]
    fold_seed: int
    sample_seed: int
    companies: tuple[CompanySpec, ...]
    reserves: tuple[CompanySpec, ...]

    def everyone(self) -> tuple[CompanySpec, ...]:
        return self.companies + self.reserves


def load_roster(path: Path) -> Roster:
    """Read ``companies.toml``.

    Raises:
        ValueError: on a missing key, a repeated ticker or an unknown split.
    """
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        companies = tuple(_spec(entry, entry["split"]) for entry in payload["company"])
        reserves = tuple(_spec(entry, None) for entry in payload.get("reserve", []))
        roster = Roster(
            fiscal_years=tuple(int(year) for year in payload["fiscal_years"]),
            fold_seed=int(payload["fold_seed"]),
            sample_seed=int(payload["sample_seed"]),
            companies=companies,
            reserves=reserves,
        )
    except KeyError as error:
        raise ValueError(f"{path.name} is missing {error}") from error
    tickers = [spec.ticker for spec in roster.everyone()]
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"{path.name} lists a ticker twice")
    return roster


def _spec(entry: dict[str, Any], split: object) -> CompanySpec:
    if split not in ("dev", "eval", None):
        raise ValueError(f"{entry.get('ticker')!r} has an unknown split {split!r}")
    return CompanySpec(
        ticker=str(entry["ticker"]),
        cik=int(entry["cik"]),
        name=str(entry["name"]),
        split=split,  # checked against the literal above
    )


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LockedFiling(_Frozen):
    ticker: str
    cik: int = Field(gt=0)
    fiscal_year: int
    accession: str
    form: str
    filing_date: date
    report_date: date
    primary_document: str
    url: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_bytes: int = Field(gt=0)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_chars: int = Field(gt=0)


class LockedCompany(_Frozen):
    ticker: str
    cik: int = Field(gt=0)
    name: str
    split: Split
    entity_name: str
    companyfacts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Exclusion(_Frozen):
    ticker: str
    cik: int
    split: Split
    reason: str
    replaced_by: str | None


class FilingsLock(_Frozen):
    format_version: int = LOCK_FORMAT_VERSION
    fiscal_years: tuple[int, ...]
    company_tickers_sha256: str
    companies: tuple[LockedCompany, ...]
    filings: tuple[LockedFiling, ...]
    excluded: tuple[Exclusion, ...] = ()

    def filing(self, ticker: str, fiscal_year: int) -> LockedFiling:
        for filing in self.filings:
            if filing.ticker == ticker and filing.fiscal_year == fiscal_year:
                return filing
        raise KeyError(f"no locked {FORM} of {ticker} for fiscal {fiscal_year}")

    def company(self, ticker: str) -> LockedCompany:
        for company in self.companies:
            if company.ticker == ticker:
                return company
        raise KeyError(f"{ticker} is not a locked company")


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode()


def write_lock(path: Path, lock: FilingsLock) -> None:
    path.write_bytes(canonical_json(lock.model_dump(mode="json")))


def read_lock(path: Path) -> FilingsLock:
    return FilingsLock.model_validate_json(path.read_bytes())


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class SnapshotPaths:
    root: Path

    @property
    def tickers(self) -> Path:
        return self.root / "company_tickers.json"

    def companyfacts(self, ticker: str) -> Path:
        return self.root / "companyfacts" / f"{ticker}.json.gz"

    def text(self, filing: LockedFiling) -> Path:
        name = f"{filing.ticker}-{filing.fiscal_year}-{filing.accession}.txt.xz"
        return self.root / "filings" / name


def gzip_bytes(content: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0, filename="") as out:
        out.write(content)
    return buffer.getvalue()


def xz_bytes(content: bytes) -> bytes:
    return lzma.compress(content, format=lzma.FORMAT_XZ, preset=9)


def trim_companyfacts(raw: bytes, tags: Iterable[str]) -> bytes:
    """Cut ``companyfacts`` to the 10-K facts of ``tags``, keeping its shape.

    ``usGaapConcepts`` lists every us-gaap concept the company ever reported,
    which is what an "unreported concept" question is checked against.
    """
    payload = json.loads(raw)
    us_gaap: dict[str, Any] = payload.get("facts", {}).get("us-gaap", {})
    wanted = set(tags)
    kept: dict[str, Any] = {}
    for concept in sorted(wanted & set(us_gaap)):
        units = {
            unit: [fact for fact in facts if fact.get("form") == FORM]
            for unit, facts in us_gaap[concept].get("units", {}).items()
        }
        kept[concept] = {"units": {unit: facts for unit, facts in units.items() if facts}}
    trimmed = {
        "cik": payload["cik"],
        "entityName": payload["entityName"],
        "facts": {"us-gaap": kept},
        "usGaapConcepts": sorted(us_gaap),
    }
    return canonical_json(trimmed)


def read_text_snapshot(paths: SnapshotPaths, filing: LockedFiling) -> str:
    """The filing's parsed text, checked against the lock.

    Raises:
        ValueError: when the snapshot is not the text the lock recorded.
    """
    content = _decompress(paths.text(filing), lzma.decompress)
    if sha256(content) != filing.text_sha256:
        raise ValueError(f"{paths.text(filing).name} does not match its SHA-256 in the lock")
    return content.decode("utf-8")


def read_companyfacts_snapshot(paths: SnapshotPaths, company: LockedCompany) -> bytes:
    """The trimmed ``companyfacts`` JSON, checked against the lock.

    Raises:
        ValueError: when the snapshot is not the one the lock recorded.
    """
    content = _decompress(paths.companyfacts(company.ticker), gzip.decompress)
    if sha256(content) != company.snapshot_sha256:
        raise ValueError(f"the companyfacts snapshot of {company.ticker} does not match the lock")
    return content


def _decompress(path: Path, decompress: Callable[[bytes], bytes]) -> bytes:
    """Read and decompress ``path``; a corrupt or truncated file is a ``ValueError``."""
    try:
        return decompress(path.read_bytes())
    except (OSError, EOFError, lzma.LZMAError) as error:
        raise ValueError(f"{path.name} cannot be read: {error}") from error


def reported_concepts(trimmed: bytes) -> frozenset[str]:
    return frozenset(json.loads(trimmed).get("usGaapConcepts", []))


def read_tickers(paths: SnapshotPaths) -> dict[str, int]:
    """Ticker to CIK, from the snapshot of ``company_tickers.json``."""
    payload = json.loads(paths.tickers.read_bytes())
    return {ticker: int(entry["cik"]) for ticker, entry in payload.items()}


def check_roster_ciks(roster: Roster, tickers: dict[str, int]) -> None:
    """Fail when a listed CIK differs from what the SEC maps its ticker to.

    Raises:
        ValueError: naming every divergent or unknown ticker.
    """
    problems = [
        f"{spec.ticker}: listed {spec.cik}, SEC maps {tickers.get(spec.ticker)}"
        for spec in roster.everyone()
        if tickers.get(spec.ticker) != spec.cik
    ]
    if problems:
        raise ValueError("CIKs differ from company_tickers.json: " + "; ".join(problems))
