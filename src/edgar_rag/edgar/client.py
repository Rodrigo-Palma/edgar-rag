"""Download filings and XBRL facts from SEC EDGAR.

``EdgarClient`` is the one way this project talks to the SEC. It refuses to
start without a declared User-Agent, paces and retries every GET (see
``fetch``), caps every download, and validates everything EDGAR returns before
it builds a URL or a path from it.

A filing is fetched by its ``FilingRef``, which pins the accession and the
document: the same reference downloads the same bytes next year, which is what
makes an evaluation built on it reproducible. Filing documents never change
once accepted, so they are cached on disk by accession; submissions and
company facts do change and are always fetched fresh.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx

from edgar_rag.edgar.errors import EdgarError
from edgar_rag.edgar.fetch import (
    MAX_DOWNLOAD_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    Clock,
    Fetcher,
    SystemClock,
)
from edgar_rag.edgar.submissions import (
    FilingRef,
    Submissions,
    SubmissionsPage,
    parse_submissions,
    parse_submissions_page,
    validate_cik,
    validate_page_name,
)
from edgar_rag.edgar.user_agent import validate_user_agent
from edgar_rag.edgar.xbrl import CompanyFacts, parse_companyfacts

__all__ = ["EdgarClient", "EdgarError", "Filing", "decode_document", "fetch_latest_filing"]

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


@dataclass(frozen=True, slots=True)
class Filing:
    """A filing's primary document, with the reference that pins it."""

    ref: FilingRef
    company: str
    html: str

    @property
    def form(self) -> str:
        return self.ref.form

    @property
    def filing_date(self) -> str:
        return self.ref.filing_date.isoformat()

    @property
    def document_url(self) -> str:
        return self.ref.document_url


class EdgarClient:
    """A fair-access client for the EDGAR endpoints this project reads.

    Pass ``http`` to supply the transport (the tests pass a mock one) and
    ``clock`` to control the pacing; the client closes only what it created.
    ``cache_dir`` turns on the on-disk cache of filing documents.

    Raises:
        ValueError: at construction, when ``user_agent`` does not declare a sender.
    """

    def __init__(
        self,
        user_agent: str,
        *,
        http: httpx.Client | None = None,
        clock: Clock | None = None,
        cache_dir: Path | None = None,
        max_download_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> None:
        headers = {
            "User-Agent": validate_user_agent(user_agent),
            "Accept-Encoding": "gzip, deflate",
        }
        self._owns_http = http is None
        self._http = http or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False)
        self._fetcher = Fetcher(self._http, headers, clock or SystemClock(), max_download_bytes)
        self._cache_dir = cache_dir

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def submissions(self, cik: int) -> Submissions:
        """The company's name, its recent filings and the pages holding older ones.

        Raises:
            EdgarError: when EDGAR fails or returns a malformed payload.
        """
        validate_cik(cik)
        try:
            content = self._fetcher.get(SUBMISSIONS_URL.format(cik=cik))
        except EdgarError as error:
            raise EdgarError(f"could not read the submissions of CIK {cik}: {error}") from error
        return parse_submissions(content, cik)

    def submissions_page(self, submissions: Submissions, page: SubmissionsPage) -> Submissions:
        """One older page of ``submissions``, as listed in its ``older_pages``.

        Raises:
            EdgarError: when EDGAR fails or returns a malformed page.
        """
        name = validate_page_name(page.name, submissions.cik)
        content = self._fetcher.get(SUBMISSIONS_PAGE_URL.format(name=name))
        return parse_submissions_page(content, submissions)

    def fetch_filing(self, ref: FilingRef) -> bytes:
        """The primary document of the pinned filing, from the cache when present.

        Raises:
            EdgarError: when EDGAR fails or the document passes the size cap.
        """
        cached = self._cache_path(ref)
        if cached is not None and cached.is_file():
            return cached.read_bytes()
        content = self._fetcher.get(ref.document_url)
        if cached is not None:
            _write_atomically(cached, content)
        return content

    def companyfacts_bytes(self, cik: int) -> bytes:
        """The raw ``companyfacts`` JSON, for a caller that keeps a snapshot of it.

        Raises:
            EdgarError: when EDGAR fails or the payload passes the size cap.
        """
        validate_cik(cik)
        return self._fetcher.get(COMPANYFACTS_URL.format(cik=cik))

    def companyfacts(self, cik: int) -> CompanyFacts:
        """Every XBRL fact the company reported, typed.

        Raises:
            EdgarError: when EDGAR fails or the payload is malformed.
        """
        return parse_companyfacts(self.companyfacts_bytes(cik))

    def latest_filing(self, cik: int, form: str = "10-K") -> Filing:
        """The newest ``form`` among the company's recent filings.

        Raises:
            EdgarError: when EDGAR fails or the company filed no such form recently.
        """
        submissions = self.submissions(cik)
        refs = submissions.filings(form)
        if not refs:
            raise EdgarError(f"no {form} among the recent filings of CIK {cik}")
        content = self.fetch_filing(refs[0])
        return Filing(ref=refs[0], company=submissions.company, html=decode_document(content))

    def _cache_path(self, ref: FilingRef) -> Path | None:
        if self._cache_dir is None:
            return None
        # Every component was validated by FilingRef, so none can leave the cache
        return self._cache_dir / "filings" / str(ref.cik) / ref.folder / ref.primary_document


def decode_document(content: bytes) -> str:
    """Decode a filing document; older filings are Windows-1252, not UTF-8."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def fetch_latest_filing(cik: int, user_agent: str, form: str = "10-K") -> Filing:
    """Fetch the most recent ``form`` filed by ``cik`` with a client of its own.

    Raises:
        EdgarError: when EDGAR refuses the request or files no such form.
        ValueError: when ``user_agent`` does not declare a sender.
    """
    with EdgarClient(user_agent) as client:
        return client.latest_filing(cik, form)


def _write_atomically(path: Path, content: bytes) -> None:
    """Write through a temporary file so a crash never leaves half a document cached."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
