"""Download a filing from SEC EDGAR."""

from dataclasses import dataclass
from typing import Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
REQUEST_TIMEOUT_SECONDS = 30.0


class EdgarError(RuntimeError):
    """Raised when EDGAR cannot serve the filing we asked for."""


@dataclass(frozen=True, slots=True)
class Filing:
    cik: int
    company: str
    form: str
    filing_date: str
    accession: str
    document_url: str
    html: str


class _RecentFilings(BaseModel):
    """EDGAR's recent filings: parallel arrays, newest first, one entry per filing."""

    model_config = ConfigDict(strict=True)

    forms: list[str] = Field(default_factory=list, alias="form")
    accessions: list[str] = Field(default_factory=list, alias="accessionNumber")
    documents: list[str] = Field(default_factory=list, alias="primaryDocument")
    filing_dates: list[str] = Field(default_factory=list, alias="filingDate")

    @model_validator(mode="after")
    def _arrays_are_parallel(self) -> Self:
        # Arrays of different lengths would pair a form with another filing's document
        lengths = {len(self.forms), len(self.accessions), len(self.documents)}
        if len(lengths | {len(self.filing_dates)}) > 1:
            raise ValueError("the recent filings arrays differ in length")
        return self


class _Filings(BaseModel):
    model_config = ConfigDict(strict=True)

    recent: _RecentFilings = Field(default_factory=_RecentFilings)


class _Submissions(BaseModel):
    """The part of ``CIK##########.json`` this client reads."""

    model_config = ConfigDict(strict=True)

    name: str = ""
    filings: _Filings = Field(default_factory=_Filings)


def _headers(user_agent: str) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _parse_submissions(content: bytes, cik: int) -> _Submissions:
    """Validate the submissions payload before any field of it is trusted.

    Raises:
        EdgarError: when the payload is not the shape EDGAR documents.
    """
    try:
        return _Submissions.model_validate_json(content)
    except ValidationError as error:
        raise EdgarError(f"the submissions of CIK {cik} are malformed: {error}") from error


def _pick_filing(submissions: _Submissions, form: str) -> tuple[str, str, str]:
    """Return (accession, primary document, filing date) of the newest ``form``."""
    recent = submissions.filings.recent
    if not recent.forms:
        raise EdgarError("submissions payload carries no recent filings")

    for position, filed_form in enumerate(recent.forms):
        if filed_form != form:
            continue
        return (
            recent.accessions[position].replace("-", ""),
            recent.documents[position],
            recent.filing_dates[position],
        )
    raise EdgarError(f"no {form} among the recent filings")


def fetch_latest_filing(
    cik: int,
    user_agent: str,
    form: str = "10-K",
    *,
    client: httpx.Client | None = None,
) -> Filing:
    """Fetch the most recent ``form`` filed by ``cik``.

    A caller may pass its own ``client``, which is how the tests run this
    without touching the network.

    Raises:
        EdgarError: when EDGAR refuses the request or files no such form.
    """
    owned = client is None
    client = client or httpx.Client(headers=_headers(user_agent), timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        try:
            response = client.get(SUBMISSIONS_URL.format(cik=cik)).raise_for_status()
        except httpx.HTTPError as error:
            raise EdgarError(f"could not read the submissions of CIK {cik}: {error}") from error
        submissions = _parse_submissions(response.content, cik)

        accession, document, filing_date = _pick_filing(submissions, form)
        url = ARCHIVE_URL.format(cik=cik, accession=accession, document=document)
        try:
            html = client.get(url).raise_for_status().text
        except httpx.HTTPError as error:
            raise EdgarError(f"could not download {url}: {error}") from error
    finally:
        if owned:
            client.close()

    return Filing(
        cik=cik,
        company=submissions.name,
        form=form,
        filing_date=filing_date,
        accession=accession,
        document_url=url,
        html=html,
    )
