"""What EDGAR says a company filed, validated before any of it builds a URL.

``CIK##########.json`` lists the newest filings as parallel arrays and points at
older pages for the rest. A large filer moves a year-old 10-K into those pages
(JPMorgan files thousands of documents a year), so a pinned filing may need one.

The accession and the document name end up in a download URL and in a cache
path, so each is checked against the shape EDGAR uses before it is trusted. Only
the rows a caller asks for are checked: EDGAR's own rows for other forms carry
paths such as ``xslF345X06/form4.xml`` that a 10-K never does.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from edgar_rag.edgar.errors import EdgarError

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{document}"
MAX_CIK = 9_999_999_999
ACCESSION_PATTERN = re.compile(r"(\d{10})-?(\d{2})-?(\d{6})", re.ASCII)
DOCUMENT_PATTERN = re.compile(r"(?!.*\.\.)[\w\-][\w.\-]*", re.ASCII)
PAGE_NAME_PATTERN = re.compile(r"CIK\d{10}-submissions-\d{3}\.json", re.ASCII)


def validate_cik(cik: int) -> int:
    """Return ``cik`` if it can be a company's Central Index Key.

    Raises:
        EdgarError: for a non-integer, zero, negative or over-long key.
    """
    if isinstance(cik, bool) or not isinstance(cik, int) or not 0 < cik <= MAX_CIK:
        raise EdgarError(f"{cik!r} is not a CIK: expected an integer from 1 to {MAX_CIK}")
    return cik


def parse_accession(accession: str) -> str:
    """Return the accession number in EDGAR's dashed form, ``0000320193-25-000079``.

    Both the dashed form and the 18 bare digits are accepted.

    Raises:
        EdgarError: for anything else.
    """
    match = ACCESSION_PATTERN.fullmatch(accession)
    if match is None or accession.count("-") not in (0, 2):
        raise EdgarError(f"{accession!r} is not an accession number")
    return "-".join(match.groups())


def validate_document(document: str) -> str:
    """Return ``document`` if it is a plain file name inside a filing's folder.

    Raises:
        EdgarError: for a path, a parent reference or a hidden name.
    """
    if DOCUMENT_PATTERN.fullmatch(document) is None:
        raise EdgarError(f"{document!r} is not a document name inside a filing")
    return document


@dataclass(frozen=True, slots=True)
class FilingRef:
    """One filing, pinned: enough to download exactly this document again.

    ``report_date`` is the end of the period the filing reports on, which is
    what ties an XBRL fact to a fiscal year; ``None`` for forms with no period.
    """

    cik: int
    accession: str
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str

    def __post_init__(self) -> None:
        validate_cik(self.cik)
        if parse_accession(self.accession) != self.accession:
            raise EdgarError(f"{self.accession!r} is not in the dashed accession form")
        validate_document(self.primary_document)

    @property
    def folder(self) -> str:
        """The accession without dashes, as the archive path spells it."""
        return self.accession.replace("-", "")

    @property
    def document_url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, folder=self.folder, document=self.primary_document)


@dataclass(frozen=True, slots=True)
class SubmissionsPage:
    """An older page of a company's filings, covering ``filing_from`` to ``filing_to``."""

    name: str
    filing_from: date
    filing_to: date


class _FilingRows(BaseModel):
    """EDGAR's filings as parallel arrays, newest first, one entry per filing."""

    model_config = ConfigDict(strict=True)

    forms: list[str] = Field(default_factory=list, alias="form")
    accessions: list[str] = Field(default_factory=list, alias="accessionNumber")
    documents: list[str] = Field(default_factory=list, alias="primaryDocument")
    filing_dates: list[str] = Field(default_factory=list, alias="filingDate")
    report_dates: list[str] = Field(default_factory=list, alias="reportDate")

    @model_validator(mode="after")
    def _arrays_are_parallel(self) -> Self:
        # Arrays of different lengths would pair a form with another filing's document
        lengths = {len(self.forms), len(self.accessions), len(self.documents)}
        if len(lengths | {len(self.filing_dates), len(self.report_dates)}) > 1:
            raise ValueError("the recent filings arrays differ in length")
        return self


class _Page(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    filing_from: str = Field(alias="filingFrom")
    filing_to: str = Field(alias="filingTo")


class _Filings(BaseModel):
    model_config = ConfigDict(strict=True)

    recent: _FilingRows = Field(default_factory=_FilingRows)
    files: list[_Page] = Field(default_factory=list)


class _Submissions(BaseModel):
    """The part of ``CIK##########.json`` this client reads."""

    model_config = ConfigDict(strict=True)

    name: str = ""
    filings: _Filings = Field(default_factory=_Filings)


@dataclass(frozen=True, slots=True)
class Submissions:
    """A company's filings as EDGAR lists them, newest first.

    ``older_pages`` is empty for a page fetched with ``EdgarClient.submissions_page``.
    """

    cik: int
    company: str
    older_pages: tuple[SubmissionsPage, ...]
    _rows: _FilingRows

    def filings(self, form: str) -> tuple[FilingRef, ...]:
        """Every filing of ``form`` on this page, newest first, each validated.

        Raises:
            EdgarError: when a row of that form has a malformed accession,
                document name or date.
        """
        rows = self._rows
        return tuple(
            self._ref(position) for position, filed in enumerate(rows.forms) if filed == form
        )

    def filing(self, accession: str) -> FilingRef:
        """The filing with ``accession``, in either form.

        Raises:
            EdgarError: when it is not on this page or its row is malformed.
        """
        wanted = parse_accession(accession)
        for position, listed in enumerate(self._rows.accessions):
            if listed == wanted:
                return self._ref(position)
        raise EdgarError(f"accession {wanted} is not among these filings of CIK {self.cik}")

    def _ref(self, position: int) -> FilingRef:
        rows = self._rows
        return FilingRef(
            cik=self.cik,
            accession=parse_accession(rows.accessions[position]),
            form=rows.forms[position],
            filing_date=_parse_date(rows.filing_dates[position]),
            report_date=_parse_date(rows.report_dates[position])
            if rows.report_dates[position]
            else None,
            primary_document=validate_document(rows.documents[position]),
        )


def parse_submissions(content: bytes, cik: int) -> Submissions:
    """Validate ``CIK##########.json`` before any field of it is trusted.

    Raises:
        EdgarError: when the payload is not the shape EDGAR documents.
    """
    try:
        payload = _Submissions.model_validate_json(content)
    except ValidationError as error:
        raise EdgarError(f"the submissions of CIK {cik} are malformed: {error}") from error
    pages = tuple(_page(page, cik) for page in payload.filings.files)
    return Submissions(cik, payload.name, pages, payload.filings.recent)


def parse_submissions_page(content: bytes, parent: Submissions) -> Submissions:
    """Validate one older page; it has the recent filings' shape at its top level.

    Raises:
        EdgarError: when the payload is not that shape.
    """
    try:
        rows = _FilingRows.model_validate_json(content)
    except ValidationError as error:
        raise EdgarError(f"a submissions page of CIK {parent.cik} is malformed: {error}") from error
    return Submissions(parent.cik, parent.company, (), rows)


def validate_page_name(name: str, cik: int) -> str:
    """Return ``name`` if it is an older page of ``cik``'s submissions.

    Raises:
        EdgarError: for any other file name.
    """
    if PAGE_NAME_PATTERN.fullmatch(name) is None or not name.startswith(f"CIK{cik:010d}-"):
        raise EdgarError(f"{name!r} is not a submissions page of CIK {cik}")
    return name


def _page(page: _Page, cik: int) -> SubmissionsPage:
    return SubmissionsPage(
        name=validate_page_name(page.name, cik),
        filing_from=_parse_date(page.filing_from),
        filing_to=_parse_date(page.filing_to),
    )


def _parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise EdgarError(f"{text!r} is not a date") from error
