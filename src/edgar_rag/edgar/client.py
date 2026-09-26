"""Download a filing from SEC EDGAR."""

from dataclasses import dataclass

import httpx

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


def _headers(user_agent: str) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _pick_filing(submissions: dict, form: str) -> tuple[str, str, str]:
    """Return (accession, primary document, filing date) of the newest ``form``.

    EDGAR returns the recent filings as parallel arrays, newest first.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    if not forms:
        raise EdgarError("submissions payload carries no recent filings")

    for position, filed_form in enumerate(forms):
        if filed_form != form:
            continue
        return (
            recent["accessionNumber"][position].replace("-", ""),
            recent["primaryDocument"][position],
            recent["filingDate"][position],
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
            submissions = client.get(SUBMISSIONS_URL.format(cik=cik)).raise_for_status().json()
        except httpx.HTTPError as error:
            raise EdgarError(f"could not read the submissions of CIK {cik}: {error}") from error

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
        company=submissions.get("name", ""),
        form=form,
        filing_date=filing_date,
        accession=accession,
        document_url=url,
        html=html,
    )
