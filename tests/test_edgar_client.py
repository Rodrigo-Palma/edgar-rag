import itertools
from datetime import date
from pathlib import Path

import httpx
import pytest

from edgar_rag.edgar.client import EdgarClient, EdgarError, decode_document, fetch_latest_filing
from edgar_rag.edgar.fetch import MIN_REQUEST_INTERVAL_SECONDS
from edgar_rag.edgar.submissions import FilingRef, SubmissionsPage, parse_accession
from tests.edgar_fakes import FakeClock, RecordingTransport

USER_AGENT = "Test Runner tests@ledgerworks.io"
DATA = Path(__file__).parent / "data"

SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["4", "10-K", "10-Q", "10-K"],
            "accessionNumber": [
                "0000320193-25-000090",
                "0000320193-25-000079",
                "0000320193-25-000073",
                "0000320193-24-000123",
            ],
            "primaryDocument": [
                "xslF345X06/form4.xml",
                "aapl-20250927.htm",
                "aapl-20250628.htm",
                "aapl-20240928.htm",
            ],
            "filingDate": ["2025-11-02", "2025-10-31", "2025-08-01", "2024-11-01"],
            "reportDate": ["", "2025-09-27", "2025-06-28", "2024-09-28"],
        },
        "files": [
            {
                "name": "CIK0000320193-submissions-001.json",
                "filingCount": 1256,
                "filingFrom": "1994-01-26",
                "filingTo": "2015-08-23",
            }
        ],
    },
}

OLDER_PAGE = {
    "form": ["10-K"],
    "accessionNumber": ["0001193125-15-356351"],
    "primaryDocument": ["d17062d10k.htm"],
    "filingDate": ["2015-10-28"],
    "reportDate": ["2015-09-26"],
}

PINNED = FilingRef(
    cik=320193,
    accession="0000320193-24-000123",
    form="10-K",
    filing_date=date(2024, 11, 1),
    report_date=date(2024, 9, 28),
    primary_document="aapl-20240928.htm",
)


class FakeSec:
    """Serves submissions, one older page, the fixture company facts and one HTML body."""

    def __init__(self, html: bytes = b"<html>filing</html>") -> None:
        self.html = html

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/submissions/CIK0000320193.json"):
            return httpx.Response(200, json=SUBMISSIONS)
        if url.endswith("/submissions/CIK0000320193-submissions-001.json"):
            return httpx.Response(200, json=OLDER_PAGE)
        if "/api/xbrl/companyfacts/" in url:
            content = (DATA / "companyfacts-aapl-trimmed.json").read_bytes()
            return httpx.Response(200, content=content)
        if "/Archives/" in url:
            return httpx.Response(200, content=self.html)
        return httpx.Response(404)


def _client(handler=None, **kwargs) -> tuple[EdgarClient, RecordingTransport]:
    clock = FakeClock()
    transport = RecordingTransport(handler or FakeSec(), clock)
    client = EdgarClient(USER_AGENT, http=httpx.Client(transport=transport), clock=clock, **kwargs)
    return client, transport


def test_downloads_the_newest_filing_of_the_requested_form():
    client, transport = _client()

    filing = client.latest_filing(320193)

    assert filing.company == "Apple Inc."
    assert filing.form == "10-K"
    assert filing.filing_date == "2025-10-31"
    assert filing.ref.report_date == date(2025, 9, 27)
    # The accession loses its dashes in the archive path, and the Form 4 is skipped
    assert filing.document_url.endswith("/320193/000032019325000079/aapl-20250927.htm")
    assert filing.html == "<html>filing</html>"
    assert all(request.headers["User-Agent"] == USER_AGENT for request in transport.requests)


def test_the_client_refuses_to_start_without_a_declared_sender():
    with pytest.raises(ValueError, match="placeholder"):
        EdgarClient("Your Name your.email@example.com")


def test_a_form_the_company_never_filed_is_an_edgar_error():
    client, _ = _client()

    with pytest.raises(EdgarError, match="no S-1"):
        client.latest_filing(320193, "S-1")


def test_an_empty_submissions_payload_is_an_edgar_error():
    client, _ = _client(lambda request: httpx.Response(200, json={"name": "Empty"}))

    with pytest.raises(EdgarError, match="no 10-K"):
        client.latest_filing(320193)


def test_a_rejected_request_is_reported_with_the_cik():
    client, _ = _client(lambda request: httpx.Response(403, text="denied"))

    with pytest.raises(EdgarError, match="CIK 320193"):
        client.latest_filing(320193)


def test_a_missing_document_names_the_url_that_failed():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/Archives/" in str(request.url):
            return httpx.Response(404, text="gone")
        return FakeSec()(request)

    client, _ = _client(handler)

    with pytest.raises(EdgarError, match="aapl-20250927.htm"):
        client.latest_filing(320193)


def test_consecutive_calls_share_one_pace():
    client, transport = _client()

    client.latest_filing(320193)
    client.companyfacts(320193)

    gaps = [later - earlier for earlier, later in itertools.pairwise(transport.times)]
    assert gaps == pytest.approx([MIN_REQUEST_INTERVAL_SECONDS] * 2)


def _recent(**overrides) -> dict:
    recent = {**SUBMISSIONS["filings"]["recent"], **overrides}
    return {"name": "Apple Inc.", "filings": {"recent": recent}}


def _documents(newest_10k: str) -> list[str]:
    return ["x.xml", newest_10k, "aapl-20250628.htm", "aapl-20240928.htm"]


MALFORMED_SUBMISSIONS = {
    "accessions-shorter-than-forms": _recent(accessionNumber=["0000320193-25-000090"]),
    "documents-missing": {
        "name": "Apple Inc.",
        "filings": {"recent": {"form": ["10-K"], "accessionNumber": ["0000320193-25-000079"]}},
    },
    "forms-not-a-list": _recent(form="10-K"),
    "accession-not-text": _recent(accessionNumber=[1, 2, 3, 4]),
    "recent-not-an-object": {"name": "Apple Inc.", "filings": {"recent": ["10-K"]}},
    "name-not-text": {**SUBMISSIONS, "name": ["Apple"]},
    # Adversarial case 21: values that would build another URL or cache path
    "document-climbs-out": _recent(primaryDocument=_documents("../../x")),
    "document-is-a-path": _recent(primaryDocument=_documents("a/b.htm")),
    "document-is-hidden": _recent(primaryDocument=_documents(".env")),
    "accession-not-a-number": _recent(
        accessionNumber=["0000320193-25-000090", "abc", "0000320193-25-000073", "1"]
    ),
    "accession-with-a-path": _recent(
        accessionNumber=["0000320193-25-000090", "../000079", "0000320193-25-000073", "1"]
    ),
    "report-date-not-a-date": _recent(reportDate=["", "soon", "2025-06-28", "2024-09-28"]),
    "older-page-elsewhere": {
        **SUBMISSIONS,
        "filings": {
            **SUBMISSIONS["filings"],
            "files": [{"name": "../x.json", "filingFrom": "2000-01-01", "filingTo": "2001-01-01"}],
        },
    },
}


@pytest.mark.parametrize(
    "payload", MALFORMED_SUBMISSIONS.values(), ids=MALFORMED_SUBMISSIONS.keys()
)
def test_malformed_submissions_are_an_edgar_error_before_any_download(payload):
    client, transport = _client(lambda request: httpx.Response(200, json=payload))

    with pytest.raises(EdgarError):
        client.latest_filing(320193)
    assert len(transport.requests) == 1
    assert "/Archives/" not in transport.urls[0]


@pytest.mark.parametrize(
    "content", [b"<html>rate limited</html>", b'["not", "an", "object"]', b"null"]
)
def test_submissions_that_are_not_a_json_object_are_an_edgar_error(content):
    client, _ = _client(lambda request: httpx.Response(200, content=content))

    with pytest.raises(EdgarError, match="CIK 320193"):
        client.submissions(320193)


def test_rows_of_other_forms_are_not_held_to_the_10k_shape():
    # A Form 4's primary document is "xslF345X06/form4.xml": legitimate, and never fetched
    client, _ = _client()

    refs = client.submissions(320193).filings("10-K")

    assert [ref.accession for ref in refs] == ["0000320193-25-000079", "0000320193-24-000123"]


@pytest.mark.parametrize("cik", [0, -320193, 10**10, True])
def test_a_cik_that_cannot_exist_is_refused_before_any_request(cik):
    client, transport = _client()

    with pytest.raises(EdgarError, match="not a CIK"):
        client.submissions(cik)
    with pytest.raises(EdgarError, match="not a CIK"):
        client.companyfacts(cik)
    assert transport.requests == []


def test_a_pinned_filing_downloads_exactly_its_document():
    client, transport = _client()

    content = client.fetch_filing(PINNED)

    assert content == b"<html>filing</html>"
    assert transport.urls == [
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    ]


def test_a_filing_is_found_by_accession_in_either_form():
    client, _ = _client()
    submissions = client.submissions(320193)

    assert submissions.filing("000032019324000123") == PINNED
    assert submissions.filing("0000320193-24-000123") == PINNED
    with pytest.raises(EdgarError, match="not among"):
        submissions.filing("0000320193-20-000001")


def test_an_older_filing_is_read_from_the_page_that_lists_it():
    client, transport = _client()
    submissions = client.submissions(320193)

    assert submissions.older_pages == (
        SubmissionsPage("CIK0000320193-submissions-001.json", date(1994, 1, 26), date(2015, 8, 23)),
    )
    page = client.submissions_page(submissions, submissions.older_pages[0])

    assert page.company == "Apple Inc."
    assert page.older_pages == ()
    assert page.filings("10-K")[0].accession == "0001193125-15-356351"
    assert transport.urls[-1] == (
        "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
    )


def test_a_page_of_another_company_is_refused():
    client, transport = _client()
    submissions = client.submissions(320193)
    foreign = SubmissionsPage(
        "CIK0000019617-submissions-001.json", date(2025, 1, 1), date(2025, 2, 1)
    )

    with pytest.raises(EdgarError, match="not a submissions page of CIK 320193"):
        client.submissions_page(submissions, foreign)
    assert len(transport.requests) == 1


def test_a_malformed_page_is_an_edgar_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if "submissions-001" in str(request.url):
            return httpx.Response(200, json={"form": "10-K"})
        return FakeSec()(request)

    client, _ = _client(handler)
    submissions = client.submissions(320193)

    with pytest.raises(EdgarError, match="page of CIK 320193 is malformed"):
        client.submissions_page(submissions, submissions.older_pages[0])


@pytest.mark.parametrize(
    "fields",
    [
        {"cik": 0},
        {"accession": "000032019324000123"},  # bare digits are not the canonical form
        {"accession": "0000320193-24-00012"},
        {"primary_document": "../aapl.htm"},
        {"primary_document": "dir/aapl.htm"},
        {"primary_document": "a..htm"},
    ],
)
def test_a_pinned_reference_is_validated_when_built(fields):
    values = {
        "cik": PINNED.cik,
        "accession": PINNED.accession,
        "form": PINNED.form,
        "filing_date": PINNED.filing_date,
        "report_date": PINNED.report_date,
        "primary_document": PINNED.primary_document,
    }
    with pytest.raises(EdgarError):
        FilingRef(**{**values, **fields})


@pytest.mark.parametrize("accession", ["0000320193-24-000123", "000032019324000123"])
def test_an_accession_in_either_form_parses_to_the_dashed_one(accession):
    assert parse_accession(accession) == "0000320193-24-000123"


@pytest.mark.parametrize(
    "accession", ["0000320193-24000123", "0000320193-24-000123\n", "", "0000320193_24_000123"]
)
def test_an_accession_with_any_other_shape_is_refused(accession):
    with pytest.raises(EdgarError, match="not an accession"):
        parse_accession(accession)


def test_the_cache_serves_a_filing_a_second_time_without_a_request(tmp_path):
    client, transport = _client(cache_dir=tmp_path)

    first = client.fetch_filing(PINNED)
    second = client.fetch_filing(PINNED)

    assert first == second == b"<html>filing</html>"
    assert len(transport.requests) == 1
    cached = tmp_path / "filings" / "320193" / "000032019324000123" / "aapl-20240928.htm"
    assert cached.read_bytes() == first
    # No temporary file is left next to the document
    assert [path.name for path in cached.parent.iterdir()] == ["aapl-20240928.htm"]


def test_a_failed_download_leaves_nothing_in_the_cache(tmp_path):
    client, _ = _client(lambda request: httpx.Response(404), cache_dir=tmp_path)

    with pytest.raises(EdgarError):
        client.fetch_filing(PINNED)
    assert list(tmp_path.rglob("*")) == []


def test_a_failed_cache_write_removes_its_partial_file(tmp_path, monkeypatch):
    def fail(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr("edgar_rag.edgar.client.os.replace", fail)
    client, _ = _client(cache_dir=tmp_path)

    with pytest.raises(OSError, match="disk full"):
        client.fetch_filing(PINNED)
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


def test_a_filing_over_the_size_cap_is_an_edgar_error():
    client, _ = _client(FakeSec(html=b"x" * 2_000), max_download_bytes=1_000)

    with pytest.raises(EdgarError, match="cap"):
        client.fetch_filing(PINNED)


def test_company_facts_are_fetched_from_the_xbrl_api_and_typed():
    client, transport = _client()

    facts = client.companyfacts(320193)

    assert facts.entity_name == "Apple Inc."
    assert len(facts.facts) == 26
    assert transport.urls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]


def test_a_windows_1252_document_is_decoded():
    assert decode_document("Café’s".encode("cp1252")) == "Café’s"
    assert decode_document("Café".encode()) == "Café"


def test_the_one_shot_helper_closes_its_own_client(monkeypatch):
    closed: list[bool] = []

    def latest(self, cik, form):
        return f"{cik}:{form}"

    monkeypatch.setattr(EdgarClient, "latest_filing", latest)
    monkeypatch.setattr(EdgarClient, "close", lambda self: closed.append(True))

    assert fetch_latest_filing(320193, USER_AGENT, "10-Q") == "320193:10-Q"
    assert closed == [True]


def test_a_client_given_a_transport_leaves_it_open():
    transport = httpx.Client(transport=httpx.MockTransport(FakeSec()))

    with EdgarClient(USER_AGENT, http=transport, clock=FakeClock()):
        pass

    assert not transport.is_closed


def test_a_client_that_built_its_transport_closes_it():
    client = EdgarClient(USER_AGENT)
    client.close()

    assert client._http.is_closed
