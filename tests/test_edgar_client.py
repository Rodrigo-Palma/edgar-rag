import httpx
import pytest

from edgar_rag.edgar.client import EdgarError, fetch_latest_filing

SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["8-K", "10-K", "10-Q"],
            "accessionNumber": ["0000-00-000001", "0000320193-25-000079", "0000-00-000003"],
            "primaryDocument": ["x.htm", "aapl-20250927.htm", "y.htm"],
            "filingDate": ["2025-11-01", "2025-10-31", "2025-08-01"],
        }
    },
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_downloads_the_newest_filing_of_the_requested_form():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "submissions" in str(request.url):
            return httpx.Response(200, json=SUBMISSIONS)
        return httpx.Response(200, text="<html>filing</html>")

    filing = fetch_latest_filing(320193, "Tester test@example.com", client=_client(handler))

    assert filing.company == "Apple Inc."
    assert filing.form == "10-K"
    assert filing.filing_date == "2025-10-31"
    # The accession loses its dashes in the archive path, and the 8-K is skipped
    assert filing.document_url.endswith("/000032019325000079/aapl-20250927.htm")
    assert filing.html == "<html>filing</html>"


def test_a_form_the_company_never_filed_is_an_edgar_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SUBMISSIONS)

    with pytest.raises(EdgarError, match="no S-1"):
        fetch_latest_filing(320193, "Tester test@example.com", "S-1", client=_client(handler))


def test_an_empty_submissions_payload_is_an_edgar_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "Empty", "filings": {"recent": {}}})

    with pytest.raises(EdgarError, match="no recent filings"):
        fetch_latest_filing(1, "Tester test@example.com", client=_client(handler))


def test_a_rejected_request_is_reported_with_the_cik():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="denied")

    with pytest.raises(EdgarError, match="CIK 320193"):
        fetch_latest_filing(320193, "Tester test@example.com", client=_client(handler))


def test_a_missing_document_names_the_url_that_failed():
    def handler(request: httpx.Request) -> httpx.Response:
        if "submissions" in str(request.url):
            return httpx.Response(200, json=SUBMISSIONS)
        return httpx.Response(404, text="gone")

    with pytest.raises(EdgarError, match="aapl-20250927.htm"):
        fetch_latest_filing(320193, "Tester test@example.com", client=_client(handler))


def _recent(**overrides) -> dict:
    recent = {**SUBMISSIONS["filings"]["recent"], **overrides}
    return {"name": "Apple Inc.", "filings": {"recent": recent}}


MALFORMED_SUBMISSIONS = {
    "accessions-shorter-than-forms": _recent(accessionNumber=["0000-00-000001"]),
    "documents-missing": {
        "name": "Apple Inc.",
        "filings": {"recent": {"form": ["10-K"], "accessionNumber": ["0000320193-25-000079"]}},
    },
    "forms-not-a-list": _recent(form="10-K"),
    "accession-not-text": _recent(accessionNumber=[1, 2, 3]),
    "recent-not-an-object": {"name": "Apple Inc.", "filings": {"recent": ["10-K"]}},
    "name-not-text": {**SUBMISSIONS, "name": ["Apple"]},
}


@pytest.mark.parametrize(
    "payload", MALFORMED_SUBMISSIONS.values(), ids=MALFORMED_SUBMISSIONS.keys()
)
def test_malformed_submissions_are_an_edgar_error_before_any_download(payload):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=payload)

    with pytest.raises(EdgarError, match="CIK 320193"):
        fetch_latest_filing(320193, "Tester test@example.com", client=_client(handler))
    assert len(seen) == 1


@pytest.mark.parametrize(
    "content", [b"<html>rate limited</html>", b'["not", "an", "object"]', b"null"]
)
def test_submissions_that_are_not_a_json_object_are_an_edgar_error(content):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    with pytest.raises(EdgarError, match="CIK 320193"):
        fetch_latest_filing(320193, "Tester test@example.com", client=_client(handler))
