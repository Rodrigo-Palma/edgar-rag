import json

import httpx
import pytest

from edgar_rag import cli
from edgar_rag.config import ServiceSettings
from edgar_rag.index import FilingIndex

USER_AGENT = "Test Runner tests@ledgerworks.io"
SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["10-K"],
            "accessionNumber": ["0000320193-24-000123"],
            "primaryDocument": ["aapl-20240928.htm"],
            "filingDate": ["2024-11-01"],
            "reportDate": ["2024-09-28"],
        },
        "files": [],
    },
}
FILING_HTML = """
<html><body>
<p>Item 1. Business</p><p>{business}</p>
<p>Item 1A. Risk Factors</p><p>{risks}</p>
</body></html>
""".format(business="The Company designs phones. " * 60, risks="Supply chains may fail. " * 60)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """No .env from the working tree and no variable from the shell."""
    monkeypatch.chdir(tmp_path)
    for name in ("EDGAR_RAG_EDGAR_USER_AGENT", "EDGAR_RAG_INDEX_DIR", "EDGAR_RAG_PORT"):
        monkeypatch.delenv(name, raising=False)


def _fake_sec_and_ollama(*, edgar_status: int = 200, ollama_status: int = 200):
    """One transport standing in for EDGAR and for Ollama, routed by host."""
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "data.sec.gov":
            return httpx.Response(edgar_status, json=SUBMISSIONS)
        if request.url.host == "www.sec.gov":
            return httpx.Response(edgar_status, text=FILING_HTML)
        if request.url.path == "/api/embed":
            texts = json.loads(request.content)["input"]
            vectors = [[float(position + 1), 1.0] for position, _ in enumerate(texts)]
            return httpx.Response(ollama_status, json={"embeddings": vectors})
        raise AssertionError(f"unexpected request to {request.url}")

    return httpx.MockTransport(handle), seen


def test_the_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    assert raised.value.code == 0
    out = capsys.readouterr().out
    for command in ("ingest", "serve", "eval"):
        assert command in out


def test_the_eval_help_lists_its_commands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["eval", "--help"])

    assert "power" in capsys.readouterr().out


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main([])

    assert raised.value.code == 2


def test_eval_power_prints_the_report(capsys):
    code = cli.main(
        ["eval", "power", "--replicates", "20", "--resamples", "200", "--gate-per-class", "400"]
    )

    assert code == 0
    assert "| Comparison | Design | Result |" in capsys.readouterr().out


def test_ingest_downloads_embeds_and_saves_the_latest_filing(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EDGAR_RAG_EDGAR_USER_AGENT", USER_AGENT)
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", str(tmp_path / "index"))
    transport, seen = _fake_sec_and_ollama()

    code = cli.main(["ingest", "--cik", "320193"], transport=transport)

    assert code == 0
    index = FilingIndex.load(tmp_path / "index")
    assert index.source["company"] == "Apple Inc."
    assert len(index.chunks) > 2
    assert seen[0] == "https://data.sec.gov/submissions/CIK0000320193.json"
    out = capsys.readouterr().out
    assert "Apple Inc. 10-K 2024-11-01" in out
    assert f"{len(index.chunks)} chunks" in out


def test_ingest_without_a_declared_sender_stops_before_any_request(capsys):
    transport, seen = _fake_sec_and_ollama()

    code = cli.main(["ingest", "--cik", "320193"], transport=transport)

    assert code == 1
    assert seen == []
    assert "edgar_user_agent" in capsys.readouterr().err


def test_ingest_reports_an_edgar_failure_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("EDGAR_RAG_EDGAR_USER_AGENT", USER_AGENT)
    transport, _ = _fake_sec_and_ollama(edgar_status=404)

    code = cli.main(["ingest", "--cik", "320193"], transport=transport)

    assert code == 1
    assert capsys.readouterr().err.startswith("EDGAR:")


def test_ingest_reports_an_embedding_failure_and_writes_no_index(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EDGAR_RAG_EDGAR_USER_AGENT", USER_AGENT)
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", str(tmp_path / "index"))
    transport, _ = _fake_sec_and_ollama(ollama_status=500)

    code = cli.main(["ingest", "--cik", "320193"], transport=transport)

    assert code == 1
    assert capsys.readouterr().err.startswith("embedding:")
    assert not (tmp_path / "index").exists()


def test_serve_runs_the_service_with_the_settings_from_the_environment(monkeypatch):
    monkeypatch.setenv("EDGAR_RAG_PORT", "8123")
    served: list[ServiceSettings] = []
    monkeypatch.setattr(cli, "serve", served.append)

    assert cli.main(["serve"]) == 0
    assert [settings.port for settings in served] == [8123]


def test_serve_refuses_invalid_settings_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("EDGAR_RAG_PORT", "0")
    monkeypatch.setattr(cli, "serve", lambda settings: pytest.fail("served bad settings"))

    assert cli.main(["serve"]) == 1
    assert "port" in capsys.readouterr().err
