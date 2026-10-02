import json

import httpx
import pytest

from edgar_rag import cli
from edgar_rag.config import ServiceSettings
from edgar_rag.domain import EmbedderSpec
from edgar_rag.index import CorpusIndex

USER_AGENT = "Test Runner tests@ledgerworks.io"
SPEC = EmbedderSpec(model="nomic-embed-text", lowercase=True)
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
    for name in (
        "EDGAR_RAG_EDGAR_USER_AGENT",
        "EDGAR_RAG_INDEX_DIR",
        "EDGAR_RAG_PORT",
        "EDGAR_RAG_EMBEDDING_MODEL",
    ):
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
    index = CorpusIndex.load(tmp_path / "index", SPEC)
    [filing] = index.filings
    assert (filing.company, filing.cik, filing.fiscal_year) == ("Apple Inc.", 320193, 2024)
    assert index.chunk_count > 2
    assert seen[0] == "https://data.sec.gov/submissions/CIK0000320193.json"
    out = capsys.readouterr().out
    assert "Apple Inc. 10-K 2024-11-01, fiscal 2024" in out
    assert f"{index.chunk_count} chunks" in out


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


def test_ingest_adds_to_the_index_dir_given_on_the_command_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EDGAR_RAG_EDGAR_USER_AGENT", USER_AGENT)
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", str(tmp_path / "from-env"))
    transport, _ = _fake_sec_and_ollama()
    chosen = tmp_path / "chosen"

    first = cli.main(["ingest", "--cik", "320193", "--index-dir", str(chosen)], transport=transport)
    again = cli.main(["ingest", "--cik", "320193", "--index-dir", str(chosen)], transport=transport)

    assert (first, again) == (0, 0)
    assert len(CorpusIndex.load(chosen, SPEC).filings) == 1
    assert not (tmp_path / "from-env").exists()
    assert f"0000320193-24-000123 added to the index in {chosen}" in capsys.readouterr().out


def test_ingest_refuses_to_add_to_an_index_of_another_embedder(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EDGAR_RAG_EDGAR_USER_AGENT", USER_AGENT)
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("EDGAR_RAG_EMBEDDING_MODEL", "mxbai-embed-large")
    transport, _ = _fake_sec_and_ollama()
    assert cli.main(["ingest", "--cik", "320193"], transport=transport) == 0
    monkeypatch.delenv("EDGAR_RAG_EMBEDDING_MODEL")

    code = cli.main(["ingest", "--cik", "320193"], transport=transport)

    assert code == 1
    assert capsys.readouterr().err.startswith("index: the index in")


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
