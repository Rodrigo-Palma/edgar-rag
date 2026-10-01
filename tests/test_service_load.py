"""The service under load: what it reads, what it holds, what it turns away.

Security finding M5: the index was read from disk on every request, and a
request held a worker for as long as the models took, so a few slow requests
could stall the service. These tests pin the fixes with fakes and no network.
"""

import httpx
from fastapi.testclient import TestClient

from edgar_rag.api import create_app
from edgar_rag.config import ServiceSettings
from edgar_rag.index import FilingIndex
from tests.fakes import CITED_REPLY, ON_TOPIC


def _fake_ollama(request: httpx.Request) -> httpx.Response:
    """Ollama as the service talks to it: embed, generate, and its version."""
    if request.url.path == "/api/embed":
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
    if request.url.path == "/api/generate":
        return httpx.Response(200, json={"response": CITED_REPLY})
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.18.0"})
    return httpx.Response(404)


def _settings_over(directory) -> ServiceSettings:
    return ServiceSettings(index_dir=directory, min_retrieval_score=0.5)


def test_the_index_is_read_once_for_ten_requests(index, tmp_path, monkeypatch):
    """It used to be read on every /ask and every /health."""
    index.save(tmp_path)
    reads: list[object] = []
    load = FilingIndex.load

    def counting_load(cls, directory):
        reads.append(directory)
        return load(directory)

    monkeypatch.setattr(FilingIndex, "load", classmethod(counting_load))
    app = create_app(_settings_over(tmp_path), transport=httpx.MockTransport(_fake_ollama))

    with TestClient(app) as client:
        statuses = [client.post("/ask", json={"question": ON_TOPIC}).status_code for _ in range(5)]
        statuses += [client.get("/health").status_code for _ in range(5)]

    assert statuses == [200] * 10
    assert reads == [tmp_path]


def test_the_models_are_called_through_one_client_closed_at_shutdown(index, tmp_path):
    index.save(tmp_path)
    paths: list[str] = []

    def recording(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return _fake_ollama(request)

    app = create_app(_settings_over(tmp_path), transport=httpx.MockTransport(recording))

    with TestClient(app) as client:
        body = client.post("/ask", json={"question": ON_TOPIC}).json()
        opened = app.state.http
        assert opened.is_closed is False

    assert body["text"] == CITED_REPLY
    assert paths == ["/api/embed", "/api/generate"]
    assert opened.is_closed is True
