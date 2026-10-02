"""The service under load: what it reads, what it holds, what it turns away.

Security finding M5: the index was read from disk on every request, and a
request held a worker for as long as the models took, so a few slow requests
could stall the service. These tests pin the fixes with fakes and no network.
"""

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import httpx
from fastapi.testclient import TestClient

from edgar_rag.config import ServiceSettings
from edgar_rag.gate import CosineGate
from edgar_rag.index import FilingIndex
from edgar_rag.service.app import create_app
from tests.fakes import CITED_REPLY, ON_TOPIC, fake_answerer

# Long enough that a passing run never comes near it, short enough that a
# request left hanging fails the test instead of the whole run.
PATIENCE_SECONDS = 5.0


class HeldGenerator:
    """A slow generator: every call waits until the test releases it.

    Counting the calls that got in, rather than sleeping, is what makes the
    concurrency tests deterministic.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self._entered = 0
        self._lock = threading.Lock()

    @property
    def entered(self) -> int:
        with self._lock:
            return self._entered

    def generate(self, prompt: str) -> str:
        with self._lock:
            self._entered += 1
        if not self.release.wait(PATIENCE_SECONDS):
            raise AssertionError("the test never released the generation")
        return CITED_REPLY


def _until(condition, what: str) -> None:
    deadline = time.monotonic() + PATIENCE_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"gave up waiting for {what}")
        time.sleep(0.01)


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


def test_four_simultaneous_requests_serve_two_and_turn_two_away_at_once(index):
    """Two generations at a time; the rest get 503 now rather than a queue."""
    generator = HeldGenerator()
    settings = ServiceSettings(max_concurrent_generations=2)
    app = create_app(settings, fake_answerer(index, generator=generator))

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=4) as pool:
        asks: list[Future[httpx.Response]] = [
            pool.submit(client.post, "/ask", json={"question": ON_TOPIC}) for _ in range(4)
        ]
        try:
            _until(lambda: sum(ask.done() for ask in asks) == 2, "two requests turned away")
            early = [ask for ask in asks if ask.done()]
            assert generator.entered == 2
        finally:
            generator.release.set()
        turned_away = [ask.result() for ask in early]
        served = [ask.result(timeout=PATIENCE_SECONDS) for ask in asks if ask not in early]

    assert sorted(r.status_code for r in [*turned_away, *served]) == [200, 200, 503, 503]
    assert {r.status_code for r in turned_away} == {503}
    assert all(r.headers["retry-after"].isdigit() for r in turned_away)
    assert generator.entered == 2


def test_a_slot_is_given_back_when_its_generation_ends(index):
    generator = HeldGenerator()
    generator.release.set()
    settings = ServiceSettings(max_concurrent_generations=1)

    with TestClient(create_app(settings, fake_answerer(index, generator=generator))) as client:
        statuses = [client.post("/ask", json={"question": ON_TOPIC}).status_code for _ in range(3)]

    assert statuses == [200, 200, 200]


def test_a_request_past_its_time_budget_is_answered_with_504(index):
    """The worker may still be generating; the client is not kept waiting for it."""
    generator = HeldGenerator()
    settings = ServiceSettings(request_timeout_seconds=0.2)
    app = create_app(settings, fake_answerer(index, generator=generator))

    with TestClient(app) as client:
        started = time.monotonic()
        try:
            response = client.post("/ask", json={"question": ON_TOPIC})
        finally:
            generator.release.set()
        waited = time.monotonic() - started

    assert response.status_code == 504
    assert waited < PATIENCE_SECONDS
    assert "took too long" in response.json()["detail"]


def _telemetry(caplog) -> list[dict[str, object]]:
    lines = [
        record.getMessage() for record in caplog.records if record.name == "edgar_rag.telemetry"
    ]
    return [json.loads(line) for line in lines]


def test_every_request_is_logged_as_one_line_of_json(index, caplog):
    """Stages, reason and degraded per request, the 4xx included (ADR 0012)."""
    app = create_app(
        ServiceSettings(min_retrieval_score=0.5),
        fake_answerer(index),
        transport=httpx.MockTransport(_fake_ollama),
    )

    with caplog.at_level(logging.INFO, logger="edgar_rag.telemetry"), TestClient(app) as client:
        client.post("/ask", json={"question": ON_TOPIC, "top_k": 2})
        client.post("/ask", json={"question": "hi"})
        client.get("/health")

    answered, rejected, health = _telemetry(caplog)
    assert answered["path"] == "/ask"
    assert answered["status"] == 200
    assert set(answered["stages"]) == {"embed", "search", "gate", "generate"}
    assert answered["abstained"] is False
    assert answered["reason"] is None
    assert answered["degraded"] is False
    assert answered["top_k"] == 2
    assert {"prompt_tokens", "completion_tokens", "seconds"} <= set(answered)
    assert rejected["status"] == 422
    assert rejected["stages"] == {}
    assert health["path"] == "/health"


def test_an_abstention_is_logged_with_its_reason_and_no_generation(index, caplog):
    app = create_app(ServiceSettings(), fake_answerer(index, gate=CosineGate(1.1)))

    with caplog.at_level(logging.INFO, logger="edgar_rag.telemetry"), TestClient(app) as client:
        client.post("/ask", json={"question": ON_TOPIC})

    [line] = _telemetry(caplog)
    assert line["reason"] == "gate_rejected"
    assert line["abstained"] is True
    assert "generate" not in line["stages"]


def test_a_request_turned_away_or_timed_out_is_still_logged(index, caplog):
    generator = HeldGenerator()
    settings = ServiceSettings(request_timeout_seconds=0.2, max_concurrent_generations=1)
    app = create_app(settings, fake_answerer(index, generator=generator))

    with caplog.at_level(logging.INFO, logger="edgar_rag.telemetry"), TestClient(app) as client:
        try:
            client.post("/ask", json={"question": ON_TOPIC})
            client.post("/ask", json={"question": ON_TOPIC})
        finally:
            generator.release.set()

    assert [line["status"] for line in _telemetry(caplog)] == [504, 503]


def test_the_question_itself_is_not_logged(index, caplog):
    """A question is whatever a client typed; the log keeps what it cost."""
    app = create_app(ServiceSettings(), fake_answerer(index))

    with caplog.at_level(logging.INFO, logger="edgar_rag.telemetry"), TestClient(app) as client:
        client.post("/ask", json={"question": ON_TOPIC})

    assert ON_TOPIC not in caplog.text


def test_a_crash_is_logged_as_a_500(index, caplog):
    class CrashingGenerator:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("a bug, not a backend failure")

    app = create_app(ServiceSettings(), fake_answerer(index, generator=CrashingGenerator()))
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.INFO, logger="edgar_rag.telemetry"), client:
        assert client.post("/ask", json={"question": ON_TOPIC}).status_code == 500

    [line] = _telemetry(caplog)
    assert line["status"] == 500
    assert "generate" in line["stages"]
