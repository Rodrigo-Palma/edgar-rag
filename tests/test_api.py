import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import Response

import edgar_rag.gate as gate_module
from edgar_rag.api import (
    app,
    provide_embedder,
    provide_gate,
    provide_generator,
    provide_index,
    provide_settings,
)
from edgar_rag.config import ServiceSettings
from edgar_rag.embeddings import ModelError
from edgar_rag.gate import BrierGate, CosineGate, GateError
from tests.fakes import FakeEmbedder, FakeGenerator

ON_TOPIC = "what does the company design?"
TABLE = {ON_TOPIC: [1.0, 0.0]}


def _settings(**overrides) -> ServiceSettings:
    base = {"min_retrieval_score": 0.5, "index_dir": "unused"}
    return ServiceSettings(**{**base, **overrides})


@pytest.fixture
def client(index):
    app.dependency_overrides = {
        provide_settings: lambda: _settings(),
        provide_index: lambda: index,
        provide_embedder: lambda: FakeEmbedder(TABLE),
        provide_generator: lambda: FakeGenerator("The Company designs phones [1]."),
    }
    yield TestClient(app)
    app.dependency_overrides = {}


def test_ask_returns_the_answer_its_citations_and_the_filing(client):
    response = client.post("/ask", json={"question": ON_TOPIC, "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "The Company designs phones [1]."
    assert body["abstained"] is False
    assert len(body["citations"]) == 1
    assert body["source"]["company"] == "Example Inc"


def test_ask_rejects_a_question_that_is_too_short(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_ask_rejects_a_top_k_outside_the_allowed_range(client):
    assert client.post("/ask", json={"question": ON_TOPIC, "top_k": 99}).status_code == 422


@pytest.mark.parametrize("question", ["   ", "\t\n  \n", "  hi  "])
def test_ask_rejects_a_question_that_is_blank_once_trimmed(client, question):
    """Whitespace passed the length check and reached the core as a 500."""
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_a_value_error_from_the_core_is_a_client_error_not_a_crash(index):
    class RefusingEmbedder:
        def embed(self, texts):
            raise ValueError("nothing to embed")

    app.dependency_overrides = {
        provide_settings: lambda: _settings(),
        provide_index: lambda: index,
        provide_embedder: lambda: RefusingEmbedder(),
        provide_generator: lambda: FakeGenerator("unused"),
    }
    response = TestClient(app).post("/ask", json={"question": ON_TOPIC})
    app.dependency_overrides = {}

    assert response.status_code == 422
    assert "nothing to embed" not in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"question": "x" * 501},
        {"question": ON_TOPIC, "top_k": 0},
        {"question": ON_TOPIC, "top_k": 11},
    ],
    ids=["question-501-chars", "top_k-0", "top_k-11"],
)
def test_ask_rejects_inputs_just_outside_the_limits(client, body):
    assert client.post("/ask", json=body).status_code == 422


def _assert_no_topology(text: str) -> None:
    """Nothing a client could use to map the machines behind the service."""
    for leak in ("http://", "https://", "localhost", "11434", "8100", "Error"):
        assert leak not in text, f"{leak!r} leaked into {text!r}"


def _ask_with(index, overrides: dict) -> Response:
    app.dependency_overrides = {
        provide_settings: lambda: _settings(),
        provide_index: lambda: index,
        provide_embedder: lambda: FakeEmbedder(TABLE),
        provide_generator: lambda: FakeGenerator("The Company designs phones [1]."),
        **overrides,
    }
    try:
        return TestClient(app).post("/ask", json={"question": ON_TOPIC, "top_k": 2})
    finally:
        app.dependency_overrides = {}


def test_ask_reports_a_model_failure_as_a_bad_gateway_without_its_url(index, caplog):
    """Adversarial case 15: the detail is generic, the log keeps what failed."""

    class BrokenGenerator:
        def generate(self, prompt: str) -> str:
            raise ModelError(
                "http://localhost:11434/api/generate did not answer: ConnectError: refused"
            )

    with caplog.at_level(logging.ERROR, logger="edgar_rag.api"):
        response = _ask_with(index, {provide_generator: lambda: BrokenGenerator()})

    assert response.status_code == 502
    _assert_no_topology(response.text)
    assert "http://localhost:11434/api/generate" in caplog.text


def test_an_embedder_failure_is_a_bad_gateway_too(index):
    class BrokenEmbedder:
        def embed(self, texts):
            raise ModelError("http://localhost:11434/api/embed did not answer")

    response = _ask_with(index, {provide_embedder: lambda: BrokenEmbedder()})

    assert response.status_code == 502
    _assert_no_topology(response.text)


def test_a_gate_that_cannot_reach_its_model_is_a_bad_gateway_not_a_crash(index, caplog):
    """A ``GateError`` used to escape as a 500."""

    class UnreachableGate:
        def admits(self, question, passages):
            raise GateError("http://localhost:8100 did not answer: ConnectError")

    with caplog.at_level(logging.ERROR, logger="edgar_rag.api"):
        response = _ask_with(index, {provide_gate: lambda: UnreachableGate()})

    assert response.status_code == 502
    _assert_no_topology(response.text)
    assert "http://localhost:8100" in caplog.text


def test_a_degraded_brier_answer_tells_the_client_nothing_about_the_brier_host(index, monkeypatch):
    """Adversarial case 14, end to end: the gate reason is part of the response."""

    def refuse(url, json, timeout):
        raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr(gate_module.httpx, "post", refuse)
    degraded_gate = BrierGate("http://localhost:8100", fallback=CosineGate(0.5))

    response = _ask_with(index, {provide_gate: lambda: degraded_gate})

    assert response.status_code == 200
    assert "relevance model unavailable" in response.json()["detail"]
    _assert_no_topology(response.text)


def test_health_says_which_filing_is_indexed(index, tmp_path):
    index.save(tmp_path)
    app.dependency_overrides = {provide_settings: lambda: _settings(index_dir=str(tmp_path))}
    response = TestClient(app).get("/health")
    app.dependency_overrides = {}

    body = response.json()
    assert body["status"] == "ready"
    assert body["chunks"] == 2


def test_health_reports_a_missing_index_instead_of_failing(tmp_path):
    app.dependency_overrides = {provide_settings: lambda: _settings(index_dir=str(tmp_path))}
    response = TestClient(app).get("/health")
    app.dependency_overrides = {}

    assert response.status_code == 200
    assert response.json() == {"status": "no index", "indexed_filing": None, "chunks": 0}


def test_ask_without_an_index_is_unavailable_not_a_crash(tmp_path):
    app.dependency_overrides = {provide_settings: lambda: _settings(index_dir=str(tmp_path))}
    response = TestClient(app).post("/ask", json={"question": ON_TOPIC})
    app.dependency_overrides = {}

    assert response.status_code == 503
    assert "ingest" in response.json()["detail"]
    # Adversarial case 16: the configured path stays on the server
    assert str(tmp_path) not in response.text


def test_the_gate_is_cosine_only_while_no_brier_url_is_configured():
    settings = ServiceSettings(brier_url=None)

    assert isinstance(provide_gate(settings), CosineGate)


def test_configuring_a_brier_url_puts_the_model_in_front_with_cosine_behind_it():
    settings = ServiceSettings(
        brier_url="http://brier.test",
        brier_min_confidence=0.8,
        min_retrieval_score=0.4,
    )

    gate = provide_gate(settings)

    assert isinstance(gate, BrierGate)
    assert gate.min_confidence == 0.8
    assert gate.fallback == CosineGate(0.4)


def _brier_replying(*confidences: float):
    """A brier service that judges passage n with confidences[n-1], or fails past them."""
    asked: list[int] = []

    def post(url, json, timeout):  # noqa: ARG001 - mirrors httpx.post
        asked.append(1)
        request = httpx.Request("POST", url)
        if len(asked) > len(confidences):
            return httpx.Response(503, text="down", request=request)
        yes = confidences[len(asked) - 1]
        return httpx.Response(
            200, json={"answers": [{"probabilities": [1 - yes, yes]}]}, request=request
        )

    return post


def test_a_partly_judged_brier_refusal_reaches_the_client_as_degraded(index, monkeypatch):
    """Red flag 8: passage 2 was never judged, and the response used to hide it."""
    monkeypatch.setattr(gate_module.httpx, "post", _brier_replying(0.4))
    gate = BrierGate("http://brier.test", min_confidence=0.7, fallback=CosineGate(0.5))

    response = _ask_with(index, {provide_gate: lambda: gate})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["text"] is None
    assert body["reason"] == "gate_rejected"
    assert body["degraded"] is True
    assert body["gate_score"] == 0.4
    _assert_no_topology(response.text)


def test_a_brier_outage_answered_by_cosine_reaches_the_client_as_degraded(index, monkeypatch):
    """The fallback is never reported as a model decision."""
    monkeypatch.setattr(gate_module.httpx, "post", _brier_replying())
    gate = BrierGate("http://brier.test", fallback=CosineGate(0.5))

    body = _ask_with(index, {provide_gate: lambda: gate}).json()

    assert body["abstained"] is False
    assert body["reason"] is None
    assert body["degraded"] is True
    assert body["gate_score"] == 1.0


def test_a_healthy_answer_carries_every_field_of_the_contract(client):
    body = client.post("/ask", json={"question": ON_TOPIC, "top_k": 2}).json()

    assert set(body) == {
        "question",
        "text",
        "citations",
        "abstained",
        "reason",
        "detail",
        "retrieval_score",
        "gate_score",
        "degraded",
        "source",
    }
    assert body["degraded"] is False
    assert set(body["citations"][0]) == {"marker", "item", "title", "quote", "score"}


def test_the_ask_response_is_published_in_the_openapi_schema(client):
    """``response_model`` is what makes the schema a promise rather than a sample."""
    schema = client.get("/openapi.json").json()

    ok = schema["paths"]["/ask"]["post"]["responses"]["200"]["content"]["application/json"]
    assert ok["schema"] == {"$ref": "#/components/schemas/AskResponse"}
    reason = schema["components"]["schemas"]["AbstentionReason"]
    assert "model_declined" in reason["enum"]
