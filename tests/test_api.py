import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from edgar_rag.config import ServiceSettings
from edgar_rag.domain import GateDecision, Generation
from edgar_rag.gate import AllOf, CosineGate, GateError, NoGate
from edgar_rag.index import CorpusIndex, IndexFormatError
from edgar_rag.models import ModelError
from edgar_rag.period import PeriodGuard
from edgar_rag.service.app import build_gate, create_app, serve
from tests.fakes import (
    CIK,
    CITED_REPLY,
    EXAMPLE,
    ON_TOPIC,
    SPEC,
    FakeEmbedder,
    FakeGenerator,
    fake_answerer,
)


def _settings(**overrides) -> ServiceSettings:
    base = {"min_retrieval_score": 0.5, "index_dir": "unused"}
    return ServiceSettings(**{**base, **overrides})


@contextmanager
def _serving(index, **collaborators) -> Iterator[TestClient]:
    """A fresh app per test, with its lifespan run: nothing global is overridden."""
    app = create_app(_settings(), fake_answerer(index, **collaborators))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client(index) -> Iterator[TestClient]:
    with _serving(index) as served:
        yield served


def test_ask_returns_the_answer_its_citations_and_the_filing(client):
    response = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC, "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "The Company designs phones [1]."
    assert body["abstained"] is False
    assert len(body["citations"]) == 1
    assert body["source"]["company"] == "Example Inc"


def test_ask_rejects_a_question_that_is_too_short(client):
    assert client.post("/ask", json={"cik": CIK, "question": "hi"}).status_code == 422


def test_ask_rejects_a_top_k_outside_the_allowed_range(client):
    assert (
        client.post("/ask", json={"cik": CIK, "question": ON_TOPIC, "top_k": 99}).status_code == 422
    )


@pytest.mark.parametrize("question", ["   ", "\t\n  \n", "  hi  "])
def test_ask_rejects_a_question_that_is_blank_once_trimmed(client, question):
    """Whitespace passed the length check and reached the core as a 500."""
    assert client.post("/ask", json={"cik": CIK, "question": question}).status_code == 422


def test_a_value_error_from_the_core_is_a_client_error_not_a_crash(index):
    class RefusingEmbedder:
        def embed(self, texts):
            raise ValueError("nothing to embed")

    with _serving(index, embedder=RefusingEmbedder()) as client:
        response = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC})

    assert response.status_code == 422
    assert "nothing to embed" not in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"cik": CIK, "question": "x" * 501},
        {"cik": CIK, "question": ON_TOPIC, "top_k": 0},
        {"cik": CIK, "question": ON_TOPIC, "top_k": 11},
        {"question": ON_TOPIC},
        {"cik": 0, "question": ON_TOPIC},
        {"cik": 10_000_000_000, "question": ON_TOPIC},
        {"cik": "Apple", "question": ON_TOPIC},
        {"cik": CIK, "fiscal_year": 1992, "question": ON_TOPIC},
        {"cik": CIK, "fiscal_year": 2101, "question": ON_TOPIC},
    ],
    ids=[
        "question-501-chars",
        "top_k-0",
        "top_k-11",
        "no-cik",
        "cik-0",
        "cik-11-digits",
        "cik-not-a-number",
        "year-before-edgar",
        "year-2101",
    ],
)
def test_ask_rejects_inputs_just_outside_the_limits(client, body):
    assert client.post("/ask", json=body).status_code == 422


def _assert_no_topology(text: str) -> None:
    """Nothing a client could use to map the machines behind the service.

    The filing's own address on EDGAR is public and meant to be returned, so
    ``source`` is left out of the search.
    """
    body = json.loads(text)
    body.pop("source", None)
    shown = json.dumps(body)
    for leak in ("http://", "https://", "localhost", "11434", "8100", "Error"):
        assert leak not in shown, f"{leak!r} leaked into {shown!r}"


def _ask_with(index, **collaborators) -> httpx.Response:
    with _serving(index, **collaborators) as client:
        return client.post("/ask", json={"cik": CIK, "question": ON_TOPIC, "top_k": 2})


def test_ask_reports_a_model_failure_as_a_bad_gateway_without_its_url(index, caplog):
    """Adversarial case 15: the detail is generic, the log keeps what failed."""

    class BrokenGenerator:
        def generate(self, prompt: str) -> Generation:
            raise ModelError(
                "http://localhost:11434/api/generate did not answer: ConnectError: refused"
            )

    with caplog.at_level(logging.ERROR, logger="edgar_rag.service"):
        response = _ask_with(index, generator=BrokenGenerator())

    assert response.status_code == 502
    _assert_no_topology(response.text)
    assert "http://localhost:11434/api/generate" in caplog.text


def test_an_embedder_failure_is_a_bad_gateway_too(index):
    class BrokenEmbedder:
        def embed(self, texts):
            raise ModelError("http://localhost:11434/api/embed did not answer")

    response = _ask_with(index, embedder=BrokenEmbedder())

    assert response.status_code == 502
    _assert_no_topology(response.text)


def test_a_gate_that_cannot_reach_its_model_is_a_bad_gateway_not_a_crash(index, caplog):
    """A ``GateError`` used to escape as a 500."""

    class UnreachableGate:
        def admits(self, question, passages, filing):
            raise GateError("http://localhost:8100 did not answer: ConnectError")

    with caplog.at_level(logging.ERROR, logger="edgar_rag.service"):
        response = _ask_with(index, gate=UnreachableGate())

    assert response.status_code == 502
    _assert_no_topology(response.text)
    assert "http://localhost:8100" in caplog.text


def _ollama_answering(status: int, asked: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if asked is not None:
            asked.append(request.url.path)
        return httpx.Response(status, json={"version": "0.18.0"})

    return httpx.MockTransport(handler)


def test_health_says_which_filing_is_indexed(index, tmp_path):
    index.save(tmp_path)
    app = create_app(_settings(index_dir=tmp_path), transport=_ollama_answering(200))
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ready"
    assert body["chunks"] == 2
    assert body["filings"] == [
        {
            "cik": EXAMPLE.cik,
            "fiscal_year": EXAMPLE.fiscal_year,
            "accession": EXAMPLE.accession,
            "company": "Example Inc",
        }
    ]
    assert body["ollama_reachable"] is True


def test_health_carries_the_fingerprint_of_the_loaded_index(index, tmp_path):
    """Which embedding model, how many dimensions, and a digest of the vectors."""
    index.save(tmp_path)
    app = create_app(_settings(index_dir=tmp_path), transport=_ollama_answering(200))
    with TestClient(app) as client:
        fingerprint = client.get("/health").json()["fingerprint"]

    assert fingerprint["embedding_model"] == "nomic-embed-text"
    assert fingerprint["lowercase"] is True
    assert fingerprint["dimensions"] == 2
    assert len(fingerprint["digest"]) == 16


def test_the_fingerprint_changes_with_the_vectors(index, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    index.save(first)
    [shard] = index.shards
    swapped = replace(shard, vectors=shard.vectors[::-1].copy())
    CorpusIndex.of(SPEC, (swapped,)).save(second)

    def digest(directory) -> str:
        app = create_app(_settings(index_dir=directory), transport=_ollama_answering(200))
        with TestClient(app) as client:
            return client.get("/health").json()["fingerprint"]["digest"]

    assert digest(first) != digest(second)


def test_health_says_when_ollama_cannot_be_reached(index):
    app = create_app(_settings(), fake_answerer(index), transport=_ollama_answering(500))
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ready"
    assert body["ollama_reachable"] is False


def test_health_asks_ollama_once_for_polls_within_ten_seconds(index):
    asked: list[str] = []
    app = create_app(_settings(), fake_answerer(index), transport=_ollama_answering(200, asked))
    with TestClient(app) as client:
        for _ in range(5):
            client.get("/health")

    assert asked == ["/api/version"]


def test_health_reports_a_missing_index_instead_of_failing(tmp_path):
    app = create_app(_settings(index_dir=tmp_path), transport=_ollama_answering(200))
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no index"
    assert body["filings"] == []
    assert body["chunks"] == 0
    assert body["fingerprint"] is None


def _start(app) -> None:
    """Run the lifespan, where the index is loaded, and shut down again."""
    with TestClient(app):
        pass


@pytest.mark.parametrize(
    "embedding_model", ["mxbai-embed-large", "nomic-embed-text:v1.5"], ids=["model", "tag"]
)
def test_the_service_refuses_to_start_on_an_index_from_another_embedder(
    index, tmp_path, embedding_model
):
    """Questions embedded by another model would search the index and return noise."""
    index.save(tmp_path)
    app = create_app(
        _settings(index_dir=tmp_path, embedding_model=embedding_model),
        transport=_ollama_answering(200),
    )

    with pytest.raises(IndexFormatError, match="was built with nomic-embed-text"):
        _start(app)


def test_the_service_refuses_to_start_on_a_shard_that_changed(index, tmp_path):
    index.save(tmp_path)
    (tmp_path / EXAMPLE.accession / "chunks.json").write_text("[]")
    app = create_app(_settings(index_dir=tmp_path), transport=_ollama_answering(200))

    with pytest.raises(IndexFormatError, match="SHA-256"):
        _start(app)


def test_the_service_refuses_to_start_on_a_single_filing_index(tmp_path):
    (tmp_path / "vectors.npy").write_bytes(b"")
    (tmp_path / "chunks.json").write_text("{}")
    app = create_app(_settings(index_dir=tmp_path), transport=_ollama_answering(200))

    with pytest.raises(IndexFormatError, match="format 1"):
        _start(app)


def test_a_question_about_a_company_not_indexed_abstains_as_out_of_scope(client):
    response = client.post("/ask", json={"cik": CIK + 1, "question": ON_TOPIC})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["reason"] == "out_of_scope"
    assert body["source"] is None
    assert body["text"] is None


def test_a_question_about_a_year_not_indexed_abstains_as_out_of_scope(client):
    body = client.post("/ask", json={"cik": CIK, "fiscal_year": 2019, "question": ON_TOPIC}).json()

    assert body["reason"] == "out_of_scope"


def test_the_response_names_the_filing_the_scope_resolved_to(client):
    body = client.post("/ask", json={"cik": CIK, "fiscal_year": 2024, "question": ON_TOPIC}).json()

    assert body["source"] == {
        "company": "Example Inc",
        "cik": CIK,
        "fiscal_year": 2024,
        "accession": EXAMPLE.accession,
        "form": "10-K",
        "filing_date": "2024-11-01",
        "url": EXAMPLE.url,
    }


def test_ask_without_an_index_is_unavailable_not_a_crash(tmp_path):
    with TestClient(create_app(_settings(index_dir=tmp_path))) as client:
        response = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC})

    assert response.status_code == 503
    assert "ingest" in response.json()["detail"]
    # Adversarial case 16: the configured path stays on the server
    assert str(tmp_path) not in response.text


def test_the_default_gate_puts_the_period_guard_in_front_of_cosine():
    settings = ServiceSettings(min_retrieval_score=0.4)

    assert build_gate(settings) == AllOf(PeriodGuard(), CosineGate(0.4))


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("none", NoGate()),
        ("cosine", CosineGate(0.4)),
        ("period+cosine", AllOf(PeriodGuard(), CosineGate(0.4))),
    ],
)
def test_each_gate_is_built_as_named(choice, expected):
    assert build_gate(ServiceSettings(gate=choice, min_retrieval_score=0.4)) == expected


def test_a_question_about_another_year_is_declined_out_of_period_by_default(index):
    """End to end: the default service never asks the model about fiscal 2019."""
    generator = FakeGenerator(CITED_REPLY)
    app = create_app(ServiceSettings(), _default_gate_answerer(index, generator))

    with TestClient(app) as client:
        body = client.post("/ask", json={"cik": CIK, "question": IN_2019}).json()

    assert body["abstained"] is True
    assert body["reason"] == "out_of_period"
    assert "fiscal 2019" in body["detail"]
    assert body["source"]["fiscal_year"] == 2024
    assert generator.prompts == []


def test_a_question_about_a_reported_year_is_answered_by_default(index):
    generator = FakeGenerator(CITED_REPLY)
    app = create_app(ServiceSettings(), _default_gate_answerer(index, generator))

    with TestClient(app) as client:
        body = client.post("/ask", json={"cik": CIK, "question": IN_2023}).json()

    assert body["abstained"] is False
    assert len(generator.prompts) == 1


IN_2019 = "what did the company design in fiscal 2019?"
IN_2023 = "what did the company design in fiscal 2023?"


def _default_gate_answerer(index, generator):
    """The gate the default settings build, over a fake embedder and generator."""
    gate = build_gate(ServiceSettings())
    embedder = FakeEmbedder({IN_2019: [1.0, 0.0], IN_2023: [1.0, 0.0]})
    return fake_answerer(index, embedder=embedder, generator=generator, gate=gate)


class _PartlyJudgedGate:
    """A gate that declined on part of its evidence, as one asking a service can."""

    def admits(self, question, passages, filing):
        return GateDecision(
            admitted=False, confidence=0.4, reason="1 of 2 passages unjudged", degraded=True
        )


def test_a_partly_judged_refusal_reaches_the_client_as_degraded(index):
    """Red flag 8: a decision on part of the evidence used to look like a whole one."""
    response = _ask_with(index, gate=_PartlyJudgedGate())

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["text"] is None
    assert body["reason"] == "gate_rejected"
    assert body["degraded"] is True
    assert body["gate_score"] == 0.4


def test_a_healthy_answer_carries_every_field_of_the_contract(client):
    body = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC, "top_k": 2}).json()

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
        "replayed",
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


def test_the_service_is_served_on_the_local_machine_unless_configured_otherwise():
    """No authentication and no rate limit: only 127.0.0.1 can reach it by default."""
    calls: list[dict[str, object]] = []

    serve(_settings(), run=lambda app, **address: calls.append(address))
    telemetry = logging.getLogger("edgar_rag.telemetry")
    for handler in telemetry.handlers[:]:
        telemetry.removeHandler(handler)
    telemetry.propagate = True
    telemetry.setLevel(logging.NOTSET)

    assert calls == [{"host": "127.0.0.1", "port": 8000}]
