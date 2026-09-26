import pytest
from fastapi.testclient import TestClient

from edgar_rag.api import app, provide_embedder, provide_generator, provide_index, provide_settings
from edgar_rag.config import Settings
from edgar_rag.embeddings import ModelError
from tests.conftest import FakeEmbedder, FakeGenerator

ON_TOPIC = "what does the company design?"
TABLE = {ON_TOPIC: [1.0, 0.0]}


def _settings(**overrides) -> Settings:
    base = {
        "edgar_user_agent": "Test Runner test@example.com",
        "min_retrieval_score": 0.5,
        "index_dir": "unused",
    }
    return Settings(**{**base, **overrides})


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
    assert len(body["citations"]) == 2
    assert body["source"]["company"] == "Example Inc"


def test_ask_rejects_a_question_that_is_too_short(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_ask_rejects_a_top_k_outside_the_allowed_range(client):
    assert client.post("/ask", json={"question": ON_TOPIC, "top_k": 99}).status_code == 422


def test_ask_reports_a_model_failure_as_a_bad_gateway(index):
    class BrokenGenerator:
        def generate(self, prompt: str) -> str:
            raise ModelError("ollama is not running")

    app.dependency_overrides = {
        provide_settings: lambda: _settings(),
        provide_index: lambda: index,
        provide_embedder: lambda: FakeEmbedder(TABLE),
        provide_generator: lambda: BrokenGenerator(),
    }
    response = TestClient(app).post("/ask", json={"question": ON_TOPIC, "top_k": 2})
    app.dependency_overrides = {}

    assert response.status_code == 502
    assert "ollama is not running" in response.json()["detail"]


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
