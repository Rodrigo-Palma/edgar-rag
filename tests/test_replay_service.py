"""EDGAR_RAG_MODE=replay: recorded replies through the real endpoints, and nothing else."""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
from fastapi.testclient import TestClient

from edgar_rag.config import ServiceSettings
from edgar_rag.domain import Generation, NotRecorded, Scope
from edgar_rag.prompt import NonceSource
from edgar_rag.service.app import Replay, create_app
from tests.fakes import CIK, CITED_REPLY, ON_TOPIC, FakeEmbedder, FakeGenerator, fake_answerer

RECORDED_NONCE = "c0ffee42"


class FakeRecorded:
    """Knows ``ON_TOPIC`` about the example filing, recorded with ``RECORDED_NONCE``."""

    def nonce_for(self, question: str, scope: Scope) -> NonceSource | None:
        if question != ON_TOPIC or scope.cik != CIK:
            return None
        return lambda: RECORDED_NONCE


class UnrecordedGenerator:
    def generate(self, prompt: str) -> Generation:
        raise NotRecorded("no generate recorded for this prompt")


def _settings() -> ServiceSettings:
    return ServiceSettings(
        mode="replay", gate="cosine", min_retrieval_score=0.5, index_dir="unused"
    )


@contextmanager
def _replaying(index, generator=None) -> Iterator[tuple[TestClient, FakeGenerator]]:
    recorded = FakeGenerator(CITED_REPLY)
    replay = Replay(
        index=index,
        embedder=FakeEmbedder({ON_TOPIC: [1.0, 0.0], "what is in item 1a?": [0.0, 1.0]}),
        generator=generator or recorded,
        questions=FakeRecorded(),
    )
    with TestClient(create_app(_settings(), replay=replay)) as client:
        yield client, recorded


def test_a_recorded_question_is_answered_from_the_tape_with_its_recorded_nonce(index):
    with _replaying(index) as (client, generator):
        response = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC})

    assert response.status_code == 200
    body = response.json()
    assert body["replayed"] is True
    assert body["text"] == CITED_REPLY
    assert f"<passages-{RECORDED_NONCE}>" in generator.prompts[0]


def test_a_question_the_tape_does_not_hold_is_404_and_reaches_no_model(index):
    with _replaying(index) as (client, generator):
        response = client.post("/ask", json={"cik": CIK, "question": "what is in item 1a?"})

    assert response.status_code == 404
    assert response.json()["detail"].startswith("not recorded")
    assert generator.prompts == []


def test_a_prompt_the_tape_never_recorded_is_404_not_a_crash(index):
    with _replaying(index, generator=UnrecordedGenerator()) as (client, _):
        response = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC})

    assert response.status_code == 404
    assert "no generate recorded" not in response.text


def test_health_says_the_service_is_replaying_and_does_not_probe_ollama(index):
    with _replaying(index) as (client, _):
        health = client.get("/health").json()

    assert health["mode"] == "replay"
    assert health["ollama_reachable"] is None


def test_a_live_service_says_it_is_live_and_that_nothing_was_replayed(index):
    app = create_app(
        ServiceSettings(min_retrieval_score=0.5),
        fake_answerer(index),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    with TestClient(app) as client:
        answer = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC}).json()
        health = client.get("/health").json()

    assert answer["replayed"] is False
    assert health["mode"] == "live"
