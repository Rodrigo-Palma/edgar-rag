"""What the service refuses before an endpoint runs: big bodies and foreign hosts."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from edgar_rag.config import ServiceSettings
from edgar_rag.service.app import create_app
from edgar_rag.service.limits import MAX_BODY_BYTES, BodySizeLimit
from tests.fakes import CIK, ON_TOPIC, FakeGenerator, fake_answerer

ONE_MEBIBYTE = 1024 * 1024


def _settings(**overrides) -> ServiceSettings:
    return ServiceSettings(**{"index_dir": "unused", "min_retrieval_score": 0.5, **overrides})


def _big_question() -> bytes:
    return json.dumps({"cik": CIK, "question": "x" * ONE_MEBIBYTE}).encode()


def test_a_one_mebibyte_body_is_refused_with_413_before_the_endpoint_runs(index):
    generator = FakeGenerator("The Company designs phones [1].")
    app = create_app(_settings(), fake_answerer(index, generator=generator))

    with TestClient(app) as client:
        response = client.post(
            "/ask", content=_big_question(), headers={"content-type": "application/json"}
        )

    assert response.status_code == 413
    assert len(response.content) < 200
    assert generator.prompts == []


def test_a_chunked_body_with_no_length_is_counted_and_refused(index):
    app = create_app(_settings(), fake_answerer(index))

    with TestClient(app) as client:
        response = client.post("/ask", content=iter([_big_question()]))

    assert "content-length" not in response.request.headers
    assert response.status_code == 413


def test_a_body_longer_than_its_declared_length_is_refused_while_it_streams():
    reached: list[bytes] = []

    async def endpoint(scope, receive, send):
        reached.append((await receive())["body"])

    sent: list[dict] = []
    chunk = b"x" * (MAX_BODY_BYTES // 2 + 1)
    messages = iter(
        [
            {"type": "http.request", "body": chunk, "more_body": True},
            {"type": "http.request", "body": chunk, "more_body": True},
        ]
    )

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": [(b"content-length", b"10")]}
    asyncio.run(BodySizeLimit(endpoint)(scope, receive, send))

    assert sent[0]["status"] == 413
    assert reached == []


def test_a_body_under_the_limit_reaches_the_endpoint_whole():
    reached: list[bytes] = []

    async def endpoint(scope, receive, send):
        message = await receive()
        reached.append(message["body"])
        assert message["more_body"] is False

    messages = iter(
        [
            {"type": "http.request", "body": b"ab", "more_body": True},
            {"type": "http.request", "body": b"cd", "more_body": False},
        ]
    )

    async def receive():
        return next(messages)

    async def send(message):
        raise AssertionError("nothing is sent for a body that fits")

    asyncio.run(BodySizeLimit(endpoint)({"type": "http", "headers": []}, receive, send))

    assert reached == [b"abcd"]


def test_an_invalid_body_is_not_echoed_back_in_the_422(index):
    secret = "a question the server should not repeat " * 3
    app = create_app(_settings(), fake_answerer(index))

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            content=json.dumps({"cik": CIK, "question": secret}),
            headers={"content-type": "text/plain"},
        )

    assert response.status_code == 422
    assert secret not in response.text
    assert "input" not in response.text
    assert response.json()["detail"][0]["type"]


def test_a_field_error_still_says_which_field(index):
    app = create_app(_settings(), fake_answerer(index))

    with TestClient(app) as client:
        response = client.post("/ask", json={"cik": CIK, "question": ON_TOPIC, "top_k": 99})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "top_k"]
    assert "99" not in response.text


@pytest.fixture
def default_hosts(monkeypatch):
    monkeypatch.delenv("EDGAR_RAG_ALLOWED_HOSTS")


@pytest.mark.usefixtures("default_hosts")
def test_only_loopback_names_are_allowed_by_default():
    assert _settings().allowed_hosts == ("127.0.0.1", "localhost", "[::1]")


@pytest.mark.usefixtures("default_hosts")
@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000"])
def test_a_loopback_host_is_served(index, host):
    with TestClient(create_app(_settings(), fake_answerer(index))) as client:
        assert client.get("/openapi.json", headers={"host": host}).status_code == 200


@pytest.mark.usefixtures("default_hosts")
def test_a_foreign_host_is_refused_against_dns_rebinding(index):
    with TestClient(create_app(_settings(), fake_answerer(index))) as client:
        response = client.get("/openapi.json", headers={"host": "attacker.example"})

    assert response.status_code == 400


def test_the_allowed_hosts_are_read_comma_separated(monkeypatch):
    monkeypatch.setenv("EDGAR_RAG_ALLOWED_HOSTS", " edgar.internal , 127.0.0.1 ")

    assert _settings().allowed_hosts == ("edgar.internal", "127.0.0.1")
