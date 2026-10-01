import httpx
import numpy as np
import pytest

from edgar_rag import embeddings
from edgar_rag.embeddings import ModelError, OllamaEmbedder, OllamaGenerator


def _reply(monkeypatch, payload: dict) -> list[dict]:
    """Capture what the client posts, and answer with ``payload``."""
    sent: list[dict] = []

    def fake_post(url, json, timeout):
        sent.append({"url": url, "json": json})
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(embeddings.httpx, "post", fake_post)
    return sent


def test_embedder_returns_one_vector_per_text(monkeypatch):
    sent = _reply(monkeypatch, {"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    vectors = OllamaEmbedder("http://localhost:11434/", "nomic").embed(("a", "b"))

    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32
    assert sent[0]["url"] == "http://localhost:11434/api/embed"
    assert sent[0]["json"]["input"] == ["a", "b"]


def test_embedder_refuses_a_short_reply_instead_of_misaligning_chunks(monkeypatch):
    _reply(monkeypatch, {"embeddings": [[1.0, 0.0]]})

    with pytest.raises(ModelError, match="1 of 2 vectors"):
        OllamaEmbedder("http://localhost:11434", "nomic").embed(("a", "b"))


def test_embedding_nothing_is_a_caller_error(monkeypatch):
    with pytest.raises(ValueError):
        OllamaEmbedder("http://localhost:11434", "nomic").embed(())


def test_generator_returns_the_trimmed_answer(monkeypatch):
    sent = _reply(monkeypatch, {"response": "  an answer [1]  "})

    answer = OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")

    assert answer == "an answer [1]"
    assert sent[0]["json"]["stream"] is False


def test_an_empty_generation_is_an_error_not_an_empty_answer(monkeypatch):
    _reply(monkeypatch, {"response": "   "})

    with pytest.raises(ModelError, match="empty answer"):
        OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")


def test_a_server_that_is_not_running_is_reported_with_its_url(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(embeddings.httpx, "post", fake_post)

    with pytest.raises(ModelError, match="api/embed did not answer"):
        OllamaEmbedder("http://localhost:11434", "nomic").embed(("a",))


def test_passages_are_lower_cased_before_they_are_sent(monkeypatch):
    """Ollama 0.18.0 collapses every capitalised token onto one vector."""
    sent = _reply(monkeypatch, {"embeddings": [[1.0, 0.0]]})

    OllamaEmbedder("http://localhost:11434", "nomic").embed(("The Company designs Phones",))

    assert sent[0]["json"]["input"] == ["the company designs phones"]


def test_lowercasing_can_be_turned_off_when_the_tokenizer_is_fixed(monkeypatch):
    sent = _reply(monkeypatch, {"embeddings": [[1.0, 0.0]]})

    OllamaEmbedder("http://localhost:11434", "nomic", lowercase=False).embed(("The Company",))

    assert sent[0]["json"]["input"] == ["The Company"]


def _raw_reply(monkeypatch, content: bytes) -> None:
    """Answer every post with ``content`` as the body, whatever it is."""

    def fake_post(url, json, timeout):
        return httpx.Response(200, content=content, request=httpx.Request("POST", url))

    monkeypatch.setattr(embeddings.httpx, "post", fake_post)


NOT_A_JSON_OBJECT = {
    "not-json": b"<html>502 Bad Gateway</html>",
    "truncated": b'{"embeddings": [[1.0',
    "a-list": b"[[1.0, 0.0]]",
    "a-string": b'"ok"',
    "null": b"null",
}


@pytest.mark.parametrize("content", NOT_A_JSON_OBJECT.values(), ids=NOT_A_JSON_OBJECT.keys())
def test_an_embedder_reply_that_is_not_a_json_object_is_a_model_error(monkeypatch, content):
    _raw_reply(monkeypatch, content)

    with pytest.raises(ModelError):
        OllamaEmbedder("http://localhost:11434", "nomic").embed(("a",))


@pytest.mark.parametrize("content", NOT_A_JSON_OBJECT.values(), ids=NOT_A_JSON_OBJECT.keys())
def test_a_generator_reply_that_is_not_a_json_object_is_a_model_error(monkeypatch, content):
    _raw_reply(monkeypatch, content)

    with pytest.raises(ModelError):
        OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")


@pytest.mark.parametrize(
    "vectors",
    [[[1.0, 0.0], [1.0]], [[1.0, "x"], [0.0, 1.0]], [None, [0.0, 1.0]], "ab", [1.0, 0.0]],
    ids=["ragged", "text-component", "null-vector", "a-string", "flat"],
)
def test_malformed_vectors_are_a_model_error(monkeypatch, vectors):
    _reply(monkeypatch, {"embeddings": vectors})

    with pytest.raises(ModelError):
        OllamaEmbedder("http://localhost:11434", "nomic").embed(("a", "b"))


@pytest.mark.parametrize("response", [42, ["an answer"], {"text": "an answer"}])
def test_a_generation_that_is_not_text_is_a_model_error(monkeypatch, response):
    _reply(monkeypatch, {"response": response})

    with pytest.raises(ModelError):
        OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")


def _client_answering(payload: dict) -> tuple[httpx.Client, list[httpx.Request]]:
    """A client that answers every request with ``payload`` and records it."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def _no_module_post(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the module-level httpx.post was used instead of the client")

    monkeypatch.setattr(embeddings.httpx, "post", refuse)


def test_an_embedder_given_a_client_sends_through_it(monkeypatch):
    """One pooled client for the life of the service, not a connection per call."""
    _no_module_post(monkeypatch)
    client, seen = _client_answering({"embeddings": [[1.0, 0.0]]})

    with client:
        OllamaEmbedder("http://ollama.test", "nomic", client=client).embed(("a",))

    assert [str(request.url) for request in seen] == ["http://ollama.test/api/embed"]


def test_a_generator_given_a_client_sends_through_it(monkeypatch):
    _no_module_post(monkeypatch)
    client, seen = _client_answering({"response": "an answer [1]"})

    with client:
        answer = OllamaGenerator("http://ollama.test", "qwen3", client=client).generate("p")

    assert answer == "an answer [1]"
    assert [str(request.url) for request in seen] == ["http://ollama.test/api/generate"]


def test_a_client_that_cannot_connect_is_a_model_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(refuse)) as client, pytest.raises(ModelError):
        OllamaEmbedder("http://ollama.test", "nomic", client=client).embed(("a",))
