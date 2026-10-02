import httpx
import numpy as np
import pytest

from edgar_rag import models
from edgar_rag.models import ModelError, OllamaEmbedder, OllamaGenerator, OllamaProbe


def _reply(monkeypatch, payload: dict) -> list[dict]:
    """Capture what the client posts, and answer with ``payload``."""
    sent: list[dict] = []

    def fake_post(url, json, timeout):
        sent.append({"url": url, "json": json})
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(models.httpx, "post", fake_post)
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

    generation = OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")

    assert generation.text == "an answer [1]"
    assert sent[0]["json"]["stream"] is False


def test_the_generator_asks_for_the_same_answer_every_time(monkeypatch):
    """Replay keys recorded generations on the prompt, so sampling must not vary."""
    sent = _reply(monkeypatch, {"response": "an answer [1]"})

    OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")

    body = sent[0]["json"]
    assert body["think"] is False
    assert body["options"] == {"temperature": 0, "seed": 0, "num_ctx": 8192}


def test_the_generator_reports_its_tokens_and_the_seconds_it_took(monkeypatch):
    _reply(monkeypatch, {"response": "an answer [1]", "prompt_eval_count": 812, "eval_count": 9})
    ticks = iter((100.0, 102.5))

    generation = OllamaGenerator(
        "http://localhost:11434", "qwen3", clock=lambda: next(ticks)
    ).generate("a prompt")

    assert (generation.prompt_tokens, generation.completion_tokens) == (812, 9)
    assert generation.seconds == 2.5


def test_token_counts_the_server_left_out_are_unknown_not_zero(monkeypatch):
    """Ollama omits ``prompt_eval_count`` when the whole prompt came from its cache."""
    _reply(monkeypatch, {"response": "an answer [1]", "eval_count": 0})

    generation = OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")

    assert generation.prompt_tokens is None
    assert generation.completion_tokens == 0


@pytest.mark.parametrize("count", [-1, "812", 8.5, True], ids=["negative", "text", "float", "bool"])
def test_a_token_count_that_is_not_a_count_is_a_model_error(monkeypatch, count):
    _reply(monkeypatch, {"response": "an answer [1]", "eval_count": count})

    with pytest.raises(ModelError, match="eval_count"):
        OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")


def test_an_empty_generation_is_an_error_not_an_empty_answer(monkeypatch):
    _reply(monkeypatch, {"response": "   "})

    with pytest.raises(ModelError, match="empty answer"):
        OllamaGenerator("http://localhost:11434", "qwen3").generate("a prompt")


def test_a_server_that_is_not_running_is_reported_with_its_url(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(models.httpx, "post", fake_post)

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

    monkeypatch.setattr(models.httpx, "post", fake_post)


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

    monkeypatch.setattr(models.httpx, "post", refuse)


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
        generation = OllamaGenerator("http://ollama.test", "qwen3", client=client).generate("p")

    assert generation.text == "an answer [1]"
    assert [str(request.url) for request in seen] == ["http://ollama.test/api/generate"]


def test_a_client_that_cannot_connect_is_a_model_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(refuse)) as client, pytest.raises(ModelError):
        OllamaEmbedder("http://ollama.test", "nomic", client=client).embed(("a",))


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _version_server(*statuses: int) -> tuple[httpx.Client, list[str]]:
    """Ollama answering /api/version with each status in turn, the last one after."""
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        status = statuses[min(len(asked), len(statuses)) - 1]
        return httpx.Response(status, json={"version": "0.18.0"})

    return httpx.Client(transport=httpx.MockTransport(handler)), asked


def test_the_probe_asks_ollama_for_its_version():
    client, asked = _version_server(200)

    with client:
        assert OllamaProbe("http://ollama.test/", client).reachable() is True

    assert asked == ["/api/version"]


def test_the_probe_reuses_its_answer_for_ten_seconds():
    """/health may be polled every second; Ollama is asked at most every ten."""
    clock = ManualClock()
    client, asked = _version_server(200, 500)

    with client:
        probe = OllamaProbe("http://ollama.test", client, clock=clock)
        first = probe.reachable()
        clock.now = 9.9
        cached = probe.reachable()
        clock.now = 10.0
        fresh = probe.reachable()

    assert (first, cached, fresh) == (True, True, False)
    assert len(asked) == 2


def test_an_ollama_that_cannot_be_reached_is_reported_not_raised():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(transport=httpx.MockTransport(refuse)) as client:
        assert OllamaProbe("http://ollama.test", client).reachable() is False
