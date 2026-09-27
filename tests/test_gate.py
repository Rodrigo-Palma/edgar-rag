import httpx
import pytest

import edgar_rag.gate as gate_module
from edgar_rag.chunking import Chunk
from edgar_rag.gate import BrierGate, CosineGate, GateError
from edgar_rag.index import ScoredChunk

QUESTION = "what does the company design?"


def _passages(*scores: float) -> tuple[ScoredChunk, ...]:
    return tuple(
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"c{position}",
                item=f"Item {position}",
                title="t",
                text=f"passage {position}",
            ),
            score=score,
        )
        for position, score in enumerate(scores, start=1)
    )


def _posting(handler):
    def post(url, json, timeout):  # noqa: ARG001 - mirrors httpx.post
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return client.post(url, json=json)

    return post


def _answering(*confidences: float):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as json_module

        body = json_module.loads(request.content)
        calls.append(body["state"])
        confidence = confidences[len(calls) - 1]
        return httpx.Response(
            200,
            json={
                "answers": [
                    {
                        "name": "relevance",
                        "kind": "bool",
                        "value": "yes" if confidence >= 0.5 else "no",
                        "confidence": max(confidence, 1 - confidence),
                        "abstained": False,
                        "probabilities": [1 - confidence, confidence],
                    }
                ],
                "temperature": 1.0,
            },
        )

    return handler, calls


def test_cosine_admits_a_close_passage():
    decision = CosineGate(0.5).admits(QUESTION, _passages(0.8, 0.4))

    assert decision.admitted is True
    assert decision.confidence == 0.8


def test_cosine_refuses_and_says_which_threshold_it_missed():
    decision = CosineGate(0.9).admits(QUESTION, _passages(0.6))

    assert decision.admitted is False
    assert "below the 0.9 threshold" in decision.reason


def test_cosine_refuses_when_retrieval_found_nothing():
    assert CosineGate(0.5).admits(QUESTION, ()).admitted is False


def test_brier_stops_at_the_first_passage_that_clears_the_threshold(monkeypatch):
    handler, calls = _answering(0.95, 0.99)
    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    decision = BrierGate("http://brier.test", min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7)
    )

    assert decision.admitted is True
    assert len(calls) == 1
    assert "passage 1 answers the question" in decision.reason


def test_brier_keeps_looking_past_a_passage_it_rejects(monkeypatch):
    handler, calls = _answering(0.2, 0.9)
    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    decision = BrierGate("http://brier.test", min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7)
    )

    assert decision.admitted is True
    assert len(calls) == 2
    assert "passage 2" in decision.reason


def test_brier_refuses_and_reports_the_closest_it_saw(monkeypatch):
    handler, _ = _answering(0.3, 0.55)
    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    decision = BrierGate("http://brier.test", min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7)
    )

    assert decision.admitted is False
    assert decision.confidence == 0.55
    assert "the closest was passage 2 at 0.550" in decision.reason


def test_brier_refuses_when_retrieval_found_nothing():
    decision = BrierGate("http://brier.test").admits(QUESTION, ())

    assert decision.admitted is False
    assert "nothing to judge" in decision.reason


def test_an_unreachable_brier_falls_back_to_cosine_and_says_it_did(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    decision = BrierGate("http://brier.test", fallback=CosineGate(0.5)).admits(
        QUESTION, _passages(0.8)
    )

    assert decision.admitted is True
    assert decision.degraded is True
    assert "degraded" in decision.reason


def test_an_unreachable_brier_without_a_fallback_is_an_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    with pytest.raises(GateError, match="did not answer"):
        BrierGate("http://brier.test").admits(QUESTION, _passages(0.8))


def test_a_malformed_reply_is_treated_as_the_service_being_broken(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    with pytest.raises(GateError, match="did not answer"):
        BrierGate("http://brier.test").admits(QUESTION, _passages(0.8))


def test_a_partial_failure_keeps_the_confidence_already_gathered(monkeypatch):
    """One bad request used to discard a confident yes from an earlier passage."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "answers": [
                        {
                            "name": "relevance",
                            "kind": "bool",
                            "value": "no",
                            "confidence": 0.6,
                            "abstained": False,
                            "probabilities": [0.6, 0.4],
                        }
                    ],
                    "temperature": 1.0,
                },
            )
        return httpx.Response(503, text="down")

    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    decision = BrierGate("http://brier.test", min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7)
    )

    assert decision.admitted is False
    assert decision.confidence == 0.4
    assert decision.degraded is False


def test_a_total_failure_still_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    with pytest.raises(GateError):
        BrierGate("http://brier.test").admits(QUESTION, _passages(0.8, 0.7))


def test_the_reason_never_names_a_passage_that_does_not_exist(monkeypatch):
    """With every confidence at zero the message used to say "passage 0"."""
    handler, _ = _answering(0.0, 0.0)
    monkeypatch.setattr(gate_module.httpx, "post", _posting(handler))

    decision = BrierGate("http://brier.test").admits(QUESTION, _passages(0.8, 0.7))

    assert "passage 0" not in decision.reason
    assert "passage 1" in decision.reason
