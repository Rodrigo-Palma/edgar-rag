import logging

import httpx
import pytest

import edgar_rag.gate as gate_module
from edgar_rag.domain import AbstentionReason, Chunk, GateDecision, ScoredChunk
from edgar_rag.gate import BrierGate, CosineGate, GateError
from tests.fakes import EXAMPLE

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


def _client(handler) -> httpx.Client:
    """A client whose requests ``handler`` answers, with no network behind it."""
    return httpx.Client(transport=httpx.MockTransport(handler))


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
    decision = CosineGate(0.5).admits(QUESTION, _passages(0.8, 0.4), EXAMPLE)

    assert decision.admitted is True
    assert decision.confidence == 0.8


def test_cosine_refuses_and_says_which_threshold_it_missed():
    decision = CosineGate(0.9).admits(QUESTION, _passages(0.6), EXAMPLE)

    assert decision.admitted is False
    assert "below the 0.9 threshold" in decision.reason


def test_cosine_refuses_when_retrieval_found_nothing():
    assert CosineGate(0.5).admits(QUESTION, (), EXAMPLE).admitted is False


def test_brier_asks_about_every_passage_and_decides_on_the_most_confident():
    """It used to stop at the first passage over the bar and report that one."""
    handler, calls = _answering(0.95, 0.99)
    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7), EXAMPLE
    )

    assert decision.admitted is True
    assert len(calls) == 2
    assert decision.confidence == 0.99
    assert "passage 2 answers the question with confidence 0.990" in decision.reason


@pytest.mark.parametrize("threshold", [0.1, 0.7, 0.96, 0.995])
def test_brier_s_score_is_the_same_whatever_its_threshold(threshold):
    """A threshold is chosen afterwards from these scores, so they must not depend on one.

    With the early stop, a 0.7 gate scored this question 0.95 and a 0.96 gate
    scored it 0.99, and no threshold swept over the first score reproduced
    what the second gate decided.
    """
    handler, _ = _answering(0.95, 0.99, 0.2)
    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=threshold).admits(
        QUESTION, _passages(0.8, 0.7, 0.6), EXAMPLE
    )

    assert decision.confidence == 0.99
    assert decision.scores == {"brier": 0.99}
    assert decision.admitted is (threshold <= 0.99)


def test_brier_files_no_score_of_its_own_when_its_fallback_decided():
    client = _client(lambda request: httpx.Response(503, text="down"))

    decision = BrierGate("http://brier.test", client, fallback=CosineGate(0.5)).admits(
        QUESTION, _passages(0.8), EXAMPLE
    )

    assert decision.scores == {"cosine": 0.8}


def test_brier_needs_a_client_to_be_built():
    """The module-level ``httpx.post`` opened a connection per passage, unpooled."""
    with pytest.raises(TypeError):
        BrierGate("http://brier.test")  # type: ignore[call-arg]


def test_brier_keeps_looking_past_a_passage_it_rejects():
    handler, calls = _answering(0.2, 0.9)
    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7), EXAMPLE
    )

    assert decision.admitted is True
    assert len(calls) == 2
    assert "passage 2" in decision.reason


def test_brier_refuses_and_reports_the_closest_it_saw():
    handler, _ = _answering(0.3, 0.55)
    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7), EXAMPLE
    )

    assert decision.admitted is False
    assert decision.confidence == 0.55
    assert "the closest was passage 2 at 0.550" in decision.reason


def test_brier_refuses_when_retrieval_found_nothing():
    unused = _client(lambda request: httpx.Response(500))
    decision = BrierGate("http://brier.test", unused).admits(QUESTION, (), EXAMPLE)

    assert decision.admitted is False
    assert "nothing to judge" in decision.reason


def test_an_unreachable_brier_falls_back_to_cosine_and_says_it_did():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    client = _client(handler)

    decision = BrierGate("http://brier.test", client, fallback=CosineGate(0.5)).admits(
        QUESTION, _passages(0.8), EXAMPLE
    )

    assert decision.admitted is True
    assert decision.degraded is True
    assert "degraded" in decision.reason


def test_an_unreachable_brier_without_a_fallback_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    client = _client(handler)

    with pytest.raises(GateError, match="did not answer"):
        BrierGate("http://brier.test", client).admits(QUESTION, _passages(0.8), EXAMPLE)


def test_a_malformed_reply_is_treated_as_the_service_being_broken():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = _client(handler)

    with pytest.raises(GateError, match="did not answer"):
        BrierGate("http://brier.test", client).admits(QUESTION, _passages(0.8), EXAMPLE)


def test_a_partial_failure_keeps_the_confidence_already_gathered():
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

    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7), EXAMPLE
    )

    assert decision.admitted is False
    assert decision.confidence == 0.4
    # Passage 2 was never judged, so the refusal rests on part of the evidence
    assert decision.degraded is True
    assert "1 of 2 passages could not be judged" in decision.reason


def test_a_total_failure_still_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    client = _client(handler)

    with pytest.raises(GateError):
        BrierGate("http://brier.test", client).admits(QUESTION, _passages(0.8, 0.7), EXAMPLE)


def test_the_reason_never_names_a_passage_that_does_not_exist():
    """With every confidence at zero the message used to say "passage 0"."""
    handler, _ = _answering(0.0, 0.0)
    client = _client(handler)

    decision = BrierGate("http://brier.test", client).admits(QUESTION, _passages(0.8, 0.7), EXAMPLE)

    assert "passage 0" not in decision.reason
    assert "passage 1" in decision.reason


def _replying(status: int, content: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content, headers={"content-type": "application/json"})

    return handler


def _assert_no_topology(text: str) -> None:
    """Nothing a client could use to map the machines behind the service."""
    lowered = text.lower()
    for leak in ("http", "localhost", "8100", "error", "exception", "connect"):
        assert leak not in lowered, f"{leak!r} leaked into {text!r}"


MALFORMED_REPLIES = {
    "one-probability": b'{"answers": [{"probabilities": [0.1]}]}',
    "nan": b'{"answers": [{"probabilities": [0.5, NaN]}]}',
    "infinity": b'{"answers": [{"probabilities": [0.5, Infinity]}]}',
    "above-one": b'{"answers": [{"probabilities": [0.1, 1.7]}]}',
    "below-zero": b'{"answers": [{"probabilities": [0.1, -0.2]}]}',
    "null-probabilities": b'{"answers": [{"probabilities": null}]}',
    "no-probabilities": b'{"answers": [{"name": "relevance"}]}',
    "text-probability": b'{"answers": [{"probabilities": [0.1, "high"]}]}',
    "no-answers": b'{"answers": []}',
    "answers-not-a-list": b'{"answers": {"probabilities": [0.1, 0.9]}}',
    "not-an-object": b"[0.1, 0.9]",
    "not-json": b"<html>bad gateway</html>",
}


@pytest.mark.parametrize("content", MALFORMED_REPLIES.values(), ids=MALFORMED_REPLIES.keys())
def test_a_malformed_probability_falls_back_instead_of_crashing(content):
    client = _client(_replying(200, content))

    decision = BrierGate("http://brier.test", client, fallback=CosineGate(0.5)).admits(
        QUESTION, _passages(0.8), EXAMPLE
    )

    assert decision.admitted is True
    assert decision.degraded is True


@pytest.mark.parametrize("content", MALFORMED_REPLIES.values(), ids=MALFORMED_REPLIES.keys())
def test_a_malformed_probability_without_a_fallback_is_a_gate_error(content):
    client = _client(_replying(200, content))

    with pytest.raises(GateError):
        BrierGate("http://brier.test", client).admits(QUESTION, _passages(0.8), EXAMPLE)


def test_a_confidence_of_zero_is_an_answer_not_a_missing_one():
    """``not best_confidence`` used to read a judged 0.0 as nothing judged."""
    calls: list[int] = []
    zero = b'{"answers": [{"probabilities": [1.0, 0.0]}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return _replying(200, zero)(request)
        return httpx.Response(503, text="down")

    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7), EXAMPLE
    )

    assert decision.admitted is False
    assert decision.confidence == 0.0
    assert decision.degraded is True


def test_an_admission_after_a_failed_passage_is_still_marked_degraded():
    calls: list[int] = []
    confident = b'{"answers": [{"probabilities": [0.1, 0.9]}]}'

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="down")
        return _replying(200, confident)(request)

    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7), EXAMPLE
    )

    assert decision.admitted is True
    assert decision.degraded is True


def test_a_fully_judged_refusal_is_not_degraded():
    handler, _ = _answering(0.3, 0.55)
    client = _client(handler)

    decision = BrierGate("http://brier.test", client, min_confidence=0.7).admits(
        QUESTION, _passages(0.8, 0.7), EXAMPLE
    )

    assert decision.degraded is False


def test_the_degraded_reason_names_no_url_and_no_exception(caplog):
    """Adversarial case 14: the reason reaches the client, the detail only the log."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    client = _client(handler)

    with caplog.at_level(logging.WARNING, logger="edgar_rag.gate"):
        decision = BrierGate("http://localhost:8100", client, fallback=CosineGate(0.5)).admits(
            QUESTION, _passages(0.8), EXAMPLE
        )

    assert decision.degraded is True
    assert "relevance model unavailable" in decision.reason
    _assert_no_topology(decision.reason)
    assert "http://localhost:8100" in caplog.text
    assert "Connection refused" in caplog.text


def test_brier_asks_through_the_client_it_was_given(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the module-level httpx.post was used instead of the client")

    monkeypatch.setattr(gate_module.httpx, "post", refuse)
    handler, calls = _answering(0.9)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        decision = BrierGate("http://brier.test", client).admits(QUESTION, _passages(0.6), EXAMPLE)

    assert decision.admitted is True
    assert calls == ["passage 1"]


def test_the_client_does_not_take_part_in_comparing_gates():
    with httpx.Client() as client:
        assert BrierGate("http://brier.test", client) == BrierGate(
            "http://brier.test", httpx.Client()
        )


def test_a_decision_s_scores_cannot_be_changed_after_it_is_returned():
    scores = {"cosine": 0.8}
    decision = GateDecision(admitted=True, confidence=0.8, reason="r", scores=scores)

    scores["cosine"] = 0.1

    assert decision.scores == {"cosine": 0.8}
    with pytest.raises(TypeError):
        decision.scores["cosine"] = 0.1  # type: ignore[index]


def test_a_relevance_rejection_says_gate_rejected_by_default():
    decision = CosineGate(0.9).admits(QUESTION, _passages(0.6), EXAMPLE)

    assert decision.rejection is AbstentionReason.GATE_REJECTED


@pytest.mark.parametrize("best", [0.6, 0.95], ids=["rejected", "admitted"])
def test_cosine_files_its_score_whatever_it_decides(best):
    decision = CosineGate(0.9).admits(QUESTION, _passages(best), EXAMPLE)

    assert decision.scores == {"cosine": best}
