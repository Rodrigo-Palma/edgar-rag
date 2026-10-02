import pytest

from edgar_rag.domain import AbstentionReason, Chunk, GateDecision, ScoredChunk
from edgar_rag.gate import CosineGate
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
