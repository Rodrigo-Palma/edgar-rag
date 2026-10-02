import pytest

from edgar_rag.domain import AbstentionReason, Chunk, GateDecision, ScoredChunk
from edgar_rag.gate import AllOf, CosineGate, NoGate
from edgar_rag.period import PeriodGuard
from tests.fakes import EXAMPLE, FakeGate

QUESTION = "what does the company design?"
IN_PERIOD = "What was revenue in fiscal 2024?"
OUT_OF_PERIOD = "What was revenue in fiscal 2019?"


def _passages(*scores: float) -> tuple[ScoredChunk, ...]:
    return tuple(
        ScoredChunk(Chunk(f"c{position}", f"Item {position}", "t", f"passage {position}"), score)
        for position, score in enumerate(scores, start=1)
    )


def _period_and_cosine(min_score: float = 0.5) -> AllOf:
    return AllOf(PeriodGuard(), CosineGate(min_score))


def test_every_gate_admitting_admits_with_the_last_gate_s_score():
    decision = _period_and_cosine().admits(IN_PERIOD, _passages(0.8), EXAMPLE)

    assert decision.admitted is True
    assert decision.confidence == 0.8
    assert decision.reason == (
        "the question names fiscal 2024; this filing reports fiscal 2022 to 2024; "
        "best passage scored 0.800"
    )


def test_a_period_rejection_is_out_of_period_whatever_the_relevance_gate_thought():
    decision = _period_and_cosine().admits(OUT_OF_PERIOD, _passages(0.8), EXAMPLE)

    assert decision.admitted is False
    assert decision.rejection is AbstentionReason.OUT_OF_PERIOD
    assert (
        decision.reason == "the question names fiscal 2019; this filing reports fiscal 2022 to 2024"
    )


def test_a_relevance_rejection_in_period_is_gate_rejected():
    decision = _period_and_cosine(0.9).admits(IN_PERIOD, _passages(0.6), EXAMPLE)

    assert decision.admitted is False
    assert decision.rejection is AbstentionReason.GATE_REJECTED
    assert decision.reason == "best passage scored 0.600, below the 0.9 threshold"


def test_when_both_decline_the_first_names_the_reason_and_both_are_told():
    decision = _period_and_cosine(0.9).admits(OUT_OF_PERIOD, _passages(0.6), EXAMPLE)

    assert decision.rejection is AbstentionReason.OUT_OF_PERIOD
    assert "fiscal 2019" in decision.reason
    assert "below the 0.9 threshold" in decision.reason


@pytest.mark.parametrize(
    ("question", "best"),
    [(IN_PERIOD, 0.8), (IN_PERIOD, 0.3), (OUT_OF_PERIOD, 0.8), (OUT_OF_PERIOD, 0.3)],
    ids=["both-admit", "cosine-declines", "period-declines", "both-decline"],
)
def test_every_gate_is_asked_and_scored_whatever_the_decision(question, best):
    """A threshold is chosen afterwards from the scores, so none may go missing."""
    first, last = FakeGate.admitting(), FakeGate.admitting(0.4)
    guarded = AllOf(PeriodGuard(), first, CosineGate(0.5), last)

    decision = guarded.admits(question, _passages(best), EXAMPLE)

    assert len(first.calls) == len(last.calls) == 1
    assert set(decision.scores) == {"period", "cosine"}
    assert decision.scores["cosine"] == best
    assert decision.confidence == 0.4


def test_the_score_of_the_learned_gate_is_reported_even_when_the_rule_declines():
    decision = _period_and_cosine().admits(OUT_OF_PERIOD, _passages(0.8), EXAMPLE)

    assert decision.confidence == 0.8
    assert decision.scores == {"period": 0.0, "cosine": 0.8}


@pytest.mark.parametrize("position", [0, 1])
def test_one_degraded_gate_degrades_the_composition(position):
    degraded = FakeGate(
        GateDecision(admitted=True, confidence=0.7, reason="fell back", degraded=True)
    )
    gates = [FakeGate.admitting(), FakeGate.admitting()]
    gates[position] = degraded

    assert AllOf(*gates).admits(QUESTION, _passages(0.8), EXAMPLE).degraded is True


def test_no_gate_degraded_leaves_the_composition_undegraded():
    assert _period_and_cosine().admits(IN_PERIOD, _passages(0.8), EXAMPLE).degraded is False


def test_a_composition_of_one_gate_decides_as_that_gate_does():
    alone = CosineGate(0.5).admits(QUESTION, _passages(0.4), EXAMPLE)

    composed = AllOf(CosineGate(0.5)).admits(QUESTION, _passages(0.4), EXAMPLE)

    assert composed == alone


def test_a_composition_needs_a_gate():
    with pytest.raises(ValueError, match="at least one gate"):
        AllOf()


def test_compositions_of_equal_gates_are_equal():
    assert _period_and_cosine(0.55) == AllOf(PeriodGuard(), CosineGate(0.55))
    assert _period_and_cosine(0.55) != AllOf(CosineGate(0.55), PeriodGuard())


@pytest.mark.parametrize("passages", [(), _passages(0.01)], ids=["nothing-retrieved", "far"])
def test_no_gate_admits_everything(passages):
    decision = NoGate().admits(QUESTION, passages, EXAMPLE)

    assert decision.admitted is True
    assert decision.scores == {}
