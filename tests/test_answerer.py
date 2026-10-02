import pytest

from edgar_rag.answer import Answerer
from edgar_rag.api import AskRequest
from edgar_rag.domain import DEFAULT_TOP_K
from edgar_rag.gate import CosineGate
from edgar_rag.telemetry import StageTimer
from tests.fakes import FakeEmbedder, FakeGate, FakeGenerator, FixedNonce

ON_TOPIC = "what does the company design?"
OFF_TOPIC = "who won the league in 1998?"
TABLE = {ON_TOPIC: [1.0, 0.0], OFF_TOPIC: [0.7071, 0.7071]}


def _answerer(index, reply: str = "The Company designs phones [1].") -> Answerer:
    return Answerer(
        index=index,
        embedder=FakeEmbedder(TABLE),
        generator=FakeGenerator(reply),
        gate=CosineGate(0.9),
        nonce=FixedNonce(),
    )


def test_the_answerer_answers_with_the_collaborators_it_was_built_with(index):
    answer = _answerer(index).ask(ON_TOPIC, top_k=2)

    assert answer.text == "The Company designs phones [1]."
    assert [citation.marker for citation in answer.citations] == [1]


def test_an_answered_question_is_timed_through_every_stage(index):
    stages = StageTimer()

    _answerer(index).ask(ON_TOPIC, top_k=2, stages=stages)

    assert set(stages.seconds()) == {"embed", "search", "gate", "generate"}


def test_a_gate_rejection_is_timed_without_a_generation_stage(index):
    """No generate stage is the evidence the model was never asked."""
    stages = StageTimer()

    answer = _answerer(index).ask(OFF_TOPIC, top_k=2, stages=stages)

    assert answer.abstained is True
    assert set(stages.seconds()) == {"embed", "search", "gate"}


def test_the_answerer_reports_the_filing_it_answers_from(index):
    assert _answerer(index).source == {"company": "Example Inc"}


def test_without_a_top_k_the_answerer_retrieves_its_own_default(index):
    gate = FakeGate.admitting()
    answerer = Answerer(
        index=index,
        embedder=FakeEmbedder(TABLE),
        generator=FakeGenerator("The Company designs phones [1]."),
        gate=gate,
        top_k=1,
    )

    answerer.ask(ON_TOPIC)

    [(_, passages)] = gate.calls
    assert len(passages) == 1


def test_a_top_k_of_zero_is_refused_not_replaced_by_the_default(index):
    """Zero is a value the caller passed, not the absence of one."""
    with pytest.raises(ValueError, match="top_k"):
        _answerer(index).ask(ON_TOPIC, top_k=0)


def test_the_default_top_k_is_the_one_the_service_offers():
    """One number, read by the answerer and the HTTP schema alike."""
    assert Answerer.__dataclass_fields__["top_k"].default == DEFAULT_TOP_K
    assert AskRequest.model_fields["top_k"].default == DEFAULT_TOP_K
