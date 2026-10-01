from edgar_rag.answer import Answerer
from edgar_rag.gate import CosineGate
from edgar_rag.telemetry import StageTimer
from tests.fakes import FakeEmbedder, FakeGenerator, FixedNonce

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
