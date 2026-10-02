"""What an ``Answer`` tells its caller: a typed reason, the gate's score, and
whether the gate ran on a fallback.

These used to die inside ``answer_question``: the gate's confidence and its
``degraded`` flag were dropped, every abstention carried the same message, and
the reason was free prose a caller could only grep.
"""

from edgar_rag.answer import answer_question
from edgar_rag.domain import AbstentionReason, GateDecision, abstained_message
from edgar_rag.gate import CosineGate
from tests.fakes import FakeEmbedder, FakeGate, FakeGenerator, FixedNonce

QUESTION = "what does the company design?"
TABLE = {QUESTION: [1.0, 0.0]}
NONCE = "0badc0de"


def _ask(index, gate, reply: str = "The Company designs phones [1]."):
    return answer_question(
        QUESTION,
        index,
        FakeEmbedder(TABLE),
        FakeGenerator(reply),
        gate,
        top_k=2,
        nonce=FixedNonce(NONCE),
    )


def test_a_gate_rejection_says_so_with_the_gate_score(index):
    answer = _ask(index, CosineGate(1.1))

    assert answer.abstained is True
    assert answer.text is None
    assert answer.reason is AbstentionReason.GATE_REJECTED
    assert answer.gate_score == 1.0
    assert "below the 1.1 threshold" in answer.detail


def test_a_model_refusal_is_not_reported_as_missing_passages(index):
    """Red flag 7: every abstention said no passage was close enough."""
    answer = _ask(index, CosineGate(0.5), reply=f"REFUSE-{NONCE}")

    assert answer.text is None
    assert answer.reason is AbstentionReason.MODEL_DECLINED
    assert answer.detail.startswith(abstained_message(AbstentionReason.MODEL_DECLINED))
    assert "close enough" not in answer.detail


def test_an_uncited_answer_abstains_as_no_valid_citation(index):
    answer = _ask(index, CosineGate(0.5), reply="The company designs phones.")

    assert answer.text is None
    assert answer.reason is AbstentionReason.NO_VALID_CITATION
    assert answer.detail.startswith(abstained_message(AbstentionReason.NO_VALID_CITATION))


def test_each_abstention_reason_has_its_own_message():
    messages = [abstained_message(reason) for reason in AbstentionReason]

    assert len(set(messages)) == len(messages)
    assert all(message.endswith(".") for message in messages)


def test_the_reasons_are_the_published_wire_values():
    assert [reason.value for reason in AbstentionReason] == [
        "gate_rejected",
        "out_of_period",
        "model_declined",
        "no_valid_citation",
        "unsupported_claim",
        "out_of_scope",
    ]


def test_an_answer_has_no_abstention_reason_and_keeps_the_gate_detail(index):
    answer = _ask(index, FakeGate.admitting(confidence=0.83))

    assert answer.abstained is False
    assert answer.reason is None
    assert answer.text == "The Company designs phones [1]."
    assert answer.gate_score == 0.83
    assert answer.detail == "fake: admitted"


def test_a_degraded_gate_is_reported_on_an_answer(index):
    """A gate that swapped itself for a weaker one used to look like a model decision."""
    gate = FakeGate(GateDecision(admitted=True, confidence=0.7, reason="fell back", degraded=True))

    answer = _ask(index, gate)

    assert answer.abstained is False
    assert answer.degraded is True


def test_a_degraded_gate_is_reported_on_a_rejection(index):
    gate = FakeGate(GateDecision(admitted=False, confidence=0.2, reason="fell back", degraded=True))

    answer = _ask(index, gate)

    assert answer.reason is AbstentionReason.GATE_REJECTED
    assert answer.degraded is True
    assert answer.gate_score == 0.2


def test_a_degraded_gate_is_reported_when_the_model_then_declines(index):
    gate = FakeGate(GateDecision(admitted=True, confidence=0.7, reason="fell back", degraded=True))

    answer = _ask(index, gate, reply=f"REFUSE-{NONCE}")

    assert answer.reason is AbstentionReason.MODEL_DECLINED
    assert answer.degraded is True
    assert answer.gate_score == 0.7


def test_a_healthy_gate_is_not_degraded(index):
    assert _ask(index, CosineGate(0.5)).degraded is False
