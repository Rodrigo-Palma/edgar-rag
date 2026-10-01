"""The doubles other tests rely on behave the way those tests assume."""

import pytest

from edgar_rag.answer import answer_question
from tests.fakes import FakeEmbedder, FakeGate, FakeGenerator, FixedNonce

QUESTION = "what does the company design?"


def test_a_rejecting_gate_is_asked_once_and_the_generator_never(index):
    gate = FakeGate.rejecting()
    generator = FakeGenerator("unused [1]")

    answer = answer_question(
        QUESTION, index, FakeEmbedder({QUESTION: [1.0, 0.0]}), generator, gate, top_k=2
    )

    assert answer.abstained is True
    assert [question for question, _ in gate.calls] == [QUESTION]
    assert generator.prompts == []


def test_an_admitting_gate_lets_the_generator_run(index):
    generator = FakeGenerator("The Company designs phones [1].")

    answer = answer_question(
        QUESTION,
        index,
        FakeEmbedder({QUESTION: [1.0, 0.0]}),
        generator,
        FakeGate.admitting(),
        top_k=2,
    )

    assert answer.abstained is False
    assert len(generator.prompts) == 1


def test_the_fake_embedder_fails_on_text_it_was_not_given():
    with pytest.raises(KeyError):
        FakeEmbedder({}).embed(("unplanned",))


def test_a_fixed_nonce_returns_the_same_value_and_counts_draws():
    nonce = FixedNonce("cafe1234")

    assert (nonce(), nonce()) == ("cafe1234", "cafe1234")
    assert nonce.calls == 2


@pytest.mark.parametrize("value", ["", "a b", "x<y"])
def test_a_fixed_nonce_refuses_a_value_that_could_break_a_delimiter(value):
    with pytest.raises(ValueError):
        FixedNonce(value)
