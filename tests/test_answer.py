import numpy as np
import pytest

from edgar_rag.answer import ABSTAINED_MESSAGE, answer_question
from edgar_rag.index import build_index
from tests.conftest import FakeEmbedder, FakeGenerator

ON_TOPIC = "what does the company design?"
OFF_TOPIC = "who won the league in 1998?"
TABLE = {ON_TOPIC: [1.0, 0.0], OFF_TOPIC: [0.7071, 0.7071]}


def test_answers_with_a_citation_per_passage(index):
    generator = FakeGenerator("The Company designs phones [1].")

    answer = answer_question(
        ON_TOPIC, index, FakeEmbedder(TABLE), generator, min_score=0.5, top_k=2
    )

    assert answer.abstained is False
    assert answer.text == "The Company designs phones [1]."
    assert [citation.marker for citation in answer.citations] == [1, 2]
    assert answer.citations[0].item == "Item 1"
    assert answer.retrieval_score == 1.0


def test_abstains_before_generating_when_retrieval_is_weak(index):
    generator = FakeGenerator("something invented")

    answer = answer_question(
        OFF_TOPIC, index, FakeEmbedder(TABLE), generator, min_score=0.9, top_k=2
    )

    assert answer.abstained is True
    assert answer.text == ABSTAINED_MESSAGE
    assert answer.citations == ()
    # The model is never asked, so it cannot invent an answer
    assert generator.prompts == []


def test_abstains_when_the_model_says_the_filing_does_not_cover_it(index):
    generator = FakeGenerator("NOT IN THE FILING")

    answer = answer_question(
        ON_TOPIC, index, FakeEmbedder(TABLE), generator, min_score=0.5, top_k=2
    )

    assert answer.abstained is True
    assert answer.citations == ()
    assert generator.prompts != []


def test_the_prompt_carries_the_passages_the_answer_must_use(index):
    generator = FakeGenerator("answer [1]")

    answer_question(ON_TOPIC, index, FakeEmbedder(TABLE), generator, min_score=0.5, top_k=2)

    prompt = generator.prompts[0]
    assert "[1] (Item 1) The Company designs phones." in prompt
    assert "[2] (Item 1A) Supply chains may fail." in prompt


def test_rejects_an_empty_question(index):
    with pytest.raises(ValueError):
        answer_question("   ", index, FakeEmbedder(TABLE), FakeGenerator("x"), min_score=0.5)


def test_an_index_needs_its_vectors_to_line_up():
    chunks = build_index.__doc__  # sanity: the helper is documented
    assert chunks

    with pytest.raises(ValueError):
        build_index({}, (), np.zeros((0, 2), dtype=np.float32))
