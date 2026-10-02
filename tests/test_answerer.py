from dataclasses import replace

import numpy as np
import pytest

from edgar_rag.answer import Answerer
from edgar_rag.domain import DEFAULT_TOP_K, AbstentionReason, Chunk, GateDecision, Scope
from edgar_rag.gate import CosineGate
from edgar_rag.index import CorpusIndex, build_shard
from edgar_rag.service.schemas import AskRequest
from edgar_rag.telemetry import StageTimer
from tests.fakes import (
    CIK,
    EXAMPLE,
    SCOPE,
    SPEC,
    FakeEmbedder,
    FakeGate,
    FakeGenerator,
    FixedNonce,
)

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
    answer = _answerer(index).ask(ON_TOPIC, SCOPE, top_k=2)

    assert answer.text == "The Company designs phones [1]."
    assert [citation.marker for citation in answer.citations] == [1]


def test_an_answered_question_is_timed_through_every_stage(index):
    stages = StageTimer()

    _answerer(index).ask(ON_TOPIC, SCOPE, top_k=2, stages=stages)

    assert set(stages.seconds()) == {"embed", "search", "gate", "generate"}


def test_a_gate_rejection_is_timed_without_a_generation_stage(index):
    """No generate stage is the evidence the model was never asked."""
    stages = StageTimer()

    answer = _answerer(index).ask(OFF_TOPIC, SCOPE, top_k=2, stages=stages)

    assert answer.abstained is True
    assert set(stages.seconds()) == {"embed", "search", "gate"}


def test_an_answer_names_the_filing_it_came_from(index):
    assert _answerer(index).ask(ON_TOPIC, SCOPE, top_k=2).source == EXAMPLE


def test_an_abstention_names_the_filing_it_searched(index):
    answer = _answerer(index).ask(OFF_TOPIC, SCOPE, top_k=2)

    assert answer.reason is AbstentionReason.GATE_REJECTED
    assert answer.source == EXAMPLE


@pytest.mark.parametrize("scope", [Scope(CIK + 1), Scope(CIK, 2019)], ids=["company", "year"])
def test_a_scope_with_no_indexed_filing_abstains_before_anything_runs(index, scope):
    embedder, gate, generator = FakeEmbedder(TABLE), FakeGate.admitting(), FakeGenerator("x [1]")
    stages = StageTimer()

    answer = Answerer(index, embedder, generator, gate).ask(ON_TOPIC, scope, stages=stages)

    assert answer.abstained is True
    assert answer.reason is AbstentionReason.OUT_OF_SCOPE
    assert answer.source is None
    assert answer.text is None
    assert (answer.gate_score, answer.retrieval_score, answer.degraded) == (0.0, 0.0, False)
    assert str(scope) in answer.detail
    assert embedder.calls == gate.calls == generator.prompts == []
    assert stages.seconds() == {}


def _two_companies() -> CorpusIndex:
    """Example Inc says little about design; another company says exactly that."""
    passage = (Chunk(chunk_id="Item 1#0", item="Item 1", title="Business", text="Fine print."),)
    other = replace(EXAMPLE, accession="0000000007-24-000001", cik=7, company="Other Corp")
    return CorpusIndex.of(
        SPEC,
        (
            build_shard(EXAMPLE, passage, np.asarray([[0.0, 1.0]], dtype=np.float32)),
            build_shard(other, passage, np.asarray([[1.0, 0.0]], dtype=np.float32)),
        ),
    )


def test_a_question_is_never_answered_from_another_company_s_filing():
    gate = FakeGate.admitting()
    answerer = Answerer(_two_companies(), FakeEmbedder(TABLE), FakeGenerator("x [1]"), gate)

    answer = answerer.ask(ON_TOPIC, SCOPE, top_k=4)

    [(_, passages)] = gate.calls
    assert [scored.chunk.chunk_id for scored in passages] == [f"{EXAMPLE.accession}:Item 1#0"]
    assert answer.retrieval_score == pytest.approx(0.0)
    assert answer.source == EXAMPLE


def test_without_a_top_k_the_answerer_retrieves_its_own_default(index):
    gate = FakeGate.admitting()
    answerer = Answerer(
        index=index,
        embedder=FakeEmbedder(TABLE),
        generator=FakeGenerator("The Company designs phones [1]."),
        gate=gate,
        top_k=1,
    )

    answerer.ask(ON_TOPIC, SCOPE)

    [(_, passages)] = gate.calls
    assert len(passages) == 1


def test_a_top_k_of_zero_is_refused_not_replaced_by_the_default(index):
    """Zero is a value the caller passed, not the absence of one."""
    with pytest.raises(ValueError, match="top_k"):
        _answerer(index).ask(ON_TOPIC, SCOPE, top_k=0)


def test_the_default_top_k_is_the_one_the_service_offers():
    """One number, read by the answerer and the HTTP schema alike."""
    assert Answerer.__dataclass_fields__["top_k"].default == DEFAULT_TOP_K
    assert AskRequest.model_fields["top_k"].default == DEFAULT_TOP_K


def test_an_answer_carries_the_trace_of_what_it_cost(index):
    answer = _answerer(index).ask(ON_TOPIC, SCOPE, top_k=2)

    assert set(answer.trace.stages) == {"embed", "search", "gate", "generate"}
    assert answer.trace.generation is not None
    assert answer.trace.generation.completion_tokens == len(answer.text.split())
    assert answer.trace.generation.prompt_tokens > 0


def test_a_gate_rejection_traces_no_generation(index):
    answer = _answerer(index).ask(OFF_TOPIC, SCOPE, top_k=2)

    assert answer.trace.generation is None
    assert "generate" not in answer.trace.stages


def test_a_model_refusal_is_traced_with_the_generation_it_cost(index):
    """A refusal still spent the tokens; the cost report must not hide it."""
    answer = _answerer(index, reply="REFUSE-0badc0de").ask(ON_TOPIC, SCOPE, top_k=2)

    assert answer.abstained is True
    assert answer.trace.generation is not None
    assert answer.trace.generation.completion_tokens == 1


def test_the_gate_judges_against_the_filing_the_scope_resolved_to(index):
    """A scope without a year resolves to the latest filing; the gate sees that one."""
    gate = FakeGate.admitting()

    Answerer(index, FakeEmbedder(TABLE), FakeGenerator("x [1]"), gate).ask(ON_TOPIC, SCOPE)

    assert gate.filings == [EXAMPLE]


def test_a_rejection_carries_the_reason_the_gate_gave(index):
    gate = FakeGate(
        GateDecision(
            admitted=False,
            confidence=0.8,
            reason="fiscal 2019 is outside 2022 to 2024",
            rejection=AbstentionReason.OUT_OF_PERIOD,
        )
    )
    generator = FakeGenerator("x [1]")

    answer = Answerer(index, FakeEmbedder(TABLE), generator, gate).ask(ON_TOPIC, SCOPE)

    assert answer.reason is AbstentionReason.OUT_OF_PERIOD
    assert "fiscal 2019 is outside 2022 to 2024" in answer.detail
    assert generator.prompts == []


@pytest.mark.parametrize("question", [ON_TOPIC, OFF_TOPIC], ids=["admitted", "rejected"])
def test_the_trace_keeps_every_gate_score_whatever_the_decision(index, question):
    answer = _answerer(index).ask(question, SCOPE, top_k=2)

    assert set(answer.trace.gate_scores) == {"cosine"}
    assert answer.trace.gate_scores["cosine"] == answer.gate_score


def test_a_question_outside_the_index_has_no_gate_scores(index):
    answer = _answerer(index).ask(ON_TOPIC, Scope(CIK + 1))

    assert answer.trace.gate_scores == {}
