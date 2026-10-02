from decimal import Decimal

from edgar_rag.answer import Answerer
from edgar_rag.domain import Chunk, GateDecision, ScoredChunk
from edgar_rag.eval.golden import NarrativeCase
from edgar_rag.eval.grading import from_golden, from_narrative, split_is_sane
from edgar_rag.eval.runner import (
    CapturingIndex,
    Harness,
    ScoreEvery,
    repeat_generations,
    run_cases,
)
from edgar_rag.gate import AllOf, CosineGate
from edgar_rag.period import PeriodGuard
from tests.eval import harness_fakes as world
from tests.fakes import EXAMPLE, FakeGate, FakeGenerator


def _harness(reply: str = world.ANSWER) -> tuple[Harness, FakeGenerator]:
    generator = FakeGenerator(reply)
    gates = AllOf(PeriodGuard(), CosineGate(min_score=0.0))
    answerer = Answerer(
        index=CapturingIndex(world.index()),
        embedder=world.embedder(),
        generator=generator,
        gate=gates,
    )
    return Harness(answerer, gates=gates), generator


def _questions():
    return tuple(from_golden(case) for case in world.golden())


def _records(reply: str = world.ANSWER):
    harness, generator = _harness(reply)
    records = {r.id: r for r in run_cases(harness, _questions(), generate=True, progress=_Null())}
    return records, generator


class _Null:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


def test_score_every_keeps_the_scores_and_overrides_the_decision():
    declining = FakeGate(
        GateDecision(admitted=False, confidence=0.1, reason="no", scores={"cosine": 0.1})
    )
    passage = ScoredChunk(world.CHUNKS[0], 0.1)

    admitted = ScoreEvery(declining, admit=True).admits("q", (passage,), EXAMPLE)

    assert admitted.admitted is True
    assert dict(admitted.scores) == {"cosine": 0.1}


def test_an_e2e_case_is_generated_and_a_gate_only_case_is_only_scored():
    records, generator = _records()

    assert records["revenue:EX:2024"].generated
    assert records["wrong_year:EX:2024"].generated
    assert not records["off_domain:EX:2024"].generated
    assert records["off_domain:EX:2024"].outcome == "scored_only"
    assert len(generator.prompts) == 2


def test_every_gate_score_is_recorded_whatever_the_gate_would_have_decided():
    records, _ = _records()

    wrong_year = records["wrong_year:EX:2024"]
    # The period guard would decline it, and the model was asked anyway: arm A.
    assert wrong_year.scores == {"cosine": 0.9939, "period": 0.0}
    assert wrong_year.generated
    assert records["revenue:EX:2024"].scores == {"cosine": 1.0, "period": 1.0}
    assert records["off_domain:EX:2024"].scores["period"] == 1.0


def test_an_answer_stating_the_gold_value_from_the_cited_passage_is_correct():
    records, _ = _records()

    revenue = records["revenue:EX:2024"]
    assert revenue.outcome == "answered"
    assert revenue.numeric_correct and revenue.citation_supported and revenue.gold_retrieved
    assert revenue.correct
    assert revenue.cited_items == ("Item 7",)
    assert revenue.retrieved == (
        f"{EXAMPLE.accession}:Item 7#0",
        f"{EXAMPLE.accession}:Item 1A#0",
    )


def test_a_wrong_figure_is_withheld_and_left_ungraded():
    records, _ = _records("Total net sales were $12 million [1].")

    revenue = records["revenue:EX:2024"]
    # The citation check withholds a figure its passage does not print.
    assert revenue.outcome == "unsupported_claim"
    assert revenue.generation == "Total net sales were $12 million [1]."
    assert revenue.numeric_correct is None and not revenue.correct


def test_the_prompt_of_a_case_is_the_same_on_every_run():
    _, first = _records()
    _, second = _records()

    assert first.prompts == second.prompts
    assert first.prompts[0] != first.prompts[1]  # one nonce per case, not per run


def test_a_record_holds_stage_times_and_generation_cost():
    records, _ = _records()

    revenue = records["revenue:EX:2024"]
    assert set(revenue.stages) == {"embed", "search", "gate", "generate"}
    assert revenue.prompt_tokens is not None and revenue.generator_seconds == 0.0
    assert "generate" not in records["off_domain:EX:2024"].stages


def test_a_gate_only_run_never_asks_the_model():
    harness, generator = _harness()

    records = list(run_cases(harness, _questions(), generate=False, progress=_Null()))

    assert generator.prompts == []
    assert {r.outcome for r in records} == {"scored_only"}


def test_repeats_send_the_recorded_prompt_to_the_live_model_and_compare_the_text():
    harness, generator = _harness()
    records = {r.id: r for r in run_cases(harness, _questions(), generate=True, progress=_Null())}

    same = repeat_generations(harness, _questions(), records, FakeGenerator(world.ANSWER), 5)
    other = repeat_generations(harness, _questions(), records, FakeGenerator("different"), 1)

    assert sorted(r.id for r in same) == ["revenue:EX:2024", "wrong_year:EX:2024"]
    assert all(r.identical for r in same)
    assert [r.identical for r in other] == [False]
    assert generator.prompts[-1] in generator.prompts[:2]


def test_a_narrative_records_the_items_cited_and_whether_its_split_can_grade_them():
    harness, _ = _harness()
    narrative = NarrativeCase(
        id="narrative:answerable:01",
        ticker="EX",
        cik=EXAMPLE.cik,
        fiscal_year=EXAMPLE.fiscal_year,
        accession=EXAMPLE.accession,
        question=world.REVENUE,
        answerable=True,
        expected_item="Item 7",
        evidence_terms=("net sales",),
    )

    record = harness.ask(from_narrative(narrative, "dev", None), generate=True)

    assert record.kind == "narrative" and record.e2e
    assert record.cited_items == ("Item 7",)
    # "Item 7" holds most of this filing's text, so the split cannot grade the item.
    assert record.item_split_sane is False


def test_split_sanity_follows_the_pre_registered_rule():
    def chunk(item: str, size: int) -> Chunk:
        return Chunk(chunk_id=f"{item}#0", item=item, title=item, text="x" * size)

    balanced = [chunk("Item 1", 40), chunk("Item 1A", 30), chunk("Item 7", 30)]
    assert split_is_sane(balanced, "Item 1A")
    assert not split_is_sane(balanced, "Item 1C")
    assert not split_is_sane([chunk("Item 1", 60), chunk("Item 7", 40)], None)
    assert not split_is_sane([chunk("Full filing", 50), chunk("Item 1", 50)], None)


def test_a_per_share_value_is_matched_to_the_cent():
    case = world.golden()[0].model_copy(
        update={"expected_value": Decimal("6.08"), "unit": "USD/shares"}
    )

    assert from_golden(case).per_share
