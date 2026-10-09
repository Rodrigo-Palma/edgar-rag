"""The citation check replayed over recorded generations, and the v1.1 report built from it."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from edgar_rag.domain import Chunk, ScoredChunk
from edgar_rag.eval.citation_replay import (
    Recorded,
    ReplayError,
    item_of,
    passages_shown,
    pinned_sha256,
    reconstruct,
    states_figures_only_in_words,
    verify_v1_replay,
    with_first_figure_changed,
)
from edgar_rag.eval.citation_report import Count, Pins, arm_counts, case_results, render
from edgar_rag.prompt import as_data, build_prompt, case_nonce
from tests.eval.harness_fakes import record

ROOT = Path(__file__).resolve().parents[2]
RND = "Research and development expense was $31,370 million, see [3]."
LABELS = "Total net sales increased. See Note 3 on page 21."


def _prompt(case_id: str, chunks: tuple[Chunk, ...]) -> str:
    passages = tuple(ScoredChunk(chunk, 0.9) for chunk in chunks)
    return build_prompt("how much was R&D?", passages, case_nonce(case_id)())


def _chunks() -> tuple[Chunk, ...]:
    return (
        Chunk(chunk_id="0001:Item 7#0", item="Item 7", title="MD&A", text=RND),
        Chunk(chunk_id="0001:Item 8#2", item="Item 8", title="FS", text=LABELS),
    )


def _case(case_id: str, generation: str, outcome: str = "answered", **fields: object):
    retrieved = tuple(chunk.chunk_id for chunk in _chunks())
    return record(
        case_id,
        answerable=True,
        cosine=0.9,
        outcome=outcome,
        generation=generation,
        retrieved=fields.pop("retrieved", retrieved),
        **fields,
    )


def _tape(*cases) -> tuple[Recorded, ...]:
    return tuple(Recorded(_prompt(case.id, _chunks()), case.generation) for case in cases)


def test_the_passages_are_read_back_as_the_prompt_showed_them():
    nonce = case_nonce("c1")()

    shown = passages_shown(_prompt("c1", _chunks()), nonce)

    assert shown == (("Item 7", as_data(RND)), ("Item 8", LABELS))


def test_a_prompt_without_the_case_s_block_is_refused():
    with pytest.raises(ReplayError, match="no passages block"):
        passages_shown(_prompt("c1", _chunks()), case_nonce("other")())


def test_passages_not_numbered_one_to_k_are_refused():
    nonce = case_nonce("c1")()
    prompt = _prompt("c1", _chunks()).replace("\n\n[2] (", "\n\n[3] (")

    with pytest.raises(ReplayError, match="not numbered"):
        passages_shown(prompt, nonce)


def test_the_item_of_a_chunk_id_is_between_the_accession_and_the_index():
    assert item_of("0000320193-24-000123:Item 1C#3") == "Item 1C"


def test_the_protocol_pin_is_found_by_file_name():
    protocol = "| frozen run `eval/runs/v1/cases.jsonl` | sha256 `" + "a" * 64 + "` |"

    assert pinned_sha256(protocol, "cases.jsonl") == "a" * 64
    with pytest.raises(ReplayError, match="pins no sha256"):
        pinned_sha256(protocol, "generate.jsonl")


def test_each_case_with_text_is_matched_to_its_tape_entry():
    answered = _case("c1", "R&D was $31,370 million [1].")
    declined = _case("c2", "REFUSE-x", outcome="model_declined")

    (replayable,) = reconstruct((answered, declined), _tape(answered))

    assert replayable.record.id == "c1"
    assert [p.chunk.item for p in replayable.passages] == ["Item 7", "Item 8"]


def test_a_case_with_no_tape_entry_is_refused():
    case = _case("c1", "R&D was $31,370 million [1].")

    with pytest.raises(ReplayError, match="0 tape entries"):
        reconstruct((case,), ())


def test_passages_that_are_not_the_chunks_retrieved_are_refused():
    case = _case("c1", "R&D was $31,370 million [1].", retrieved=("0001:Item 1#0", "0001:Item 8#2"))

    with pytest.raises(ReplayError, match="not the chunks retrieved"):
        reconstruct((case,), _tape(case))


def test_the_v1_replay_must_give_back_the_recorded_outcome():
    agrees = _case("c1", "Services revenue was $3 billion [2].")
    disagrees = _case("c2", "R&D was $9 billion [1].")

    verify_v1_replay(reconstruct((agrees,), _tape(agrees)))
    with pytest.raises(ReplayError, match="do not reproduce 1 cases"):
        verify_v1_replay(reconstruct((disagrees,), _tape(disagrees)))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue was ninety billion dollars [1].", True),
        ("Revenue was $90 billion, or ninety billion dollars [1].", False),
        ("The company designs phones [1].", False),
    ],
)
def test_an_answer_states_figures_only_in_words_when_none_has_a_digit(text, expected):
    assert states_figures_only_in_words(text) is expected


@pytest.mark.parametrize(
    ("text", "changed"),
    [
        ("Dividends were $6.1 billion [1].", "Dividends were $6.2 billion [1]."),
        ("It has 19 stores [1].", "It has 10 stores [1]."),
        ("The company designs phones [1].", None),
    ],
)
def test_the_positive_control_moves_the_last_digit_of_the_first_figure(text, changed):
    assert with_first_figure_changed(text) == changed


def test_a_count_prints_its_wilson_interval_and_n_a_when_empty():
    assert Count(0, 117).cell() == "0/117 = 0.0% [0.0%, 3.2%]"
    assert Count(1, 0).cell() == "n/a (n=0)"


def test_the_report_counts_every_measure_per_arm_and_lists_the_cases_that_change():
    clean = _case("c1", "R&D was $31,370 million [1].")
    by_label = _case("c2", "Services revenue was $3 billion [2].")
    in_words = _case("c3", "R&D was forty billion dollars [1].")
    cases = (clean, by_label, in_words)
    results = case_results(reconstruct(cases, _tape(*cases)))
    masks = {"A": np.array([True, True, True]), "B": np.array([True, False, True])}

    rows = arm_counts(results, cases, masks)
    report = render(rows, results, cases, masks, Pins("a" * 64, "b" * 64, "protocol-v1.1.md"))

    a, b = rows
    assert (a.m1.k, a.m1.n, a.m2.k, a.lost.k) == (1, 3, 1, 2)
    assert (b.m1.k, b.m1.n, b.lost.k) == (0, 2, 1)
    assert "| `c2` | A | answered | unsupported_claim | unsupported_claim |" in report
    assert "at most" in report
    assert report.count("exploratory") >= 4


def _script() -> ModuleType:
    path = ROOT / "scripts" / "measure_citation_support.py"
    spec = importlib.util.spec_from_file_location("measure_citation_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _script()


def test_the_committed_report_is_what_the_frozen_run_gives():
    assert measure.build() == measure.REPORT.read_text("utf-8")


def test_the_script_refuses_a_run_the_protocol_does_not_pin(tmp_path, capsys):
    protocol = tmp_path / "protocol.md"
    protocol.write_text(
        measure.PROTOCOL.read_text("utf-8").replace(
            pinned_sha256(measure.PROTOCOL.read_text("utf-8"), "cases.jsonl"), "0" * 64
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReplayError, match="cases.jsonl has sha256"):
        measure.build(protocol=protocol)


def test_the_script_exits_1_and_writes_nothing_on_a_replay_error(tmp_path, monkeypatch, capsys):
    def broken() -> str:
        raise ReplayError("broken")

    monkeypatch.setattr(measure, "build", broken)
    out = tmp_path / "report.md"

    assert measure.main(["measure", str(out)]) == 1
    assert not out.exists()
    assert "broken" in capsys.readouterr().err
