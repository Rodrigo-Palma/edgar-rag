"""The positive controls of the v1.1 citation report: what each one plants, and what it counts."""

import pytest

from edgar_rag.domain import Chunk, ScoredChunk
from edgar_rag.eval.citation_controls import (
    in_words,
    reference_probe,
    with_reference_planted,
    words_probe,
)
from edgar_rag.eval.citation_replay import Replayable
from tests.eval.harness_fakes import record

TABLE = "Research and development (in millions) 31,370. See Note 12 on page 48."


def _replayable(text: str, *passages: tuple[str, str]) -> Replayable:
    chunks = tuple(
        ScoredChunk(Chunk(chunk_id=f"{item}#{n}", item=item, title="", text=body), 0.9)
        for n, (item, body) in enumerate(passages)
    )
    case = record("c1", answerable=True, cosine=0.9, generation=text)
    return Replayable(case, chunks)


def test_a_reference_number_replaces_the_first_cited_figure():
    case = _replayable("R&D was $31.4 billion [1].", ("Item 7", TABLE))

    assert with_reference_planted(case) == "R&D was $7 billion [1]."


def test_a_reference_number_the_passage_also_prints_as_a_figure_is_not_planted():
    case = _replayable("R&D was $31.4 billion [1].", ("Item 31", "R&D was 31 and 31,370."))

    assert with_reference_planted(case) is None


def test_no_reference_and_no_figure_plant_nothing():
    assert with_reference_planted(_replayable("It designs phones [1].", ("Item 7", TABLE))) is None
    assert with_reference_planted(_replayable("R&D was $31.4 billion [1].", ("", "31,370"))) is None


def test_the_reference_probe_sees_what_v1_0_accepts_and_m1_withholds():
    probe = reference_probe(
        _replayable("R&D was $31.4 billion [1].", ("Item 7", "R&D (in millions) 31,370."))
    )

    assert probe is not None
    assert (probe.v1_0_accepts, probe.m1_withholds, probe.current_withholds) == (True, True, True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("R&D was $31.4 billion [1].", "R&D was thirty-one point four billion dollars [1]."),
        (
            "Margin was 46.2% on 7 stores [1].",
            "Margin was forty-six point two percent on seven stores [1].",
        ),
        ("It has 7 stores [1].", None),
        ("A loss of (1.2) billion [1].", None),
        ("It designs phones [1].", None),
    ],
    ids=["dollars-and-scale", "percent-and-count", "bare-count-first", "parentheses", "no-figure"],
)
def test_in_words_writes_every_figure_in_words_or_refuses(text, expected):
    assert in_words(text, 1) == expected


def test_the_words_probe_counts_m2_and_judges_right_and_wrong():
    probe = words_probe(_replayable("R&D was $31.4 billion [1].", ("Item 7", TABLE)))

    assert probe is not None
    assert (
        probe.counted_by_m2,
        probe.right_answered,
        probe.wrong_withheld,
        probe.wrong_accepted_by_v1_0,
    ) == (True, True, True, True)


def test_the_words_probe_skips_an_answer_it_cannot_write_in_words():
    assert words_probe(_replayable("It has 7 stores [1].", ("Item 7", "It has 7 stores."))) is None
