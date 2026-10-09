"""Positive controls of the v1.1 citation measurement, not pre-registered.

A zero in M1 or M2 means something only if the replay would have seen the
defect had it been there. Each control plants one defect into the answers the
v1 run accepted, judges them on their own passages, and counts how often the
measurement sees it:

- **reference**: the first figure of a cited sentence is replaced by a note,
  page, item, exhibit or section number of a passage that sentence cites, one
  that passage prints nowhere else as a figure. The v1.0.0 rules should accept
  it. The M1 rules withhold it unless the passage text backs the new figure
  anyway: a figure of one or two digits is held only to that precision, so
  ``$8 million`` is backed by ``8,250`` in a table in thousands. What M1
  withholds here is its power to see the defect.
- **words**: every figure in digits is written in words. M2 should count the
  answer, the v1.1 check should still answer it, and with its first figure
  moved up by one digit the v1.1 check should withhold it while v1.0.0 does not.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import partial

from edgar_rag.amounts import Amount, number_in_words
from edgar_rag.citations import CURRENT, V1_0, cited_figures, passage_figures
from edgar_rag.domain import ScoredChunk
from edgar_rag.eval.citation_replay import (
    REFERENCES_STRIPPED,
    Replayable,
    judge,
    states_figures_only_in_words,
    with_first_figure_changed,
)

PERCENT_RULE_OFF = replace(CURRENT, percent_backs_only_percent=False)
"""The v1.1 check without "a percentage backs only a percentage"."""
RESCALE_FLOOR_OFF = replace(CURRENT, min_digits_to_rescale=0)
"""The v1.1 check rescaling a number of any length, as v1.0.0 did."""

# A figure the controls can rewrite: an optional "$", digits, an optional scale word.
_PLAIN = re.compile(
    r"(?P<dollar>\$\s*)?(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<scale>\s*(?:thousand|million|billion|trillion)s?)?",
    re.IGNORECASE,
)
# The references M1 removes from a passage (``citations._REFERENCE``), with the number captured.
_REFERENCE_NUMBER = re.compile(
    r"\b(?:notes?|pages?)\s+(\d{1,4})[a-z]?\b(?![.,]\d|\s*%)"
    r"|\b(?:items?|exhibits?|sections?)\s+(\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ReferenceProbe:
    """One answer with a reference number planted as its first cited figure."""

    id: str
    v1_0_accepts: bool
    m1_withholds: bool
    current_withholds: bool


@dataclass(frozen=True, slots=True)
class WordsProbe:
    """One answer with its figures written in words, right and then one digit off."""

    id: str
    counted_by_m2: bool
    right_answered: bool
    wrong_withheld: bool
    wrong_accepted_by_v1_0: bool


def _is_plain(amount: Amount) -> bool:
    return _PLAIN.fullmatch(amount.text) is not None


def _swap(text: str, old: str, new: Callable[[bool], str]) -> str | None:
    """``text`` with the first ``old`` standing alone replaced; ``new`` learns if ``%`` followed."""
    pattern = re.compile(rf"(?<![\w.,]){re.escape(old)}(?![\d,]*\d)(?P<percent>\s*%)?")
    changed, count = pattern.subn(lambda found: new(bool(found["percent"])), text, count=1)
    return changed if count else None


def _references(scored: ScoredChunk) -> list[str]:
    label = re.match(r"Item\s+(\d+)", scored.chunk.item)
    found = [a or b for a, b in _REFERENCE_NUMBER.findall(scored.chunk.text)]
    return ([label.group(1)] if label else []) + found


def _only_in_references(number: str, scored: ScoredChunk) -> bool:
    """Whether the passage prints ``number`` only as a reference, never as a figure of its text."""
    value = Decimal(number)
    return all(abs(found.value) != value for found in passage_figures(scored, REFERENCES_STRIPPED))


def with_reference_planted(case: Replayable) -> str | None:
    """The answer with its first cited figure in digits replaced by a reference number.

    The number comes from a passage the figure's own sentence cites, and that
    passage prints it only as a reference. ``None`` when there is no such
    figure or number.
    """
    for figure, markers in cited_figures(case.text, len(case.passages)):
        if not _is_plain(figure):
            continue
        cited = [case.passages[i - 1] for i in sorted(markers)]
        numbers = [
            n
            for scored in cited
            for n in _references(scored)
            if all(_only_in_references(n, other) for other in cited)
        ]
        if numbers:
            return _plant(case.text, figure, numbers[0])
    return None


def _plant(text: str, figure: Amount, number: str) -> str | None:
    found = _PLAIN.fullmatch(figure.text)
    assert found is not None  # checked by with_reference_planted
    planted = f"{found['dollar'] or ''}{number}{found['scale'] or ''}"
    return _swap(text, figure.text, lambda percent: planted + ("%" if percent else ""))


def in_words(text: str, available: int) -> str | None:
    """The answer with every figure in digits written in words.

    ``None`` when a figure is not plain (negative, in parentheses), or when the
    first one would not stay a figure in words (a bare count: ``12 stores``).
    """
    figures = [f for f, _ in cited_figures(text, available) if any(c.isdigit() for c in f.text)]
    if not figures or not all(_is_plain(f) for f in figures):
        return None
    first = _PLAIN.fullmatch(figures[0].text)
    if first is None or not (first["dollar"] or first["scale"] or figures[0].is_percent):
        return None
    changed: str | None = text
    for figure in figures:
        if changed is None:
            return None
        changed = _swap(changed, figure.text, partial(_spelled, figure))
    return changed


def _spelled(figure: Amount, percent: bool) -> str:
    plain = _PLAIN.fullmatch(figure.text)
    assert plain is not None  # checked by in_words
    words = number_in_words(plain["number"]) + (plain["scale"] or "")
    return words + (" dollars" if plain["dollar"] else "") + (" percent" if percent else "")


def reference_probe(case: Replayable) -> ReferenceProbe | None:
    planted = with_reference_planted(case)
    if planted is None:
        return None
    return ReferenceProbe(
        id=case.record.id,
        v1_0_accepts=judge(planted, case.passages, V1_0).outcome == "answered",
        m1_withholds=judge(planted, case.passages, REFERENCES_STRIPPED).outcome
        == "unsupported_claim",
        current_withholds=judge(planted, case.passages, CURRENT).outcome != "answered",
    )


def words_probe(case: Replayable) -> WordsProbe | None:
    available = len(case.passages)
    right = in_words(case.text, available)
    bumped = with_first_figure_changed(case.text)
    wrong = in_words(bumped, available) if bumped is not None else None
    if right is None or wrong is None:
        return None
    return WordsProbe(
        id=case.record.id,
        counted_by_m2=states_figures_only_in_words(right),
        right_answered=judge(right, case.passages, CURRENT).outcome == "answered",
        wrong_withheld=judge(wrong, case.passages, CURRENT).outcome != "answered",
        wrong_accepted_by_v1_0=judge(wrong, case.passages, V1_0).outcome == "answered",
    )
