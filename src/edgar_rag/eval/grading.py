"""The questions an evaluation asks, and how an answer to one is graded.

Golden and narrative cases become one ``EvalQuestion``, so the runner asks
both the same way. Grading reads only the answer and the passages the model
was given:

* ``numeric_correct``: the answer states the gold value (``is_numeric_match``,
  with the case's tolerance);
* ``citation_supported``: a passage the answer cites prints the gold value, by
  the strict rule the golden-set builder used to keep the case at all
  (``PrintedNumbers.prints``), so "supported" means what "answerable" meant;
* ``gold_retrieved``: any retrieved passage prints it, which separates a
  retrieval miss from a generation miss.

``split_is_sane`` is the rule that decides whether a narrative's filing was
split into items well enough for "cited the expected item" to mean anything.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from edgar_rag.domain import Answer, Chunk, ScoredChunk
from edgar_rag.eval.golden import GoldenCase, NarrativeCase, NegativeKind, Split
from edgar_rag.eval.numeric import is_numeric_match
from edgar_rag.eval.templates import USD_PER_SHARE
from edgar_rag.eval.textmatch import PrintedNumbers

FULL_FILING = "Full filing"
MIN_SECTIONS = 2
MAX_SECTION_SHARE = 0.5


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """A case as the runner asks it, golden or narrative."""

    id: str
    kind: str
    split: Split
    fold: int | None
    ticker: str
    cik: int
    fiscal_year: int
    question: str
    answerable: bool
    negative_kind: NegativeKind | None
    e2e: bool
    expected_value: Decimal | None = None
    tolerance: Decimal = Decimal("0.005")
    per_share: bool = False
    expected_item: str | None = None


def from_golden(case: GoldenCase) -> EvalQuestion:
    return EvalQuestion(
        id=case.id,
        kind="golden",
        split=case.split,
        fold=case.fold,
        ticker=case.ticker,
        cik=case.cik,
        fiscal_year=case.fiscal_year,
        question=case.question,
        answerable=case.answerable,
        negative_kind=case.negative_kind,
        e2e=case.e2e,
        expected_value=case.expected_value,
        tolerance=case.tolerance,
        per_share=case.unit == USD_PER_SHARE,
    )


def from_narrative(case: NarrativeCase, split: Split, fold: int | None) -> EvalQuestion:
    """A narrative case, which is always generated and never has a gold number."""
    return EvalQuestion(
        id=case.id,
        kind="narrative",
        split=split,
        fold=fold,
        ticker=case.ticker,
        cik=case.cik,
        fiscal_year=case.fiscal_year,
        question=case.question,
        answerable=case.answerable,
        negative_kind=None,
        e2e=True,
        expected_item=case.expected_item,
    )


@dataclass(frozen=True, slots=True)
class Grade:
    numeric_correct: bool | None
    citation_supported: bool | None
    gold_retrieved: bool | None
    cited_items: tuple[str, ...]


def grade(question: EvalQuestion, answer: Answer, passages: tuple[ScoredChunk, ...]) -> Grade:
    """Grade an answer against the passages it was written from.

    Only an answered case with a gold value gets the three checks; elsewhere
    they are ``None``, because "not correct" would read as a wrong answer.
    """
    cited = tuple(citation.marker for citation in answer.citations)
    items = tuple(citation.item for citation in answer.citations)
    gold = question.expected_value
    if answer.abstained or gold is None or answer.text is None:
        return Grade(None, None, None, items)
    cited_texts = [passages[marker - 1].chunk.text for marker in cited if marker <= len(passages)]
    return Grade(
        numeric_correct=is_numeric_match(answer.text, gold, question.tolerance),
        citation_supported=_prints(cited_texts, gold, question.per_share),
        gold_retrieved=_prints((p.chunk.text for p in passages), gold, question.per_share),
        cited_items=items,
    )


def _prints(texts: Iterable[str], value: Decimal, per_share: bool) -> bool:
    return any(PrintedNumbers.of(text).prints(value, per_share=per_share) for text in texts)


def split_is_sane(chunks: Iterable[Chunk], expected_item: str | None) -> bool:
    """Whether a filing's split into items can grade a cited item.

    The filing has at least ``MIN_SECTIONS`` items besides the full-filing
    fallback, no item holds more than ``MAX_SECTION_SHARE`` of its text, and
    the expected item is one of them.
    """
    sizes: Counter[str] = Counter()
    for chunk in chunks:
        sizes[chunk.item] += len(chunk.text)
    sections = {item for item in sizes if item != FULL_FILING}
    total = sum(sizes.values())
    if len(sections) < MIN_SECTIONS or total == 0:
        return False
    if max(sizes.values()) > MAX_SECTION_SHARE * total:
        return False
    return expected_item is None or expected_item in sections


def folds_by_ticker(cases: Iterable[GoldenCase]) -> Mapping[str, int | None]:
    """The cross-fitting fold of each company, as its golden cases carry it."""
    return {case.ticker: case.fold for case in cases}
