"""The golden set's schema: one JSON object per line, validated on read and write.

A ``GoldenCase`` is a question scoped to one pinned filing (``cik``,
``fiscal_year``, ``accession``) and its label:

* answerable: ``expected_value`` is what that filing's XBRL reported for
  ``concept`` over the fiscal year ``period_year``, and the builder checked the
  number is printed in the filing's indexed text;
* not answerable: ``negative_kind`` says which way the filing cannot answer it.

Every case is in the gate-only tier; ``e2e`` marks the subset that also gets a
generation. ``fold`` is the cross-fitting fold of an eval-split company and is
``None`` in the dev split.

Serialisation is canonical (sorted keys, no spaces, decimals as strings), so the
same cases always produce the same bytes and a file hash means something.
"""

import json
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Split = Literal["dev", "eval"]
NegativeKind = Literal["wrong_year", "other_company", "unreported_concept", "off_domain"]
NEGATIVE_KINDS: tuple[NegativeKind, ...] = (
    "wrong_year",
    "other_company",
    "unreported_concept",
    "off_domain",
)
DEFAULT_TOLERANCE = Decimal("0.005")


class GoldenCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    split: Split
    fold: int | None = Field(default=None, ge=0, le=1)
    e2e: bool
    ticker: str
    cik: int = Field(gt=0)
    fiscal_year: int
    accession: str
    question: str = Field(min_length=3)
    template: str
    answerable: bool
    negative_kind: NegativeKind | None = None
    concept: str | None = None
    period_year: int | None = None
    period_end: date | None = None
    expected_value: Decimal | None = None
    unit: str | None = None
    tolerance: Decimal = DEFAULT_TOLERANCE
    asked_company: str | None = None

    @model_validator(mode="after")
    def _label_is_consistent(self) -> Self:
        if self.answerable:
            if self.negative_kind is not None:
                raise ValueError("an answerable case has no negative kind")
            if self.expected_value is None or self.unit is None or self.concept is None:
                raise ValueError("an answerable case needs concept, expected_value and unit")
            if self.period_end is None or self.period_year is None:
                raise ValueError("an answerable case needs the period it asks about")
        else:
            if self.negative_kind is None:
                raise ValueError("an unanswerable case needs its negative kind")
            if self.expected_value is not None:
                raise ValueError("an unanswerable case has no expected value")
        if (self.split == "eval") != (self.fold is not None):
            raise ValueError("eval-split cases carry a fold and dev-split cases do not")
        if (self.negative_kind == "other_company") != (self.asked_company is not None):
            raise ValueError("asked_company is set exactly on other_company cases")
        return self


class NarrativeCase(BaseModel):
    """A question written by hand, reported apart from the generated cases.

    An answerable one names the ``expected_item`` and ``evidence_terms`` that
    item must contain; an unanswerable one lists ``absent_terms`` the whole
    filing must not contain. The builder checks both against the snapshots.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    ticker: str
    cik: int = Field(gt=0)
    fiscal_year: int
    accession: str
    question: str = Field(min_length=3)
    answerable: bool
    expected_item: str | None = None
    evidence_terms: tuple[str, ...] = ()
    absent_terms: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _evidence_matches_label(self) -> Self:
        if self.answerable and (self.expected_item is None or not self.evidence_terms):
            raise ValueError("an answerable narrative needs expected_item and evidence_terms")
        if not self.answerable and (self.expected_item is not None or not self.absent_terms):
            raise ValueError("an unanswerable narrative needs absent_terms and no expected_item")
        return self


def dumps_line(case: BaseModel) -> str:
    """One canonical JSON line, without the newline."""
    payload = case.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dumps_cases(cases: Iterable[BaseModel]) -> str:
    return "".join(f"{dumps_line(case)}\n" for case in cases)


def load_cases[T: BaseModel](path: Path, model: type[T]) -> tuple[T, ...]:
    """Every line of ``path`` validated as ``model``; ids must be unique.

    Raises:
        ValueError: naming the line that does not validate, or a repeated id.
    """
    cases: list[T] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = model.model_validate_json(line)
        except ValueError as error:
            raise ValueError(f"{path.name}:{number}: {error}") from error
        identifier = str(getattr(case, "id", ""))
        if identifier in seen:
            raise ValueError(f"{path.name}:{number}: repeated id {identifier!r}")
        seen.add(identifier)
        cases.append(case)
    return tuple(cases)
