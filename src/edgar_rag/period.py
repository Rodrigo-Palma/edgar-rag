"""Declining a question about a year the filing does not report on.

A relevance score cannot see a year. "What was revenue in fiscal 2019?" sits
next to the revenue passage of a 2024 10-K whether it embeds the words or asks
a model whether the passage answers it, and the passage does answer it, for
another year. The guard is the cheap rule in front: no model, no XBRL value,
only the years the question names and the fiscal year the filing's metadata
says it covers.
"""

import re
from dataclasses import dataclass

from edgar_rag.domain import (
    AbstentionReason,
    GateDecision,
    IndexedFiling,
    ScoredChunk,
)

PERIOD = "period"
"""The name the guard files its score under in ``GateDecision.scores``."""

YEARS_REPORTED = 3
"""Fiscal years a 10-K reports: its own and the two before it, as the income
statement and the cash flow statement show them."""

# "2024", "fiscal 2024", "in 2019": a four-digit year standing on its own.
_FOUR_DIGIT_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
# "FY2024", "FY 2024", "FY24", "fy'24": the abbreviation, with two or four digits.
_FISCAL_ABBREVIATION = re.compile(r"\bFY ?['’]?(\d{4}|\d{2})\b", re.IGNORECASE)
_CENTURY = 100


def years_named(question: str, near: int) -> frozenset[int]:
    """The years ``question`` names, with a two-digit ``FY24`` read near ``near``.

    A two-digit year is read as the year with those last two digits closest
    to ``near``, the filing's fiscal year, so ``FY24`` asked of a 2025 filing
    is 2024 and ``FY99`` is 1999, not 2099.
    """
    years = {int(match) for match in _FOUR_DIGIT_YEAR.findall(question)}
    for digits in _FISCAL_ABBREVIATION.findall(question):
        number = int(digits)
        years.add(number if len(digits) == 4 else _closest_with_last_two_digits(number, near))
    return frozenset(years)


def _closest_with_last_two_digits(last_two: int, near: int) -> int:
    candidate = near - near % _CENTURY + last_two
    return min(
        (candidate - _CENTURY, candidate, candidate + _CENTURY),
        key=lambda year: abs(year - near),
    )


@dataclass(frozen=True, slots=True)
class PeriodGuard:
    """Decline a question that names a fiscal year the filing does not report.

    The filing covers its own fiscal year and the ``years_reported - 1``
    before it. A question that names no year passes: most questions mean the
    filing's own year and say so only through the scope. A question naming
    several years passes only when the filing reports every one of them,
    because an answer for part of the question is not an answer to it.

    The score is 1.0 when the guard admits and 0.0 when it declines: it is a
    rule, with no threshold to tune, so it ranks nothing.

    What it cannot tell apart: a year that is not a period ("notes due 2030",
    "the 2019 Omnibus Plan") is read as one, and a question naming a fiscal
    year in words ("last year") names none.

    Raises:
        ValueError: when ``years_reported`` is below one.
    """

    years_reported: int = YEARS_REPORTED

    def __post_init__(self) -> None:
        if self.years_reported < 1:
            raise ValueError("a filing reports at least its own fiscal year")

    def covered(self, filing: IndexedFiling) -> range:
        """The fiscal years ``filing`` reports, oldest first."""
        return range(filing.fiscal_year - self.years_reported + 1, filing.fiscal_year + 1)

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision:
        covered = self.covered(filing)
        span = f"this filing reports fiscal {covered[0]} to {covered[-1]}"
        named = years_named(question, near=filing.fiscal_year)
        outside = sorted(year for year in named if year not in covered)
        if outside:
            return GateDecision(
                admitted=False,
                confidence=0.0,
                reason=f"the question names fiscal {_listed(outside)}; {span}",
                rejection=AbstentionReason.OUT_OF_PERIOD,
                scores={PERIOD: 0.0},
            )
        said = f"the question names fiscal {_listed(sorted(named))}" if named else "no year named"
        return GateDecision(
            admitted=True, confidence=1.0, reason=f"{said}; {span}", scores={PERIOD: 1.0}
        )


def _listed(years: list[int]) -> str:
    return ", ".join(str(year) for year in years)
