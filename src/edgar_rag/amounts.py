"""Amounts as a filing writes them, read into values a check can compare.

Two checks need the same reading of a number: the evaluation, which asks
whether an answer states the gold XBRL value, and the citation support check,
which asks whether a number in an answer appears in the passage it cites. One
parser serves both, so a figure the evaluation counts as stated is a figure
the support check can find.

Parsing rules, deliberately few and all tested: thousands commas; an optional
``$``; scale words ``thousand``, ``million``, ``billion``, ``trillion`` (and
plurals); a leading minus (hyphen, minus sign or en dash) or accounting
parentheses wrapping the amount (``(1.2) billion`` or ``(1.2 billion)``) mark a
negative. A bare four-digit number from 1900 to 2099 with no ``$``, scale or
decimals is read as a year and ignored.

``amount_matches`` holds the two ways an amount states a value: within a
relative tolerance, or equal to the value rounded to the digits the amount
displays ("$0.4 billion" states 383 million).
"""

import re
from dataclasses import dataclass
from decimal import Decimal

DEFAULT_RELATIVE_TOLERANCE = Decimal("0.005")

_SCALES = {
    "thousand": Decimal(10) ** 3,
    "million": Decimal(10) ** 6,
    "billion": Decimal(10) ** 9,
    "trillion": Decimal(10) ** 12,
}

_AMOUNT = re.compile(
    r"""
    (?<![\w.,])
    (?P<open>\(\s*)?
    (?P<minus>[-−–]\s*)?
    (?P<dollar>\$\s*)?
    (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?![\d,]*\d)
    (?P<close_number>\s*\))?
    (?:\s*(?P<scale>thousand|million|billion|trillion)s?\b)?
    (?P<close_scale>\s*\))?
    """,
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Amount:
    """A number as written: its value and the smallest step its digits can show."""

    value: Decimal
    unit: Decimal
    text: str


def _looks_like_year(number: str, has_context: bool) -> bool:
    return not has_context and len(number) == 4 and number.isdigit() and 1900 <= int(number) <= 2099


def amounts_in(text: str) -> tuple[Amount, ...]:
    """Every amount written in ``text``, in order, skipping bare years."""
    found = []
    for match in _AMOUNT.finditer(text):
        number = match["number"]
        scale_word = match["scale"]
        has_context = bool(match["dollar"] or scale_word or "." in number or "," in number)
        if _looks_like_year(number, has_context):
            continue
        digits = Decimal(number.replace(",", ""))
        scale = _SCALES[scale_word.lower()] if scale_word else Decimal(1)
        decimals = len(number.split(".")[1]) if "." in number else 0
        is_negative = bool(match["minus"]) or bool(
            match["open"] and (match["close_number"] or match["close_scale"])
        )
        value = digits * scale
        found.append(
            Amount(
                value=-value if is_negative else value,
                unit=scale * Decimal(10) ** -decimals,
                text=match.group(0).strip(),
            )
        )
    return tuple(found)


def amount_matches(
    amount: Amount, gold: Decimal, relative_tolerance: Decimal = DEFAULT_RELATIVE_TOLERANCE
) -> bool:
    """Whether one written amount states ``gold`` under the two rules above."""
    error = abs(amount.value - gold)
    return error <= relative_tolerance * abs(gold) or error <= amount.unit / 2
