"""Numeric accuracy: does an answer state the gold value, as a reader would check it.

The gold value is the exact figure from XBRL (``391035000000``); answers write it
the way a filing does (``$391.0 billion``, ``(1,234) million``). An amount in
the answer matches when either

* it is within a relative tolerance of the gold value (0.5% by default), or
* it equals the gold value rounded to the digits the answer displays: "$0.4
  billion" matches 383 million, because that is how 383 million is written to
  one decimal in billions.

Parsing rules, deliberately few and all tested: thousands commas; an optional
``$``; scale words ``thousand``, ``million``, ``billion``, ``trillion`` (and
plurals); a leading minus (hyphen, minus sign or en dash) or accounting
parentheses wrapping the amount (``(1.2) billion`` or ``(1.2 billion)``) mark a
negative. A bare four-digit number from 1900 to 2099 with no ``$``, scale or
decimals is read as a year and ignored.

Known limit: an answer matches if any amount in it matches, so an answer that
lists several figures gets credit for the right one among them.
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


def is_numeric_match(
    answer: str,
    gold: Decimal | int | str,
    relative_tolerance: Decimal = DEFAULT_RELATIVE_TOLERANCE,
) -> bool:
    """Whether any amount stated in ``answer`` matches the gold value."""
    gold_value = Decimal(gold)
    return any(
        amount_matches(amount, gold_value, relative_tolerance) for amount in amounts_in(answer)
    )
