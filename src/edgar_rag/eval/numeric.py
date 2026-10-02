"""Numeric accuracy: does an answer state the gold value, as a reader would check it.

The gold value is the exact figure from XBRL (``391035000000``); answers write it
the way a filing does (``$391.0 billion``, ``(1,234) million``). An amount in
the answer matches when either

* it is within a relative tolerance of the gold value (0.5% by default), or
* it equals the gold value rounded to the digits the answer displays: "$0.4
  billion" matches 383 million, because that is how 383 million is written to
  one decimal in billions.

The parsing rules live in ``edgar_rag.amounts``, shared with the citation
support check.

Known limit: an answer matches if any amount in it matches, so an answer that
lists several figures gets credit for the right one among them.
"""

from decimal import Decimal

from edgar_rag.amounts import DEFAULT_RELATIVE_TOLERANCE, Amount, amount_matches, amounts_in

__all__ = [
    "DEFAULT_RELATIVE_TOLERANCE",
    "Amount",
    "amount_matches",
    "amounts_in",
    "is_numeric_match",
]


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
