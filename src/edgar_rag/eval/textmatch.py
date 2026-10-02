"""Is an XBRL value printed in a filing's text, in any of the ways a 10-K prints it?

A 10-K prints the same figure several ways: ``391,035`` in a statement headed
"in millions", ``$391.0 billion`` in the discussion, ``6.08`` for a per-share
amount. ``PrintedNumbers`` reads every amount in a text once (with the parser
the numeric accuracy metric uses) and indexes each under the values it can
stand for: a bare number as units, thousands or millions; a number with a scale
word only as written. A value is printed when rounding it to an indexed
amount's displayed precision gives that amount.

Two strengths, because the two uses fail in opposite directions:

* ``prints`` (keeping a positive) is strict: a bare table number needs at
  least ``STRICT_BARE_DIGITS`` significant digits and a scaled one
  ``STRICT_SCALED_DIGITS``, so a page number or a "$1 billion" does not stand in
  for the figure;
* ``may_print`` (keeping a negative) is loose: any match with
  ``LOOSE_DIGITS`` digits counts, so a negative is only kept when no plausible
  rendering of the value is anywhere in the text.

Per-share amounts are matched only to the cent and only as written.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from edgar_rag.eval.numeric import Amount, amounts_in

STRICT_BARE_DIGITS = 4
STRICT_SCALED_DIGITS = 3
LOOSE_DIGITS = 2
# Statements print in units, thousands or millions; the heading is not parsed
BARE_MULTIPLIERS = (Decimal(1), Decimal(10) ** 3, Decimal(10) ** 6)
CENT = Decimal("0.01")
_DIGITS = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NOT_WORD = re.compile(r"[^0-9a-z]+")

Key = tuple[Decimal, Decimal]  # (absolute value, displayed step)


def significant_digits(amount: Amount) -> int:
    """Digits the amount was written with, leading zeros excluded."""
    match = _DIGITS.search(amount.text)
    digits = "" if match is None else re.sub(r"\D", "", match.group(0))
    return len(digits.lstrip("0"))


def _is_scaled(amount: Amount) -> bool:
    return re.search(r"thousand|million|billion|trillion", amount.text, re.IGNORECASE) is not None


@dataclass(frozen=True, slots=True)
class PrintedNumbers:
    """Every amount in a text, indexed by the values it can stand for."""

    bare: dict[Key, int]
    scaled: dict[Key, int]
    cents: frozenset[Decimal]
    bare_steps: tuple[Decimal, ...]
    scaled_steps: tuple[Decimal, ...]

    @classmethod
    def of(cls, text: str) -> "PrintedNumbers":
        bare: dict[Key, int] = {}
        scaled: dict[Key, int] = {}
        cents: set[Decimal] = set()
        for amount in amounts_in(text):
            digits = significant_digits(amount)
            value = abs(amount.value)
            if _is_scaled(amount):
                _keep_max(scaled, (value, amount.unit), digits)
                continue
            if amount.unit == CENT:
                cents.add(value)
            for multiplier in BARE_MULTIPLIERS:
                _keep_max(bare, (value * multiplier, amount.unit * multiplier), digits)
        return cls(
            bare=bare,
            scaled=scaled,
            cents=frozenset(cents),
            bare_steps=_steps(bare),
            scaled_steps=_steps(scaled),
        )

    def prints(self, value: Decimal, *, per_share: bool = False) -> bool:
        """Whether ``value`` is printed, by the strict rule."""
        if per_share:
            return abs(value).quantize(CENT) == abs(value) and abs(value) in self.cents
        return self._bare_digits(value) >= STRICT_BARE_DIGITS or (
            self._scaled_digits(value) >= STRICT_SCALED_DIGITS
        )

    def may_print(self, value: Decimal, *, per_share: bool = False) -> bool:
        """Whether any plausible rendering of ``value`` is in the text (loose rule)."""
        if per_share and abs(value).quantize(CENT) in self.cents:
            return True
        best = max(self._bare_digits(value), self._scaled_digits(value))
        return best >= LOOSE_DIGITS

    def _bare_digits(self, value: Decimal) -> int:
        return _digits(self.bare, self.bare_steps, value)

    def _scaled_digits(self, value: Decimal) -> int:
        return _digits(self.scaled, self.scaled_steps, value)


def _digits(index: dict[Key, int], steps: tuple[Decimal, ...], value: Decimal) -> int:
    """The most digits of any indexed amount that ``value`` rounds to."""
    target = abs(value)
    best = 0
    for printed, step in _candidates(index, steps, target):
        if abs(printed - target) <= step / 2:
            best = max(best, index[(printed, step)])
    return best


def _steps(index: dict[Key, int]) -> tuple[Decimal, ...]:
    return tuple(sorted({step for _, step in index if step > 0}))


def _candidates(
    index: dict[Key, int], steps: tuple[Decimal, ...], target: Decimal
) -> Iterable[Key]:
    """The keys ``target`` could round to: its neighbours at every step in use."""
    for step in steps:
        for rounding in (ROUND_FLOOR, ROUND_CEILING):
            printed = (target / step).to_integral_value(rounding=rounding) * step
            if (printed, step) in index:
                yield printed, step


def _keep_max(index: dict[Key, int], key: Key, digits: int) -> None:
    index[key] = max(index.get(key, 0), digits)


def normalize_words(text: str) -> str:
    """Lower case, every run of non-alphanumerics as one space, padded."""
    return f" {_NOT_WORD.sub(' ', text.lower()).strip()} "


def contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Whether ``phrase`` occurs as whole words in a ``normalize_words`` text."""
    return normalize_words(phrase) in normalized_text
