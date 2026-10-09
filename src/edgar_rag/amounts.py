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
decimals is read as a year and ignored. An amount followed by ``%`` or
``percent`` is marked as a percentage; its value and text are unchanged.

``spelled_amounts_in`` reads a figure written in words, but only one that says
what it counts: number words with a scale word, ``percent`` or ``dollars``
(``ninety billion dollars``, ``twelve percent``). Number words alone (``two
segments``) are wording, not a figure. The words compose as English writes
them: ``hundred`` multiplies, a scale word steps down (``one billion two
hundred million``) or multiplies what is all below it (``one thousand two
hundred million``), ``point`` takes digit words (``one point five billion``), and ``a
half`` adds one half (``two and a half billion``). Words that do not compose
under those rules (``three quarters percent``, ``two and three percent``) are
kept as a figure marked unreadable, so a check holding it to a passage fails
closed instead of reading part of it.

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
_PERCENT_AFTER = re.compile(r"\s*(?:%|percent\b|per\s+cent\b)", re.IGNORECASE)

_UNIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS_WORDS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_WORD_VALUES = {
    **{word: value for value, word in enumerate(_UNIT_WORDS)},
    **{word: 10 * value for value, word in enumerate(_TENS_WORDS, start=2)},
}
_NUMBER_WORD = "(?:" + "|".join(sorted(_WORD_VALUES, key=len, reverse=True)) + ")"
_SCALE_WORD = r"(?:thousand|million|billion|trillion)s?"
_WORD = rf"(?:{_NUMBER_WORD}|hundred|{_SCALE_WORD}|half|quarters?)"
# A figure starts at a number word, or at "a" before "hundred" or a scale
# word; "and", "a" and "point" only ever join two words of it.
_SPELLED = re.compile(
    rf"""
    (?<![\w-])
    (?P<number>
        (?:{_NUMBER_WORD}|a[\s-]+(?:hundred|{_SCALE_WORD}))
        (?:[\s-]+(?:and[\s-]+)?(?:a[\s-]+)?(?:point[\s-]+)?{_WORD})*
    )
    (?:\s+(?P<kind>percent|per\s+cent|dollars?))?
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)
# "three quarters of a billion", "half a billion": the figure is a fraction of
# the words that follow, which this parser does not compute.
_FRACTION_BEFORE = re.compile(
    r"\b(?:half|quarters?|thirds?|fifths?|tenths?)[\s-]+(?:of[\s-]+)?\Z", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class Amount:
    """A number as written: its value and the smallest step its digits can show."""

    value: Decimal
    unit: Decimal
    text: str
    is_percent: bool = False
    is_readable: bool = True
    """False for a figure in words that does not compose: it states a figure, but no value."""


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
                is_percent=bool(_PERCENT_AFTER.match(text, match.end())),
            )
        )
    return tuple(found)


def spelled_amounts_in(text: str) -> tuple[Amount, ...]:
    """Every figure written in words in ``text`` that names a scale, a percentage or dollars.

    The unit is the smallest step the words show (``ninety billion`` shows
    billions, ``one point five billion`` a tenth of one), so the figure is held
    to the precision it was written at, like one in digits.
    """
    found = []
    for match in _SPELLED.finditer(text):
        words = _words(match["number"])
        kind = match["kind"]
        if not (kind or any(_scale_of(word) for word in words)):
            continue
        is_fraction = bool(_FRACTION_BEFORE.search(text, 0, match.start()))
        parsed = None if is_fraction else _spelled_value(words)
        value, unit = parsed if parsed is not None else (Decimal(0), Decimal(0))
        found.append(
            Amount(
                value=value,
                unit=unit,
                text=match.group(0).strip(),
                is_percent=bool(kind and kind.lower().startswith("per")),
                is_readable=parsed is not None,
            )
        )
    return tuple(found)


def _words(number: str) -> list[str]:
    return [word for word in re.split(r"[\s-]+", number.lower()) if word]


def _scale_of(word: str) -> Decimal | None:
    return _SCALES.get(word.removesuffix("s"))


def _spelled_value(words: list[str]) -> tuple[Decimal, Decimal] | None:
    """The value and unit of a figure in words, or ``None`` when the words do not compose."""
    if "point" in words:
        return _decimal_value(words)
    return _integer_value(words)


def _decimal_value(words: list[str]) -> tuple[Decimal, Decimal] | None:
    """``one point five billion``: an integer, ``point``, digit words, then at most one scale."""
    at = words.index("point")
    whole = _integer_value(words[:at])
    tail = words[at + 1 :]
    scale = _scale_of(tail[-1]) if tail else None
    digits = tail[:-1] if scale is not None else tail
    if whole is None or whole[1] != 1 or not digits:
        return None
    if any(_WORD_VALUES.get(word, 10) > 9 for word in digits):
        return None
    fraction = "".join(str(_WORD_VALUES[word]) for word in digits)
    step = scale or Decimal(1)
    return Decimal(f"{whole[0]}.{fraction}") * step, step * Decimal(10) ** -len(digits)


def _integer_value(words: list[str]) -> tuple[Decimal, Decimal] | None:
    """Groups below a thousand, each closed by a scale word (see ``_close_group``)."""
    total, group, half = Decimal(0), Decimal(0), False
    last_scale: Decimal | None = None
    previous = ""
    for position, word in enumerate(words):
        following = words[position + 1 : position + 3]
        scale = _scale_of(word)
        if not _follows(previous, word, following):
            return None
        if scale is not None:
            closed = _close_group(total, group, scale, last_scale)
            if closed is None:
                return None
            total, group, last_scale = closed, Decimal(0), scale
        elif word == "hundred":
            group = max(group, Decimal(1)) * 100
        elif word == "half":
            group, half = group + Decimal("0.5"), True
        elif word in _WORD_VALUES:
            group += _WORD_VALUES[word]
        elif word == "a" and following[:1] != ["half"]:
            group = Decimal(1)
        previous = word
    unit = Decimal(1) if group or last_scale is None else last_scale
    step = unit / 10 if half else unit
    return total + group, step


def _close_group(
    total: Decimal, group: Decimal, scale: Decimal, last_scale: Decimal | None
) -> Decimal | None:
    """The total once ``scale`` closes ``group``, or ``None`` when the scales do not compose.

    A smaller scale adds its group (``one billion two hundred million``); a
    larger one multiplies everything before it, but only when all of that is
    below it (``one thousand two hundred million``, not ``five million six
    million``).
    """
    if last_scale is None or scale < last_scale:
        return total + group * scale if group else None
    if total + group < scale:
        return (total + group) * scale
    return None


def _follows(previous: str, word: str, following: list[str]) -> bool:
    """Whether ``word`` may come after ``previous`` in a figure written in words."""
    if previous == "half":
        return _scale_of(word) is not None
    if word == "and":
        closes_group = previous == "hundred" or _scale_of(previous) is not None
        return closes_group or following == ["a", "half"]
    if word == "a":
        return previous in ("", "and") and bool(following) and following[0] != "a"
    if word == "half":
        return previous == "a"
    if word == "hundred":
        return previous in _WORD_VALUES or previous == "a"
    if word in _WORD_VALUES:
        return _after_number(previous, word)
    return _scale_of(word) is not None and previous not in ("", "and")


def _after_number(previous: str, word: str) -> bool:
    if previous in ("", "and", "hundred") or _scale_of(previous) is not None:
        return previous != "hundred" or _WORD_VALUES[word] > 0
    return previous in _TENS_WORDS and 0 < _WORD_VALUES[word] < 10


def amount_matches(
    amount: Amount, gold: Decimal, relative_tolerance: Decimal = DEFAULT_RELATIVE_TOLERANCE
) -> bool:
    """Whether one written amount states ``gold`` under the two rules above."""
    error = abs(amount.value - gold)
    return error <= relative_tolerance * abs(gold) or error <= amount.unit / 2
