"""Figures in words, and the percentage mark, as the citation check reads them."""

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgar_rag.amounts import amounts_in, number_in_words, spelled_amounts_in


@pytest.mark.parametrize(
    ("text", "value", "unit", "is_percent"),
    [
        ("ninety billion dollars", Decimal(90) * 10**9, Decimal(10) ** 9, False),
        ("ninety-one billion", Decimal(91) * 10**9, Decimal(10) ** 9, False),
        ("one hundred and twenty-five million", Decimal(125) * 10**6, Decimal(10) ** 6, False),
        ("twelve percent", Decimal(12), Decimal(1), True),
        ("five per cent", Decimal(5), Decimal(1), True),
        ("seven dollars", Decimal(7), Decimal(1), False),
        ("a hundred thousand", Decimal(100) * 10**3, Decimal(10) ** 3, False),
        ("one point five billion dollars", Decimal("1.5") * 10**9, Decimal(10) ** 8, False),
        ("two point zero five billion", Decimal("2.05") * 10**9, Decimal(10) ** 7, False),
        ("two and a half billion dollars", Decimal("2.5") * 10**9, Decimal(10) ** 8, False),
        ("twelve and a half percent", Decimal("12.5"), Decimal("0.1"), True),
        ("two thousand five hundred dollars", Decimal(2500), Decimal(1), False),
        ("one hundred and five million", Decimal(105) * 10**6, Decimal(10) ** 6, False),
        ("one billion two hundred million", Decimal("1.2") * 10**9, Decimal(10) ** 6, False),
        ("a billion dollars", Decimal(10) ** 9, Decimal(10) ** 9, False),
        ("one thousand two hundred million", Decimal(1200) * 10**6, Decimal(10) ** 6, False),
        ("two thousand million dollars", Decimal(2) * 10**9, Decimal(10) ** 6, False),
    ],
)
def test_a_figure_in_words_is_read_at_the_scale_it_names(text, value, unit, is_percent):
    (amount,) = spelled_amounts_in(text)

    assert (amount.value, amount.unit, amount.is_percent) == (value, unit, is_percent)


@pytest.mark.parametrize(
    "text", ["two segments", "one of the risks", "someone paid", "the tenth billion"]
)
def test_number_words_that_count_nothing_are_not_a_figure(text):
    assert spelled_amounts_in(text) == ()


def test_a_percentage_is_marked_without_changing_its_value_or_text():
    found = amounts_in("margin was 46.2% and then 12 percent, on 7 stores")

    assert [(a.value, a.text, a.is_percent) for a in found] == [
        (Decimal("46.2"), "46.2", True),
        (Decimal(12), "12", True),
        (Decimal(7), "7", False),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "three quarters percent",
        "two and three percent",
        "five million six million dollars",
        "one half billion",
        "twenty thirty percent",
        "five five dollars",
        "three quarters of a billion dollars",
        "half a billion",
        "one point billion",
        "one point twelve billion",
        "one million point five billion",
    ],
)
def test_a_figure_in_words_it_cannot_compose_is_unreadable_not_misread(text):
    """Fail closed: words that do not compose stay a figure, one with no value."""
    (amount,) = spelled_amounts_in(text)[-1:]

    assert amount.is_readable is False


def test_a_figure_in_digits_is_always_readable():
    assert all(amount.is_readable for amount in amounts_in("$1.5 billion, 12%, 7 stores"))


_UNITS = [
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
]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _below_thousand(n: int, with_and: bool) -> str:
    hundreds, rest = divmod(n, 100)
    words = [f"{_UNITS[hundreds]} hundred"] if hundreds else []
    if rest and hundreds and with_and:
        words.append("and")
    if rest >= 20:
        tens, unit = divmod(rest, 10)
        words.append(_TENS[tens - 2] + (f"-{_UNITS[unit]}" if unit else ""))
    elif rest:
        words.append(_UNITS[rest])
    return " ".join(words)


def _in_words(n: int, with_and: bool) -> str:
    words, remaining = [], n
    for scale, size in (("billion", 10**9), ("million", 10**6), ("thousand", 10**3)):
        count, remaining = divmod(remaining, size)
        if count:
            words.append(f"{_below_thousand(count, with_and)} {scale}")
    if remaining:
        words.append(_below_thousand(remaining, with_and))
    return " ".join(words)


@given(st.integers(min_value=1, max_value=999_999_999_999), st.booleans())
def test_every_whole_number_in_words_reads_back_to_itself(n, with_and):
    (amount,) = spelled_amounts_in(f"{_in_words(n, with_and)} dollars")

    assert (amount.is_readable, amount.value) == (True, Decimal(n))


_ALPHABET = [*_UNITS, *_TENS, "hundred", "thousand", "million", "billion", "and", "a", "half"]
_ALPHABET += ["point", "quarter", "quarters", "percent", "dollars", "of"]


@given(st.lists(st.sampled_from(_ALPHABET), min_size=1, max_size=12))
def test_any_run_of_number_words_is_read_or_marked_unreadable_never_raises(words):
    for amount in spelled_amounts_in(" ".join(words)):
        assert amount.is_readable or (amount.value, amount.unit) == (Decimal(0), Decimal(0))


@given(st.integers(min_value=0, max_value=999_999_999_999), st.integers(0, 3))
def test_a_number_written_in_words_reads_back_to_its_value(n, decimals):
    number = f"{n / 10**decimals:,.{decimals}f}"

    (amount,) = spelled_amounts_in(f"{number_in_words(number)} dollars")

    assert amount.value == Decimal(number.replace(",", ""))


@pytest.mark.parametrize("number", ["1.2.3", "-5", "1,23", "1000000000000", ""])
def test_number_in_words_refuses_what_is_not_a_number_below_a_trillion(number):
    with pytest.raises(ValueError, match=r"not a number|trillion"):
        number_in_words(number)
