"""Figures in words, and the percentage mark, as the citation check reads them."""

from decimal import Decimal

import pytest

from edgar_rag.amounts import amounts_in, spelled_amounts_in


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
