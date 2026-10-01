from decimal import Decimal

import pytest

from edgar_rag.eval.numeric import Amount, amount_matches, amounts_in, is_numeric_match


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("$391.0 billion", "391000000000", "100000000"),
        ("391,035 million", "391035000000", "1000000"),
        ("$1,234,567", "1234567", "1"),
        ("6.08", "6.08", "0.01"),
        ("2.5 Thousand", "2500", "100"),
        ("1.1 trillions", "1100000000000", "100000000000"),
        ("(1,234) million", "-1234000000", "1000000"),
        ("$(1.2) billion", "-1200000000", "100000000"),
        ("(1.2 billion)", "-1200000000", "100000000"),
        ("-$5 million", "-5000000", "1000000"),
        ("−3.5 million", "-3500000", "100000"),
        ("– 7 million", "-7000000", "1000000"),
        ("US$1.2 billion", "1200000000", "100000000"),
    ],
)
def test_one_amount_is_parsed_with_its_displayed_precision(text, value, unit):
    (amount,) = amounts_in(text)

    assert amount.value == Decimal(value)
    assert amount.unit == Decimal(unit)


def test_an_unclosed_parenthesis_is_not_a_negative():
    (amount,) = amounts_in("(revenue was $5 billion")

    assert amount.value == Decimal(5) * 10**9


@pytest.mark.parametrize(
    "text",
    ["fiscal 2024", "2023–2024", "1,2345", "in 1999 and 2099", "the year"],
)
def test_years_and_malformed_numbers_are_not_amounts(text):
    assert amounts_in(text) == ()


def test_a_year_with_a_currency_or_scale_is_an_amount():
    assert [a.value for a in amounts_in("$2024 and 2024 million")] == [
        Decimal(2024),
        Decimal(2024) * 10**6,
    ]


def test_every_amount_in_a_sentence_is_found_in_order():
    text = "Net sales were $391.0 billion in fiscal 2024, up from $383.3 billion."

    assert [a.text for a in amounts_in(text)] == ["$391.0 billion", "$383.3 billion"]


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("$391.0 billion", 391_035_000_000, True),  # rounding to one decimal in billions
        ("$391 billion", 391_035_000_000, True),
        ("$392.0 billion", 391_035_000_000, True),  # does not round to gold, but 0.25% off
        ("$393.0 billion", 391_035_000_000, False),  # 0.503% off and not a rounding
        ("$0.4 billion", 383_000_000, True),  # 4% off, but it is 383m written to 0.1bn
        ("$0.5 billion", 383_000_000, False),
        ("$1,000", 1_004, True),  # within 0.5% relative tolerance
        ("$1,000", 1_006, False),  # 0.6%: neither rule
        ("(1,234) million", -1_234_000_000, True),
        ("1,234 million", -1_234_000_000, False),  # sign matters
        ("$6.08", "6.08", True),
        ("about 6.1", "6.08", True),  # 6.08 rounds to 6.1
        ("6.13", "6.08", False),
        ("no figures here", 100, False),
        ("in fiscal 2024 it was $5 million", 2_024, False),  # the year is not an amount
    ],
)
def test_numeric_match(answer, gold, expected):
    assert is_numeric_match(answer, gold) is expected


def test_any_matching_amount_in_the_answer_counts():
    assert is_numeric_match("$383.3 billion, up to $391.0 billion", 391_035_000_000)


def test_the_tolerance_is_configurable():
    amount = Amount(value=Decimal(1000), unit=Decimal(1), text="1000")

    assert not amount_matches(amount, Decimal(1010))
    assert amount_matches(amount, Decimal(1010), relative_tolerance=Decimal("0.01"))


def test_a_zero_gold_value_only_matches_within_displayed_precision():
    assert is_numeric_match("$0.0 million", 0)
    assert not is_numeric_match("$0.04 million", 0)
