"""Is an XBRL value printed in a filing: strict to keep a positive, loose to keep a negative."""

from decimal import Decimal

import pytest

from edgar_rag.eval.textmatch import PrintedNumbers, contains_phrase, normalize_words

STATEMENT = """
CONSOLIDATED BALANCE SHEETS (in millions)
Total assets
$
364,980
352,583
Diluted earnings per share
$
6.08
6.13
See Note 105 on page 212.
Total net sales increased 2% to $391.0 billion.
Dividends of $15 billion were paid.
Capital expenditures were ( 9,447 ) in the year.
"""


@pytest.fixture(scope="module")
def printed():
    return PrintedNumbers.of(STATEMENT)


@pytest.mark.parametrize(
    "value",
    [
        Decimal("364980000000"),  # a statement line in millions
        Decimal("352583000000"),
        Decimal("391035000000"),  # $391.0 billion rounds from it
        Decimal("9447000000"),  # accounting parentheses with spaces
    ],
)
def test_a_value_printed_as_a_statement_or_in_prose_is_found(printed, value):
    assert printed.prints(value)


def test_a_per_share_amount_is_matched_to_the_cent(printed):
    assert printed.prints(Decimal("6.08"), per_share=True)
    assert not printed.prints(Decimal("6.07"), per_share=True)
    assert not printed.prints(Decimal("6.075"), per_share=True)


def test_a_value_that_is_not_printed_is_not_found(printed):
    assert not printed.prints(Decimal("383285000000"))
    assert not printed.may_print(Decimal("383285000000"))


def test_a_short_bare_number_does_not_stand_in_for_a_figure(printed):
    # "105" is a note number; read as 105 million it must not keep a positive
    assert not printed.prints(Decimal("105000000"))
    # but a negative is dropped when it could be that figure
    assert printed.may_print(Decimal("105000000"))


def test_a_two_digit_scaled_amount_only_counts_for_the_loose_rule(printed):
    # "$15 billion" is too coarse to confirm 15.2 billion, too close to call it absent
    assert not printed.prints(Decimal("15200000000"))
    assert printed.may_print(Decimal("15200000000"))
    assert not printed.may_print(Decimal("16000000000"))


def test_the_sign_does_not_matter(printed):
    assert printed.prints(Decimal("-9447000000"))


def test_a_year_is_not_read_as_an_amount():
    assert not PrintedNumbers.of("In fiscal 2024 we grew.").may_print(Decimal("2024"))


def test_phrases_match_whole_words_whatever_the_punctuation():
    words = normalize_words("Net-interest income rose; NONINTEREST income fell.")

    assert contains_phrase(words, "net interest income")
    assert contains_phrase(words, "noninterest income")
    assert not contains_phrase(words, "interest inc")
