from dataclasses import replace

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from edgar_rag.domain import AbstentionReason
from edgar_rag.period import PeriodGuard, years_named
from tests.fakes import EXAMPLE

# EXAMPLE is fiscal 2024, so it reports 2022, 2023 and 2024
FISCAL_2024 = EXAMPLE


@pytest.mark.parametrize(
    ("question", "years"),
    [
        ("What was revenue?", set()),
        ("What was revenue in 2024?", {2024}),
        ("What was revenue in fiscal 2019?", {2019}),
        ("What was revenue for fiscal year 1998?", {1998}),
        ("How did revenue change from 2022 to 2024?", {2022, 2024}),
        ("Revenue in FY2023?", {2023}),
        ("Revenue in FY 2023?", {2023}),
        ("Revenue in fy2023?", {2023}),
        ("Revenue in FY23?", {2023}),
        ("Revenue in FY 23?", {2023}),
        ("Revenue in FY'23?", {2023}),
        ("Revenue in FY’23?", {2023}),
        ("Revenue in FY99?", {1999}),
        ("What was net income (2024)?", {2024}),
        ("What was revenue in 2024, 2023 and 2022?", {2022, 2023, 2024}),
        # Not years: digits that only look like one, and years run into other text
        ("Revenue of $2,024 million?", set()),
        ("How many of the 12024 shares were sold?", set()),
        ("What does note 2024a say?", set()),
        ("What happened in 1850 or 2150?", set()),
        ("Revenue in FY123?", set()),
        ("Revenue in FYI 2023?", {2023}),
    ],
)
def test_the_years_a_question_names(question, years):
    assert years_named(question, near=2024) == years


@pytest.mark.parametrize(
    ("near", "year"), [(2025, 2024), (2001, 1999), (1999, 2001), (2050, 2024), (2075, 2124)]
)
def test_a_two_digit_year_is_the_closest_one_to_the_filing(near, year):
    assert years_named(f"Revenue in FY{year % 100:02d}?", near=near) == {year}


@pytest.mark.parametrize(
    "question",
    [
        "What was Example's revenue?",
        "What was Example's revenue in fiscal 2024?",
        "What was Example's revenue in fiscal 2022?",
        "What was the change in revenue from FY22 to FY24?",
    ],
)
def test_a_question_about_a_reported_year_or_no_year_is_admitted(question):
    decision = PeriodGuard().admits(question, (), FISCAL_2024)

    assert decision.admitted is True
    assert decision.scores == {"period": 1.0}


@pytest.mark.parametrize(
    ("question", "named"),
    [
        ("What was Example's revenue in fiscal 2019?", "2019"),
        ("What was Example's revenue in FY2021?", "2021"),
        ("What will Example's revenue be in fiscal 2025?", "2025"),
        ("How did revenue change from 2019 to 2024?", "2019"),
        ("Revenue in 2020 and 2021?", "2020, 2021"),
    ],
)
def test_a_question_about_a_year_the_filing_does_not_report_is_out_of_period(question, named):
    decision = PeriodGuard().admits(question, (), FISCAL_2024)

    assert decision.admitted is False
    assert decision.rejection is AbstentionReason.OUT_OF_PERIOD
    assert decision.reason == (
        f"the question names fiscal {named}; this filing reports fiscal 2022 to 2024"
    )
    assert decision.confidence == 0.0
    assert decision.scores == {"period": 0.0}
    assert decision.degraded is False


def test_the_guard_judges_against_the_filing_s_own_fiscal_year():
    fiscal_2019 = replace(EXAMPLE, fiscal_year=2019)

    decision = PeriodGuard().admits("Revenue in fiscal 2019?", (), fiscal_2019)

    assert decision.admitted is True


def test_the_years_reported_can_be_narrowed_to_the_balance_sheet_s_two():
    guard = PeriodGuard(years_reported=2)

    assert list(guard.covered(FISCAL_2024)) == [2023, 2024]
    assert guard.admits("Assets at fiscal 2022 year end?", (), FISCAL_2024).admitted is False


@pytest.mark.parametrize("years", [0, -1])
def test_a_filing_reports_at_least_its_own_year(years):
    with pytest.raises(ValueError, match="at least its own"):
        PeriodGuard(years_reported=years)


# A question in a reported year must never be declined by the guard,
# whatever the wording around the year and however the year is written.
_PHRASINGS = (
    "{}",
    "fiscal {}",
    "fiscal year {}",
    "FY{}",
    "FY {}",
    "FY'{}",
    "the year ended {}",
)
_FILLER = st.text(alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz ?,.'$"), max_size=30)


@st.composite
def _questions_in_reported_years(draw: st.DrawFn) -> tuple[str, int]:
    fiscal_year = draw(st.integers(min_value=1900, max_value=2099))
    # Written out rather than read from the guard, so narrowing it fails here
    reported = [fiscal_year - 2, fiscal_year - 1, fiscal_year]
    years = draw(st.lists(st.sampled_from(reported), min_size=1, max_size=3))
    parts = [draw(_FILLER)]
    for year in years:
        phrasing = draw(st.sampled_from(_PHRASINGS))
        two_digits = phrasing.startswith("FY") and draw(st.booleans())
        written = f"{year % 100:02d}" if two_digits else str(year)
        parts += [" ", phrasing.format(written), " ", draw(_FILLER)]
    return "".join(parts), fiscal_year


@given(_questions_in_reported_years())
def test_a_question_in_a_reported_year_is_never_declined(case):
    question, fiscal_year = case
    filing = replace(EXAMPLE, fiscal_year=fiscal_year)

    assert PeriodGuard().admits(question, (), filing).admitted is True


@given(
    fiscal_year=st.integers(min_value=1902, max_value=2099),
    distance=st.integers(min_value=1, max_value=50),
    later=st.booleans(),
)
def test_a_four_digit_year_outside_the_reported_ones_is_always_declined(
    fiscal_year, distance, later
):
    year = fiscal_year + distance if later else fiscal_year - 2 - distance
    assume(1900 <= year <= 2099)
    filing = replace(EXAMPLE, fiscal_year=fiscal_year)

    decision = PeriodGuard().admits(f"What was revenue in fiscal {year}?", (), filing)

    assert decision.admitted is False
    assert decision.rejection is AbstentionReason.OUT_OF_PERIOD
