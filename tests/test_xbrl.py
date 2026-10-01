"""Company facts: typed values tied to a year by their dates, never by ``fy``.

The fixture is Apple's real ``companyfacts`` cut down to three concepts and the
facts of three filings: the 10-Ks for fiscal 2024 and 2025 and the 10-Q for the
third quarter of fiscal 2025.
"""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from edgar_rag.edgar.errors import EdgarError
from edgar_rag.edgar.xbrl import XbrlError, parse_companyfacts, period_end, reported_fact

FIXTURE = Path(__file__).parent / "data" / "companyfacts-aapl-trimmed.json"
REVENUE = "RevenueFromContractWithCustomerExcludingAssessedTax"
FY2024_10K = "0000320193-24-000123"
FY2024_REPORT_DATE = date(2024, 9, 28)
Q3_2025_10Q = "0000320193-25-000073"


@pytest.fixture(scope="module")
def apple():
    return parse_companyfacts(FIXTURE.read_bytes())


def _revenue(apple, **kwargs):
    arguments = {
        "concept": REVENUE,
        "unit": "USD",
        "accession": FY2024_10K,
        "report_date": FY2024_REPORT_DATE,
        **kwargs,
    }
    return reported_fact(apple, **arguments)


def test_the_fixture_is_typed_into_facts(apple):
    assert apple.cik == 320193
    assert apple.entity_name == "Apple Inc."
    assert {fact.concept for fact in apple.facts} == {REVENUE, "Assets", "EarningsPerShareDiluted"}
    eps = apple.of("EarningsPerShareDiluted")
    assert {fact.unit for fact in eps} == {"USD/shares"}
    # Decimals stay exact: 6.08 is not 6.0799999999999996
    assert Decimal("6.08") in {fact.value for fact in eps}


def test_every_fact_of_a_10k_carries_the_filing_year_whatever_period_it_measures(apple):
    # The trap this module exists for: three years of revenue, one fy
    in_10k = [fact for fact in apple.of(REVENUE) if fact.accession == FY2024_10K]

    assert {fact.fiscal_year for fact in in_10k} == {2024}
    assert sorted(fact.end.year for fact in in_10k) == [2022, 2023, 2024]


def test_the_year_a_filing_covers_comes_from_its_report_date(apple):
    fact = _revenue(apple)

    assert fact is not None
    assert fact.value == Decimal(391_035_000_000)
    assert (fact.start, fact.end) == (date(2023, 10, 1), date(2024, 9, 28))


def test_a_comparative_year_is_found_by_moving_back_from_the_report_date(apple):
    # fy says 2024 for both; only the dates say this one is fiscal 2023
    fact = _revenue(apple, years_before=1)

    assert fact is not None
    assert fact.value == Decimal(383_285_000_000)
    assert fact.fiscal_year == 2024
    assert fact.end == date(2023, 9, 30)


def test_a_balance_is_matched_as_an_instant(apple):
    fact = reported_fact(
        apple,
        concept="Assets",
        unit="USD",
        accession=FY2024_10K,
        report_date=FY2024_REPORT_DATE,
    )

    assert fact is not None
    assert fact.is_instant
    assert fact.value == Decimal(364_980_000_000)


def test_a_later_filing_never_answers_for_the_pinned_one(apple):
    # The fiscal 2025 10-K repeats fiscal 2024 revenue; the pin decides which filing speaks
    fact = _revenue(apple)

    assert fact is not None
    assert fact.accession == FY2024_10K


def test_a_year_the_filing_does_not_report_is_none(apple):
    assert _revenue(apple, years_before=3) is None
    assert _revenue(apple, unit="EUR") is None
    assert _revenue(apple, concept="Revenues") is None


def test_quarterly_and_year_to_date_figures_are_not_a_fiscal_year(apple):
    # The 10-Q tags three- and nine-month revenue; none of it spans a year
    in_10q = [fact for fact in apple.of(REVENUE) if fact.accession == Q3_2025_10Q]

    assert in_10q
    assert not any(fact.is_annual for fact in in_10q)
    assert _revenue(apple, accession=Q3_2025_10Q, report_date=date(2025, 6, 28)) is None


def _payload(*facts: dict) -> bytes:
    return json.dumps(
        {
            "cik": 1,
            "entityName": "Example Co",
            "facts": {"us-gaap": {REVENUE: {"units": {"USD": list(facts)}}}},
        }
    ).encode()


def _fact(
    start: str | None, end: str, val: int | float, accn: str = "0000000001-25-000001"
) -> dict:
    fact = {
        "end": end,
        "val": val,
        "accn": accn,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-11-01",
    }
    return fact if start is None else {**fact, "start": start}


def _synthetic_revenue(*facts: dict, report_date: date = date(2025, 9, 27)):
    company = parse_companyfacts(_payload(*facts))
    return reported_fact(
        company,
        concept=REVENUE,
        unit="USD",
        accession="0000000001-25-000001",
        report_date=report_date,
    )


def test_a_fourth_quarter_tagged_with_the_year_end_is_not_the_year():
    quarter = _fact("2025-06-29", "2025-09-27", 100)
    year = _fact("2024-09-29", "2025-09-27", 400)

    fact = _synthetic_revenue(quarter, year)

    assert fact is not None
    assert fact.value == 400


def test_a_53_week_year_still_counts_as_a_year():
    fact = _synthetic_revenue(_fact("2022-09-25", "2023-09-30", 383), report_date=date(2023, 9, 30))

    assert fact is not None
    assert fact.duration_days == 370


def test_a_year_end_that_drifts_by_days_still_matches():
    # 52/53-week years end on a Saturday: 2024-09-28, then 2025-09-27
    fact = _synthetic_revenue(_fact("2023-10-01", "2024-09-28", 391), report_date=date(2025, 9, 27))

    assert fact is None  # the year itself is not the comparative one
    comparative = reported_fact(
        parse_companyfacts(_payload(_fact("2023-10-01", "2024-09-28", 391))),
        concept=REVENUE,
        unit="USD",
        accession="0000000001-25-000001",
        report_date=date(2025, 9, 27),
        years_before=1,
    )
    assert comparative is not None
    assert comparative.value == 391


def test_two_different_values_for_one_period_in_one_filing_is_an_error():
    first = _fact("2024-09-29", "2025-09-27", 400)
    second = _fact("2024-09-29", "2025-09-27", 401)

    with pytest.raises(XbrlError, match="2 values"):
        _synthetic_revenue(first, second)


def test_the_same_value_tagged_twice_is_one_fact():
    fact = _fact("2024-09-29", "2025-09-27", 400)

    assert _synthetic_revenue(fact, {**fact, "frame": "CY2025"}) is not None


def test_a_leap_day_report_date_moves_back_to_february():
    assert period_end(date(2024, 2, 29), years_before=1) == date(2023, 2, 28)
    assert period_end(date(2025, 1, 31), years_before=2) == date(2023, 1, 31)


def test_period_end_cannot_look_forward():
    with pytest.raises(ValueError, match="cannot be negative"):
        period_end(date(2025, 9, 27), years_before=-1)


@pytest.mark.parametrize(
    "content",
    [
        b"not json",
        b"[]",
        b'{"entityName": "No CIK"}',
        _payload({"end": "2025-09-27", "val": 1}),  # no accession, form or filing date
        _payload(_fact(None, "not a date", 1)),
        _payload(_fact(None, "2025-09-27", "12")),
        _payload(_fact(None, "2025-09-27", 1) | {"val": None}),
        _payload(_fact(None, "2025-09-27", 1)).replace(b'"val": 1', b'"val": NaN'),
        _payload(_fact(None, "2025-09-27", 1)).replace(b'"val": 1', b'"val": Infinity'),
    ],
)
def test_malformed_company_facts_are_an_xbrl_error(content):
    with pytest.raises(XbrlError, match="malformed") as raised:
        parse_companyfacts(content)
    # One except clause catches every EDGAR failure
    assert isinstance(raised.value, EdgarError)
