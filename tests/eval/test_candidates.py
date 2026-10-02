"""Candidate pools: a positive only when its value is printed, a negative only when verified."""

from decimal import Decimal

import pytest

from edgar_rag.eval.build import EvalPaths, load_corpus
from edgar_rag.eval.candidates import (
    Drops,
    off_domain,
    other_companies,
    positives,
    unreported_concepts,
    wrong_years,
)
from tests.eval.golden_fakes import DEV, EVAL, FakeCompany, rewrite_text, write_eval_root

ALDER = DEV[0]


@pytest.fixture
def corpus(tmp_path):
    write_eval_root(tmp_path, omit={"BBB": frozenset({"Assets"})})
    _, _, companies = load_corpus(EvalPaths(tmp_path))
    return {company.company.ticker: company for company in companies}


def _facts(cases):
    return {(case.fiscal_year, case.concept, case.period_year) for case in cases}


def test_every_printed_fact_becomes_four_phrasings_with_the_filing_value(corpus):
    drops = Drops()
    cases = list(positives(corpus["AAA"], drops))

    # 3 concepts x the filing's year and the year before x 2 filings
    assert len(_facts(cases)) == 12
    assert len(cases) == 48
    revenue = next(c for c in cases if c.id == "pos:AAA:2025:revenue:y1:p0")
    assert revenue.question == "What was Alder Works's total revenue in fiscal 2024?"
    assert revenue.expected_value == ALDER.revenue(2024)
    assert revenue.period_end.year == 2024
    assert revenue.concept == "us-gaap:Revenues"
    eps = next(c for c in cases if c.template.startswith("diluted_eps"))
    assert eps.unit == "USD/shares"
    assert drops.counts[("positive", "net_income", "no_xbrl_fact")] == 4


def test_a_fact_the_filing_does_not_print_is_dropped_and_counted(corpus):
    birch = corpus["BBB"]
    # BBB's text omits the assets table and its facts omit Assets entirely
    drops = Drops()
    cases = list(positives(birch, drops))

    assert not any(case.template.startswith("total_assets") for case in cases)
    assert drops.counts[("positive", "total_assets", "no_xbrl_fact")] == 4


def test_a_reported_value_missing_from_the_text_is_dropped_as_not_printed(tmp_path):
    write_eval_root(tmp_path)
    rewrite_text(tmp_path, "AAA", 2025, lambda text: text.replace("11,175", "11,999"))
    _, _, companies = load_corpus(EvalPaths(tmp_path))
    drops = Drops()

    cases = list(positives(companies[0], drops))

    assert drops.counts[("positive", "revenue", "not_in_indexed_text")] == 1
    assert "pos:AAA:2025:revenue:y0:p0" not in {case.id for case in cases}
    assert "pos:AAA:2025:revenue:y1:p0" in {case.id for case in cases}


def test_wrong_year_asks_four_to_seven_years_back_about_values_not_printed(corpus):
    drops = Drops()
    cases = list(wrong_years(corpus["AAA"], drops))

    assert {case.period_year for case in cases if case.fiscal_year == 2025} == {
        2018,
        2019,
        2020,
        2021,
    }
    assert all(not case.answerable and case.expected_value is None for case in cases)
    # net income is never reported: 4 years back x 2 filings, each counted once
    assert drops.counts[("wrong_year", "net_income", "no_value_for_year")] == 8


def test_wrong_year_drops_an_old_value_the_filing_prints(tmp_path):
    write_eval_root(tmp_path)
    old_revenue = f"{int(ALDER.revenue(2019) / 10**6):,}"
    rewrite_text(tmp_path, "AAA", 2025, lambda text: f"{text}\nFive-year revenue {old_revenue}\n")
    _, _, companies = load_corpus(EvalPaths(tmp_path))
    drops = Drops()

    cases = list(wrong_years(companies[0], drops))

    assert drops.counts[("wrong_year", "revenue", "value_in_text")] == 1
    assert not any(c.fiscal_year == 2025 and c.id.endswith("revenue:2019:p0") for c in cases)


def test_other_company_names_a_peer_and_its_figure_for_a_covered_year(corpus):
    peers = (corpus["AAA"], corpus["BBB"])
    drops = Drops()
    cases = list(other_companies(corpus["AAA"], peers, drops))

    assert {case.asked_company for case in cases} == {"BBB"}
    assert all("Birch Mills" in case.question for case in cases)
    assert {case.period_year for case in cases} == {2023, 2024, 2025}
    # BBB never reported Assets, so no question asks for them
    assert drops.counts[("other_company", "total_assets", "no_value_for_asked_company")] == 4


def test_other_company_drops_a_peer_value_the_filing_happens_to_print(tmp_path):
    twin = FakeCompany("EEE", 1005, "Elm Twin", "dev", base=ALDER.base)
    write_eval_root(tmp_path, (ALDER, twin))
    _, _, companies = load_corpus(EvalPaths(tmp_path))
    drops = Drops()

    cases = list(other_companies(companies[0], companies, drops))

    # the twin's figures are Alder's own, so every one of them is in Alder's filing
    assert cases == []
    assert drops.counts[("other_company", "revenue", "value_in_text")] == 4


def test_unreported_concepts_skip_what_the_company_reports_or_mentions(corpus):
    drops = Drops()
    cases = list(unreported_concepts(corpus["AAA"], drops))

    assert len(cases) == 13 * 2 * 2 * 4
    assert all(case.concept.startswith("us-gaap:") for case in cases)
    assert "net interest income" in next(
        c.question for c in cases if c.concept == "us-gaap:InterestIncomeExpenseNet"
    )


def test_off_domain_drops_a_question_whose_marker_is_in_the_filing(corpus):
    drops = Drops()
    cases = list(off_domain(corpus["AAA"], drops))

    assert len(cases) == 80
    assert "Draw Alder Works's logo in ASCII art." in {case.question for case in cases}
    assert all(case.period_year is None for case in cases)


def test_eval_split_cases_carry_a_fold(corpus):
    cases = list(positives(corpus["CCC"], Drops())) + list(positives(corpus["DDD"], Drops()))

    assert {case.fold for case in cases} == {0, 1}
    assert {case.split for case in cases} == {"eval"}
    assert corpus["CCC"].fold != corpus["DDD"].fold
    assert EVAL[0].eps(2025) == Decimal("37.25")
