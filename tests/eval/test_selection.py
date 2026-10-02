"""Sampling: subtype totals that come out equal, spread over facts, the same every run."""

import random
from collections import Counter

import pytest

from edgar_rag.eval.golden import NEGATIVE_KINDS, GoldenCase
from edgar_rag.eval.selection import (
    E2eQuota,
    by_company_and_kind,
    even_allocation,
    kind_quota,
    select_e2e,
    select_gate_negatives,
    spread_sample,
)


def _case(ticker, kind, concept, year, phrasing):
    negative = kind != "positive"
    return GoldenCase.model_validate(
        {
            "id": f"{kind}:{ticker}:{concept}:{year}:p{phrasing}",
            "split": "dev",
            "e2e": False,
            "ticker": ticker,
            "cik": 1,
            "fiscal_year": 2025,
            "accession": "0000000001-26-000001",
            "question": f"{concept} {year} {phrasing}?",
            "template": f"{concept}/{phrasing}",
            "answerable": not negative,
            "negative_kind": kind if negative else None,
            "asked_company": "ZZZ" if kind == "other_company" else None,
            "concept": f"us-gaap:{concept}",
            "period_year": year,
            "period_end": None if negative else f"{year}-12-31",
            "expected_value": None if negative else "1",
            "unit": None if negative else "USD",
        }
    )


def _pool(tickers=("AAA", "BBB"), concepts=("A", "B", "C"), years=(2024, 2025)):
    return [
        _case(ticker, kind, concept, year, phrasing)
        for ticker in tickers
        for kind in ("positive", *NEGATIVE_KINDS)
        for concept in concepts
        for year in years
        for phrasing in range(4)
    ]


@pytest.mark.parametrize(("companies", "per_company", "per_kind"), [(20, 15, 75), (4, 25, 25)])
def test_rotating_quotas_give_every_subtype_the_same_total(companies, per_company, per_kind):
    totals = Counter()
    for position in range(companies):
        quota = kind_quota(per_company, position)
        assert sum(quota.values()) == per_company
        totals.update(quota)

    assert totals == dict.fromkeys(NEGATIVE_KINDS, per_kind)


def test_a_sample_takes_one_phrasing_per_fact_and_rotates_concepts_first():
    pool = [case for case in _pool(tickers=("AAA",)) if case.answerable]

    picks = spread_sample(pool, 3, random.Random(1))

    assert len({case.concept for case in picks}) == 3
    six = spread_sample(pool, 6, random.Random(1))
    assert len({(case.concept, case.period_year) for case in six}) == 6


def test_a_sample_is_the_same_for_the_same_seed_whatever_the_input_order():
    pool = [case for case in _pool(tickers=("AAA",)) if case.answerable]
    shuffled = list(pool)
    random.Random(5).shuffle(shuffled)

    assert spread_sample(pool, 10, random.Random(3)) == spread_sample(
        shuffled, 10, random.Random(3)
    )


def test_a_sample_past_the_distinct_facts_takes_further_phrasings():
    pool = [case for case in _pool(tickers=("AAA",)) if case.answerable]

    picks = spread_sample(pool, len(pool), random.Random(1))

    assert sorted(case.id for case in picks) == sorted(case.id for case in pool)
    with pytest.raises(ValueError, match="pool of"):
        spread_sample(pool, len(pool) + 1, random.Random(1))


def test_even_allocation_respects_capacities_and_refuses_too_little():
    assert even_allocation(7, {"a": 2, "b": 10, "c": 10}, ["a", "b", "c"]) == {
        "a": 2,
        "b": 3,
        "c": 2,
    }
    with pytest.raises(ValueError, match="cannot place"):
        even_allocation(5, {"a": 2, "b": 2}, ["a", "b"])


def test_e2e_negatives_are_inside_the_gate_only_ones():
    pools = by_company_and_kind(_pool())
    e2e = select_e2e(pools, ["AAA", "BBB"], E2eQuota(positives=3, negatives=5), seed=9)
    gate = select_gate_negatives(pools, ["AAA", "BBB"], 40, e2e, seed=9)

    e2e_negatives = {case_id for case_id in e2e if not case_id.startswith("positive")}
    assert len(e2e) == 2 * (3 + 5)
    assert e2e_negatives <= gate
    assert len(gate) == 40
    assert Counter(case_id.split(":")[0] for case_id in gate) == dict.fromkeys(NEGATIVE_KINDS, 10)


def test_e2e_refuses_a_pool_without_enough_distinct_facts():
    pools = by_company_and_kind(_pool(concepts=("A",), years=(2025,)))

    with pytest.raises(ValueError, match="e2e AAA positive"):
        select_e2e(pools, ["AAA"], E2eQuota(positives=2, negatives=4), seed=1)
