import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from edgar_rag.eval.arms import (
    ARM_BY_KEY,
    ARMS,
    IN_SAMPLE,
    admitted,
    answered,
    fit_threshold,
    runnable,
)
from tests.eval.harness_fakes import record

OUTCOMES = ("answered", "model_declined", "unsupported_claim", "no_valid_citation")


cases = st.lists(
    st.builds(
        lambda i, answerable, cosine, period, brier, outcome, fold: record(
            f"c{i}",
            answerable=answerable,
            cosine=cosine,
            period=period,
            brier=brier,
            outcome=outcome,
            fold=fold,
            ticker=f"T{i % 4}",
        ),
        st.integers(0, 10_000),
        st.booleans(),
        st.floats(0.0, 1.0),
        st.sampled_from((0.0, 1.0)),
        st.floats(0.0, 1.0),
        st.sampled_from(OUTCOMES),
        st.just(None),
    ),
    min_size=1,
    max_size=40,
).filter(lambda rows: any(r.answerable for r in rows))


@given(cases)
def test_every_gated_arm_answers_a_subset_of_what_arm_a_answers(records):
    thresholds = {score: fit_threshold(records, score) for score in ("cosine", "brier")}
    unguarded = answered(ARM_BY_KEY["A"], records, thresholds)

    for arm in ARMS:
        assert not np.any(answered(arm, records, thresholds) & ~unguarded), arm.key


@given(cases)
def test_adding_the_period_guard_never_answers_more(records):
    thresholds = {score: fit_threshold(records, score) for score in ("cosine", "brier")}
    for alone, guarded in (("B", "F"), ("C", "E")):
        lone = answered(ARM_BY_KEY[alone], records, thresholds)
        both = answered(ARM_BY_KEY[guarded], records, thresholds)
        assert not np.any(both & ~lone)


def test_arm_a_answers_exactly_what_was_recorded_as_answered():
    records = [
        record("a", answerable=True, cosine=0.9),
        record("b", answerable=False, cosine=0.1, outcome="model_declined"),
    ]

    assert answered(ARM_BY_KEY["A"], records, {}).tolist() == [True, False]


def test_the_period_guard_admits_on_its_rule_alone():
    records = [
        record("in", answerable=True, cosine=0.0, period=1.0),
        record("out", answerable=False, cosine=0.9, period=0.0),
    ]

    assert admitted(ARM_BY_KEY["D"], records, {}).tolist() == [True, False]


def test_without_folds_the_threshold_is_r90_in_sample():
    positives = [record(f"p{i}", answerable=True, cosine=i / 10) for i in range(1, 11)]
    negatives = [record("n", answerable=False, cosine=0.95)]

    threshold = fit_threshold(positives + negatives, "cosine")

    assert not threshold.cross_fitted
    assert threshold.by_fold == {IN_SAMPLE: pytest.approx(0.2)}


def test_with_folds_each_fold_gets_the_threshold_fitted_on_the_other():
    fold0 = [record(f"a{i}", answerable=True, cosine=0.5, fold=0, ticker="A") for i in range(10)]
    fold1 = [record(f"b{i}", answerable=True, cosine=0.8, fold=1, ticker="B") for i in range(10)]

    threshold = fit_threshold(fold0 + fold1, "cosine")

    assert threshold.cross_fitted
    assert threshold.by_fold == {0: 0.8, 1: 0.5}
    mask = admitted(ARM_BY_KEY["B"], fold0 + fold1, {"cosine": threshold})
    assert not mask[:10].any() and mask[10:].all()


def test_mixing_cases_with_and_without_folds_is_refused():
    rows = [
        record("a", answerable=True, cosine=0.5, fold=0),
        record("b", answerable=True, cosine=0.5),
    ]

    with pytest.raises(ValueError, match="mix"):
        fit_threshold(rows, "cosine")


def test_an_arm_whose_gate_did_not_run_is_not_runnable():
    rows = [record("a", answerable=True, cosine=0.5)]

    assert runnable(ARM_BY_KEY["F"], rows)
    assert not runnable(ARM_BY_KEY["C"], rows)
    assert not runnable(ARM_BY_KEY["E"], rows)
    assert not runnable(ARM_BY_KEY["C"], [])
