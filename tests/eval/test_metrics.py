import itertools
import math

import numpy as np
import pytest

from edgar_rag.eval.metrics import (
    admit,
    auroc,
    crossfit_thresholds,
    discordant_counts,
    far_at_recall,
    mcnemar_exact,
    risk_coverage_curve,
    threshold_at_recall,
    wilson_interval,
)

# Newcombe (1998), "Two-sided confidence intervals for the single proportion",
# Statistics in Medicine 17:857-872, Table I, method 3 (Wilson score, no correction).
NEWCOMBE_WILSON = [
    (81, 263, 0.2553, 0.3662),
    (15, 148, 0.0624, 0.1605),
    (0, 20, 0.0000, 0.1611),
    (1, 29, 0.0061, 0.1718),
    (29, 29, 0.8830, 1.0000),
]


@pytest.mark.parametrize(("successes", "n", "low", "high"), NEWCOMBE_WILSON)
def test_wilson_matches_newcombe_table(successes, n, low, high):
    estimate = wilson_interval(successes, n)

    assert estimate.low == pytest.approx(low, abs=5e-5)
    assert estimate.high == pytest.approx(high, abs=5e-5)
    assert estimate.point == successes / n


def test_wilson_at_the_edges_matches_the_plan_values():
    none = wilson_interval(0, 5)
    every = wilson_interval(5, 5)

    # The plan quotes [0, 0.435] and [0.566, 1]; the exact bounds are 0.43448 and 0.56552.
    assert none.low == 0.0 and none.high == pytest.approx(0.435, abs=1e-3)
    assert every.low == pytest.approx(0.566, abs=1e-3) and every.high == 1.0


def test_wilson_upper_edge_has_the_closed_form_z2_over_n_plus_z2():
    z = 1.959963984540054
    n = 7

    assert wilson_interval(0, n).high == pytest.approx(z * z / (n + z * z), rel=1e-12)


def test_wilson_widens_with_confidence_and_stays_inside_the_unit_interval():
    narrow = wilson_interval(3, 10, confidence=0.90)
    wide = wilson_interval(3, 10, confidence=0.99)

    assert wide.low < narrow.low < narrow.high < wide.high
    assert wide.low >= 0.0 and wide.high <= 1.0
    assert narrow.half_width == pytest.approx((narrow.high - narrow.low) / 2)


@pytest.mark.parametrize(
    ("successes", "n", "confidence"), [(1, 0, 0.95), (-1, 5, 0.95), (6, 5, 0.95), (1, 5, 1.0)]
)
def test_wilson_rejects_impossible_inputs(successes, n, confidence):
    with pytest.raises(ValueError):
        wilson_interval(successes, n, confidence)


def _pair_count_auroc(scores, labels):
    positives = [s for s, y in zip(scores, labels, strict=True) if y]
    negatives = [s for s, y in zip(scores, labels, strict=True) if not y]
    wins = sum(
        1.0 if p > q else 0.5 if p == q else 0.0 for p, q in itertools.product(positives, negatives)
    )
    return wins / (len(positives) * len(negatives))


def test_auroc_of_trivial_cases():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_auroc_equals_the_brute_force_pair_count_with_ties():
    rng = np.random.default_rng(3)
    scores = rng.integers(0, 6, size=60).astype(float)  # heavy ties on purpose
    labels = rng.random(60) < 0.4

    assert auroc(scores, labels) == pytest.approx(_pair_count_auroc(scores, labels), abs=1e-12)


def test_auroc_needs_both_classes_and_matching_lengths():
    with pytest.raises(ValueError, match="positive and one negative"):
        auroc([0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="same length"):
        auroc([0.1, 0.2], [1])
    with pytest.raises(ValueError, match="finite"):
        auroc([0.1, float("nan")], [1, 0])


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 0, 1.0),
        (0, 5, 2 / 32),  # 2 * P(X <= 0), X ~ Bin(5, 1/2)
        (1, 9, 2 * 11 / 1024),  # 2 * (1 + 10) / 2**10
        (5, 5, 1.0),
        (3, 7, 2 * (1 + 10 + 45 + 120) / 1024),
    ],
)
def test_mcnemar_exact_matches_the_binomial_tail(b, c, expected):
    assert mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-12)
    assert mcnemar_exact(c, b) == pytest.approx(expected, rel=1e-12)


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_discordant_counts_split_the_two_directions():
    assert discordant_counts([1, 1, 0, 0, 1], [1, 0, 1, 0, 0]) == (2, 1)
    with pytest.raises(ValueError):
        discordant_counts([1, 0], [1])


def test_threshold_at_recall_is_the_kth_largest_positive():
    scores = [float(s) for s in range(1, 11)]  # 1..10

    assert threshold_at_recall(scores, 0.9) == 2.0  # admits 9 of 10
    assert threshold_at_recall(scores, 0.85) == 2.0  # ceil(8.5) = 9 positives
    assert threshold_at_recall(scores, 1.0) == 1.0
    assert threshold_at_recall(scores, 0.1) == 10.0


def test_threshold_at_recall_validates_its_inputs():
    with pytest.raises(ValueError):
        threshold_at_recall([], 0.9)
    with pytest.raises(ValueError):
        threshold_at_recall([1.0], 0.0)


def test_crossfit_threshold_for_a_fold_never_sees_that_fold():
    scores = np.array([10.0, 9.0, 8.0, 1.0, 0.5, 0.4, 0.3, 5.0])
    labels = np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    thresholds = crossfit_thresholds(scores, labels, folds, target_recall=1.0)

    assert thresholds == {0: 0.3, 1: 8.0}


def test_crossfit_needs_two_folds_and_aligned_arrays():
    with pytest.raises(ValueError, match="two folds"):
        crossfit_thresholds([1.0, 0.0], [1, 0], [0, 0])
    with pytest.raises(ValueError, match="folds must match"):
        crossfit_thresholds([1.0, 0.0], [1, 0], [0])


def test_admit_uses_the_threshold_of_each_case_fold():
    admitted = admit([0.5, 0.5, 0.9], [0, 1, 1], {0: 0.4, 1: 0.6})

    assert admitted.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="no threshold"):
        admit([0.5], [2], {0: 0.4})
    with pytest.raises(ValueError, match="folds must match"):
        admit([0.5, 0.1], [0], {0: 0.4})


def test_far_at_recall_measures_out_of_fold():
    scores = np.array([10.0, 9.0, 8.0, 1.0, 0.5, 0.4, 0.3, 5.0])
    labels = np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    point = far_at_recall(scores, labels, folds, target_recall=1.0)

    # Fold 0 uses 0.3, so its negative (1.0) is admitted; fold 1 uses 8.0 and
    # rejects its negative (5.0) along with all three of its positives.
    assert point.false_accept_rate == 0.5
    assert point.recall == 0.5
    assert (point.n_positive, point.n_negative) == (6, 2)


def test_far_at_recall_needs_negatives():
    with pytest.raises(ValueError, match="at least one negative"):
        far_at_recall([1.0, 2.0], [1, 1], [0, 1])


def test_risk_coverage_curve_on_a_hand_checked_example():
    scores = [0.9, 0.8, 0.8, 0.3, 0.1]
    correct = [True, False, True, True, False]

    curve = risk_coverage_curve(scores, correct)

    # The tie at 0.8 enters together: no threshold answers only one of the two.
    assert curve.thresholds.tolist() == [0.9, 0.8, 0.3, 0.1]
    assert curve.coverage.tolist() == [0.2, 0.6, 0.8, 1.0]
    assert curve.risk.tolist() == pytest.approx([0.0, 1 / 3, 0.25, 0.4])
    assert curve.area == pytest.approx(0.2 * 0.0 + 0.4 * (1 / 3) + 0.2 * 0.25 + 0.2 * 0.4)


def test_risk_coverage_area_without_ties_is_the_mean_selective_risk():
    rng = np.random.default_rng(5)
    scores = rng.random(50)
    correct = rng.random(50) < 0.7
    order = np.argsort(-scores)
    errors = np.cumsum(~correct[order])
    expected = float(np.mean(errors / np.arange(1, 51)))

    assert risk_coverage_curve(scores, correct).area == pytest.approx(expected)


def test_risk_coverage_needs_cases():
    with pytest.raises(ValueError):
        risk_coverage_curve([], [])


def test_risk_at_full_coverage_is_the_overall_error_rate():
    curve = risk_coverage_curve([0.2, 0.4, 0.6], [True, False, False])

    assert math.isclose(curve.risk[-1], 2 / 3)
