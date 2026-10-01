"""Point statistics for the evaluation, each with the convention it assumes written down.

Every function here is pure and takes plain arrays, so the report, the power
simulation and the tests all go through the same arithmetic.

Score convention: a higher gate score means "more likely answerable", and a gate
admits a question when its score is at or above the threshold. Labels are
booleans where ``True`` means answerable (the positive class).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_CONFIDENCE = 0.95
DEFAULT_TARGET_RECALL = 0.90


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a two-sided interval at ``confidence``."""

    point: float
    low: float
    high: float
    confidence: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


def normal_quantile(probability: float) -> float:
    """Inverse of the standard normal CDF."""
    return NormalDist().inv_cdf(probability)


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def wilson_interval(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> Estimate:
    """Proportion ``successes / n`` with the Wilson score interval.

    Wilson rather than Wald because the samples here are small and the rates sit
    near the edges: Wald's interval leaves [0, 1] and collapses to a point at
    0/n, Wilson does neither (Brown, Cai and DasGupta, 2001).
    """
    _check_confidence(confidence)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    z = normal_quantile(0.5 + confidence / 2)
    p = successes / n
    z2n = z * z / n
    centre = (p + z2n / 2) / (1 + z2n)
    margin = z * math.sqrt(p * (1 - p) / n + z2n / (4 * n)) / (1 + z2n)
    return Estimate(
        point=p,
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        confidence=confidence,
    )


def _scores_and_labels(
    scores: ArrayLike, labels: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.bool_)
    if score_array.ndim != 1 or score_array.shape != label_array.shape:
        raise ValueError("scores and labels must be one-dimensional and the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    return score_array, label_array


def _midranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ranks starting at 1, ties sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    is_new = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    group = np.cumsum(is_new) - 1
    starts = np.flatnonzero(is_new)
    ends = np.r_[starts[1:], len(values)]
    group_rank = (starts + 1 + ends) / 2
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = group_rank[group]
    return ranks


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve: P(positive outscores negative), ties counted as half.

    Computed as the Mann-Whitney U statistic from midranks, which equals the
    pair count exactly and runs in O(n log n).
    """
    score_array, label_array = _scores_and_labels(scores, labels)
    n_pos = int(label_array.sum())
    n_neg = len(label_array) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one positive and one negative")
    rank_sum = float(_midranks(score_array)[label_array].sum())
    u_statistic = rank_sum - n_pos * (n_pos + 1) / 2
    return u_statistic / (n_pos * n_neg)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Under the null each discordant pair goes either way with probability 1/2,
    so the p-value is twice the binomial tail of the smaller count, capped at 1.
    Integer arithmetic keeps it exact at the sizes used here.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts cannot be negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / (1 << n))


def discordant_counts(first: ArrayLike, second: ArrayLike) -> tuple[int, int]:
    """``(b, c)``: pairs where only ``first`` is true, and where only ``second`` is."""
    first_array = np.asarray(first, dtype=np.bool_)
    second_array = np.asarray(second, dtype=np.bool_)
    if first_array.shape != second_array.shape:
        raise ValueError("paired outcomes must have the same shape")
    return int(np.sum(first_array & ~second_array)), int(np.sum(~first_array & second_array))


def threshold_at_recall(
    positive_scores: ArrayLike, target_recall: float = DEFAULT_TARGET_RECALL
) -> float:
    """The highest threshold that admits at least ``target_recall`` of the positives.

    It is the k-th largest positive score with k = ceil(target * n). Ties at the
    threshold are admitted too, so the realised recall can exceed the target.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    values = np.sort(np.asarray(positive_scores, dtype=np.float64))[::-1]
    if values.size == 0:
        raise ValueError("cannot fit a threshold without positives")
    k = math.ceil(target_recall * values.size - 1e-9)
    return float(values[k - 1])


def crossfit_thresholds(
    scores: ArrayLike,
    labels: ArrayLike,
    folds: ArrayLike,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> dict[int, float]:
    """Per fold, the recall-matched threshold fitted on the positives of the other folds.

    The threshold applied to a fold never saw that fold, so the false-accept
    rate measured on it is out of sample. Folds are assigned by company.
    """
    score_array, label_array = _scores_and_labels(scores, labels)
    fold_array = np.asarray(folds, dtype=np.int64)
    if fold_array.shape != score_array.shape:
        raise ValueError("folds must match scores in length")
    fold_ids = np.unique(fold_array)
    if fold_ids.size < 2:
        raise ValueError("cross-fitting needs at least two folds")
    return {
        int(fold): threshold_at_recall(
            score_array[label_array & (fold_array != fold)], target_recall
        )
        for fold in fold_ids
    }


def admit(
    scores: ArrayLike, folds: ArrayLike, thresholds: Mapping[int, float]
) -> NDArray[np.bool_]:
    """Whether each case is admitted under the threshold of its own fold."""
    score_array = np.asarray(scores, dtype=np.float64)
    fold_array = np.asarray(folds, dtype=np.int64)
    if fold_array.shape != score_array.shape:
        raise ValueError("folds must match scores in length")
    missing = set(np.unique(fold_array).tolist()) - set(thresholds)
    if missing:
        raise ValueError(f"no threshold for folds {sorted(missing)}")
    cutoffs = np.array([thresholds[int(fold)] for fold in fold_array], dtype=np.float64)
    return score_array >= cutoffs


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """A gate at a cross-fitted recall-matched threshold, measured out of fold."""

    false_accept_rate: float
    recall: float
    thresholds: Mapping[int, float]
    n_positive: int
    n_negative: int


def far_at_recall(
    scores: ArrayLike,
    labels: ArrayLike,
    folds: ArrayLike,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> OperatingPoint:
    """False-accept rate on the negatives at a cross-fitted threshold (FAR@R90 by default).

    The realised recall is reported next to it, because a cross-fitted threshold
    only hits the target recall on average.
    """
    score_array, label_array = _scores_and_labels(scores, labels)
    thresholds = crossfit_thresholds(score_array, label_array, folds, target_recall)
    admitted = admit(score_array, folds, thresholds)
    n_pos = int(label_array.sum())
    n_neg = len(label_array) - n_pos
    if n_neg == 0:
        raise ValueError("a false-accept rate needs at least one negative")
    return OperatingPoint(
        false_accept_rate=float(admitted[~label_array].mean()),
        recall=float(admitted[label_array].mean()),
        thresholds=thresholds,
        n_positive=n_pos,
        n_negative=n_neg,
    )


@dataclass(frozen=True, slots=True)
class RiskCoverage:
    """Selective risk against coverage, one point per distinct score threshold.

    Point i answers every case scoring at or above ``thresholds[i]``; ``risk`` is
    the error rate among those answered. Coverage increases along the arrays.
    """

    thresholds: NDArray[np.float64]
    coverage: NDArray[np.float64]
    risk: NDArray[np.float64]

    @property
    def area(self) -> float:
        """Area under the risk-coverage curve (AURC), as the mean risk over the points
        weighted by the coverage each one adds. Lower is better."""
        steps = np.diff(np.r_[0.0, self.coverage])
        return float(np.sum(steps * self.risk))


def risk_coverage_curve(scores: ArrayLike, correct: ArrayLike) -> RiskCoverage:
    """Risk-coverage curve of answering in decreasing score order.

    ``correct`` is whether answering that case would be right. For a question
    with no answer in the filing, answering at all is an error, so the caller
    passes ``False`` there.
    """
    score_array, correct_array = _scores_and_labels(scores, correct)
    if score_array.size == 0:
        raise ValueError("a risk-coverage curve needs at least one case")
    order = np.argsort(-score_array, kind="mergesort")
    sorted_scores = score_array[order]
    errors = np.cumsum(~correct_array[order])
    is_last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    answered = np.flatnonzero(is_last_of_tie) + 1
    return RiskCoverage(
        thresholds=sorted_scores[answered - 1],
        coverage=answered / score_array.size,
        risk=errors[answered - 1] / answered,
    )
