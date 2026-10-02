"""The numbers the report prints, computed from a run's records and nothing else.

Every function is pure: the same records give the same numbers, bootstrap
included (seeded). Rates are reported with Wilson intervals; differences and
AUROC with a cluster bootstrap by company, stratified by cross-fitting fold
when there are folds. Next to every comparison sits the minimum detectable
effect at 80% power implied by the bootstrap standard error, so a reader can
tell "no difference" from "not resolvable at this n".
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edgar_rag.eval.arms import Arm, Threshold, admitted, answered
from edgar_rag.eval.bootstrap import (
    DEFAULT_RESAMPLES,
    BootstrapResult,
    bootstrap_mean,
    cluster_bootstrap,
)
from edgar_rag.eval.metrics import (
    Estimate,
    auroc,
    discordant_counts,
    far_at_recall,
    mcnemar_exact,
    normal_quantile,
    wilson_interval,
)
from edgar_rag.eval.records import CaseRecord

REPORT_SEED = 20261001
TARGET_POWER = 0.80


@dataclass(frozen=True, slots=True)
class Rate:
    """``hits`` of ``n``, with its Wilson interval; ``None`` interval when ``n`` is zero."""

    hits: int
    n: int
    interval: Estimate | None


def rate(mask: NDArray[np.bool_]) -> Rate:
    n = int(mask.size)
    hits = int(mask.sum())
    return Rate(hits, n, wilson_interval(hits, n) if n else None)


@dataclass(frozen=True, slots=True)
class Difference:
    """A difference of rates (or AUROCs) with its bootstrap interval and MDE."""

    estimate: Estimate
    mde: float


def _layout(records: Sequence[CaseRecord]) -> tuple[NDArray[np.str_], NDArray[np.int64] | None]:
    companies = np.array([record.ticker for record in records])
    folds = [record.fold for record in records]
    if any(fold is None for fold in folds):
        return companies, None
    return companies, np.array(folds, dtype=np.int64)


def _mde(result: BootstrapResult, confidence: float) -> float:
    se = float(np.std(result.replicates, ddof=1)) if result.replicates.size > 1 else 0.0
    return (normal_quantile(0.5 + confidence / 2) + normal_quantile(TARGET_POWER)) * se


def mean_difference(
    values: NDArray[np.float64],
    records: Sequence[CaseRecord],
    confidence: float,
    resamples: int = DEFAULT_RESAMPLES,
) -> Difference:
    """Bootstrap of the mean of per-case values (a paired difference of 0/1 outcomes)."""
    companies, folds = _layout(records)
    result = bootstrap_mean(
        values,
        companies,
        strata=folds,
        n_resamples=resamples,
        confidence=confidence,
        seed=REPORT_SEED,
    )
    return Difference(result.estimate, _mde(result, confidence))


def clustered_rate(
    mask: NDArray[np.bool_], records: Sequence[CaseRecord], resamples: int = DEFAULT_RESAMPLES
) -> Estimate:
    return mean_difference(mask.astype(np.float64), records, 0.95, resamples).estimate


@dataclass(frozen=True, slots=True)
class ArmRow:
    """One arm on the end-to-end tier."""

    arm: Arm
    false_answers: Rate
    false_answers_clustered: Estimate
    recall_cost: Rate
    correct: Rate
    numeric_accuracy: Rate
    citation_support: Rate
    generator_seconds_per_question: float
    gate_recall: float


def arm_row(
    arm: Arm,
    e2e: Sequence[CaseRecord],
    gate_tier: Sequence[CaseRecord],
    thresholds: dict[str, Threshold],
    resamples: int = DEFAULT_RESAMPLES,
) -> ArmRow:
    negatives = [record for record in e2e if not record.answerable]
    positives = [record for record in e2e if record.answerable]
    neg_answered = answered(arm, negatives, thresholds)
    pos_answered = answered(arm, positives, thresholds)
    pos_admitted = admitted(arm, positives, thresholds)
    a_correct = np.array([record.correct for record in positives], dtype=np.bool_)
    answered_positives = [r for r, a in zip(positives, pos_answered, strict=True) if a]
    let_through = admitted(arm, e2e, thresholds)
    seconds = sum(
        record.generator_seconds or 0.0
        for record, admit in zip(e2e, let_through, strict=True)
        if admit
    )
    gate_positives = [record for record in gate_tier if record.answerable]
    return ArmRow(
        arm=arm,
        false_answers=rate(neg_answered),
        false_answers_clustered=clustered_rate(neg_answered, negatives, resamples),
        recall_cost=rate(a_correct & ~pos_admitted),
        correct=rate(a_correct & pos_admitted),
        numeric_accuracy=rate(np.array([bool(r.numeric_correct) for r in answered_positives])),
        citation_support=rate(np.array([bool(r.citation_supported) for r in answered_positives])),
        generator_seconds_per_question=seconds / len(e2e) if e2e else 0.0,
        gate_recall=float(admitted(arm, gate_positives, thresholds).mean())
        if gate_positives
        else 0.0,
    )


@dataclass(frozen=True, slots=True)
class Paired:
    """FAR(first) - FAR(second) on the same negatives, with the discordant pairs."""

    difference: Difference
    first_only: int
    second_only: int
    mcnemar_p: float


def paired_far(
    first: Arm,
    second: Arm,
    negatives: Sequence[CaseRecord],
    thresholds: dict[str, Threshold],
    confidence: float,
    resamples: int = DEFAULT_RESAMPLES,
) -> Paired:
    one = answered(first, negatives, thresholds)
    two = answered(second, negatives, thresholds)
    b, c = discordant_counts(one, two)
    values = one.astype(np.float64) - two.astype(np.float64)
    return Paired(
        mean_difference(values, negatives, confidence, resamples), b, c, mcnemar_exact(b, c)
    )


def recall_cost_values(
    arm: Arm, positives: Sequence[CaseRecord], thresholds: dict[str, Threshold]
) -> NDArray[np.float64]:
    a_correct = np.array([record.correct for record in positives], dtype=np.bool_)
    return (a_correct & ~admitted(arm, positives, thresholds)).astype(np.float64)


@dataclass(frozen=True, slots=True)
class GateOnly:
    """A learned gate's scores on every golden case of the split."""

    score: str
    auroc: Estimate
    far: float
    recall: float
    n_positive: int
    n_negative: int
    cross_fitted: bool


def _scores(records: Sequence[CaseRecord], score: str) -> NDArray[np.float64]:
    return np.array([record.scores[score] for record in records], dtype=np.float64)


def _labels(records: Sequence[CaseRecord]) -> NDArray[np.bool_]:
    return np.array([record.answerable for record in records], dtype=np.bool_)


def _auroc_on(
    values: NDArray[np.float64], labels: NDArray[np.bool_]
) -> Callable[[NDArray[np.intp]], float]:
    def statistic(rows: NDArray[np.intp]) -> float:
        chosen = labels[rows]
        if chosen.all() or not chosen.any():
            return float("nan")
        return auroc(values[rows], chosen)

    return statistic


def gate_only(
    score: str,
    records: Sequence[CaseRecord],
    threshold: Threshold,
    resamples: int = DEFAULT_RESAMPLES,
) -> GateOnly:
    values, labels = _scores(records, score), _labels(records)
    companies, folds = _layout(records)
    result = cluster_bootstrap(
        _auroc_on(values, labels), companies, strata=folds, n_resamples=resamples, seed=REPORT_SEED
    )
    if threshold.cross_fitted and folds is not None:
        point = far_at_recall(values, labels, folds, threshold.target_recall)
        far, recall = point.false_accept_rate, point.recall
    else:
        cutoff = next(iter(threshold.by_fold.values()))
        far, recall = (
            float((values[~labels] >= cutoff).mean()),
            float((values[labels] >= cutoff).mean()),
        )
    return GateOnly(
        score=score,
        auroc=result.estimate,
        far=far,
        recall=recall,
        n_positive=int(labels.sum()),
        n_negative=int((~labels).sum()),
        cross_fitted=threshold.cross_fitted,
    )


def delta_auroc(
    first: str,
    second: str,
    records: Sequence[CaseRecord],
    resamples: int = DEFAULT_RESAMPLES,
) -> Difference:
    """AUROC(first) - AUROC(second) on the same cases, by company bootstrap, 95%."""
    one, two, labels = _scores(records, first), _scores(records, second), _labels(records)
    on_one, on_two = _auroc_on(one, labels), _auroc_on(two, labels)
    companies, folds = _layout(records)
    result = cluster_bootstrap(
        lambda rows: on_one(rows) - on_two(rows),
        companies,
        strata=folds,
        n_resamples=resamples,
        seed=REPORT_SEED,
    )
    return Difference(result.estimate, _mde(result, 0.95))


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else 0.0
