"""The six arms of the protocol, as masks over one recording.

A gate only decides whether the model is called; it never changes the
passages or the prompt. So arm A (no gate, the model's own refusal only) is
what the runner recorded, and a gated arm answers a case exactly when A
answered it and the arm's gate admits it. Each gated arm therefore answers a
subset of what A answers, by construction, and the six arms cost one
generation per case.

    A  no gate             D  period guard
    B  cosine              E  period guard + brier
    C  brier               F  period guard + cosine

A learned gate (cosine, brier) admits when its score reaches a threshold
chosen to admit 90% of the answerable questions of the gate-only tier (R90).
With cross-fitting folds (the eval split) each company's threshold is fitted
on the other fold, so no case is judged by a threshold that saw it. The dev
split has no folds; its threshold is fitted in-sample and the report says so.
E and F reuse the threshold of their learned gate, so E against C and F
against B isolate what the period guard adds.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edgar_rag.eval.metrics import DEFAULT_TARGET_RECALL, crossfit_thresholds, threshold_at_recall
from edgar_rag.eval.records import CaseRecord

COSINE = "cosine"
BRIER = "brier"
PERIOD = "period"
IN_SAMPLE = -1
"""The fold key of a threshold fitted on every case, when there are no folds."""


@dataclass(frozen=True, slots=True)
class Arm:
    key: str
    label: str
    period: bool
    score: str | None

    @property
    def gated(self) -> bool:
        return self.period or self.score is not None


ARMS: tuple[Arm, ...] = (
    Arm("A", "no gate, model refusal only", period=False, score=None),
    Arm("B", "cosine", period=False, score=COSINE),
    Arm("C", "brier", period=False, score=BRIER),
    Arm("D", "period guard", period=True, score=None),
    Arm("E", "period guard + brier", period=True, score=BRIER),
    Arm("F", "period guard + cosine", period=True, score=COSINE),
)
ARM_BY_KEY: Mapping[str, Arm] = {arm.key: arm for arm in ARMS}


@dataclass(frozen=True, slots=True)
class Threshold:
    """A learned gate's threshold per fold, and how it was fitted."""

    score: str
    by_fold: Mapping[int, float]
    cross_fitted: bool
    target_recall: float


def fold_key(record: CaseRecord) -> int:
    return IN_SAMPLE if record.fold is None else record.fold


def has_score(records: Sequence[CaseRecord], score: str) -> bool:
    """Whether every record carries ``score``: a gate that did not run has no arm."""
    return bool(records) and all(score in record.scores for record in records)


def runnable(arm: Arm, records: Sequence[CaseRecord]) -> bool:
    needs = [score for score in (arm.score, PERIOD if arm.period else None) if score is not None]
    return all(has_score(records, score) for score in needs)


def fit_threshold(
    records: Sequence[CaseRecord], score: str, target_recall: float = DEFAULT_TARGET_RECALL
) -> Threshold:
    """The R90 threshold of ``score``, fitted on the answerable golden cases of ``records``.

    Cross-fitted when every record has a fold, in-sample otherwise.

    Raises:
        ValueError: when the records mix cases with and without a fold, or
            carry no answerable case.
    """
    golden = [record for record in records if record.kind == "golden"]
    folds = {record.fold for record in golden}
    if None in folds and len(folds) > 1:
        raise ValueError("records mix cases with and without a cross-fitting fold")
    values = np.array([record.scores[score] for record in golden], dtype=np.float64)
    labels = np.array([record.answerable for record in golden], dtype=np.bool_)
    if None in folds:
        cutoff = threshold_at_recall(values[labels], target_recall)
        return Threshold(
            score, {IN_SAMPLE: cutoff}, cross_fitted=False, target_recall=target_recall
        )
    keys = np.array([fold_key(record) for record in golden], dtype=np.int64)
    fitted = crossfit_thresholds(values, labels, keys, target_recall)
    return Threshold(score, fitted, cross_fitted=True, target_recall=target_recall)


def admitted(
    arm: Arm, records: Sequence[CaseRecord], thresholds: Mapping[str, Threshold]
) -> NDArray[np.bool_]:
    """Whether the arm's gate lets each case through to the model."""
    mask = np.ones(len(records), dtype=np.bool_)
    if arm.period:
        mask &= np.array([record.scores[PERIOD] >= 1.0 for record in records], dtype=np.bool_)
    if arm.score is not None:
        threshold = thresholds[arm.score]
        mask &= np.array(
            [record.scores[arm.score] >= threshold.by_fold[fold_key(record)] for record in records],
            dtype=np.bool_,
        )
    return mask


def answered(
    arm: Arm, records: Sequence[CaseRecord], thresholds: Mapping[str, Threshold]
) -> NDArray[np.bool_]:
    """Whether the arm answers each case: arm A answered it and the arm's gate admits it."""
    unguarded = np.array([record.answered for record in records], dtype=np.bool_)
    return unguarded & admitted(arm, records, thresholds)
