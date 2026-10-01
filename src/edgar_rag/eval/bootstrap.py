"""Cluster bootstrap by company, with percentile intervals.

Questions about the same company share a filing, a writing style and a
retrieval index, so their outcomes are correlated. Resampling questions as if
they were independent understates the variance; resampling whole companies does
not. When the analysis cross-fits thresholds in folds of companies, the folds
are passed as strata and each replicate keeps every fold at its own size.

Two entry points share one resampling routine, so with the same seed they draw
the same companies: ``cluster_bootstrap`` takes any statistic of the row
indices (AUROC, a cross-fitted FAR), and ``bootstrap_mean`` is a vectorised
shortcut for a mean of per-row values (a rate, or a paired difference), fast
enough for the power simulation to run thousands of times.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from edgar_rag.eval.metrics import DEFAULT_CONFIDENCE, Estimate

DEFAULT_RESAMPLES = 5_000
MAX_INVALID_SHARE = 0.01


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """The estimate on the full data, its percentile interval and the replicates behind it.

    ``n_invalid`` counts replicates where the statistic was undefined (an AUROC
    on a resample with no positives, say); they are dropped from the interval.
    """

    estimate: Estimate
    replicates: NDArray[np.float64]
    n_invalid: int


@dataclass(frozen=True, slots=True)
class _Clusters:
    rows: tuple[NDArray[np.intp], ...]
    strata: tuple[NDArray[np.intp], ...]


def _group(clusters: ArrayLike, strata: ArrayLike | None) -> _Clusters:
    cluster_array = np.asarray(clusters)
    if cluster_array.ndim != 1 or cluster_array.size == 0:
        raise ValueError("clusters must be a non-empty one-dimensional array")
    cluster_ids, row_cluster = np.unique(cluster_array, return_inverse=True)
    rows = tuple(np.flatnonzero(row_cluster == k) for k in range(cluster_ids.size))
    if strata is None:
        return _Clusters(rows=rows, strata=(np.arange(cluster_ids.size),))
    strata_array = np.asarray(strata)
    if strata_array.shape != cluster_array.shape:
        raise ValueError("strata must match clusters in length")
    cluster_stratum = []
    for members in rows:
        values = np.unique(strata_array[members])
        if values.size != 1:
            raise ValueError("every cluster must sit inside a single stratum")
        cluster_stratum.append(values[0])
    stratum_of = np.asarray(cluster_stratum)
    groups = tuple(np.flatnonzero(stratum_of == s) for s in np.unique(stratum_of))
    return _Clusters(rows=rows, strata=groups)


def _draw(groups: _Clusters, n_resamples: int, seed: int) -> NDArray[np.intp]:
    """Cluster indices for each replicate, shape (n_resamples, n_clusters).

    Each stratum is resampled with replacement to its own size, in a fixed
    order, so the draw depends only on the seed and the cluster layout.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    rng = np.random.default_rng(seed)
    columns = [
        members[rng.integers(0, members.size, size=(n_resamples, members.size))]
        for members in groups.strata
    ]
    return np.concatenate(columns, axis=1)


def _percentile(
    point: float, replicates: NDArray[np.float64], confidence: float, n_invalid: int
) -> BootstrapResult:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    total = replicates.size + n_invalid
    if replicates.size == 0 or n_invalid > MAX_INVALID_SHARE * total:
        raise ValueError(
            f"{n_invalid} of {total} replicates were undefined; "
            "the statistic is too fragile for this many clusters"
        )
    alpha = 1.0 - confidence
    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        estimate=Estimate(point=point, low=float(low), high=float(high), confidence=confidence),
        replicates=replicates,
        n_invalid=n_invalid,
    )


def cluster_bootstrap(
    statistic: Callable[[NDArray[np.intp]], float],
    clusters: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int,
) -> BootstrapResult:
    """Percentile interval for ``statistic`` under resampling of whole clusters.

    ``statistic`` receives the row indices of a replicate (a cluster drawn twice
    contributes its rows twice) and returns a float, or NaN when undefined on
    that replicate. The point estimate is the statistic on all rows.
    """
    groups = _group(clusters, strata)
    draws = _draw(groups, n_resamples, seed)
    point = float(statistic(np.concatenate(groups.rows)))
    values = np.array(
        [statistic(np.concatenate([groups.rows[k] for k in draw])) for draw in draws],
        dtype=np.float64,
    )
    valid = np.isfinite(values)
    return _percentile(point, values[valid], confidence, int((~valid).sum()))


def bootstrap_mean(
    values: ArrayLike,
    clusters: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int,
) -> BootstrapResult:
    """Cluster bootstrap of the mean of per-row values, vectorised.

    For a rate pass 0/1 outcomes; for a paired difference between two arms pass
    ``first - second`` per row. Each replicate is the ratio of resampled cluster
    sums to resampled cluster sizes, which is exactly the mean of the rows of
    that replicate, so it agrees with ``cluster_bootstrap(np.mean...)``.
    """
    value_array = np.asarray(values, dtype=np.float64)
    groups = _group(clusters, strata)
    if value_array.shape != (sum(r.size for r in groups.rows),):
        raise ValueError("values must match clusters in length")
    sums = np.array([value_array[r].sum() for r in groups.rows])
    sizes = np.array([r.size for r in groups.rows], dtype=np.float64)
    draws = _draw(groups, n_resamples, seed)
    replicates = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return _percentile(float(value_array.mean()), replicates, confidence, 0)
