import numpy as np
import pytest

from edgar_rag.eval.bootstrap import bootstrap_mean, cluster_bootstrap
from edgar_rag.eval.metrics import auroc


def _clustered_outcomes(rng, n_clusters=20, per_cluster=15, a=2.0, b=8.0):
    """Binary outcomes whose cluster rates are Beta(a, b): population mean a / (a + b)."""
    rates = rng.beta(a, b, size=n_clusters)
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    return (rng.random(clusters.size) < rates[clusters]).astype(float), clusters


def test_same_seed_same_interval_and_different_seed_different_draws():
    rng = np.random.default_rng(0)
    values, clusters = _clustered_outcomes(rng)

    first = bootstrap_mean(values, clusters, n_resamples=500, seed=11)
    again = bootstrap_mean(values, clusters, n_resamples=500, seed=11)
    other = bootstrap_mean(values, clusters, n_resamples=500, seed=12)

    assert first.estimate == again.estimate
    np.testing.assert_array_equal(first.replicates, again.replicates)
    assert not np.array_equal(first.replicates, other.replicates)


def test_vectorised_mean_agrees_with_the_generic_bootstrap_draw_for_draw():
    rng = np.random.default_rng(1)
    values, clusters = _clustered_outcomes(rng)
    strata = clusters % 2

    fast = bootstrap_mean(values, clusters, strata=strata, n_resamples=300, seed=5)
    generic = cluster_bootstrap(
        lambda rows: float(values[rows].mean()),
        clusters,
        strata=strata,
        n_resamples=300,
        seed=5,
    )

    np.testing.assert_allclose(fast.replicates, generic.replicates, rtol=1e-12)
    assert fast.estimate.point == pytest.approx(generic.estimate.point)


def test_stratified_resampling_keeps_every_stratum_at_its_own_size():
    clusters = np.repeat(np.arange(6), 2)
    strata = np.repeat([0, 0, 0, 0, 1, 1], 2)  # 4 clusters in stratum 0, 2 in stratum 1
    seen = []

    def record(rows):
        seen.append(np.bincount(strata[rows], minlength=2))
        return 0.0

    cluster_bootstrap(record, clusters, strata=strata, n_resamples=50, seed=0)

    assert all(counts.tolist() == [8, 4] for counts in seen)


def test_whole_clusters_are_resampled_never_split():
    clusters = np.repeat(np.arange(5), 3)
    sizes = []

    def record(rows):
        sizes.append(np.bincount(clusters[rows], minlength=5) % 3)
        return 0.0

    cluster_bootstrap(record, clusters, n_resamples=50, seed=0)

    assert all(not counts.any() for counts in sizes)


def test_cluster_percentile_interval_covers_near_nominal_where_iid_does_not():
    """Coverage by simulation, fixed seed: 20 clusters x 15 rows, ICC = 1/11.

    The design effect is 1 + 14/11, about 2.3, so a bootstrap that treats rows as
    independent should cover far less than 95%, and the cluster one close to it
    (the percentile interval runs a little short with 20 clusters).
    """
    rng = np.random.default_rng(20261001)
    truth = 0.2
    replications = 300
    cluster_hits = iid_hits = 0
    for replication in range(replications):
        values, clusters = _clustered_outcomes(rng)
        by_cluster = bootstrap_mean(values, clusters, n_resamples=1_000, seed=replication)
        by_row = bootstrap_mean(values, np.arange(values.size), n_resamples=1_000, seed=replication)
        cluster_hits += by_cluster.estimate.low <= truth <= by_cluster.estimate.high
        iid_hits += by_row.estimate.low <= truth <= by_row.estimate.high

    # Standard error of a coverage near 0.95 over 300 replications is ~1.3 p.p.
    assert 0.90 <= cluster_hits / replications <= 0.98
    assert iid_hits / replications < 0.88


def test_undefined_replicates_are_dropped_and_counted():
    calls = []

    def undefined_once(rows):
        calls.append(rows.size)
        return float("nan") if len(calls) == 2 else float(len(calls))

    result = cluster_bootstrap(undefined_once, np.arange(10), n_resamples=200, seed=0)

    assert result.n_invalid == 1
    assert result.replicates.size == 199
    assert result.estimate.point == 1.0  # the full-data call comes first


def test_auroc_runs_through_the_generic_bootstrap():
    rng = np.random.default_rng(2)
    labels = np.tile([1, 0], 40).astype(bool)
    scores = rng.normal(size=labels.size) + labels
    clusters = np.repeat(np.arange(20), 4)

    result = cluster_bootstrap(
        lambda rows: auroc(scores[rows], labels[rows]), clusters, n_resamples=300, seed=0
    )

    assert result.estimate.point == auroc(scores, labels)
    assert result.estimate.low < result.estimate.point < result.estimate.high


def test_a_statistic_undefined_too_often_is_an_error_not_a_narrow_interval():
    clusters = np.arange(3)

    def mostly_nan(rows):
        return float("nan") if rows[0] != 0 else 1.0

    with pytest.raises(ValueError, match="undefined"):
        cluster_bootstrap(mostly_nan, clusters, n_resamples=100, seed=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"clusters": np.array([])}, "non-empty"),
        ({"strata": np.array([0])}, "strata must match"),
        ({"strata": np.array([0, 1, 0, 1]), "clusters": np.array([0, 0, 1, 1])}, "single stratum"),
        ({"n_resamples": 0}, "n_resamples"),
        ({"confidence": 1.5}, "confidence"),
        ({"values": np.ones(3)}, "values must match"),
    ],
)
def test_bootstrap_mean_rejects_inconsistent_inputs(kwargs, message):
    arguments = {"values": np.ones(4), "clusters": np.array([0, 0, 1, 1]), "seed": 0}
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        bootstrap_mean(**arguments)
