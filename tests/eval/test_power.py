import math

import numpy as np
import pytest

from edgar_rag.eval.power import (
    Design,
    DifferenceSimulation,
    clustered_delta_auroc_mde,
    delta_auroc_mde,
    hanley_mcneil_se,
    main,
    marginal_intercept,
    minimum_detectable,
    simulate_nested_difference,
    simulate_paired_difference,
    wilson_half_width,
)

SMALL = Design(replicates=150, resamples=1_000)


def test_marginal_intercept_without_spread_is_the_logit():
    assert marginal_intercept(0.2, 0.0) == pytest.approx(math.log(0.2 / 0.8), abs=1e-9)


def test_marginal_intercept_keeps_the_average_rate_once_companies_vary():
    intercept = marginal_intercept(0.05, 0.5)
    effects = np.random.default_rng(0).normal(0.0, 0.5, size=400_000)

    assert np.mean(1 / (1 + np.exp(-(intercept + effects)))) == pytest.approx(0.05, abs=5e-4)
    assert intercept < math.log(0.05 / 0.95)  # the spread pushes the mean up, so b goes down


def test_marginal_intercept_rejects_rates_outside_the_open_interval():
    with pytest.raises(ValueError):
        marginal_intercept(0.0, 0.5)


@pytest.mark.parametrize(
    ("rate", "n", "plan_pp"),
    [
        (0.10, 300, 3.4),
        (0.20, 300, 4.5),
        (0.10, 75, 7.1),
        (0.20, 75, 8.9),
        (0.85, 180, 5.2),
        (0.70, 180, 6.6),
    ],
)
def test_wilson_half_widths_reproduce_the_plan_table(rate, n, plan_pp):
    assert 100 * wilson_half_width(rate, n) == pytest.approx(plan_pp, abs=0.05)


def test_hanley_mcneil_at_chance_has_the_closed_form():
    # At A = 0.5, Q1 = Q2 = 1/3, so each extra case adds 1/3 - 1/4 = 1/12.
    n_pos, n_neg = 40, 60
    expected = math.sqrt((0.25 + (n_pos - 1) / 12 + (n_neg - 1) / 12) / (n_pos * n_neg))

    assert hanley_mcneil_se(0.5, n_pos, n_neg) == pytest.approx(expected, rel=1e-12)


def test_delta_auroc_mde_reproduces_the_plan_and_shrinks_with_correlation():
    assert delta_auroc_mde(0.80, 2_000, 2_000) == pytest.approx(0.020, abs=0.001)
    assert delta_auroc_mde(0.80, 2_000, 2_000, correlation=0.8) < delta_auroc_mde(
        0.80, 2_000, 2_000
    )


def test_company_shift_makes_the_auroc_difference_harder_to_detect():
    design = Design(replicates=120, seed=1)
    clustered = clustered_delta_auroc_mde(0.80, per_company=100, design=design)

    assert clustered > delta_auroc_mde(0.80, 2_000, 2_000)


def test_nested_difference_half_width_matches_the_plan_at_five_points():
    result = simulate_nested_difference(0.05, SMALL)

    # Plan: 2.7 at 5 p.p.; acceptance is within 0.5 p.p.
    assert 100 * result.mean_half_width == pytest.approx(2.7, abs=0.5)
    assert result.power == 1.0  # the lower bound clears zero: a nested gate only subtracts
    assert result.share_with_low_above(0.05) < 0.10  # but not the 5 p.p. margin


def test_paired_power_grows_with_the_effect_and_respects_size_roughly():
    null = simulate_paired_difference(0.0, SMALL)
    small = simulate_paired_difference(0.02, SMALL)
    large = simulate_paired_difference(0.06, SMALL)

    assert null.true_difference == 0.0
    assert null.power < 0.06  # nominal one-sided size 1.25%; the simulation measured ~2.2%
    assert null.power < small.power < large.power
    assert large.power > 0.85


def test_simulations_are_reproducible_from_the_seed():
    first = simulate_paired_difference(0.04, Design(replicates=20, resamples=200))
    again = simulate_paired_difference(0.04, Design(replicates=20, resamples=200))

    np.testing.assert_array_equal(first.intervals, again.intervals)


def _fake(effect: float, power: float) -> DifferenceSimulation:
    return DifferenceSimulation(effect, 0.0, 0.0, power, np.zeros((1, 2)))


def test_minimum_detectable_is_the_smallest_effect_reaching_the_power():
    grid = [_fake(-0.02, 0.4), _fake(-0.05, 0.82), _fake(-0.04, 0.79), _fake(-0.06, 0.95)]

    assert minimum_detectable(grid) == 0.05
    assert minimum_detectable([_fake(-0.02, 0.4)]) is None


def test_the_command_prints_every_comparison(capsys):
    exit_code = main(["--replicates", "20", "--resamples", "200", "--gate-per-class", "400"])

    output = capsys.readouterr().out
    assert exit_code == 0
    for label in (
        "FAR of one arm, n=300",
        "H1 FAR(A)-FAR(F)",
        "H1 MDE at 80% power, lower bound above 5.0 p.p.",
        "H2 MDE",
        "97.5% CI",
        "99.0% CI",
        "ΔAUROC",
    ):
        assert label in output
