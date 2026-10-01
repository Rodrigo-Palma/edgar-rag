"""Power and minimum detectable effects for the pre-registered evaluation.

Run before the evaluation, so the report can say which differences this sample
can and cannot resolve:

    python -m edgar_rag.eval.power --n-companies 20 --neg-per-company 15

The simulated comparisons go through the same estimator as the report
(``bootstrap_mean``: cluster bootstrap by company, stratified by
cross-fitting fold, percentile interval), so the half-widths and power printed
here are those of the analysis, not of a textbook approximation of it.

Every assumption is a parameter with its default printed in the output: the
spread between companies is a random intercept on the logit scale (sd 0.5), and
for the paired comparison the rate at which the new gate does worse than the
baseline is fixed (1 p.p.) while its rate of doing better varies.
"""

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from edgar_rag.eval.bootstrap import bootstrap_mean
from edgar_rag.eval.metrics import auroc, normal_quantile, wilson_interval

CONFIRMATORY_CONFIDENCE = 0.975  # Bonferroni over the two confirmatory hypotheses
TARGET_POWER = 0.80
H1_MARGIN = 0.05
H1_EFFECTS = (0.05, 0.075, 0.10)
H2_EFFECTS = (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
# Nominal level whose percentile interval reached ~97.5% coverage in this design's
# simulation (the 97.5% percentile interval covered ~95%). See ``build_report``.
CALIBRATED_CONFIDENCE = 0.99
_QUADRATURE_NODES = 64


@dataclass(frozen=True, slots=True)
class Design:
    """The clustered layout of the end-to-end negatives and the simulation settings."""

    n_companies: int = 20
    negatives_per_company: int = 15
    n_folds: int = 2
    company_sd: float = 0.5
    replicates: int = 600
    resamples: int = 5_000
    confidence: float = CONFIRMATORY_CONFIDENCE
    seed: int = 20261001

    @property
    def n_negatives(self) -> int:
        return self.n_companies * self.negatives_per_company


@dataclass(frozen=True, slots=True)
class DifferenceSimulation:
    """What the analysis would report for one true difference, over many replicates."""

    true_difference: float
    mean_half_width: float
    coverage: float
    power: float
    intervals: NDArray[np.float64]

    def share_with_low_above(self, margin: float) -> float:
        """Share of replicates whose lower bound clears ``margin``."""
        return float(np.mean(self.intervals[:, 0] > margin))


def _expit(x: NDArray[np.float64]) -> NDArray[np.float64]:
    result: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-x))
    return result


def marginal_intercept(marginal_rate: float, sd: float) -> float:
    """Logit intercept b with E[expit(b + u)] = ``marginal_rate`` for u ~ N(0, sd).

    Without this, adding a company effect would also move the average rate, and
    the simulation would test a different effect from the one it names.
    """
    if not 0.0 < marginal_rate < 1.0:
        raise ValueError(f"marginal_rate must be in (0, 1), got {marginal_rate}")
    nodes, weights = np.polynomial.hermite_e.hermegauss(_QUADRATURE_NODES)
    weights = weights / weights.sum()
    low, high = -30.0, 30.0
    for _ in range(100):
        middle = (low + high) / 2
        if float(np.sum(weights * _expit(middle + sd * nodes))) < marginal_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _company_rates(
    rng: np.random.Generator, marginal_rate: float, design: Design
) -> NDArray[np.float64]:
    intercept = marginal_intercept(marginal_rate, design.company_sd)
    effects = rng.normal(0.0, design.company_sd, size=design.n_companies)
    return _expit(intercept + effects)


def _layout(design: Design) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    companies = np.repeat(np.arange(design.n_companies), design.negatives_per_company)
    folds = companies % design.n_folds
    return companies, folds


def _summarise(
    true_difference: float,
    intervals: NDArray[np.float64],
    rejects: NDArray[np.bool_],
) -> DifferenceSimulation:
    low, high = intervals[:, 0], intervals[:, 1]
    return DifferenceSimulation(
        true_difference=true_difference,
        mean_half_width=float(np.mean((high - low) / 2)),
        coverage=float(np.mean((low <= true_difference) & (true_difference <= high))),
        power=float(np.mean(rejects)),
        intervals=intervals,
    )


def simulate_nested_difference(
    difference: float, design: Design, margin: float = 0.0
) -> DifferenceSimulation:
    """H1: FAR(A) - FAR(F) when F only ever removes answers A gave.

    Because the gate only subtracts, the difference per negative is 1 exactly
    when A answers and F declines, so only that rate is simulated. Power is the
    share of replicates whose lower bound clears ``margin``.
    """
    rng = np.random.default_rng(design.seed)
    companies, folds = _layout(design)
    intervals = np.empty((design.replicates, 2))
    for replicate in range(design.replicates):
        rates = _company_rates(rng, difference, design)
        outcomes = rng.random(companies.size) < rates[companies]
        result = bootstrap_mean(
            outcomes,
            companies,
            strata=folds,
            n_resamples=design.resamples,
            confidence=design.confidence,
            seed=int(rng.integers(2**32)),
        )
        intervals[replicate] = (result.estimate.low, result.estimate.high)
    return _summarise(difference, intervals, intervals[:, 0] > margin)


def simulate_paired_difference(
    net_improvement: float, design: Design, worse_rate: float = 0.01
) -> DifferenceSimulation:
    """H2: FAR(E) - FAR(F) on the same negatives, discordant both ways.

    Per negative, E alone answers (E worse) at ``worse_rate`` and F alone
    answers (E better) at ``worse_rate + net_improvement``, each with its own
    company effect. The true difference is ``-net_improvement``; power is the
    share of replicates whose upper bound is below zero.
    """
    rng = np.random.default_rng(design.seed)
    companies, folds = _layout(design)
    intervals = np.empty((design.replicates, 2))
    for replicate in range(design.replicates):
        better = _company_rates(rng, worse_rate + net_improvement, design)
        worse = _company_rates(rng, worse_rate, design)
        draw = rng.random(companies.size)
        is_better = draw < better[companies]
        is_worse = (~is_better) & (draw < better[companies] + worse[companies])
        result = bootstrap_mean(
            is_worse.astype(np.float64) - is_better.astype(np.float64),
            companies,
            strata=folds,
            n_resamples=design.resamples,
            confidence=design.confidence,
            seed=int(rng.integers(2**32)),
        )
        intervals[replicate] = (result.estimate.low, result.estimate.high)
    return _summarise(0.0 - net_improvement, intervals, intervals[:, 1] < 0.0)


def wilson_half_width(rate: float, n: int, confidence: float = 0.95) -> float:
    """Half-width of the Wilson interval at the count nearest ``rate * n``."""
    return wilson_interval(round(rate * n), n, confidence).half_width


def hanley_mcneil_se(area: float, n_positive: int, n_negative: int) -> float:
    """Standard error of an AUROC under Hanley and McNeil (1982), i.i.d. cases."""
    q1 = area / (2 - area)
    q2 = 2 * area * area / (1 + area)
    variance = (
        area * (1 - area)
        + (n_positive - 1) * (q1 - area * area)
        + (n_negative - 1) * (q2 - area * area)
    ) / (n_positive * n_negative)
    return math.sqrt(variance)


def delta_auroc_mde(
    area: float,
    n_positive: int,
    n_negative: int,
    correlation: float = 0.5,
    confidence: float = 0.95,
    power: float = TARGET_POWER,
) -> float:
    """MDE of the difference of two correlated AUROCs on the same cases, i.i.d."""
    se = hanley_mcneil_se(area, n_positive, n_negative)
    se_difference = se * math.sqrt(2 * (1 - correlation))
    return (normal_quantile(0.5 + confidence / 2) + normal_quantile(power)) * se_difference


def clustered_delta_auroc_mde(
    area: float,
    per_company: int,
    design: Design,
    correlation: float = 0.5,
    confidence: float = 0.95,
    power: float = TARGET_POWER,
) -> float:
    """MDE of the AUROC difference by simulation, with a company shift on the scores.

    Binormal scores with unit noise: each company shifts both gates' scores
    (sd ``design.company_sd``, correlated across gates) for all its questions, and
    the two gates' noise is correlated at ``correlation``. Both gates have the
    same true AUROC, so the spread of the difference is its null standard error.
    """
    rng = np.random.default_rng(design.seed)
    separation = math.sqrt(2.0 * (1.0 + design.company_sd**2)) * normal_quantile(area)
    companies = np.repeat(np.arange(design.n_companies), 2 * per_company)
    labels = np.tile(np.r_[np.ones(per_company), np.zeros(per_company)], design.n_companies)
    covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    differences = np.empty(design.replicates)
    for replicate in range(design.replicates):
        shifts = rng.multivariate_normal(
            [0.0, 0.0], covariance * design.company_sd**2, size=design.n_companies
        )
        noise = rng.multivariate_normal([0.0, 0.0], covariance, size=companies.size)
        scores = separation * labels[:, None] + shifts[companies] + noise
        differences[replicate] = auroc(scores[:, 0], labels) - auroc(scores[:, 1], labels)
    z = normal_quantile(0.5 + confidence / 2) + normal_quantile(power)
    return z * float(np.std(differences, ddof=1))


def minimum_detectable(
    simulations: Sequence[DifferenceSimulation], power: float = TARGET_POWER
) -> float | None:
    """Smallest simulated effect size reaching ``power``, or None if none does."""
    reached = [abs(s.true_difference) for s in simulations if s.power >= power]
    return min(reached) if reached else None


def _pp(value: float) -> str:
    return f"{100 * value:.1f} p.p."


def _wilson_rows(n_negatives: int, per_subtype: int, n_positives: int) -> list[str]:
    rows = []
    for label, n, rates in (
        (f"FAR of one arm, n={n_negatives}", n_negatives, (0.10, 0.20)),
        (f"FAR per subtype, n={per_subtype}", per_subtype, (0.10, 0.20)),
        (f"Recall cost / accuracy, n={n_positives}", n_positives, (0.85, 0.70)),
    ):
        widths = ", ".join(f"±{_pp(wilson_half_width(r, n))} at {r:.0%}" for r in rates)
        rows.append(f"| {label} | Wilson 95% | {widths} |")
    return rows


def _h1_rows(design: Design) -> list[str]:
    rows = []
    for effect in H1_EFFECTS:
        h1 = simulate_nested_difference(effect, design)
        above_margin = h1.share_with_low_above(H1_MARGIN)
        rows.append(
            f"| H1 FAR(A)-FAR(F) at {_pp(effect)} | nested, {design.confidence:.1%} CI | "
            f"half-width ±{_pp(h1.mean_half_width)}, coverage {h1.coverage:.3f}, "
            f"P(low>0) {h1.power:.2f}, P(low>{_pp(H1_MARGIN)}) {above_margin:.2f} |"
        )
    return rows


def _h2_rows(design: Design, worse_rate: float) -> list[str]:
    """H2 power over a grid, with the null first: its rejection rate is the real size
    of the test, to read against the nominal (1 - confidence) / 2."""
    simulations = [simulate_paired_difference(net, design, worse_rate) for net in H2_EFFECTS]
    level = f"{design.confidence:.1%} CI"
    rows = [
        f"| H2 FAR(E)-FAR(F) at {_pp(result.true_difference)} | paired, {level} | "
        f"P(high<0) {result.power:.3f}, half-width ±{_pp(result.mean_half_width)}, "
        f"coverage {result.coverage:.3f} |"
        for result in simulations
    ]
    mde = minimum_detectable([s for s in simulations if s.true_difference != 0.0])
    rows.append(
        f"| H2 MDE at {TARGET_POWER:.0%} power | paired, {level} | "
        f"{_pp(mde) if mde is not None else 'above the grid'} |"
    )
    return rows


def _auroc_rows(design: Design, gate_per_class: int, area: float) -> list[str]:
    iid = delta_auroc_mde(area, gate_per_class, gate_per_class)
    per_company = max(1, gate_per_class // design.n_companies)
    clustered = clustered_delta_auroc_mde(
        area,
        per_company,
        Design(
            n_companies=design.n_companies,
            company_sd=design.company_sd,
            replicates=min(design.replicates, 300),
            seed=design.seed,
        ),
    )
    return [
        f"| ΔAUROC (C-B), {gate_per_class}/{gate_per_class} at AUROC {area} | "
        f"Hanley-McNeil, r=0.5, 95%/80% | MDE {iid:.3f} i.i.d.; "
        f"{clustered:.3f} with company shift sd {design.company_sd} (simulated) |"
    ]


def build_report(
    design: Design,
    per_subtype: int,
    n_positives: int,
    gate_per_class: int,
    worse_rate: float,
    area: float,
    calibrated_confidence: float = CALIBRATED_CONFIDENCE,
) -> str:
    """The power table as markdown, assumptions first.

    The H2 grid runs twice: at the nominal confirmatory level, and at
    ``calibrated_confidence``. With 10 companies per fold the percentile interval
    is too narrow, so the nominal run shows the test's real size and the
    calibrated run shows the power the analysis has once its size is honest.
    """
    header = [
        f"Design: {design.n_companies} companies x {design.negatives_per_company} negatives "
        f"= {design.n_negatives}, {design.n_folds} folds; company effect sd "
        f"{design.company_sd} logit; {design.replicates} replicates x {design.resamples} "
        f"resamples; seed {design.seed}; H2 worse-rate {_pp(worse_rate)}.",
        "",
        "| Comparison | Design | Result |",
        "|---|---|---|",
    ]
    body = (
        _wilson_rows(design.n_negatives, per_subtype, n_positives)
        + _h1_rows(design)
        + _h2_rows(design, worse_rate)
        + _h2_rows(replace(design, confidence=calibrated_confidence), worse_rate)
        + _auroc_rows(design, gate_per_class, area)
    )
    return "\n".join(header + body)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--n-companies", type=int, default=20)
    parser.add_argument("--neg-per-company", type=int, default=15)
    parser.add_argument("--per-subtype", type=int, default=75)
    parser.add_argument("--positives", type=int, default=180)
    parser.add_argument("--gate-per-class", type=int, default=2_000)
    parser.add_argument("--auroc", type=float, default=0.80)
    parser.add_argument("--company-sd", type=float, default=0.5)
    parser.add_argument("--worse-rate", type=float, default=0.01)
    parser.add_argument("--replicates", type=int, default=600)
    parser.add_argument("--resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--calibrated-confidence", type=float, default=CALIBRATED_CONFIDENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    design = Design(
        n_companies=args.n_companies,
        negatives_per_company=args.neg_per_company,
        company_sd=args.company_sd,
        replicates=args.replicates,
        resamples=args.resamples,
        seed=args.seed,
    )
    print(
        build_report(
            design,
            per_subtype=args.per_subtype,
            n_positives=args.positives,
            gate_per_class=args.gate_per_class,
            worse_rate=args.worse_rate,
            area=args.auroc,
            calibrated_confidence=args.calibrated_confidence,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
