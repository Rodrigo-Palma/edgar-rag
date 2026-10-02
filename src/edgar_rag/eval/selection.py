"""Sample the golden set from the candidate pools, deterministically.

Two tiers per split, the second inside the first:

* ``e2e``: a fixed number of positives and negatives per company. The
  negatives are split over the four subtypes by a quota that rotates with the
  company's position, so the split's totals per subtype are equal (15 per
  company over 20 companies gives 75 per subtype). Each pick is a different
  fact, spread over concepts before any concept repeats.
* gate-only: every positive (all phrasings), and as many negatives, split
  equally over the subtypes and, within a subtype, as evenly over companies as
  their pools allow. The e2e negatives are always among them.

Randomness comes from ``random.Random`` seeded with a string naming the seed,
the split, the company and the purpose, so a change to one company's pool never
reshuffles another's.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from edgar_rag.eval.golden import NEGATIVE_KINDS, GoldenCase, NegativeKind


@dataclass(frozen=True, slots=True)
class E2eQuota:
    positives: int
    negatives: int


def rng_for(seed: int, *parts: str) -> random.Random:
    return random.Random(":".join((str(seed), *parts)))


def kind_quota(total: int, position: int) -> dict[NegativeKind, int]:
    """``total`` split over the subtypes; the remainder rotates with ``position``."""
    count = len(NEGATIVE_KINDS)
    base, extra = divmod(total, count)
    rotated = [NEGATIVE_KINDS[(position + step) % count] for step in range(count)]
    return {kind: base + (1 if rotated.index(kind) < extra else 0) for kind in NEGATIVE_KINDS}


def fact_group(case: GoldenCase) -> tuple[str, ...]:
    """Cases that are phrasings of one question about one fact share a group."""
    return (
        case.accession,
        case.negative_kind or "positive",
        case.asked_company or "",
        case.concept or case.template,
        str(case.period_year),
    )


def spread_key(case: GoldenCase) -> str:
    return case.concept or case.template


def spread_sample(
    cases: Sequence[GoldenCase],
    count: int,
    rng: random.Random,
    group: Callable[[GoldenCase], tuple[str, ...]] = fact_group,
) -> list[GoldenCase]:
    """``count`` cases: one per group first, groups ordered to rotate concepts.

    Raises:
        ValueError: when the pool holds fewer than ``count`` cases.
    """
    if count > len(cases):
        raise ValueError(f"asked for {count} cases from a pool of {len(cases)}")
    groups: dict[tuple[str, ...], list[GoldenCase]] = defaultdict(list)
    for case in sorted(cases, key=lambda case: case.id):
        groups[group(case)].append(case)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    seen: dict[str, int] = defaultdict(int)
    ranked = []
    for position, key in enumerate(keys):
        concept = spread_key(groups[key][0])
        ranked.append((seen[concept], position, key))
        seen[concept] += 1
    ordered = [key for _, _, key in sorted(ranked)]
    deepest = max(len(members) for members in groups.values()) if groups else 0
    picks = [
        groups[key][depth]
        for depth in range(deepest)
        for key in ordered
        if depth < len(groups[key])
    ]
    return picks[:count]


def by_company_and_kind(
    cases: Sequence[GoldenCase],
) -> dict[tuple[str, str], list[GoldenCase]]:
    pools: dict[tuple[str, str], list[GoldenCase]] = defaultdict(list)
    for case in cases:
        pools[(case.ticker, case.negative_kind or "positive")].append(case)
    return pools


def select_e2e(
    pools: dict[tuple[str, str], list[GoldenCase]],
    tickers: Sequence[str],
    quota: E2eQuota,
    seed: int,
) -> set[str]:
    """The ids of the e2e cases of one split.

    Raises:
        ValueError: naming the company and subtype whose pool is too small.
    """
    chosen: set[str] = set()
    for position, ticker in enumerate(tickers):
        wanted: dict[str, int] = {"positive": quota.positives}
        for negative_kind, count in kind_quota(quota.negatives, position).items():
            wanted[negative_kind] = count
        for kind, count in wanted.items():
            chosen.update(_e2e_picks(pools.get((ticker, kind), []), count, seed, ticker, kind))
    return chosen


def _e2e_picks(
    pool: Sequence[GoldenCase], count: int, seed: int, ticker: str, kind: str
) -> set[str]:
    try:
        picks = spread_sample(pool, count, rng_for(seed, "e2e", ticker, kind))
    except ValueError as error:
        raise ValueError(f"e2e {ticker} {kind}: {error}") from error
    if len({fact_group(case) for case in picks}) < count:
        raise ValueError(f"e2e {ticker} {kind}: fewer than {count} distinct facts")
    return {case.id for case in picks}


def even_allocation(total: int, capacity: dict[str, int], order: Sequence[str]) -> dict[str, int]:
    """``total`` spread over ``order`` one at a time, never past a capacity.

    Raises:
        ValueError: when the capacities add up to less than ``total``.
    """
    if total > sum(capacity.get(name, 0) for name in order):
        raise ValueError(f"cannot place {total} cases in pools of {sum(capacity.values())}")
    allocation = dict.fromkeys(order, 0)
    left = total
    while left:
        for name in order:
            if left and allocation[name] < capacity.get(name, 0):
                allocation[name] += 1
                left -= 1
    return allocation


def select_gate_negatives(
    pools: dict[tuple[str, str], list[GoldenCase]],
    tickers: Sequence[str],
    total: int,
    e2e: set[str],
    seed: int,
) -> set[str]:
    """The ids of the gate-only negatives of one split, the e2e ones included.

    Raises:
        ValueError: when a subtype's pools cannot supply its quota.
    """
    chosen: set[str] = set()
    for kind, quota in kind_quota(total, 0).items():
        capacity = {ticker: len(pools.get((ticker, kind), [])) for ticker in tickers}
        allocation = even_allocation(quota, capacity, tickers)
        for ticker in tickers:
            pool = pools.get((ticker, kind), [])
            fixed = [case for case in pool if case.id in e2e]
            if allocation[ticker] < len(fixed):
                raise ValueError(f"gate-only {ticker} {kind}: quota below its e2e cases")
            rest = [case for case in pool if case.id not in e2e]
            rng = rng_for(seed, "gate", ticker, kind)
            extra = spread_sample(rest, allocation[ticker] - len(fixed), rng)
            chosen.update(case.id for case in fixed + extra)
    return chosen
