"""Counts and report of the v1.1 citation measurement (``docs/eval/protocol-v1.1.md``).

Everything here is exploratory and descriptive: it judges the generations the
v1 run recorded under other citation rules, and changes no number of the v1
report. Each count is k/n per arm with a Wilson 95% interval, and a zero is
printed with the upper bound it still allows.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edgar_rag.citations import CURRENT
from edgar_rag.eval.citation_replay import (
    REFERENCES_STRIPPED,
    Judged,
    Replayable,
    judge,
    states_figures_only_in_words,
    with_first_figure_changed,
)
from edgar_rag.eval.metrics import wilson_interval
from edgar_rag.eval.records import CaseRecord

LABEL = "exploratory"
WITHHELD = ("no_valid_citation", "unsupported_claim")


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One case with text: what the run recorded, and what M1 and the v1.1 check say."""

    id: str
    recorded: str
    references_stripped: Judged
    current: Judged
    words_only: bool
    perturbed: Judged | None


def case_results(cases: Sequence[Replayable]) -> tuple[CaseResult, ...]:
    return tuple(
        CaseResult(
            id=case.record.id,
            recorded=case.record.outcome,
            references_stripped=judge(case.text, case.passages, REFERENCES_STRIPPED),
            current=judge(case.text, case.passages, CURRENT),
            words_only=states_figures_only_in_words(case.text),
            perturbed=_perturbed(case),
        )
        for case in cases
    )


def _perturbed(case: Replayable) -> Judged | None:
    if case.record.outcome != "answered":
        return None
    changed = with_first_figure_changed(case.text)
    return None if changed is None else judge(changed, case.passages, CURRENT)


@dataclass(frozen=True, slots=True)
class Count:
    k: int
    n: int

    def cell(self) -> str:
        """``k/n = p% [low, high]``, Wilson 95%; at 0/n the bound is the point."""
        if self.n == 0:
            return "n/a (n=0)"
        estimate = wilson_interval(self.k, self.n)
        span = f"[{100 * estimate.low:.1f}%, {100 * estimate.high:.1f}%]"
        return f"{self.k}/{self.n} = {100 * estimate.point:.1f}% {span}"


def positive_control(results: Sequence[CaseResult]) -> Count:
    """Answered cases with a figure in digits that the v1.1 check withholds once it is changed."""
    probed = [r.perturbed for r in results if r.perturbed is not None]
    return Count(sum(judged.outcome != "answered" for judged in probed), len(probed))


@dataclass(frozen=True, slots=True)
class ArmCounts:
    arm: str
    answered: int
    withheld: int
    m1: Count
    m2: Count
    lost: Count
    gained: Count


def arm_counts(
    results: Sequence[CaseResult],
    cases: Sequence[CaseRecord],
    masks: Mapping[str, NDArray[np.bool_]],
) -> tuple[ArmCounts, ...]:
    """The three measurements per arm, over the cases each arm admits."""
    by_id = {result.id: result for result in results}
    return tuple(
        _arm_row(
            arm,
            [by_id[c.id] for c, kept in zip(cases, mask, strict=True) if kept and c.id in by_id],
        )
        for arm, mask in masks.items()
    )


def _arm_row(arm: str, admitted: Sequence[CaseResult]) -> ArmCounts:
    answered = [r for r in admitted if r.recorded == "answered"]
    withheld = [r for r in admitted if r.recorded in WITHHELD]
    n = len(answered)
    return ArmCounts(
        arm=arm,
        answered=n,
        withheld=len(withheld),
        m1=Count(sum(r.references_stripped.outcome == "unsupported_claim" for r in answered), n),
        m2=Count(sum(r.words_only for r in answered), n),
        lost=Count(sum(r.current.outcome != "answered" for r in answered), n),
        gained=Count(sum(r.current.outcome == "answered" for r in withheld), len(withheld)),
    )


def arms_holding(
    case_id: str, cases: Sequence[CaseRecord], masks: Mapping[str, NDArray[np.bool_]]
) -> str:
    index = next(i for i, case in enumerate(cases) if case.id == case_id)
    return "".join(arm for arm, mask in masks.items() if mask[index])


@dataclass(frozen=True, slots=True)
class Pins:
    """The pinned inputs, as the report prints them."""

    cases_sha256: str
    tape_sha256: str
    protocol: str


def render(
    rows: Sequence[ArmCounts],
    results: Sequence[CaseResult],
    cases: Sequence[CaseRecord],
    masks: Mapping[str, NDArray[np.bool_]],
    pins: Pins,
) -> str:
    """The report, deterministic: the same inputs give the same bytes."""
    parts = [
        _header(results, pins),
        _table_m1(rows),
        _table_m2(rows),
        _table_m3(rows),
        _flips(results, cases, masks),
        _control(positive_control(results)),
        _limits(),
    ]
    return "\n".join(parts)


def _header(results: Sequence[CaseResult], pins: Pins) -> str:
    total = len(results)
    return f"""# Report v1.1: citation support over the v1 generations ({LABEL})

Generated by `make eval` (`scripts/measure_citation_support.py`) from the
frozen v1 run and its generation tape, with no model and no network, under
the protocol committed before it: [{pins.protocol}]({pins.protocol}).

**Exploratory and descriptive.** It changes no number of
[report-v1.md](report-v1.md), no headline, and neither H1 nor H2.

| input | sha256 |
|---|---|
| `eval/runs/v1/cases.jsonl` | `{pins.cases_sha256}` |
| `eval/runs/v1/tape/generate.jsonl` | `{pins.tape_sha256}` |

## Reconstruction

The passages of all {total} cases with text were read back from the prompts
the tape recorded; each matched one tape entry, and each passage carries the
item of the chunk the run retrieved. Replayed under the v1.0.0 rules, they
give back the outcome the run recorded for {total} of {total} cases. Every
count below rests on that reconstruction.
"""


def _table(title: str, intro: str, header: str, lines: Sequence[str]) -> str:
    body = "\n".join(lines)
    return f"## {title}\n\n{intro}\n\n{header}\n{body}\n"


def _ceiling(count: Count) -> str:
    if count.n == 0 or count.k:
        return ""
    return f"; at most {100 * wilson_interval(0, count.n).high:.1f}% (95%)"


def _with_ceiling(count: Count) -> str:
    return f"{count.cell()}{_ceiling(count)}"


def _table_m1(rows: Sequence[ArmCounts]) -> str:
    return _table(
        "M1: reference numbers as support",
        "Answered cases the v1.0.0 rules withhold as `unsupported_claim` once the "
        "item label is not read and `Note`, `page`, `Item`, `Exhibit` and `Section` "
        "numbers are removed from the cited passages.",
        f"| arm | withheld, k/n ({LABEL}, Wilson 95%) |\n|---|---|",
        [f"| {r.arm} | {_with_ceiling(r.m1)} |" for r in rows],
    )


def _table_m2(rows: Sequence[ArmCounts]) -> str:
    return _table(
        "M2: figures only in words",
        "Answered cases whose every figure is written in words (number words with a "
        "scale word, `percent` or `dollars`); v1.0.0 did not check these.",
        f"| arm | answers, k/n ({LABEL}, Wilson 95%) |\n|---|---|",
        [f"| {r.arm} | {_with_ceiling(r.m2)} |" for r in rows],
    )


def _table_m3(rows: Sequence[ArmCounts]) -> str:
    return _table(
        "M3: the v1.1 check against the recorded outcome",
        "The v1.1 check over the same generations: answered cases it withholds, and "
        "withheld cases (`no_valid_citation`, `unsupported_claim`) it would answer.",
        f"| arm | answered, now withheld ({LABEL}, Wilson 95%) "
        f"| withheld, now answered ({LABEL}, Wilson 95%) |\n|---|---|---|",
        [f"| {r.arm} | {_with_ceiling(r.lost)} | {_with_ceiling(r.gained)} |" for r in rows],
    )


def _flips(
    results: Sequence[CaseResult],
    cases: Sequence[CaseRecord],
    masks: Mapping[str, NDArray[np.bool_]],
) -> str:
    changed = [
        r
        for r in results
        if r.current.outcome != r.recorded
        or (r.recorded == "answered" and r.references_stripped.outcome != "answered")
    ]
    if not changed:
        return "## Cases that change\n\nNone: no case changes outcome under M1 or the v1.1 check.\n"
    lines = [
        f"| `{r.id}` | {arms_holding(r.id, cases, masks)} | {r.recorded} "
        f"| {r.references_stripped.outcome} | {r.current.outcome} | {r.current.finding or ''} |"
        for r in changed
    ]
    return _table(
        "Cases that change",
        "Every case whose outcome differs under M1 or the v1.1 check, by id.",
        "| case | arms | recorded | M1 | v1.1 | v1.1 finding |\n|---|---|---|---|---|---|",
        lines,
    )


def _control(control: Count) -> str:
    return f"""## Positive control (not pre-registered)

A zero only means something if the check could have said otherwise on these
same passages. For every answered case (arm A) that states a figure in
digits, the last digit of its first such figure is moved up by one and the
answer is judged again under the v1.1 check: it withholds {control.cell()}
({LABEL}, Wilson 95%). The rest are answers whose changed figure the cited
passage happens to print as well. This checks the replay, it is not a result.
"""


def _limits() -> str:
    return """## What this does not establish

- Whether the v1 answers were wrong: a withheld answer may state a correct
  figure in a form the check cannot match, and an accepted one may still
  paraphrase wrongly without figures.
- Anything about another model, prompt or corpus. With n near 100 per arm, a
  zero still allows a rate of a few percent, as printed next to it.
"""
