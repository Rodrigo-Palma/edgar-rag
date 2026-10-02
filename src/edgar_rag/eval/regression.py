"""The CI evaluation gate: a replayed dev run against the baseline committed beside it.

A replay is deterministic: the same code, index and tape give the same
record for every case. So a case whose result changed did so because the
code changed, and there is no run-to-run noise to allow for. The rule
(protocol plan, section 1, line 5, with decision D-05):

* every case is judged, under each arm the CI can run without brier, as
  right or wrong. An end-to-end case is right when the arm answers an
  answerable case correctly (the gold value, from a passage that prints it)
  or does not answer an unanswerable one; an end-to-end case the model was
  not asked about did not answer. A gate-only case has no
  generation, so it is right when the arm's gate admits an answerable case
  or declines an unanswerable one; arm A has no gate and is not judged there;
* cases are grouped by tier and class (answerable, or the kind of
  unanswerable), and every unanswerable case is also counted in its tier's
  ``unanswerable`` group, the unit of the false-answer rate, so a worsening
  spread over several kinds cannot pass under each kind's limit. Answerable is
  one class already, and there is no group of all cases: a lost answer and a
  newly refused negative would cancel there. Within a group and an arm, a case
  going from right to wrong is worse and from wrong to right is better;
* the gate fails on a net worsening of ``NET_CHANGE_LIMIT`` cases or more in
  any group and arm, and on a net improvement as large, which has to be
  written into the baseline in the same pull request
  (``make eval-ci-baseline``) so a later regression cannot hide behind it.
  It also fails when the cases or the cosine threshold are not the ones the
  baseline judged. A call missing from the tape fails earlier, in the replay.
* a change of ``outcome`` that leaves the case right, or wrong
  (``model_declined`` becoming ``no_valid_citation``), does not fail: the gate
  guards quality, and the reason a client sees is guarded by the contract
  tests. It is printed as a transition matrix and counted in a warning.

The gates' own decisions are outside what this judges, by design: the run
records every gate's raw score (cosine is asked with a threshold of 0) and
the arms are recomputed here from the scores. A bug in how a gate compares
its score to its threshold is left to the unit tests and to ``make demo``,
which asserts one cited answer and one ``out_of_period`` decline.

The cosine threshold is the service's default, so the arms are judged as
the service would answer. Passing means none of these cases got worse; with
100 answerable and 25 unanswerable cases per kind end to end, it says
nothing about differences in the population smaller than the interval
widths the report prints.
"""

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from edgar_rag.eval.arms import ARM_BY_KEY, COSINE, PERIOD, Arm
from edgar_rag.eval.metrics import wilson_interval
from edgar_rag.eval.records import CaseRecord, ModelInfo

BASELINE_FORMAT_VERSION = 1
NET_CHANGE_LIMIT = 3
E2E_ARMS = ("A", "B", "D", "F")
GATE_ARMS = ("B", "D", "F")
UPDATE = "run `make eval-ci-baseline` and commit eval/ci/baseline.json in this pull request"

Tier = Literal["e2e", "gate-only"]
UNANSWERABLE = "unanswerable"


class JudgedCase(BaseModel):
    """One case of the baseline: its group, what arm A did, and the arms that got it right."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    tier: Tier
    label: str
    outcome: str
    right: tuple[str, ...]


class Baseline(BaseModel):
    """The judged cases of a replayed dev run, and what they were judged against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: int = BASELINE_FORMAT_VERSION
    cosine_threshold: float
    golden_sha256: str
    index_digest: str
    generation: ModelInfo | None
    cases: tuple[JudgedCase, ...]


def _admits(arm: Arm, record: CaseRecord, cosine_threshold: float) -> bool:
    if arm.period and record.scores[PERIOD] < 1.0:
        return False
    return arm.score != COSINE or record.scores[COSINE] >= cosine_threshold


def judge(record: CaseRecord, cosine_threshold: float) -> JudgedCase:
    """Which arms get ``record`` right, by the rule in the module docstring."""
    # The golden set decides the tier, never what the run did: an end-to-end
    # case the pipeline stopped generating for stays end to end, and counts as
    # not answered, instead of moving to the tier judged on admission alone.
    tier: Tier = "e2e" if record.e2e else "gate-only"
    right: list[str] = []
    for key in E2E_ARMS if tier == "e2e" else GATE_ARMS:
        admitted = _admits(ARM_BY_KEY[key], record, cosine_threshold)
        if tier == "e2e":
            answers = admitted and record.answered
            ok = (answers and record.correct) if record.answerable else not answers
        else:
            ok = admitted == record.answerable
        if ok:
            right.append(key)
    label = "answerable" if record.answerable else str(record.negative_kind)
    return JudgedCase(
        id=record.id, tier=tier, label=label, outcome=record.outcome, right=tuple(right)
    )


def judge_all(records: Iterable[CaseRecord], cosine_threshold: float) -> tuple[JudgedCase, ...]:
    """Every golden case judged, sorted by id; narratives are not part of the CI tier."""
    golden = (record for record in records if record.kind == "golden")
    return tuple(sorted((judge(r, cosine_threshold) for r in golden), key=lambda c: c.id))


Group = tuple[Tier, str, str]
"""Tier, class and arm."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """What changed against the baseline, per group, and why the gate fails if it does."""

    worse: dict[Group, list[str]] = field(default_factory=dict)
    better: dict[Group, list[str]] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    transitions: dict[tuple[str, str], int] = field(default_factory=dict)
    silent_transitions: int = 0
    """Cases whose outcome changed while every arm judged them as before."""

    @property
    def passed(self) -> bool:
        return not self.failures


def compare(baseline: Baseline, current: Baseline) -> Verdict:
    """Judge ``current`` against ``baseline`` by the rule in the module docstring."""
    stale = _stale(baseline, current)
    if stale:
        return Verdict(failures=stale)
    before = {case.id: case for case in baseline.cases}
    worse: dict[Group, list[str]] = {}
    better: dict[Group, list[str]] = {}
    transitions: Counter[tuple[str, str]] = Counter()
    silent = 0
    for case in current.cases:
        old = before[case.id]
        if old.outcome != case.outcome:
            transitions[(old.outcome, case.outcome)] += 1
            silent += old.right == case.right
        for arm in E2E_ARMS if case.tier == "e2e" else GATE_ARMS:
            was, now = arm in old.right, arm in case.right
            if was != now:
                target = worse if was else better
                for label in _labels(case):
                    target.setdefault((case.tier, label, arm), []).append(case.id)
    failures = []
    for group in sorted(worse.keys() | better.keys()):
        net = len(worse.get(group, [])) - len(better.get(group, []))
        if net >= NET_CHANGE_LIMIT:
            failures.append(f"net worsening of {net} cases in {_name(group)} (limit {_limit()})")
        elif -net >= NET_CHANGE_LIMIT:
            failures.append(f"net improvement of {-net} cases in {_name(group)}: {UPDATE}")
    return Verdict(
        worse=worse,
        better=better,
        failures=tuple(failures),
        transitions=dict(sorted(transitions.items())),
        silent_transitions=silent,
    )


def _labels(case: JudgedCase) -> tuple[str, ...]:
    """The groups a case counts in: its class, and the aggregate when it is unanswerable."""
    return (case.label,) if case.label == "answerable" else (case.label, UNANSWERABLE)


def _stale(baseline: Baseline, current: Baseline) -> tuple[str, ...]:
    if baseline.cosine_threshold != current.cosine_threshold:
        return (
            f"the baseline was judged at cosine {baseline.cosine_threshold}, the service now "
            f"defaults to {current.cosine_threshold}: {UPDATE}",
        )
    old, new = {c.id for c in baseline.cases}, {c.id for c in current.cases}
    if baseline.golden_sha256 != current.golden_sha256 or old != new:
        return (
            f"the cases are not the ones the baseline judged ({len(old - new)} gone, "
            f"{len(new - old)} new): {UPDATE}",
        )
    return ()


def _name(group: Group) -> str:
    tier, label, arm = group
    return f"{tier} {label} under arm {arm} ({ARM_BY_KEY[arm].label})"


def _limit() -> str:
    return f"{NET_CHANGE_LIMIT}; a lasting change goes in the baseline: {UPDATE}"


def render(verdict: Verdict, current: Baseline, examples: int = 5) -> str:
    """The verdict as markdown: what changed, the interval widths of the tier, pass or fail."""
    lines = ["## CI evaluation: replayed dev split against eval/ci/baseline.json", ""]
    groups = sorted(verdict.worse.keys() | verdict.better.keys())
    if groups:
        lines += ["| tier | class | arm | worse | better | net |", "|---|---|---|---|---|---|"]
        for group in groups:
            worse, better = verdict.worse.get(group, []), verdict.better.get(group, [])
            tier, label, arm = group
            lines.append(
                f"| {tier} | {label} | {arm} | {len(worse)} | {len(better)} "
                f"| {len(worse) - len(better):+d} |"
            )
        lines.append("")
        for group in groups:
            worse = verdict.worse.get(group, [])
            if worse:
                shown = ", ".join(f"`{case}`" for case in worse[:examples])
                lines.append(f"- worse in {_name(group)}: {shown}")
        lines.append("")
    elif verdict.passed:
        lines += ["Every case judged as in the baseline, under every arm.", ""]
    if verdict.transitions:
        lines += [
            f"Outcome changes ({verdict.silent_transitions} of them left the case judged as "
            "before; those do not fail the gate):",
            "",
            "| from | to | cases |",
            "|---|---|---|",
        ]
        lines += [f"| {old} | {new} | {n} |" for (old, new), n in verdict.transitions.items()]
        lines.append("")
    lines += [_widths(current.cases), ""]
    if verdict.passed:
        lines.append(f"PASS: no group changed by a net {NET_CHANGE_LIMIT} cases or more.")
    else:
        lines += [f"FAIL: {failure}" for failure in verdict.failures]
    return "\n".join(lines) + "\n"


def _widths(cases: Sequence[JudgedCase]) -> str:
    """The 95% Wilson half-width at 50% for each end-to-end class: what this tier cannot see."""
    counts = Counter(case.label for case in cases if case.tier == "e2e")
    parts = []
    for label, n in sorted(counts.items()):
        estimate = wilson_interval(n // 2, n)
        parts.append(f"{label} n={n} ±{50 * (estimate.high - estimate.low):.1f} p.p.")
    return (
        "Replay is deterministic, so any change listed is a change in behaviour. It is not a "
        "population estimate: end to end, a rate in this tier is known to within "
        + "; ".join(parts)
        + " (95% Wilson, at 50%)."
    )


def dumps(baseline: Baseline) -> str:
    """Canonical JSON with one case per line, so a baseline update diffs case by case."""
    header = baseline.model_dump(mode="json", exclude={"cases"})
    head = json.dumps(header, sort_keys=True, indent=1)[:-2]
    rows = ",\n".join(
        "  " + json.dumps(case.model_dump(mode="json"), sort_keys=True) for case in baseline.cases
    )
    return f'{head},\n "cases": [\n{rows}\n ]\n}}\n'


def loads(text: str) -> Baseline:
    return Baseline.model_validate_json(text)
