"""Counts and report of the v1.1 citation measurement (``docs/eval/protocol-v1.1.md``).

Everything here is exploratory and descriptive: it judges the generations the
v1 run recorded under other citation rules, and changes no number of the v1
report. Each count is k/n per arm with a Wilson 95% interval, and a zero is
printed with the upper bound it still allows.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from edgar_rag.citations import CURRENT
from edgar_rag.eval.citation_controls import (
    PERCENT_RULE_OFF,
    RESCALE_FLOOR_OFF,
    ReferenceProbe,
    WordsProbe,
    reference_probe,
    words_probe,
)
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
    reference: ReferenceProbe | None = None
    words: WordsProbe | None = None
    without_percent_rule: Judged | None = None
    without_rescale_floor: Judged | None = None


def case_results(cases: Sequence[Replayable]) -> tuple[CaseResult, ...]:
    return tuple(_result(case) for case in cases)


def _result(case: Replayable) -> CaseResult:
    answered = case.record.outcome == "answered"
    return CaseResult(
        id=case.record.id,
        recorded=case.record.outcome,
        references_stripped=judge(case.text, case.passages, REFERENCES_STRIPPED),
        current=judge(case.text, case.passages, CURRENT),
        words_only=states_figures_only_in_words(case.text),
        perturbed=_perturbed(case),
        reference=reference_probe(case) if answered else None,
        words=words_probe(case) if answered else None,
        without_percent_rule=judge(case.text, case.passages, PERCENT_RULE_OFF),
        without_rescale_floor=judge(case.text, case.passages, RESCALE_FLOOR_OFF),
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
        _reference_control(results),
        _words_control(results),
        _deviations(results),
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

The arms are masks over the same generations
([ADR-0015](../adr/0015-one-generation-serves-every-arm.md)): arms that admit
the same cases share a row, and the counts of different rows overlap, so they
are not independent measurements.
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


def _grouped(rows: Sequence[ArmCounts], cells: Callable[[ArmCounts], str]) -> list[str]:
    """One table line per distinct count, naming every arm that gives it."""
    arms: dict[str, list[str]] = {}
    for row in rows:
        arms.setdefault(cells(row), []).append(row.arm)
    return [f"| {', '.join(names)} | {cell} |" for cell, names in arms.items()]


def _table_m1(rows: Sequence[ArmCounts]) -> str:
    return _table(
        "M1: reference numbers as support",
        "Answered cases the v1.0.0 rules withhold as `unsupported_claim` once the "
        "item label is not read and `Note`, `page`, `Item`, `Exhibit` and `Section` "
        "numbers are removed from the cited passages.",
        f"| arms | withheld, k/n ({LABEL}, Wilson 95%) |\n|---|---|",
        _grouped(rows, lambda r: _with_ceiling(r.m1)),
    )


def _table_m2(rows: Sequence[ArmCounts]) -> str:
    return _table(
        "M2: figures only in words",
        "Answered cases whose every figure is written in words (number words with a "
        "scale word, `percent` or `dollars`); v1.0.0 did not check these.",
        f"| arms | answers, k/n ({LABEL}, Wilson 95%) |\n|---|---|",
        _grouped(rows, lambda r: _with_ceiling(r.m2)),
    )


def _table_m3(rows: Sequence[ArmCounts]) -> str:
    return _table(
        "M3: the v1.1 check against the recorded outcome",
        "The v1.1 check over the same generations: answered cases it withholds, and "
        "withheld cases (`no_valid_citation`, `unsupported_claim`) it would answer.",
        f"| arms | answered, now withheld ({LABEL}, Wilson 95%) "
        f"| withheld, now answered ({LABEL}, Wilson 95%) |\n|---|---|---|",
        _grouped(rows, lambda r: f"{_with_ceiling(r.lost)} | {_with_ceiling(r.gained)}"),
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
    return f"""## Positive controls (not pre-registered)

A zero only means something if the check could have said otherwise on these
same passages. Each control below plants one defect into the answers the run
accepted (arm A) and judges them again; it measures the replay, it is not a
result about the v1 answers.

**A wrong digit.** For every answered case that states a figure in digits,
the last digit of its first such figure is moved up by one and the answer is
judged again under the v1.1 check: it withholds {control.cell()}
({LABEL}, Wilson 95%). The rest are answers whose changed figure the cited
passage happens to print as well.
"""


def _share(probes: Sequence[object], hit: Callable[[object], bool]) -> Count:
    return Count(sum(hit(probe) for probe in probes), len(probes))


def _misses(ids: Sequence[str]) -> str:
    return ", ".join(f"`{case_id}`" for case_id in ids) if ids else "none"


def _reference_control(results: Sequence[CaseResult]) -> str:
    probes = [r.reference for r in results if r.reference is not None]
    accepted = Count(sum(p.v1_0_accepts for p in probes), len(probes))
    m1 = Count(sum(p.m1_withholds for p in probes), len(probes))
    current = Count(sum(p.current_withholds for p in probes), len(probes))
    return f"""**A reference number as a figure (power of M1).** For every answered case
whose cited passage carries a note, page, item, exhibit or section number that
it prints nowhere else as a figure, the first figure of the citing sentence is
replaced by that number (`$96.2 billion` becomes `$21 billion` for `page 21`).
The v1.0.0 rules accept {accepted.cell()}; the M1 rules withhold
{m1.cell()}, and the v1.1 check {current.cell()} ({LABEL}, Wilson 95%).
The ones M1 does not withhold are backed by the text of the passage anyway:
a reference number is short, so the planted figure is held only to one or
two digits, and a figure of the passage rounds to it at some scale (`$8
million` is `8,250` in a table in thousands). M1 sees a reference number
standing in for a figure only when no figure of the text rounds to it, so its
zero bounds that case, not every use of a reference number.
"""


def _words_control(results: Sequence[CaseResult]) -> str:
    probes = [r.words for r in results if r.words is not None]
    n = len(probes)
    counted = Count(sum(p.counted_by_m2 for p in probes), n)
    right = Count(sum(p.right_answered for p in probes), n)
    wrong = Count(sum(p.wrong_withheld for p in probes), n)
    old = Count(sum(p.wrong_accepted_by_v1_0 for p in probes), n)
    lost = _misses([p.id for p in probes if not p.right_answered])
    missed_ids = [p.id for p in probes if not p.wrong_withheld]
    digit_misses = [r.id for r in results if r.perturbed and r.perturbed.outcome == "answered"]
    missed = _misses(missed_ids) + (
        ", the answers the wrong-digit control misses too" if missed_ids == digit_misses else ""
    )
    return f"""**Figures in words (power of M2).** For every answered case whose figures in
digits can all be written in words, and whose first figure names dollars, a
scale or a percentage, every figure is written in words (`$6.1 billion`
becomes `six point one billion dollars`). M2 counts {counted.cell()}; the
v1.1 check still answers {right.cell()}. With the first figure one digit off,
the v1.1 check withholds {wrong.cell()} and v1.0.0 accepts {old.cell()}
({LABEL}, Wilson 95%). Withheld although right: {lost}. Answered although
wrong: {missed}.
"""


def _deviations(results: Sequence[CaseResult]) -> str:
    total = len(results)
    percent = sum(r.without_percent_rule != r.current for r in results)
    floor = sum(r.without_rescale_floor != r.current for r in results)
    return f"""## Deviations from the protocol

1. **Two rules outside the stated scope.** Section 1 of the protocol names two
   defects: reference numbers backing figures, and figures in words. The v1.1
   check also shipped rules 3 and 4 of the fix proposed in
   [issue #10](https://github.com/Rodrigo-Palma/edgar-rag/issues/10): a
   percentage is backed only by a percentage, and a number printed without a
   scale word is rescaled only when it has at least 3 digits, trailing zeros
   counted. The protocol names neither. Both, the 3 included, are written in
   issue #10 as opened on 2026-10-02 and never edited, a week before the
   protocol (`95eb85a`) and the measurement (`3078911`); 3 is above the one-
   and two-digit reference numbers of its probe (`Item 7`, `Note 3`, `page
   21`). M3 counts them only together with the rest. Each switched off alone,
   over the same {total} generations, changes the outcome of the v1.1 check
   in {percent} (the percentage rule) and {floor} (the 3-digit floor) of
   {total} cases.
2. **Positive controls.** None of the controls above is pre-registered: the
   wrong digit was added with the measurement, the reference number and the
   figures in words after review, in v1.1.1.
3. **The parser of figures in words, v1.1.1.** v1.1.0 read `one point five
   billion dollars` as five billion, did not read `two and a half billion`,
   and read `two thousand five hundred dollars` as two figures. v1.1.1 reads
   them whole and holds words that do not compose as a figure with no value.
   This report is rebuilt with it; M1, M2 and M3 give the counts v1.1.0 gave.
"""


def _limits() -> str:
    return """## What this does not establish

- Whether the v1 answers were wrong: a withheld answer may state a correct
  figure in a form the check cannot match, and an accepted one may still
  paraphrase wrongly without figures.
- Anything about another model, prompt or corpus. With n near 100 per arm, a
  zero still allows a rate of a few percent, as printed next to it.
- That figures in words are read in every form. Number words right after
  `half`, `quarters`, `thirds`, `fifths` or `tenths` are read as a fraction
  and held with no value, so `the first three quarters of one point seven
  seven dollars` is withheld although right; it fails closed.
"""
