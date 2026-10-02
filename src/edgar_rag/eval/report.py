"""The evaluation report, as markdown, from a run directory alone.

Deterministic: no clock is read, every bootstrap is seeded, and every number
comes from the run's records and manifest, so the same run gives the same
bytes (``edgar-rag eval report --run <dir>`` twice, diffed, is the test).

What it always prints (protocol section 5.7): n and the number of companies
in every table, the MDE next to every comparison, synthetic and narrative
results apart, the fact that questions come from templates, the risk of
pre-training memorisation, and the hardware and date of the run. An arm whose
gate did not run (brier with no plugin) is printed as not run, not dropped.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from edgar_rag.eval.arms import (
    ARM_BY_KEY,
    ARMS,
    BRIER,
    COSINE,
    PERIOD,
    Threshold,
    answered,
    fit_threshold,
    has_score,
    runnable,
)
from edgar_rag.eval.bootstrap import DEFAULT_RESAMPLES
from edgar_rag.eval.metrics import Estimate, risk_coverage_curve
from edgar_rag.eval.records import CaseRecord, ModelInfo, Run
from edgar_rag.eval.summary import (
    ArmRow,
    Difference,
    Rate,
    arm_row,
    delta_auroc,
    gate_only,
    mean_difference,
    paired_far,
    percentile,
    rate,
    recall_cost_values,
)

# Nominal level of the confirmatory intervals. The protocol wants 97.5% (Bonferroni
# over H1 and H2); with 10 companies per fold the 97.5% percentile interval covered
# only 0.94-0.96 in simulation, and the 99% one covers 0.96-0.98 (docs/eval/protocol.md).
DEFAULT_CONFIRMATORY_CONFIDENCE = 0.99
H1_MARGIN = 0.05
RECALL_COST_CEILING = 0.12
STAGES = ("embed", "search", "gate", "generate")
COVERAGE_POINTS = (0.25, 0.5, 0.75, 0.9)
NOT_RUN = "not run: no brier scores in this run (EDGAR_RAG_BRIER_URL unset when recorded)"


@dataclass(frozen=True, slots=True)
class ReportOptions:
    confirmatory_confidence: float = DEFAULT_CONFIRMATORY_CONFIDENCE
    resamples: int = DEFAULT_RESAMPLES


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.1f} p.p."


def interval(estimate: Estimate | None) -> str:
    """A rate's interval in percent, or n/a."""
    if estimate is None:
        return "n/a"
    return f"[{100 * estimate.low:.1f}%, {100 * estimate.high:.1f}%]"


def rate_cell(value: Rate) -> str:
    if value.n == 0:
        return "n/a (n=0)"
    return f"{value.hits}/{value.n} = {pct(value.hits / value.n)} {interval(value.interval)}"


def _companies(records: Sequence[CaseRecord]) -> int:
    return len({record.ticker for record in records})


def _golden(records: Sequence[CaseRecord]) -> list[CaseRecord]:
    return [record for record in records if record.kind == "golden"]


def build_report(run: Run, options: ReportOptions | None = None) -> str:
    """The whole report for ``run``."""
    options = options if options is not None else ReportOptions()
    golden = _golden(run.cases)
    e2e = [record for record in golden if record.e2e and record.outcome != "scored_only"]
    thresholds = {
        score: fit_threshold(golden, score) for score in (COSINE, BRIER) if has_score(golden, score)
    }
    sections = [
        _header(run),
        _thresholds(thresholds),
        _headline(e2e, golden, thresholds, options) if e2e else "",
        _confirmatory(e2e, thresholds, options) if e2e else "",
        _subtypes(e2e, thresholds) if e2e else "",
        _gate_only(golden, thresholds, options),
        _risk_coverage(e2e, thresholds) if e2e else "",
        _narratives(run.cases),
        _determinism(run),
        _operations(run, e2e),
        _deviations(run),
        _limits(run),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _header(run: Run) -> str:
    m = run.manifest
    generation = m.generation
    lines = [
        f"# Evaluation report: {m.split} split, {m.tier} tier",
        "",
        "| field | value |",
        "|---|---|",
        f"| date, hardware | {m.date}, {m.hardware} |",
        f"| code | `{m.repo_sha[:12]}`{' (uncommitted changes)' if m.repo_dirty else ''} |",
        f"| mode | {m.mode}{f', first {m.limit} cases' if m.limit else ''} |",
        f"| golden set | sha256 `{m.golden_sha256[:16]}` |",
        f"| index | digest `{m.index_digest}`, {m.index_filings} filings |",
        f"| embedder | {m.embedding.name} `{(m.embedding.digest or 'unknown')[:12]}` |",
        f"| generator | {_model(generation)}, options {_options(m.generation_options)} |",
        f"| Ollama | {m.ollama_version or 'unknown'} |",
        f"| gates scored | {', '.join(m.gates)}; top_k {m.top_k} |",
        f"| brier | {_brier(run)} |",
        f"| protocol | {f'sha256 `{m.protocol_sha256[:16]}`' if m.protocol_sha256 else 'none'} |",
        f"| cases | {m.cases} ({_companies(run.cases)} companies); "
        f"wall time {m.wall_seconds:.0f} s |",
    ]
    if m.split == "dev":
        lines += [
            "",
            "Dev split: development and CI only. Its companies are not in any headline number, "
            "and its thresholds are fitted in-sample.",
        ]
    return "\n".join(lines)


def _model(info: ModelInfo | None) -> str:
    if info is None:
        return "not asked (gate-only tier)"
    return f"{info.name} `{(info.digest or 'unknown')[:12]}`"


def _options(options: dict[str, object]) -> str:
    return ", ".join(f"{key}={options[key]}" for key in sorted(options))


def _brier(run: Run) -> str:
    brier = run.manifest.brier
    if brier is None:
        return "not plugged in; arms C and E not run"
    return f"commit {brier.sha or 'not stated'}"


def _thresholds(thresholds: dict[str, Threshold]) -> str:
    lines = ["## Thresholds", ""]
    if not thresholds:
        return "\n".join(lines + ["No learned gate was scored."])
    lines += ["| gate | rule | threshold by fold |", "|---|---|---|"]
    for score, threshold in thresholds.items():
        rule = "cross-fitted R90" if threshold.cross_fitted else "in-sample R90 (no folds)"
        folds = ", ".join(
            f"{'all' if fold < 0 else f'fold {fold}'}: {value:.4f}"
            for fold, value in sorted(threshold.by_fold.items())
        )
        lines.append(f"| {score} | {rule}, target recall {threshold.target_recall:.2f} | {folds} |")
    return "\n".join(lines)


def _headline(
    e2e: list[CaseRecord],
    golden: list[CaseRecord],
    thresholds: dict[str, Threshold],
    options: ReportOptions,
) -> str:
    negatives = [r for r in e2e if not r.answerable]
    positives = [r for r in e2e if r.answerable]
    lines = [
        "## Six arms, end-to-end tier",
        "",
        f"{len(negatives)} unanswerable and {len(positives)} answerable questions, "
        f"{_companies(e2e)} companies. One generation per question serves every arm.",
        "",
        "| arm | false answers on unanswerable (Wilson 95%) | company bootstrap 95% "
        "| recall cost | answered correct | numeric accuracy (answered) "
        "| citation support (answered) | gate recall (gate-only) | generator s / question |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ARMS:
        if not runnable(arm, golden):
            lines.append(f"| {arm.key} {arm.label} | {NOT_RUN} |" + " |" * 7)
            continue
        row = arm_row(arm, e2e, golden, thresholds, options.resamples)
        lines.append(_arm_line(row))
    return "\n".join(lines)


def _arm_line(row: ArmRow) -> str:
    return (
        f"| {row.arm.key} {row.arm.label} | {rate_cell(row.false_answers)} "
        f"| {interval(row.false_answers_clustered)} | {rate_cell(row.recall_cost)} "
        f"| {rate_cell(row.correct)} | {rate_cell(row.numeric_accuracy)} "
        f"| {rate_cell(row.citation_support)} | {pct(row.gate_recall)} "
        f"| {row.generator_seconds_per_question:.2f} |"
    )


def _difference(diff: Difference, confidence: float) -> str:
    est = diff.estimate
    return (
        f"{pp(est.point)}, {confidence:.1%} CI [{100 * est.low:+.1f}, {100 * est.high:+.1f}] p.p.; "
        f"MDE at 80% power about {100 * diff.mde:.1f} p.p."
    )


def h1_reading(difference: Estimate, far_a: float, margin: float = H1_MARGIN) -> str:
    """The pre-registered reading of FAR(A) - FAR(F) against the margin of interest.

    A gate only removes answers arm A gave, so the difference can never exceed
    FAR(A). When arm A's own rate is under the margin the hypothesis cannot be
    met by any gate, and the report says so instead of leaving a bare "not met".
    """
    if difference.low > margin:
        return f"above the margin: the lower bound clears {pct(margin)}"
    if far_a < margin:
        return (
            f"not attainable: arm A answers {pct(far_a)} of unanswerable questions, under "
            f"the {pct(margin)} margin, and a gate cannot remove answers the model does not give"
        )
    if difference.high < margin:
        return f"below the margin: the upper bound is under {pct(margin)}"
    return f"inconclusive at this n: the interval contains {pct(margin)}"


def _confirmatory(
    e2e: list[CaseRecord], thresholds: dict[str, Threshold], options: ReportOptions
) -> str:
    negatives = [r for r in e2e if not r.answerable]
    positives = [r for r in e2e if r.answerable]
    level = options.confirmatory_confidence
    a, e, f = ARM_BY_KEY["A"], ARM_BY_KEY["E"], ARM_BY_KEY["F"]
    lines = ["## Confirmatory comparisons", ""]
    if not negatives or not positives:
        return "\n".join(lines + ["Needs answerable and unanswerable cases; not computed."])
    h1 = paired_far(a, f, negatives, thresholds, level, options.resamples)
    cost = mean_difference(
        recall_cost_values(f, positives, thresholds), positives, level, options.resamples
    )
    passes = h1.difference.estimate.low > H1_MARGIN and cost.estimate.high <= RECALL_COST_CEILING
    far_a = float(answered(a, negatives, thresholds).mean())
    lines += [
        f"H1, FAR(A) - FAR(F) on {len(negatives)} negatives: {_difference(h1.difference, level)} "
        f"Reading: {h1_reading(h1.difference.estimate, far_a)}. "
        f"Recall cost of F: {_difference(cost, level)} "
        f"Criterion (lower bound above {pct(H1_MARGIN)}, recall cost upper bound at most "
        f"{pct(RECALL_COST_CEILING)}): {'met' if passes else 'not met'}.",
        "",
    ]
    if runnable(e, e2e):
        h2 = paired_far(e, f, negatives, thresholds, level, options.resamples)
        lines.append(
            f"H2, FAR(E) - FAR(F), paired: {_difference(h2.difference, level)} "
            f"Discordant: E alone answers {h2.first_only}, F alone answers {h2.second_only} "
            f"(exact McNemar p = {h2.mcnemar_p:.3f}). Criterion (upper bound below 0): "
            f"{'met' if h2.difference.estimate.high < 0 else 'not met'}."
        )
    else:
        lines.append(f"H2, FAR(E) - FAR(F): {NOT_RUN}.")
    lines += [
        "",
        "A difference smaller than the MDE printed beside it is not distinguishable at this n.",
    ]
    return "\n".join(lines)


def _subtypes(e2e: list[CaseRecord], thresholds: dict[str, Threshold]) -> str:
    negatives = [r for r in e2e if not r.answerable]
    kinds = sorted({str(r.negative_kind) for r in negatives})
    header = "| arm | " + " | ".join(
        f"{kind} (n={sum(r.negative_kind == kind for r in negatives)})" for kind in kinds
    )
    lines = [
        "## False answers by subtype (descriptive)",
        "",
        header + " |",
        "|---|" + "---|" * len(kinds),
    ]
    for arm in ARMS:
        if not runnable(arm, e2e):
            lines.append(f"| {arm.key} | not run |" + " |" * (len(kinds) - 1))
            continue
        cells = []
        for kind in kinds:
            group = [r for r in negatives if r.negative_kind == kind]
            cells.append(rate_cell(rate(answered(arm, group, thresholds))))
        lines.append(f"| {arm.key} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Arm D on wrong_year is low by construction: the guard and the label share one "
        "definition of the period a filing covers.",
    ]
    return "\n".join(lines)


def _gate_only(
    golden: list[CaseRecord], thresholds: dict[str, Threshold], options: ReportOptions
) -> str:
    lines = [
        "## Gate only, every golden case of the split",
        "",
        "| gate | AUROC (company bootstrap 95%) | FAR at R90 | realised recall | n | companies |",
        "|---|---|---|---|---|---|",
    ]
    for score, threshold in thresholds.items():
        result = gate_only(score, golden, threshold, options.resamples)
        lines.append(
            f"| {score} | {result.auroc.point:.3f} "
            f"[{result.auroc.low:.3f}, {result.auroc.high:.3f}] "
            f"| {pct(result.far)} | {pct(result.recall)} "
            f"| {result.n_positive}/{result.n_negative} | {_companies(golden)} |"
        )
    if BRIER not in thresholds:
        lines.append(f"| brier | {NOT_RUN} |" + " |" * 4)
    if has_score(golden, PERIOD):
        admits = np.array([r.scores[PERIOD] >= 1.0 for r in golden], dtype=np.bool_)
        labels = np.array([r.answerable for r in golden], dtype=np.bool_)
        lines.append(
            f"| period guard (a rule, no score) | n/a | {pct(float(admits[~labels].mean()))} "
            f"| {pct(float(admits[labels].mean()))} | {int(labels.sum())}/{int((~labels).sum())} "
            f"| {_companies(golden)} |"
        )
    if COSINE in thresholds and BRIER in thresholds:
        diff = delta_auroc(BRIER, COSINE, golden, options.resamples)
        lines += [
            "",
            f"AUROC(brier) - AUROC(cosine): {diff.estimate.point:+.3f}, 95% CI "
            f"[{diff.estimate.low:+.3f}, {diff.estimate.high:+.3f}]; "
            f"MDE at 80% power about {diff.mde:.3f}.",
        ]
    return "\n".join(lines)


def _risk_coverage(e2e: list[CaseRecord], thresholds: dict[str, Threshold]) -> str:
    lines = [
        "## Risk and coverage, end-to-end tier",
        "",
        "Admitting in decreasing score order, then arm A. An error is an answer to an "
        "unanswerable question, or an answerable question not answered correctly.",
        "",
        "| gate | AURC | "
        + " | ".join(f"risk at {pct(c)} coverage" for c in COVERAGE_POINTS)
        + " |",
        "|---|---|" + "---|" * len(COVERAGE_POINTS),
    ]
    # Admitting an unanswerable question the model then declines is no error.
    correct = np.array([r.correct if r.answerable else not r.answered for r in e2e], dtype=np.bool_)
    for score in thresholds:
        curve = risk_coverage_curve([r.scores[score] for r in e2e], correct)
        cells = []
        for target in COVERAGE_POINTS:
            position = int(np.searchsorted(curve.coverage, target))
            position = min(position, curve.coverage.size - 1)
            cells.append(
                f"{pct(float(curve.risk[position]))} (at {pct(float(curve.coverage[position]))})"
            )
        lines.append(f"| {score} | {curve.area:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _narratives(cases: Sequence[CaseRecord]) -> str:
    narrative = [r for r in cases if r.kind == "narrative"]
    if not narrative:
        return ""
    decided = np.array([r.answered == r.answerable for r in narrative], dtype=np.bool_)
    graded = [r for r in narrative if r.answerable and r.answered and r.item_split_sane]
    excluded = sorted({r.ticker for r in narrative if r.answerable and r.item_split_sane is False})
    item_ok = np.array([r.expected_item in r.cited_items for r in graded], dtype=np.bool_)
    return "\n".join(
        [
            "## Narrative questions (written by hand, reported apart, never pooled)",
            "",
            f"Correct decision (answer when answerable, abstain when not), arm A: "
            f"{rate_cell(rate(decided))}, {_companies(narrative)} companies.",
            f"Expected item cited, answered answerable questions on filings whose item split is "
            f"sane: {rate_cell(rate(item_ok))}. Excluded for an unsound split: "
            f"{', '.join(excluded) or 'none'}.",
        ]
    )


def _determinism(run: Run) -> str:
    if not run.repeats:
        return ""
    same = np.array([r.identical for r in run.repeats], dtype=np.bool_)
    return "\n".join(
        [
            "## Determinism",
            "",
            f"Same prompt sent twice, text identical: {rate_cell(rate(same))}.",
        ]
    )


def _operations(run: Run, e2e: list[CaseRecord]) -> str:
    lines = ["## Operating cost", ""]
    if run.manifest.mode == "replay":
        return "\n".join(lines + ["Replayed run: stage times are replay times, not model times."])
    lines += ["| stage | n | P50 s | P95 s |", "|---|---|---|---|"]
    for stage in STAGES:
        values = [r.stages[stage] for r in run.cases if stage in r.stages]
        if values:
            lines.append(
                f"| {stage} | {len(values)} | {percentile(values, 0.5):.3f} "
                f"| {percentile(values, 0.95):.3f} |"
            )
    generated = [r for r in e2e if r.generated]
    if generated:
        lines += ["", "| arm A outcome | n | prompt tokens | completion tokens | generator s |"]
        lines.append("|---|---|---|---|---|")
        for label, group in (
            ("answered", [r for r in generated if r.answered]),
            ("declined after generating", [r for r in generated if not r.answered]),
        ):
            lines.append(_cost_line(label, group))
    return "\n".join(lines)


def _cost_line(label: str, group: list[CaseRecord]) -> str:
    if not group:
        return f"| {label} | 0 | n/a | n/a | n/a |"

    def mean(values: list[float]) -> str:
        return f"{float(np.mean(values)):.1f}" if values else "n/a"

    prompt = [float(r.prompt_tokens) for r in group if r.prompt_tokens is not None]
    completion = [float(r.completion_tokens) for r in group if r.completion_tokens is not None]
    seconds = [r.generator_seconds or 0.0 for r in group]
    return (
        f"| {label} | {len(group)} | {mean(prompt)} | {mean(completion)} "
        f"| {float(np.mean(seconds)):.2f} |"
    )


def _deviations(run: Run) -> str:
    noted = run.deviations.strip()
    return "\n".join(
        ["## Deviations from the protocol", "", noted or "None recorded for this run."]
    )


def _limits(run: Run) -> str:
    return "\n".join(
        [
            "## What this does not show",
            "",
            "- The golden questions are generated from XBRL templates; the hand-written "
            "narrative questions are reported apart and never pooled with them.",
            "- The generator may have seen these filings in pre-training. FY2024 and FY2025 "
            "filings and the check that a cited passage prints the gold value limit, not "
            "remove, that risk.",
            "- Company-clustered percentile intervals undercover with few companies; Wilson "
            "intervals assume independent questions.",
            f"- {run.manifest.cases} cases from {_companies(run.cases)} companies; "
            "nothing here generalises beyond filings like these.",
        ]
    )
