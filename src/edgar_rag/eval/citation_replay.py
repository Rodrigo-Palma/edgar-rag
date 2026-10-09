"""The citation check replayed over a frozen run, with no model, index or network.

A frozen run records each generation and the ids of the passages retrieved for
it, not the passages. The passages the model saw are in the prompt the tape
recorded, written as ``[n] (item) text`` inside a block tagged with the case's
nonce. This module reads them back, checks that the reconstruction is the one
the run judged (the items match the retrieved chunk ids, and the v1.0.0 rules
reproduce every recorded outcome), and then judges the same generations under
other rules. ``docs/eval/protocol-v1.1.md`` fixes what is counted.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from edgar_rag.citations import CURRENT, V1_0, SupportRules, check_claims, markers_in
from edgar_rag.citations import stated_figures as _stated_figures
from edgar_rag.domain import Chunk, ScoredChunk
from edgar_rag.eval.arms import ARMS, BRIER, COSINE, admitted, fit_threshold, runnable
from edgar_rag.eval.records import CaseRecord, Run
from edgar_rag.prompt import case_nonce

TEXT_OUTCOMES = ("answered", "no_valid_citation", "unsupported_claim")
REFERENCES_STRIPPED = replace(V1_0, read_item_label=False, strip_passage_references=True)
"""Measurement M1: the v1.0.0 rules, with the item label and references no longer read."""

_PASSAGE_HEAD = re.compile(r"(?:\A|\n\n)\[(\d+)\] \(([^)\n]*)\) ")


class ReplayError(ValueError):
    """The frozen inputs do not support a faithful replay."""


@dataclass(frozen=True, slots=True)
class Recorded:
    """One generation of the tape: the prompt sent and the text that came back."""

    prompt: str
    text: str


def read_generations(path: Path) -> tuple[Recorded, ...]:
    """Every recorded generation of a ``generate.jsonl`` tape."""
    rows = (json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip())
    return tuple(Recorded(row["request"]["prompt"], row["response"]["text"]) for row in rows)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_sha256(protocol: str, filename: str) -> str:
    """The sha256 the protocol's pin table gives for ``filename``.

    Raises:
        ReplayError: when the protocol pins no sha256 for it.
    """
    row = re.search(rf"`[^`]*{re.escape(filename)}`[^|]*\|\s*sha256 `([0-9a-f]{{64}})`", protocol)
    if row is None:
        raise ReplayError(f"the protocol pins no sha256 for {filename}")
    return row.group(1)


def item_of(chunk_id: str) -> str:
    """``0000320193-24-000123:Item 7#3`` is a chunk of ``Item 7``."""
    return chunk_id.split(":", 1)[-1].rsplit("#", 1)[0]


def passages_shown(prompt: str, nonce: str) -> tuple[tuple[str, str], ...]:
    """The ``(item, text)`` of each passage in the prompt's block, in order.

    ``as_data`` turned every ``[n]`` inside a passage into ``(n)``, so a
    blank line followed by ``[n] (`` can only open the next passage.

    Raises:
        ReplayError: when the block is missing or its passages are not 1..k.
    """
    opening, closing = f"<passages-{nonce}>\n", f"\n</passages-{nonce}>"
    start, end = prompt.find(opening), prompt.rfind(closing)
    if start < 0 or end < start:
        raise ReplayError(f"no passages block tagged {nonce} in the prompt")
    block = prompt[start + len(opening) : end]
    heads = list(_PASSAGE_HEAD.finditer(block))
    if [int(head.group(1)) for head in heads] != list(range(1, len(heads) + 1)):
        raise ReplayError(f"the passages tagged {nonce} are not numbered 1..k")
    bounds = [head.start() for head in heads[1:]] + [len(block)]
    return tuple(
        (head.group(2), block[head.end() : stop]) for head, stop in zip(heads, bounds, strict=True)
    )


@dataclass(frozen=True, slots=True)
class Replayable:
    """A case with text, and the passages its generation was asked about."""

    record: CaseRecord
    passages: tuple[ScoredChunk, ...]

    @property
    def text(self) -> str:
        return self.record.generation or ""


def reconstruct(cases: Iterable[CaseRecord], tape: Sequence[Recorded]) -> tuple[Replayable, ...]:
    """The passages of every case with text, read back from the tape and checked.

    Raises:
        ReplayError: when a case has no single tape entry, or the passages read
            back do not carry the items of the chunks the run retrieved.
    """
    by_text: dict[str, list[Recorded]] = {}
    for entry in tape:
        by_text.setdefault(entry.text, []).append(entry)
    return tuple(_one(case, by_text) for case in cases if case.outcome in TEXT_OUTCOMES)


def _one(case: CaseRecord, by_text: Mapping[str, list[Recorded]]) -> Replayable:
    nonce = case_nonce(case.id)()
    tag = f"<passages-{nonce}>"
    matches = [entry for entry in by_text.get(case.generation or "", []) if tag in entry.prompt]
    if len(matches) != 1:
        raise ReplayError(f"{case.id}: {len(matches)} tape entries, expected 1")
    shown = passages_shown(matches[0].prompt, nonce)
    if [item for item, _ in shown] != [item_of(chunk_id) for chunk_id in case.retrieved]:
        raise ReplayError(f"{case.id}: the passages read back are not the chunks retrieved")
    passages = tuple(
        ScoredChunk(Chunk(chunk_id=chunk_id, item=item, title="", text=text), 0.0)
        for chunk_id, (item, text) in zip(case.retrieved, shown, strict=True)
    )
    return Replayable(case, passages)


@dataclass(frozen=True, slots=True)
class Judged:
    """What the service would do with a generation: its outcome, and the finding if withheld."""

    outcome: str
    finding: str | None = None


def judge(text: str, passages: tuple[ScoredChunk, ...], rules: SupportRules) -> Judged:
    """The service's decision after generation (``Answerer._generate``), under ``rules``."""
    if not markers_in(text, len(passages)):
        return Judged("no_valid_citation")
    failure = check_claims(text, passages, rules)
    if failure is None:
        return Judged("answered")
    return Judged(failure.reason.value, failure.finding)


def states_figures_only_in_words(text: str) -> bool:
    """Whether every figure the answer states is spelled out, and there is at least one."""
    figures = _stated_figures(text, CURRENT)
    return bool(figures) and not any(char.isdigit() for figure in figures for char in figure.text)


def verify_v1_replay(cases: Sequence[Replayable]) -> None:
    """The v1.0.0 rules must give back every outcome the run recorded.

    Raises:
        ReplayError: naming the cases where they do not.
    """
    differ = [
        case.record.id
        for case in cases
        if judge(case.text, case.passages, V1_0).outcome != case.record.outcome
    ]
    if differ:
        raise ReplayError(f"the v1.0.0 rules do not reproduce {len(differ)} cases: {differ[:5]}")


def arm_masks(run: Run) -> dict[str, NDArray[np.bool_]]:
    """Which cases each runnable arm admits, keyed by arm, over ``run.cases``."""
    cases = run.cases
    scores = [score for score in (COSINE, BRIER) if all(score in c.scores for c in cases)]
    thresholds = {score: fit_threshold(cases, score) for score in scores}
    return {arm.key: admitted(arm, cases, thresholds) for arm in ARMS if runnable(arm, cases)}


def with_first_figure_changed(text: str) -> str | None:
    """The answer with the last digit of its first figure in digits moved up by one.

    A positive control: the check that reports zero changes on these answers
    has to withhold most of them, or a zero says nothing. ``None`` when the
    answer states no figure in digits.
    """
    figure = next(
        (f for f in _stated_figures(text, CURRENT) if any(c.isdigit() for c in f.text)), None
    )
    if figure is None:
        return None
    last = max(i for i, char in enumerate(figure.text) if char.isdigit())
    bumped = str((int(figure.text[last]) + 1) % 10)
    changed = figure.text[:last] + bumped + figure.text[last + 1 :]
    return text.replace(figure.text, changed, 1)
