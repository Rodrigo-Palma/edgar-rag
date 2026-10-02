"""Ask the golden set through the answerer the service uses, and record everything.

The runner does not reimplement the pipeline: each case goes through
``Answerer.ask`` (ADR 0008), with the case's deterministic nonce, so its
prompt, and its recorded generation, are the same on every run. What changes
is the gate. ``ScoreEvery`` asks every gate under evaluation (period guard,
cosine, and brier when it is plugged in) for its score, then admits or
declines regardless of what they said:

* for a case of the end-to-end tier it admits, so the model is asked and the
  record holds the outcome with no gate at all: arm A;
* for a case of the gate-only tier it declines, so only the scores are paid
  for.

Because a gate only decides whether the model is called, and never changes
the passages or the prompt, every gated arm is arm A with a mask over these
scores (see ``arms``): one generation serves the six arms.
"""

import dataclasses
import hashlib
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from edgar_rag.answer import Answerer
from edgar_rag.domain import (
    DEFAULT_TOP_K,
    AbstentionReason,
    Answer,
    GateDecision,
    Generator,
    IndexedFiling,
    RelevanceGate,
    Scope,
    ScoredChunk,
)
from edgar_rag.eval.grading import EvalQuestion, grade, split_is_sane
from edgar_rag.eval.records import CaseRecord, Outcome, RepeatRecord, rounded
from edgar_rag.index import CorpusIndex
from edgar_rag.prompt import build_prompt, case_nonce, draw
from edgar_rag.telemetry import StageTimer


@dataclass(frozen=True, slots=True)
class ScoreEvery:
    """Ask ``gate`` for every score, then admit when ``admit`` says so, whatever it decided."""

    gate: RelevanceGate
    admit: bool

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision:
        decision = self.gate.admits(question, passages, filing)
        return GateDecision(
            admitted=self.admit,
            confidence=decision.confidence,
            reason=decision.reason,
            degraded=decision.degraded,
            scores=decision.scores,
        )


@dataclass(slots=True)
class CapturingIndex:
    """The index, keeping the passages of the last search so they can be graded.

    The answerer returns quotes, not the passages it was given, and grading a
    citation needs the full text of the passage cited. One case is asked at a
    time, so the last search is that case's.
    """

    index: CorpusIndex
    last: tuple[ScoredChunk, ...] = field(default=())

    def resolve(self, scope: Scope) -> IndexedFiling | None:
        return self.index.resolve(scope)

    def search(
        self, query: NDArray[np.float32], scope: Scope, top_k: int = DEFAULT_TOP_K
    ) -> tuple[ScoredChunk, ...]:
        self.last = self.index.search(query, scope, top_k=top_k)
        return self.last


@dataclass(frozen=True, slots=True)
class Harness:
    """The answerer under evaluation, with the gates to score and the index to grade from."""

    answerer: Answerer[CapturingIndex]
    gates: RelevanceGate

    def ask(self, question: EvalQuestion, generate: bool) -> CaseRecord:
        """Ask one case, generating only when ``generate`` and the case is in the e2e tier."""
        asks_model = generate and question.e2e
        answerer = dataclasses.replace(
            self.answerer,
            gate=ScoreEvery(self.gates, admit=asks_model),
            nonce=case_nonce(question.id),
        )
        self.answerer.index.last = ()
        stages = StageTimer()
        answer = answerer.ask(question.question, _scope(question), stages=stages)
        passages = self.answerer.index.last
        return _record(question, answer, passages, stages.seconds(), asks_model, self._sane)

    def _sane(self, question: EvalQuestion, answer: Answer) -> bool | None:
        if question.kind != "narrative" or answer.source is None:
            return None
        accession = answer.source.accession
        shard = next(s for s in self.answerer.index.index.shards if s.filing.accession == accession)
        return split_is_sane(shard.chunks, question.expected_item)

    def prompt_for(self, question: EvalQuestion) -> str:
        """The prompt the case is generated from: the same search, the same nonce."""
        answerer = self.answerer
        query = answerer.embedder.embed((question.question,))
        passages = answerer.index.index.search(query, _scope(question), top_k=answerer.top_k)
        return build_prompt(question.question, passages, draw(case_nonce(question.id)))


def _scope(question: EvalQuestion) -> Scope:
    return Scope(cik=question.cik, fiscal_year=question.fiscal_year)


def _outcome(answer: Answer, asked_model: bool) -> Outcome:
    if not answer.abstained:
        return "answered"
    reason = answer.reason
    if not asked_model and reason == AbstentionReason.GATE_REJECTED:
        return "scored_only"
    return "out_of_scope" if reason is None else reason.value


def _record(
    question: EvalQuestion,
    answer: Answer,
    passages: tuple[ScoredChunk, ...],
    stages: dict[str, float],
    asked_model: bool,
    sane: Callable[[EvalQuestion, Answer], bool | None],
) -> CaseRecord:
    graded = grade(question, answer, passages)
    generation = answer.trace.generation
    return CaseRecord(
        id=question.id,
        kind="narrative" if question.kind == "narrative" else "golden",
        split=question.split,
        fold=question.fold,
        ticker=question.ticker,
        cik=question.cik,
        fiscal_year=question.fiscal_year,
        answerable=question.answerable,
        negative_kind=question.negative_kind,
        e2e=question.e2e,
        scores=rounded(answer.trace.gate_scores),
        degraded=answer.degraded,
        retrieval_score=answer.retrieval_score,
        retrieved=tuple(scored.chunk.chunk_id for scored in passages),
        generated=generation is not None,
        outcome=_outcome(answer, asked_model),
        generation=generation.text if generation else None,
        numeric_correct=graded.numeric_correct,
        citation_supported=graded.citation_supported,
        gold_retrieved=graded.gold_retrieved,
        cited_items=graded.cited_items,
        expected_item=question.expected_item,
        item_split_sane=sane(question, answer),
        stages=rounded(stages),
        prompt_tokens=generation.prompt_tokens if generation else None,
        completion_tokens=generation.completion_tokens if generation else None,
        generator_seconds=round(generation.seconds, 4) if generation else None,
    )


def run_cases(
    harness: Harness,
    questions: Iterable[EvalQuestion],
    *,
    generate: bool,
    progress: TextIO = sys.stderr,
) -> Iterator[CaseRecord]:
    """Ask every question in order, yielding each record as soon as it is made."""
    listed = tuple(questions)
    for position, question in enumerate(listed, start=1):
        record = harness.ask(question, generate)
        if record.generated or position % 100 == 0 or position == len(listed):
            seconds = sum(record.stages.values())
            print(
                f"[{position}/{len(listed)}] {record.id}: {record.outcome} ({seconds:.2f} s)",
                file=progress,
                flush=True,
            )
        yield record


def repeat_order(case_id: str) -> str:
    """A fixed, content-free order for choosing which cases to repeat."""
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def repeat_generations(
    harness: Harness,
    questions: Iterable[EvalQuestion],
    records: dict[str, CaseRecord],
    live: Generator,
    count: int,
) -> tuple[RepeatRecord, ...]:
    """Send ``count`` generated cases' prompts straight to ``live`` again and compare the text.

    The first text is the one recorded (it may have come from the tape); the
    second never comes from the tape, so this measures whether the model
    gives the same text for the same prompt. Cases are chosen by
    ``repeat_order``, never by outcome.
    """
    chosen = sorted(
        (q for q in questions if q.id in records and records[q.id].generation is not None),
        key=lambda q: repeat_order(q.id),
    )[:count]
    repeats = []
    for question in chosen:
        first = records[question.id]
        second = live.generate(harness.prompt_for(question))
        repeats.append(
            RepeatRecord(
                id=question.id,
                identical=first.generation == second.text,
                first_seconds=first.generator_seconds or 0.0,
                second_seconds=round(second.seconds, 4),
            )
        )
    return tuple(repeats)
