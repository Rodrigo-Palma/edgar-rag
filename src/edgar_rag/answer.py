"""Answer a question from one filing, with citations, or decline to answer."""

from dataclasses import dataclass
from types import MappingProxyType

from edgar_rag.citations import as_citations, check_claims, markers_in
from edgar_rag.domain import (
    DEFAULT_TOP_K,
    AbstentionReason,
    Answer,
    Embedder,
    GateDecision,
    Generation,
    Generator,
    IndexedFiling,
    RelevanceGate,
    Retriever,
    Scope,
    ScoredChunk,
    Trace,
    abstained_message,
)
from edgar_rag.prompt import NonceSource, build_prompt, draw, random_nonce, refusal_token
from edgar_rag.telemetry import StageTimer


@dataclass(frozen=True, slots=True)
class Answerer[Index: Retriever]:
    """Retrieve, then answer only when the gate admits the passages.

    The collaborators are bound once: the service builds one answerer at
    startup, so the index is read and the clients are opened once rather than
    on every request, and the evaluation asks the same object the service does.

    Abstaining before generation is deliberate: a model asked to answer from
    passages it was never given will invent one, and that failure is invisible
    to the caller. The gate is a collaborator rather than a threshold because
    how relevance is judged is the part of this pipeline most worth replacing.

    The model declines only by replying with the refusal token for this
    request and nothing else. Anything short of exact equality, including the
    old fixed phrase a filing could quote, goes through the citation check.

    Every question is answered from the one filing its ``Scope`` names. A
    scope naming no indexed filing abstains with ``out_of_scope`` before
    anything is embedded: answering from the closest filing instead would
    cite another company's numbers, or another year's, as the answer.
    """

    index: Index
    embedder: Embedder
    generator: Generator
    gate: RelevanceGate
    nonce: NonceSource = random_nonce
    top_k: int = DEFAULT_TOP_K

    def ask(
        self,
        question: str,
        scope: Scope,
        top_k: int | None = None,
        stages: StageTimer | None = None,
    ) -> Answer:
        """Answer ``question`` from the ``top_k`` closest passages of the scoped filing.

        ``stages``, when given, records the time spent embedding, searching,
        gating and generating; a request the gate stopped has no generate stage,
        and one outside the indexed scope has none at all.

        Raises:
            ValueError: when the question is empty, ``top_k`` is below one, or
                the nonce source returns a value that could break the delimiter.
            GateError: when the gate cannot reach the service it depends on.
        """
        if not question.strip():
            raise ValueError("question must not be empty")
        wanted = self.top_k if top_k is None else top_k
        if wanted < 1:
            raise ValueError("top_k must be positive")

        timer = stages if stages is not None else StageTimer()
        source = self.index.resolve(scope)
        if source is None:
            return _out_of_scope(question, scope, _trace(timer))
        passages = self._retrieve(question, scope, wanted, timer)
        best_score = round(passages[0].score if passages else 0.0, 4)
        with timer.measure("gate"):
            decision = self.gate.admits(question, passages)
        judged = _Judged(question, source, decision, best_score)
        if not decision.admitted:
            return judged.abstain(AbstentionReason.GATE_REJECTED, _trace(timer))
        return self._generate(judged, passages, timer)

    def _retrieve(
        self, question: str, scope: Scope, top_k: int, timer: StageTimer
    ) -> tuple[ScoredChunk, ...]:
        with timer.measure("embed"):
            query = self.embedder.embed((question,))
        with timer.measure("search"):
            return self.index.search(query, scope, top_k=top_k)

    def _generate(
        self, judged: "_Judged", passages: tuple[ScoredChunk, ...], timer: StageTimer
    ) -> Answer:
        """Ask the model, then keep its answer only if it cites what it was given."""
        drawn = draw(self.nonce)
        prompt = build_prompt(judged.question, passages, drawn)
        with timer.measure("generate"):
            generation = self.generator.generate(prompt)
        trace = _trace(timer, generation)
        generated = generation.text
        if generated.strip() == refusal_token(drawn):
            return judged.abstain(AbstentionReason.MODEL_DECLINED, trace)

        cited = markers_in(generated, len(passages))
        if not cited:
            # An answer that points at nothing cannot be checked, and the prompt
            # asked for a marker on every factual sentence. Treating it as an
            # answer would hand the caller exactly the unverifiable output this
            # service exists to avoid.
            return judged.abstain(AbstentionReason.NO_VALID_CITATION, trace)

        failure = check_claims(generated, passages)
        if failure is not None:
            # One marker used to approve the whole answer. Each sentence that
            # states something now needs its own, and its figures have to be
            # in the passage it points at, or the answer is withheld.
            return judged.abstain(failure.reason, trace, finding=failure.finding)

        decision = judged.decision
        return Answer(
            question=judged.question,
            text=generated,
            citations=as_citations(passages, judged.question, cited),
            abstained=False,
            reason=None,
            detail=decision.reason,
            retrieval_score=judged.retrieval_score,
            gate_score=decision.confidence,
            degraded=decision.degraded,
            source=judged.source,
            trace=trace,
        )


@dataclass(frozen=True, slots=True)
class _Judged:
    """A question the gate has ruled on, and the filing it was searched in."""

    question: str
    source: IndexedFiling
    decision: GateDecision
    retrieval_score: float

    def abstain(self, reason: AbstentionReason, trace: Trace, finding: str | None = None) -> Answer:
        """Withhold the answer, keeping what the gate said even when it admitted.

        A model refusal after a degraded gate is still a degraded request, so
        the gate's score and flag travel with every abstention, not only its
        own. ``finding`` names what the citation check found, so a caller can
        see which sentence failed without receiving the withheld text.
        """
        decision = self.decision
        because = decision.reason if finding is None else f"{finding}; {decision.reason}"
        return Answer(
            question=self.question,
            text=None,
            citations=(),
            abstained=True,
            reason=reason,
            detail=f"{abstained_message(reason)} ({because})",
            retrieval_score=self.retrieval_score,
            gate_score=decision.confidence,
            degraded=decision.degraded,
            source=self.source,
            trace=trace,
        )


def _out_of_scope(question: str, scope: Scope, trace: Trace) -> Answer:
    """Abstain without a search: no gate ran, so there is no score to report."""
    reason = AbstentionReason.OUT_OF_SCOPE
    return Answer(
        question=question,
        text=None,
        citations=(),
        abstained=True,
        reason=reason,
        detail=f"{abstained_message(reason)} (no filing indexed for {scope})",
        retrieval_score=0.0,
        gate_score=0.0,
        degraded=False,
        source=None,
        trace=trace,
    )


def _trace(timer: StageTimer, generation: Generation | None = None) -> Trace:
    return Trace(stages=MappingProxyType(timer.seconds()), generation=generation)
