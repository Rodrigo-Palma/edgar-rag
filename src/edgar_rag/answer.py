"""Answer a question from a filing, with citations, or decline to answer."""

from dataclasses import dataclass

from edgar_rag.citations import as_citations, markers_in
from edgar_rag.domain import (
    DEFAULT_TOP_K,
    AbstentionReason,
    Answer,
    Embedder,
    GateDecision,
    Generator,
    RelevanceGate,
    Retriever,
    ScoredChunk,
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
    """

    index: Index
    embedder: Embedder
    generator: Generator
    gate: RelevanceGate
    nonce: NonceSource = random_nonce
    top_k: int = DEFAULT_TOP_K

    @property
    def source(self) -> dict[str, str]:
        """The filing the answers come from."""
        return self.index.source

    def ask(
        self, question: str, top_k: int | None = None, stages: StageTimer | None = None
    ) -> Answer:
        """Answer ``question`` from the ``top_k`` closest passages, or abstain.

        ``stages``, when given, records the time spent embedding, searching,
        gating and generating; a request the gate stopped has no generate stage.

        Raises:
            ValueError: when the question is empty, or the nonce source returns a
                value that could break the delimiter.
            GateError: when the gate cannot reach the service it depends on.
        """
        if not question.strip():
            raise ValueError("question must not be empty")

        timer = stages if stages is not None else StageTimer()
        passages = self._retrieve(question, self.top_k if top_k is None else top_k, timer)
        best_score = round(passages[0].score if passages else 0.0, 4)
        with timer.measure("gate"):
            decision = self.gate.admits(question, passages)
        if not decision.admitted:
            return _abstention(question, AbstentionReason.GATE_REJECTED, decision, best_score)
        return self._generate(question, passages, decision, best_score, timer)

    def _retrieve(self, question: str, top_k: int, timer: StageTimer) -> tuple[ScoredChunk, ...]:
        with timer.measure("embed"):
            query = self.embedder.embed((question,))
        with timer.measure("search"):
            return self.index.search(query, top_k=top_k)

    def _generate(
        self,
        question: str,
        passages: tuple[ScoredChunk, ...],
        decision: GateDecision,
        best_score: float,
        timer: StageTimer,
    ) -> Answer:
        """Ask the model, then keep its answer only if it cites what it was given."""
        drawn = draw(self.nonce)
        prompt = build_prompt(question, passages, drawn)
        with timer.measure("generate"):
            generated = self.generator.generate(prompt).text
        if generated.strip() == refusal_token(drawn):
            return _abstention(question, AbstentionReason.MODEL_DECLINED, decision, best_score)

        cited = markers_in(generated, len(passages))
        if not cited:
            # An answer that points at nothing cannot be checked, and the prompt
            # asked for a marker on every factual sentence. Treating it as an
            # answer would hand the caller exactly the unverifiable output this
            # service exists to avoid.
            return _abstention(question, AbstentionReason.NO_VALID_CITATION, decision, best_score)

        return Answer(
            question=question,
            text=generated,
            citations=as_citations(passages, question, cited),
            abstained=False,
            reason=None,
            detail=decision.reason,
            retrieval_score=best_score,
            gate_score=decision.confidence,
            degraded=decision.degraded,
        )


def _abstention(
    question: str, reason: AbstentionReason, decision: GateDecision, retrieval_score: float
) -> Answer:
    """Withhold the answer, keeping what the gate said even when it admitted.

    A model refusal after a degraded gate is still a degraded request, so the
    gate's score and flag travel with every abstention, not only its own.
    """
    return Answer(
        question=question,
        text=None,
        citations=(),
        abstained=True,
        reason=reason,
        detail=f"{abstained_message(reason)} ({decision.reason})",
        retrieval_score=retrieval_score,
        gate_score=decision.confidence,
        degraded=decision.degraded,
    )
