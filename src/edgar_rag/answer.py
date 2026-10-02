"""Answer a question from a filing, with citations, or decline to answer."""

from dataclasses import dataclass

from edgar_rag.citations import as_citations, markers_in
from edgar_rag.domain import (
    AbstentionReason,
    Answer,
    Embedder,
    GateDecision,
    Generator,
    RelevanceGate,
    abstained_message,
)
from edgar_rag.index import FilingIndex
from edgar_rag.prompt import NonceSource, build_prompt, draw, random_nonce, refusal_token
from edgar_rag.telemetry import StageTimer


def answer_question(
    question: str,
    index: FilingIndex,
    embedder: Embedder,
    generator: Generator,
    gate: RelevanceGate,
    top_k: int = 4,
    nonce: NonceSource = random_nonce,
    stages: StageTimer | None = None,
) -> Answer:
    """Retrieve, then answer only when the gate admits the passages.

    Abstaining before generation is deliberate: a model asked to answer from
    passages it was never given will invent one, and that failure is invisible
    to the caller. The gate is an argument rather than a threshold because how
    relevance is judged is the part of this pipeline most worth replacing.

    The model declines only by replying with the refusal token for this
    request and nothing else. Anything short of exact equality, including the
    old fixed phrase a filing could quote, goes through the citation check.

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
    with timer.measure("embed"):
        query = embedder.embed((question,))
    with timer.measure("search"):
        passages = index.search(query, top_k=top_k)
    best_score = round(passages[0].score if passages else 0.0, 4)
    with timer.measure("gate"):
        decision = gate.admits(question, passages)

    def abstain(reason: AbstentionReason) -> Answer:
        return _abstention(question, reason, decision, best_score)

    if not decision.admitted:
        return abstain(AbstentionReason.GATE_REJECTED)

    drawn = draw(nonce)
    prompt = build_prompt(question, passages, drawn)
    with timer.measure("generate"):
        generated = generator.generate(prompt)
    if generated.strip() == refusal_token(drawn):
        return abstain(AbstentionReason.MODEL_DECLINED)

    cited = markers_in(generated, len(passages))
    if not cited:
        # An answer that points at nothing cannot be checked, and the prompt
        # asked for a marker on every factual sentence. Treating it as an
        # answer would hand the caller exactly the unverifiable output this
        # service exists to avoid.
        return abstain(AbstentionReason.NO_VALID_CITATION)

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


@dataclass(frozen=True, slots=True)
class Answerer:
    """The answering pipeline with its collaborators bound once.

    The service builds one at startup, so the index is read and the clients
    are opened once rather than on every request.
    """

    index: FilingIndex
    embedder: Embedder
    generator: Generator
    gate: RelevanceGate
    nonce: NonceSource = random_nonce

    @property
    def source(self) -> dict[str, str]:
        """The filing the answers come from."""
        return self.index.source

    def ask(self, question: str, top_k: int = 4, stages: StageTimer | None = None) -> Answer:
        """Answer from the bound index; see ``answer_question`` for what can raise."""
        return answer_question(
            question=question,
            index=self.index,
            embedder=self.embedder,
            generator=self.generator,
            gate=self.gate,
            top_k=top_k,
            nonce=self.nonce,
            stages=stages,
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
