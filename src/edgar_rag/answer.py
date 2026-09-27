"""Answer a question from a filing, with citations, or decline to answer."""

from dataclasses import dataclass

from edgar_rag.embeddings import Embedder, Generator
from edgar_rag.gate import RelevanceGate
from edgar_rag.index import FilingIndex, ScoredChunk

PROMPT = """You answer questions about a company's SEC filing.

Use only the passages below. Every sentence that states a fact must end with the
passage number it came from, like [1]. If the passages do not answer the
question, reply exactly: NOT IN THE FILING.

Passages:
{passages}

Question: {question}
Answer:"""

NO_ANSWER = "NOT IN THE FILING"
ABSTAINED_MESSAGE = (
    "I did not find passages close enough to the question in this filing, so I am not answering."
)


@dataclass(frozen=True, slots=True)
class Citation:
    marker: int
    item: str
    title: str
    quote: str
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    """``reason`` says why, which matters most when ``abstained`` is set."""

    question: str
    text: str
    citations: tuple[Citation, ...]
    retrieval_score: float
    abstained: bool
    reason: str = ""


def _quote(text: str, limit: int = 240) -> str:
    return text[:limit].strip() + ("..." if len(text) > limit else "")


def _as_citations(passages: tuple[ScoredChunk, ...]) -> tuple[Citation, ...]:
    return tuple(
        Citation(
            marker=position,
            item=scored.chunk.item,
            title=scored.chunk.title,
            quote=_quote(scored.chunk.text),
            score=round(scored.score, 4),
        )
        for position, scored in enumerate(passages, start=1)
    )


def build_prompt(question: str, passages: tuple[ScoredChunk, ...]) -> str:
    numbered = "\n\n".join(
        f"[{position}] ({scored.chunk.item}) {scored.chunk.text}"
        for position, scored in enumerate(passages, start=1)
    )
    return PROMPT.format(passages=numbered, question=question)


def answer_question(
    question: str,
    index: FilingIndex,
    embedder: Embedder,
    generator: Generator,
    gate: RelevanceGate,
    top_k: int = 4,
) -> Answer:
    """Retrieve, then answer only when the gate admits the passages.

    Abstaining before generation is deliberate: a model asked to answer from
    passages it was never given will invent one, and that failure is invisible
    to the caller. The gate is an argument rather than a threshold because how
    relevance is judged is the part of this pipeline most worth replacing.

    Raises:
        ValueError: when the question is empty.
        GateError: when the gate cannot reach the service it depends on.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    query = embedder.embed((question,))
    passages = index.search(query, top_k=top_k)
    best_score = passages[0].score if passages else 0.0
    decision = gate.admits(question, passages)

    if not decision.admitted:
        return Answer(
            question=question,
            text=ABSTAINED_MESSAGE,
            citations=(),
            retrieval_score=round(best_score, 4),
            abstained=True,
            reason=decision.reason,
        )

    generated = generator.generate(build_prompt(question, passages))
    declined = generated.strip().upper().startswith(NO_ANSWER)
    return Answer(
        question=question,
        text=ABSTAINED_MESSAGE if declined else generated,
        citations=() if declined else _as_citations(passages),
        retrieval_score=round(best_score, 4),
        abstained=declined,
        reason="the model said the filing does not answer it" if declined else decision.reason,
    )
