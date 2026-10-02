"""Deciding whether the retrieved passages are worth answering from.

The gate is separate from the answering because it is the part most worth
swapping. The cheap version compares the question to the passage with cosine
similarity, which is free but only measures that the words are nearby. The other
version asks a model trained for the question, which costs a call and reports a
calibrated confidence.
"""

import logging
from dataclasses import dataclass, field
from typing import Annotated

import httpx
from pydantic import BaseModel, Field, ValidationError

from edgar_rag.domain import GateDecision, IndexedFiling, RelevanceGate, ScoredChunk

REQUEST_TIMEOUT_SECONDS = 30.0
# What a client sees when the brier service failed. The URL and the exception
# go to the log only: the reason is returned to whoever asked the question.
UNAVAILABLE_REASON = "relevance model unavailable"

# The names each gate files its score under in ``GateDecision.scores``.
COSINE = "cosine"
BRIER = "brier"

logger = logging.getLogger(__name__)

Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)]


class _BrierAnswer(BaseModel):
    # One probability per option, in the order sent: ["no", "yes"]
    probabilities: Annotated[list[Probability], Field(min_length=2, max_length=2)]


class _BrierReply(BaseModel):
    answers: Annotated[list[_BrierAnswer], Field(min_length=1)]


class GateError(RuntimeError):
    """Raised when a gate cannot reach the service it depends on."""


@dataclass(frozen=True, slots=True)
class NoGate:
    """Admit every question: the model's own refusal is the only check left.

    The baseline every gate has to beat, and the setting for an operator who
    would rather spend the generation than risk a wrong refusal.
    """

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision:
        return GateDecision(admitted=True, confidence=1.0, reason="no gate configured")


@dataclass(frozen=True, slots=True, init=False)
class AllOf:
    """Admit only when every gate admits, and ask every one of them.

    Every gate is asked even after one has declined, so the decision carries
    the score of each, which is what choosing a threshold afterwards needs.
    The cost is that a question the first gate declines still pays for the
    others.

    The decision's confidence is the last gate's: put the rules first and the
    gate whose score ranks questions last, as in ``AllOf(PeriodGuard(),
    CosineGate(0.55))``. A rejection gives the reason of the first gate that
    declined, so a period rejection reads ``out_of_period`` whatever the
    relevance gate thought; its prose lists every gate that declined. The
    decision is degraded when any gate's was.

    Raises:
        ValueError: when no gate is given.
    """

    gates: tuple[RelevanceGate, ...]

    def __init__(self, *gates: RelevanceGate) -> None:
        if not gates:
            raise ValueError("AllOf needs at least one gate")
        object.__setattr__(self, "gates", gates)

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision:
        decisions = tuple(gate.admits(question, passages, filing) for gate in self.gates)
        scores = {name: score for decision in decisions for name, score in decision.scores.items()}
        confidence = decisions[-1].confidence
        degraded = any(decision.degraded for decision in decisions)
        declined = [decision for decision in decisions if not decision.admitted]
        if declined:
            return GateDecision(
                admitted=False,
                confidence=confidence,
                reason="; ".join(decision.reason for decision in declined),
                degraded=degraded,
                rejection=declined[0].rejection,
                scores=scores,
            )
        return GateDecision(
            admitted=True,
            confidence=confidence,
            reason="; ".join(decision.reason for decision in decisions),
            degraded=degraded,
            scores=scores,
        )


@dataclass(frozen=True, slots=True)
class CosineGate:
    """Admit when the closest passage clears a similarity threshold.

    Free, and it fails in a specific way: a passage that shares vocabulary with
    the question scores well whether or not it answers it.
    """

    min_score: float

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision:
        best = passages[0].score if passages else 0.0
        scores = {COSINE: round(best, 4)}
        if best < self.min_score:
            return GateDecision(
                admitted=False,
                confidence=round(best, 4),
                reason=f"best passage scored {best:.3f}, below the {self.min_score} threshold",
                scores=scores,
            )
        return GateDecision(
            admitted=True,
            confidence=round(best, 4),
            reason=f"best passage scored {best:.3f}",
            scores=scores,
        )


@dataclass(frozen=True, slots=True)
class BrierGate:
    """Ask a calibrated decision model whether a passage answers the question.

    Each retrieved passage is asked about separately, every one of them, and
    the decision rests on the most confident answer. Stopping at the first
    passage over the bar would save calls, but then the score would depend on
    the threshold: a 0.7 gate would report 0.95 for a question a 0.96 gate
    scores 0.99, and a threshold chosen afterwards from the reported scores
    would not reproduce what the gate decides. A question costs one call per
    passage, four by default.

    ``client`` is required: the service passes the pooled client its lifespan
    opened, and a test passes one over a mock transport. Every reply is
    validated before it is read, and one that is malformed counts as the
    service being unavailable.
    """

    url: str
    # Plumbing, so two gates asking the same URL the same way are equal.
    client: httpx.Client = field(compare=False, repr=False)
    min_confidence: float = 0.7
    fallback: RelevanceGate | None = None

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision:
        """Judge the passages, falling back only if a fallback was given.

        Raises:
            GateError: when the service is unreachable and no fallback was set.
        """
        if not passages:
            return GateDecision(
                False, 0.0, "retrieval returned nothing to judge", scores={BRIER: 0.0}
            )

        try:
            return self._judge(question, passages)
        except GateError as error:
            if self.fallback is None:
                raise
            logger.warning("brier gate failed, using the fallback: %s", error)
            fell_back = self.fallback.admits(question, passages, filing)
            return GateDecision(
                admitted=fell_back.admitted,
                confidence=fell_back.confidence,
                reason=f"{fell_back.reason} (degraded: {UNAVAILABLE_REASON}, fallback used)",
                degraded=True,
                rejection=fell_back.rejection,
                scores=fell_back.scores,
            )

    def _judge(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision:
        """Ask about every passage and decide on the most confident answer.

        A passage that could not be judged marks the decision ``degraded``,
        because it rests on part of the evidence, and the others still count.
        A judged confidence of 0.0 counts as judged: it is an answer, not the
        absence of one.

        Raises:
            GateError: when no passage could be judged at all.
        """
        judged: dict[int, float] = {}
        last_failure: GateError | None = None
        for position, scored in enumerate(passages, start=1):
            try:
                judged[position] = self._confidence(question, scored.chunk.text)
            except GateError as error:
                logger.warning("brier could not judge passage %d: %s", position, error)
                last_failure = error
        if not judged:
            raise last_failure or GateError("no passage could be judged")

        best_position = max(judged, key=judged.__getitem__)
        best = judged[best_position]
        failed = len(passages) - len(judged)
        unjudged = _unjudged_note(failed, len(passages))
        if best >= self.min_confidence:
            said = f"passage {best_position} answers the question with confidence {best:.3f}"
        else:
            said = (
                f"no passage cleared {self.min_confidence}; the closest was "
                f"passage {best_position} at {best:.3f}"
            )
        return GateDecision(
            admitted=best >= self.min_confidence,
            confidence=round(best, 4),
            reason=said + unjudged,
            degraded=failed > 0,
            scores={BRIER: round(best, 4)},
        )

    def _confidence(self, question: str, passage: str) -> float:
        """The probability the model puts on "yes"."""
        payload = {
            "state": passage,
            "questions": [
                {
                    "name": "relevance",
                    "kind": "bool",
                    "prompt": f"Does the passage answer this question: {question}",
                    "options": ["no", "yes"],
                }
            ],
        }
        try:
            response = self.client.post(
                f"{self.url.rstrip('/')}/decide", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
            reply = _BrierReply.model_validate_json(response.raise_for_status().content)
        except (httpx.HTTPError, ValidationError) as error:
            raise GateError(f"{self.url} did not answer: {error}") from error
        return reply.answers[0].probabilities[1]


def _unjudged_note(failed: int, asked: int) -> str:
    """Say how much of the evidence is missing, without saying why."""
    if not failed:
        return ""
    return f"; {failed} of {asked} passages could not be judged ({UNAVAILABLE_REASON})"
