"""Deciding whether the retrieved passages are worth answering from.

The gate is separate from the answering because it is the part most worth
swapping. The cheap version compares the question to the passage with cosine
similarity, which is free but only measures that the words are nearby. The other
version asks a model trained for the question, which costs a call and reports a
calibrated confidence.
"""

import logging
from dataclasses import dataclass, field
from typing import Annotated, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from edgar_rag.index import ScoredChunk

REQUEST_TIMEOUT_SECONDS = 30.0
# What a client sees when the brier service failed. The URL and the exception
# go to the log only: the reason is returned to whoever asked the question.
UNAVAILABLE_REASON = "relevance model unavailable"

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
class GateDecision:
    """Whether to answer, how sure the gate is, and why.

    ``reason`` is written for a person reading a log, because an abstention with
    no reason is indistinguishable from a bug. ``degraded`` is a field rather
    than prose in the reason so a caller can act on it: a gate that quietly
    swapped itself for a weaker one is the failure most worth surfacing.
    """

    admitted: bool
    confidence: float
    reason: str
    degraded: bool = False


class RelevanceGate(Protocol):
    def admits(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision: ...


@dataclass(frozen=True, slots=True)
class CosineGate:
    """Admit when the closest passage clears a similarity threshold.

    Free, and it fails in a specific way: a passage that shares vocabulary with
    the question scores well whether or not it answers it.
    """

    min_score: float

    def admits(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision:
        best = passages[0].score if passages else 0.0
        if best < self.min_score:
            return GateDecision(
                admitted=False,
                confidence=round(best, 4),
                reason=f"best passage scored {best:.3f}, below the {self.min_score} threshold",
            )
        return GateDecision(
            admitted=True,
            confidence=round(best, 4),
            reason=f"best passage scored {best:.3f}",
        )


@dataclass(frozen=True, slots=True)
class BrierGate:
    """Ask a calibrated decision model whether a passage answers the question.

    Each candidate passage is asked about separately and the loop stops at the
    first one that clears the threshold, so the common case costs one call. The
    passages are ordered by retrieval score, so the first one asked about is the
    one most likely to end the loop.
    """

    url: str
    min_confidence: float = 0.7
    fallback: RelevanceGate | None = None
    # The service's pooled client; a connection per call without one. It is
    # plumbing, so two gates asking the same URL the same way are equal.
    client: httpx.Client | None = field(default=None, compare=False, repr=False)

    def admits(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision:
        """Judge the passages, falling back only if a fallback was given.

        Raises:
            GateError: when the service is unreachable and no fallback was set.
        """
        if not passages:
            return GateDecision(False, 0.0, "retrieval returned nothing to judge")

        try:
            return self._judge(question, passages)
        except GateError as error:
            if self.fallback is None:
                raise
            logger.warning("brier gate failed, using the fallback: %s", error)
            fell_back = self.fallback.admits(question, passages)
            return GateDecision(
                admitted=fell_back.admitted,
                confidence=fell_back.confidence,
                reason=f"{fell_back.reason} (degraded: {UNAVAILABLE_REASON}, fallback used)",
                degraded=True,
            )

    def _judge(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision:
        """Ask about each passage, stopping at the first that clears the bar.

        Evidence already gathered is kept if a later call fails, so one bad
        request does not throw away a confident yes from an earlier passage.
        A passage that could not be judged marks the decision ``degraded``,
        because it rests on part of the evidence. A judged confidence of 0.0
        counts as judged: it is an answer, not the absence of one.

        Raises:
            GateError: when no passage could be judged at all.
        """
        best_confidence, best_position = 0.0, 1
        judged, failed = 0, 0
        last_failure: GateError | None = None

        for position, scored in enumerate(passages, start=1):
            try:
                confidence = self._confidence(question, scored.chunk.text)
            except GateError as error:
                logger.warning("brier could not judge passage %d: %s", position, error)
                failed, last_failure = failed + 1, error
                continue
            judged += 1
            if confidence > best_confidence:
                best_confidence, best_position = confidence, position
            if confidence >= self.min_confidence:
                return GateDecision(
                    admitted=True,
                    confidence=round(confidence, 4),
                    reason=(
                        f"passage {position} answers the question with confidence {confidence:.3f}"
                        + _unjudged_note(failed, judged + failed)
                    ),
                    degraded=failed > 0,
                )

        if judged == 0 and last_failure is not None:
            raise last_failure

        return GateDecision(
            admitted=False,
            confidence=round(best_confidence, 4),
            reason=(
                f"no passage cleared {self.min_confidence}; the closest was "
                f"passage {best_position} at {best_confidence:.3f}"
                + _unjudged_note(failed, judged + failed)
            ),
            degraded=failed > 0,
        )

    def confidence_for(self, question: str, passage: str) -> float:
        """The raw probability for one passage, for sweeping the threshold.

        Raises:
            GateError: when the service is unreachable.
        """
        return self._confidence(question, passage)

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
            post = self.client.post if self.client is not None else httpx.post
            response = post(
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
