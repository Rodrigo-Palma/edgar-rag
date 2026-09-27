"""Deciding whether the retrieved passages are worth answering from.

The gate is separate from the answering because it is the part most worth
swapping. The cheap version compares the question to the passage with cosine
similarity, which is free but only measures that the words are nearby. The other
version asks a model trained for the question, which costs a call and reports a
calibrated confidence.
"""

from dataclasses import dataclass
from typing import Protocol

import httpx

from edgar_rag.index import ScoredChunk

REQUEST_TIMEOUT_SECONDS = 30.0


class GateError(RuntimeError):
    """Raised when a gate cannot reach the service it depends on."""


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether to answer, how sure the gate is, and why.

    ``reason`` is written for a person reading a log, because an abstention with
    no reason is indistinguishable from a bug.
    """

    admitted: bool
    confidence: float
    reason: str


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
            fell_back = self.fallback.admits(question, passages)
            return GateDecision(
                admitted=fell_back.admitted,
                confidence=fell_back.confidence,
                reason=f"{fell_back.reason} (fell back: {error})",
            )

    def _judge(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision:
        best_confidence, best_position = 0.0, 0
        for position, scored in enumerate(passages, start=1):
            confidence = self._confidence(question, scored.chunk.text)
            if confidence > best_confidence:
                best_confidence, best_position = confidence, position
            if confidence >= self.min_confidence:
                return GateDecision(
                    admitted=True,
                    confidence=round(confidence, 4),
                    reason=(
                        f"passage {position} answers the question with confidence {confidence:.3f}"
                    ),
                )

        return GateDecision(
            admitted=False,
            confidence=round(best_confidence, 4),
            reason=(
                f"no passage cleared {self.min_confidence}; the closest was "
                f"passage {best_position} at {best_confidence:.3f}"
            ),
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
            response = httpx.post(
                f"{self.url.rstrip('/')}/decide", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
            answer = response.raise_for_status().json()["answers"][0]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            raise GateError(f"{self.url} did not answer: {error}") from error
        return float(answer["probabilities"][1])
