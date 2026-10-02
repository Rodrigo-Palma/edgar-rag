"""Deciding whether the retrieved passages are worth answering from.

The gate is separate from the answering because it is the part most worth
swapping. The cheap version compares the question to the passage with cosine
similarity, which is free but only measures that the words are nearby. A
calibrated relevance model was measured against it on the headline run and
removed for ranking worse (ADR-0014).
"""

from dataclasses import dataclass

from edgar_rag.domain import GateDecision, IndexedFiling, RelevanceGate, ScoredChunk

# The name the cosine gate files its score under in ``GateDecision.scores``.
COSINE = "cosine"


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
