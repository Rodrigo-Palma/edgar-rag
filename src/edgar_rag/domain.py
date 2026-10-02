"""What the pipeline talks about, and the ports it talks through.

Everything here is a value or a ``Protocol``, and this module imports nothing
from the package. The answering core depends on these and on nothing that
does I/O; the adapters (the index, the Ollama models, the gates) implement
the ports, and only the entry points choose which adapter is wired in.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_TOP_K = 4
"""Passages retrieved per question unless a caller asks for another number."""


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    item: str
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


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


class AbstentionReason(StrEnum):
    """Why an answer was withheld. The values are part of the HTTP contract.

    Which check fired matters to a caller as much as the fact that one did: a
    gate rejection means the filing had nothing close, while a model refusal
    means it had something close that did not hold the answer. Free prose
    could not be acted on without parsing it, so the reason is a closed set
    and the prose lives in ``Answer.detail``.

    ``out_of_period``, ``unsupported_claim`` and ``out_of_scope`` belong to
    checks that are not in the pipeline yet (the period guard, the citation
    support check and multi-filing scope). They are published now so adding
    those checks does not change the contract.
    """

    GATE_REJECTED = "gate_rejected"
    OUT_OF_PERIOD = "out_of_period"
    MODEL_DECLINED = "model_declined"
    NO_VALID_CITATION = "no_valid_citation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    OUT_OF_SCOPE = "out_of_scope"


# One sentence per reason, because a single "no passages close enough" message
# contradicted the reason whenever the model or the citation check declined.
ABSTAINED_MESSAGES: Mapping[AbstentionReason, str] = MappingProxyType(
    {
        AbstentionReason.GATE_REJECTED: (
            "No passage in this filing is close enough to the question, so the model was not asked."
        ),
        AbstentionReason.OUT_OF_PERIOD: (
            "The question asks about a period this filing does not cover, "
            "so the model was not asked."
        ),
        AbstentionReason.MODEL_DECLINED: (
            "The model read the closest passages and found no answer in them."
        ),
        AbstentionReason.NO_VALID_CITATION: (
            "The answer cited no retrieved passage, so it could not be checked and is withheld."
        ),
        AbstentionReason.UNSUPPORTED_CLAIM: (
            "The answer makes a claim its cited passage does not support, so it is withheld."
        ),
        AbstentionReason.OUT_OF_SCOPE: (
            "The question is about a filing outside the indexed scope, so the model was not asked."
        ),
    }
)


def abstained_message(reason: AbstentionReason) -> str:
    return ABSTAINED_MESSAGES[reason]


@dataclass(frozen=True, slots=True)
class Citation:
    marker: int
    item: str
    title: str
    quote: str
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    """An answer with its citations, or the reason there is none.

    ``text`` is ``None`` exactly when ``abstained`` is set, and then ``reason``
    says which check withheld it. ``detail`` is the same story in words, for a
    person: the message for the reason followed by what the gate found.
    ``gate_score`` is the gate's own confidence, which is cosine similarity or
    a probability depending on the gate, and ``degraded`` says the gate ran on
    part of its evidence or on its fallback, which a caller must be able to
    tell apart from a model decision.
    """

    question: str
    text: str | None
    citations: tuple[Citation, ...]
    abstained: bool
    reason: AbstentionReason | None
    detail: str
    retrieval_score: float
    gate_score: float
    degraded: bool


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class RelevanceGate(Protocol):
    def admits(self, question: str, passages: tuple[ScoredChunk, ...]) -> GateDecision: ...


class Retriever(Protocol):
    """Passages close to a query vector."""

    @property
    def source(self) -> dict[str, str]:
        """The filing the passages come from."""
        ...

    def search(
        self, query: NDArray[np.float32], top_k: int = DEFAULT_TOP_K
    ) -> tuple[ScoredChunk, ...]: ...
