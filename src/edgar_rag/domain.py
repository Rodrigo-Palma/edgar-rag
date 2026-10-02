"""What the pipeline talks about, and the ports it talks through.

Everything here is a value or a ``Protocol``, and this module imports nothing
from the package. The answering core depends on these and on nothing that
does I/O; the adapters (the index, the Ollama models, the gates) implement
the ports, and only the entry points choose which adapter is wired in.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
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
class Scope:
    """Which filing a question is about: a company and, optionally, a fiscal year.

    Every question is answered from one filing. Without a ``fiscal_year`` it
    is the latest year indexed for ``cik``. Working out the company or the year
    from the question itself would be entity linking, a different problem with
    its own failures, so the caller states them.

    Raises:
        ValueError: for a CIK or a fiscal year that is not a positive integer.
    """

    cik: int
    fiscal_year: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.cik, bool) or not isinstance(self.cik, int) or self.cik < 1:
            raise ValueError(f"{self.cik!r} is not a CIK")
        year = self.fiscal_year
        if year is not None and (isinstance(year, bool) or not isinstance(year, int) or year < 1):
            raise ValueError(f"{year!r} is not a fiscal year")

    def __str__(self) -> str:
        if self.fiscal_year is None:
            return f"CIK {self.cik}, the latest fiscal year"
        return f"CIK {self.cik}, fiscal {self.fiscal_year}"


@dataclass(frozen=True, slots=True)
class IndexedFiling:
    """A filing in the index: what an answer names as its source.

    ``fiscal_year`` is the calendar year the reporting period ends in, the
    convention the golden set's filings are pinned by.
    """

    accession: str
    cik: int
    fiscal_year: int
    form: str
    company: str
    filing_date: date
    url: str


@dataclass(frozen=True, slots=True)
class EmbedderSpec:
    """How passages and questions are turned into vectors.

    Both sides of a cosine have to come from the same model with the same
    preprocessing: vectors from another model, or from text that was not
    lower-cased the same way, still load and search, and return passages that
    look plausible and are not the closest.
    """

    model: str
    lowercase: bool


@dataclass(frozen=True, slots=True)
class EmbeddingFingerprint:
    """The ``EmbedderSpec`` an index was built with, and the size of its vectors."""

    model: str
    lowercase: bool
    dimensions: int

    @property
    def spec(self) -> EmbedderSpec:
        return EmbedderSpec(model=self.model, lowercase=self.lowercase)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class AbstentionReason(StrEnum):
    """Why an answer was withheld. The values are part of the HTTP contract.

    Which check fired matters to a caller as much as the fact that one did: a
    gate rejection means the filing had nothing close, while a model refusal
    means it had something close that did not hold the answer. Free prose
    could not be acted on without parsing it, so the reason is a closed set
    and the prose lives in ``Answer.detail``.

    ``out_of_scope`` means no indexed filing matches the requested company
    and fiscal year. ``out_of_period`` means the question names a year the
    filing does not report on, which the period guard catches before any
    relevance score is trusted.
    """

    GATE_REJECTED = "gate_rejected"
    OUT_OF_PERIOD = "out_of_period"
    MODEL_DECLINED = "model_declined"
    NO_VALID_CITATION = "no_valid_citation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether to answer, how sure the gate is, and why.

    ``reason`` is written for a person reading a log, because an abstention with
    no reason is indistinguishable from a bug. ``degraded`` is a field rather
    than prose in the reason so a caller can act on it: a gate that quietly
    swapped itself for a weaker one is the failure most worth surfacing.

    ``rejection`` is the abstention reason the answer carries when the gate
    does not admit: a relevance gate leaves it at ``gate_rejected``, the
    period guard says ``out_of_period``. ``scores`` holds every score that
    went into the decision, by gate name, including those of gates that did
    not decide it: a threshold is chosen afterwards from scores, so each one
    has to be there whatever the decision was.
    """

    admitted: bool
    confidence: float
    reason: str
    degraded: bool = False
    rejection: AbstentionReason = AbstentionReason.GATE_REJECTED
    scores: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # A copy, read-only: a gate cannot change a decision it already returned.
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


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
            "A sentence of the answer cites no retrieved passage, "
            "so it could not be checked and the answer is withheld."
        ),
        AbstentionReason.UNSUPPORTED_CLAIM: (
            "The answer makes a claim its cited passage does not support, so it is withheld."
        ),
        AbstentionReason.OUT_OF_SCOPE: (
            "No indexed filing matches the requested company and fiscal year, "
            "so the model was not asked."
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
class Generation:
    """What the model wrote, and what writing it cost.

    A token count is ``None`` when the backend did not report it, which is
    not the same as zero: Ollama leaves ``prompt_eval_count`` out when the
    whole prompt came from its cache. ``seconds`` is the wall-clock time the
    caller waited, whatever the backend says it spent.
    """

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float


@dataclass(frozen=True, slots=True)
class Trace:
    """What answering one question cost.

    ``stages`` holds the seconds spent embedding, searching, gating and
    generating. ``generation`` is ``None`` exactly when the model was not
    asked, so a refusal by the model still reports the tokens it spent.
    ``gate_scores`` is every score the gate computed, by gate name, whatever
    it decided; empty when no gate ran.
    """

    stages: Mapping[str, float]
    generation: Generation | None
    gate_scores: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class Answer:
    """An answer with its citations, or the reason there is none.

    ``text`` is ``None`` exactly when ``abstained`` is set, and then ``reason``
    says which check withheld it. ``detail`` is the same story in words, for a
    person: the message for the reason followed by what the gate found.
    ``gate_score`` is the gate's own confidence, which is cosine similarity or
    a probability depending on the gate, and ``degraded`` says the gate ran on
    part of its evidence or on its fallback, which a caller must be able to
    tell apart from a model decision. ``source`` is the filing the scope
    resolved to, and ``None`` only when it resolved to none. ``trace`` is for
    whoever operates or evaluates the service; it is logged, not returned to
    the client.
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
    source: IndexedFiling | None
    trace: Trace


class NotRecorded(LookupError):
    """Raised by a recorded model, or a replayed service, asked for what was never recorded."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> Generation: ...


class RelevanceGate(Protocol):
    """Decides, before the model is asked, whether ``filing`` can answer ``question``.

    ``passages`` are the closest ones retrieved from ``filing``; a gate that
    judges the question against the filing itself, like the period guard,
    reads ``filing`` and may ignore them.
    """

    def admits(
        self, question: str, passages: tuple[ScoredChunk, ...], filing: IndexedFiling
    ) -> GateDecision: ...


class Retriever(Protocol):
    """Passages close to a query vector, from the one filing a scope names."""

    def resolve(self, scope: Scope) -> IndexedFiling | None:
        """The filing ``scope`` names, or ``None`` when none is indexed."""
        ...

    def search(
        self, query: NDArray[np.float32], scope: Scope, top_k: int = DEFAULT_TOP_K
    ) -> tuple[ScoredChunk, ...]:
        """The ``top_k`` passages of that filing closest to ``query``.

        No passage of another filing is ever returned, and none at all when
        ``scope`` names no indexed filing.
        """
        ...
