"""The JSON ``/ask`` accepts and returns: the service's public contract.

``README.md`` shows a real response of each kind, and
``tests/test_readme_contract.py`` holds the two together.
"""

from dataclasses import asdict
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from edgar_rag.domain import DEFAULT_TOP_K, AbstentionReason, Answer


class AskRequest(BaseModel):
    # Trimmed before the length check, so a question of only whitespace is
    # rejected here instead of reaching the core.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)


class CitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker: int = Field(description="The [n] in the answer text that points here.")
    item: str
    title: str
    quote: str = Field(description="The window of the passage around what the question asks.")
    score: float = Field(description="Cosine similarity of the passage to the question.")


class AskResponse(BaseModel):
    """What ``/ask`` returns, declared so the OpenAPI schema is the contract.

    ``extra="forbid"`` makes a field added to ``Answer`` and not here fail the
    request in tests, instead of reaching clients undocumented.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    text: str | None = Field(description="The answer, or null when the service abstained.")
    citations: list[CitationResponse]
    abstained: bool
    reason: AbstentionReason | None = Field(
        description="Which check withheld the answer; null when there is an answer."
    )
    detail: str = Field(description="The decision in words, for a person reading it.")
    retrieval_score: float = Field(description="Cosine similarity of the closest passage.")
    gate_score: float = Field(description="The relevance gate's confidence in its decision.")
    degraded: bool = Field(
        description="The gate decided on part of its evidence or on its fallback."
    )
    source: dict[str, str] = Field(description="The filing the passages come from.")

    @classmethod
    def of(cls, answer: Answer, source: dict[str, str]) -> "AskResponse":
        return cls.model_validate({**asdict(answer), "source": source})
