"""The JSON ``/ask`` accepts and returns: the service's public contract.

``README.md`` shows a real response of each kind, and
``tests/test_readme_contract.py`` holds the two together.
"""

from dataclasses import fields
from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from edgar_rag.domain import DEFAULT_TOP_K, AbstentionReason, Answer, Scope

MAX_CIK = 9_999_999_999
# EDGAR holds no electronic filing before 1993, so no earlier year can be indexed.
FIRST_FISCAL_YEAR = 1993
LAST_FISCAL_YEAR = 2100

# What answering cost is for the operator: it goes to the request log, not to
# a client that could use the timings to probe the service.
NOT_RETURNED = frozenset({"trace"})


class AskRequest(BaseModel):
    # Trimmed before the length check, so a question of only whitespace is
    # rejected here instead of reaching the core.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    cik: int = Field(ge=1, le=MAX_CIK, description="The company's CIK: 320193 for Apple.")
    fiscal_year: int | None = Field(
        default=None,
        ge=FIRST_FISCAL_YEAR,
        le=LAST_FISCAL_YEAR,
        description="The fiscal year of the filing; the latest one indexed when left out.",
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=10)

    def scope(self) -> Scope:
        return Scope(cik=self.cik, fiscal_year=self.fiscal_year)


class CitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker: int = Field(description="The [n] in the answer text that points here.")
    item: str
    title: str
    quote: str = Field(description="The window of the passage around what the question asks.")
    score: float = Field(description="Cosine similarity of the passage to the question.")


class SourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    cik: int
    fiscal_year: int
    accession: str
    form: str
    filing_date: date
    url: str = Field(description="The filing's primary document on EDGAR.")


class AskResponse(BaseModel):
    """What ``/ask`` returns, declared so the OpenAPI schema is the contract.

    ``extra="forbid"`` makes a field added to ``Answer`` and not here fail the
    request in tests, instead of reaching clients undocumented. The fields in
    ``NOT_RETURNED`` are left out on purpose.
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
    source: SourceResponse | None = Field(
        description="The filing the passages come from; null when no indexed filing matches."
    )
    replayed: bool = Field(
        description="The models' replies came from a recorded tape (EDGAR_RAG_MODE=replay)."
    )

    @classmethod
    def of(cls, answer: Answer, *, replayed: bool = False) -> "AskResponse":
        returned = {
            field.name: getattr(answer, field.name)
            for field in fields(answer)
            if field.name not in NOT_RETURNED
        }
        # from_attributes reads the Citation and IndexedFiling dataclasses as models
        return cls.model_validate({**returned, "replayed": replayed}, from_attributes=True)
