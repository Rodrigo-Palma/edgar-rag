"""A one-filing world for the evaluation harness: an index, golden cases, fakes.

``EXAMPLE`` (CIK 42, fiscal 2024) has a revenue passage printing $391,035
million and a risk passage. Three golden cases of the dev split ask about it:
revenue (answerable, end-to-end), revenue in 2019 (wrong year, end-to-end)
and the boiling point of water (off-domain, gate-only). The embedder places
the first two next to the revenue passage and the third between the two.

A plain module, imported by name, for the same reason as ``tests/fakes.py``.
"""

from decimal import Decimal
from pathlib import Path

from edgar_rag.domain import Chunk
from edgar_rag.eval.golden import GoldenCase, dumps_cases
from edgar_rag.eval.records import CaseRecord
from edgar_rag.index import CorpusIndex
from tests.fakes import CIK, EXAMPLE, FakeEmbedder, one_filing_index

REVENUE = "What was total revenue in fiscal 2024?"
WRONG_YEAR = "What was total revenue in fiscal 2019?"
OFF_DOMAIN = "What is the boiling point of water?"
CHUNKS = (
    Chunk(
        chunk_id="Item 7#0",
        item="Item 7",
        title="MD&A",
        text="Total net sales were $391,035 million in fiscal 2024.",
    ),
    Chunk(chunk_id="Item 1A#0", item="Item 1A", title="Risk Factors", text="Supply may fail."),
)
VECTORS = {
    REVENUE: [1.0, 0.0],
    WRONG_YEAR: [0.9, 0.1],
    OFF_DOMAIN: [0.5, 0.5],
}
ANSWER = "Total net sales were $391,035 million [1]."


def index() -> CorpusIndex:
    return one_filing_index(CHUNKS, [[1.0, 0.0], [0.0, 1.0]])


def embedder() -> FakeEmbedder:
    return FakeEmbedder(VECTORS)


def _case(case_id: str, question: str, **label: object) -> GoldenCase:
    return GoldenCase.model_validate(
        {
            "id": case_id,
            "split": "dev",
            "e2e": True,
            "ticker": "EX",
            "cik": CIK,
            "fiscal_year": EXAMPLE.fiscal_year,
            "accession": EXAMPLE.accession,
            "question": question,
            "template": case_id.split(":")[0],
            **label,
        }
    )


def golden() -> tuple[GoldenCase, ...]:
    return (
        _case(
            "revenue:EX:2024",
            REVENUE,
            answerable=True,
            concept="Revenues",
            period_year=2024,
            period_end="2024-09-28",
            expected_value=Decimal("391035000000"),
            unit="USD",
        ),
        _case("wrong_year:EX:2024", WRONG_YEAR, answerable=False, negative_kind="wrong_year"),
        _case(
            "off_domain:EX:2024",
            OFF_DOMAIN,
            answerable=False,
            negative_kind="off_domain",
            e2e=False,
        ),
    )


def write_root(root: Path) -> Path:
    """An ``eval/`` root holding only the golden set, which is all ``eval run`` reads."""
    (root / "golden").mkdir(parents=True)
    (root / "golden" / "v1.jsonl").write_text(dumps_cases(golden()), encoding="utf-8")
    (root / "golden" / "narrative-v1.jsonl").write_text("", encoding="utf-8")
    return root


def record(
    case_id: str,
    *,
    answerable: bool,
    cosine: float,
    period: float = 1.0,
    brier: float | None = None,
    outcome: str = "answered",
    fold: int | None = None,
    ticker: str = "EX",
    **fields: object,
) -> CaseRecord:
    """A record of an answered case, with any field overridden by keyword."""
    scores = {"cosine": cosine, "period": period}
    if brier is not None:
        scores["brier"] = brier
    return CaseRecord.model_validate(
        {
            "id": case_id,
            "split": "dev" if fold is None else "eval",
            "fold": fold,
            "ticker": ticker,
            "cik": 1,
            "fiscal_year": 2024,
            "answerable": answerable,
            "negative_kind": None if answerable else "off_domain",
            "e2e": True,
            "scores": scores,
            "degraded": False,
            "retrieval_score": cosine,
            "retrieved": (),
            "generated": True,
            "outcome": outcome,
            "stages": {},
            **fields,
        }
    )
