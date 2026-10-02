"""What an evaluation run leaves on disk: one record per case, and a manifest.

A run directory holds::

    manifest.json        how the run was made: code, data, models, options, timing
    cases.jsonl          one ``CaseRecord`` per case, sorted by id
    determinism.jsonl    one ``RepeatRecord`` per repeated generation, when asked for
    deviations.md        what departed from the protocol, written by hand, when anything did

The record keeps every gate score whatever the gate decided, and the outcome
of the unguarded pipeline (arm A). Every other arm is a mask over these
records (see ``arms``), so the report needs nothing but this directory: no
index, no model, no network.

Files are canonical (sorted keys, no spaces, sorted rows), so the same run
always gives the same bytes and the report built from it the same text.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from edgar_rag.eval.golden import NegativeKind, Split

RUN_FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
CASES_FILE = "cases.jsonl"
DETERMINISM_FILE = "determinism.jsonl"
DEVIATIONS_FILE = "deviations.md"

Kind = Literal["golden", "narrative"]
Tier = Literal["gate-only", "e2e"]
Mode = Literal["live", "record", "replay"]
Outcome = Literal[
    "answered",
    "scored_only",
    "gate_rejected",
    "out_of_period",
    "model_declined",
    "no_valid_citation",
    "unsupported_claim",
    "out_of_scope",
]
"""What the unguarded pipeline did with the case. ``scored_only`` is a case of
the gate-only tier: its gates were scored and the model was not asked."""

ABSTAINED: frozenset[str] = frozenset(
    {
        "gate_rejected",
        "out_of_period",
        "model_declined",
        "no_valid_citation",
        "unsupported_claim",
        "out_of_scope",
    }
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CaseRecord(_Frozen):
    """One case: its label, every gate score, and what the pipeline did with it.

    ``generated`` is whether the model was asked, and ``generation`` what it
    wrote, kept whether or not a check withheld it; it is the answer exactly
    when ``outcome`` is ``answered``. On an answered answerable
    case ``numeric_correct`` says the answer states the gold value,
    ``citation_supported`` that a cited passage prints it, and
    ``gold_retrieved`` that any retrieved passage does (``None`` elsewhere).
    ``cited_items`` are the filing items of the cited passages, which a
    narrative case is graded on; ``item_split_sane`` says whether that
    filing's split into items passed the sanity rule (narratives only).
    """

    id: str = Field(min_length=1)
    kind: Kind = "golden"
    split: Split
    fold: int | None
    ticker: str
    cik: int
    fiscal_year: int
    answerable: bool
    negative_kind: NegativeKind | None
    e2e: bool
    scores: dict[str, float]
    degraded: bool
    retrieval_score: float
    retrieved: tuple[str, ...]
    generated: bool
    outcome: Outcome
    generation: str | None = None
    numeric_correct: bool | None = None
    citation_supported: bool | None = None
    gold_retrieved: bool | None = None
    cited_items: tuple[str, ...] = ()
    expected_item: str | None = None
    item_split_sane: bool | None = None
    stages: dict[str, float]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generator_seconds: float | None = None

    @property
    def answered(self) -> bool:
        return self.outcome == "answered"

    @property
    def correct(self) -> bool:
        """Answered with the gold value and a cited passage that prints it."""
        return self.answered and bool(self.numeric_correct) and bool(self.citation_supported)


class RepeatRecord(_Frozen):
    """A generation asked twice with the same prompt, and whether the text came back the same."""

    id: str
    identical: bool
    first_seconds: float
    second_seconds: float


class ModelInfo(_Frozen):
    """A model as the server reported it: the name asked for and the digest it resolved to."""

    name: str
    digest: str | None


class BrierInfo(_Frozen):
    """The brier plugin a run scored with: who says which commit, and what it reported."""

    sha: str | None
    ready: dict[str, object] | None


class RunManifest(_Frozen):
    """How a run was made. Everything a reader needs to reproduce it, nothing volatile
    except the date and the wall time, which are part of the record."""

    format_version: int = RUN_FORMAT_VERSION
    split: Split
    tier: Tier
    mode: Mode
    limit: int | None
    repo_sha: str
    repo_dirty: bool
    golden_sha256: str
    narrative_sha256: str | None
    protocol_sha256: str | None
    index_digest: str
    index_filings: int
    embedding: ModelInfo
    generation: ModelInfo | None
    generation_options: dict[str, object]
    ollama_version: str | None
    top_k: int
    gates: tuple[str, ...]
    brier: BrierInfo | None
    cases: int
    repeats: int
    date: str
    hardware: str
    wall_seconds: float


def _line(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[BaseModel], key: str = "id") -> None:
    """Write ``rows`` sorted by ``key``, one canonical JSON object per line."""
    ordered = sorted(rows, key=lambda row: str(getattr(row, key)))
    path.write_text("".join(f"{_line(row)}\n" for row in ordered), encoding="utf-8")


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> tuple[T, ...]:
    """Every line of ``path`` validated as ``model``.

    Raises:
        ValueError: naming the line that does not validate.
    """
    rows: list[T] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"{path.name}:{number}: {error}") from error
    return tuple(rows)


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    payload = manifest.model_dump(mode="json")
    text = json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False)
    (run_dir / MANIFEST_FILE).write_text(f"{text}\n", encoding="utf-8")


class Run(_Frozen):
    """A run directory read back: the manifest, the cases, the repeats and the deviations."""

    manifest: RunManifest
    cases: tuple[CaseRecord, ...]
    repeats: tuple[RepeatRecord, ...]
    deviations: str = ""


def read_run(run_dir: Path) -> Run:
    """Read and validate a run directory.

    Raises:
        FileNotFoundError: when the manifest or the cases are missing.
        ValueError: when a file does not validate or the counts disagree.
    """
    manifest = RunManifest.model_validate_json(
        (run_dir / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    cases = read_jsonl(run_dir / CASES_FILE, CaseRecord)
    if len(cases) != manifest.cases:
        raise ValueError(
            f"{run_dir}: the manifest lists {manifest.cases} cases, {CASES_FILE} holds {len(cases)}"
        )
    repeats_path = run_dir / DETERMINISM_FILE
    repeats = read_jsonl(repeats_path, RepeatRecord) if repeats_path.exists() else ()
    deviations_path = run_dir / DEVIATIONS_FILE
    deviations = deviations_path.read_text(encoding="utf-8") if deviations_path.exists() else ""
    return Run(manifest=manifest, cases=cases, repeats=repeats, deviations=deviations)


def rounded(values: Mapping[str, float], digits: int = 4) -> dict[str, float]:
    """A copy with every value rounded, sorted by key, so records diff cleanly."""
    return {name: round(value, digits) for name, value in sorted(values.items())}
