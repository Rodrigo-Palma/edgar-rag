"""What ``EDGAR_RAG_MODE=replay`` serves from: the CI index and the tape recorded over it.

The evaluation records every model call of a run on a tape, keyed by exactly
what the model received. A golden case's prompt carries a nonce derived from
the case id, so the service can replay a case only by asking it with that
nonce: ``RecordedGolden`` maps a question, as the golden set writes it, and
its scope to that nonce. The index, the search, the gates and the citation
check then run for real, and the embedding and the generation are read from
the tape, so a replayed answer is the one the recorded model gave.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from edgar_rag.domain import EmbedderSpec, Scope
from edgar_rag.eval.golden import GoldenCase, load_cases
from edgar_rag.eval.records import ModelInfo
from edgar_rag.eval.replay import GENERATION_KEY_OPTIONS, Tape, TapedEmbedder, TapedGenerator
from edgar_rag.index import CorpusIndex
from edgar_rag.prompt import NonceSource, case_nonce

QuestionKey = tuple[str, int, int]


@dataclass(frozen=True, slots=True)
class RecordedGolden:
    """Golden cases by question text, CIK and fiscal year, each to its case id."""

    case_ids: Mapping[QuestionKey, str]

    @classmethod
    def of(cls, cases: Iterable[GoldenCase]) -> "RecordedGolden":
        """Raises ``ValueError`` when two cases ask the same question about the same filing."""
        ids: dict[QuestionKey, str] = {}
        for case in cases:
            key = (case.question, case.cik, case.fiscal_year)
            if key in ids:
                raise ValueError(f"{ids[key]} and {case.id} ask the same question of one filing")
            ids[key] = case.id
        return cls(case_ids=ids)

    @classmethod
    def read(cls, path: Path) -> "RecordedGolden":
        return cls.of(load_cases(path, GoldenCase))

    def nonce_for(self, question: str, scope: Scope) -> NonceSource | None:
        """The nonce the case was recorded with; ``None`` for anything not in the golden set.

        A scope without a fiscal year is not a golden case: every case names one.
        """
        if scope.fiscal_year is None:
            return None
        case_id = self.case_ids.get((question, scope.cik, scope.fiscal_year))
        return None if case_id is None else case_nonce(case_id)


@dataclass(frozen=True, slots=True)
class ReplayParts:
    """The index and the taped models a replayed service answers with."""

    index: CorpusIndex
    embedder: TapedEmbedder
    generator: TapedGenerator
    questions: RecordedGolden
    generation: ModelInfo


def open_replay(index_dir: Path, tape_dir: Path, golden: Path, spec: EmbedderSpec) -> ReplayParts:
    """Read the index, the tape and the golden set a replayed service answers from.

    Raises:
        FileNotFoundError: when there is no index or no golden set.
        IndexFormatError: when the index cannot be trusted, or is a Git LFS
            pointer, with what to run.
        ValueError: when there is no tape, or a file of it is not a tape.
    """
    tape = Tape.open(tape_dir)
    if not tape.meta:
        raise ValueError(f"no tape in {tape_dir}; record one with `make eval-ci-record`")
    generation = ModelInfo.model_validate(tape.meta["generation"])
    index = CorpusIndex.load(index_dir, expect=spec)
    return ReplayParts(
        index=index,
        embedder=TapedEmbedder(tape, spec),
        generator=TapedGenerator(tape, generation.name, GENERATION_KEY_OPTIONS),
        questions=RecordedGolden.read(golden),
        generation=generation,
    )
