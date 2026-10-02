"""``edgar-rag eval run`` and ``edgar-rag eval report``: the composition root of the harness.

``run`` wires the production ``Answerer`` to the index and to Ollama (live,
or through a tape), asks every case of a split and writes a run directory. A
replay of a tape that holds brier scores (the frozen headline run's) scores
brier too; nothing else does, since its gate was removed (ADR-0014).
``report`` turns a run directory into markdown and needs nothing else.

Modes: ``record`` (the default) answers from the tape when it can and calls
the models for the rest, appending to the tape; ``replay`` uses the tape only,
so it runs without Ollama and fails on a miss; ``live`` uses no tape.

The eval split is held out for the pre-registered round: ``run --split eval``
refuses to start without ``--protocol`` naming the committed protocol, whose
hash goes in the manifest.
"""

import argparse
import datetime
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import ValidationError

from edgar_rag.answer import Answerer
from edgar_rag.config import EvalSettings
from edgar_rag.domain import DEFAULT_TOP_K, Embedder, EmbedderSpec, Generator, RelevanceGate
from edgar_rag.eval import provenance
from edgar_rag.eval.build import EvalPaths
from edgar_rag.eval.golden import GoldenCase, NarrativeCase, Split, load_cases
from edgar_rag.eval.grading import EvalQuestion, folds_by_ticker, from_golden, from_narrative
from edgar_rag.eval.records import (
    CASES_FILE,
    DETERMINISM_FILE,
    BrierInfo,
    CaseRecord,
    Mode,
    ModelInfo,
    RepeatRecord,
    RunManifest,
    read_run,
    write_jsonl,
    write_manifest,
)
from edgar_rag.eval.replay import (
    BRIER,
    GENERATION_KEY_OPTIONS,
    Tape,
    TapedBrierScores,
    TapedEmbedder,
    TapedGenerator,
    TapeMiss,
)
from edgar_rag.eval.report import DEFAULT_CONFIRMATORY_CONFIDENCE, ReportOptions, build_report
from edgar_rag.eval.runner import (
    CapturingIndex,
    Harness,
    repeat_generations,
    repeat_order,
    run_cases,
)
from edgar_rag.gate import AllOf, CosineGate
from edgar_rag.index import CorpusIndex, IndexFormatError
from edgar_rag.models import (
    LOWERCASE_INPUT,
    REQUEST_TIMEOUT_SECONDS,
    ModelError,
    OllamaEmbedder,
    OllamaGenerator,
)
from edgar_rag.period import PeriodGuard

FAILED = 1
RUN_HELP = "ask a split's golden set through the answerer and write a run directory"
REPORT_HELP = "print the markdown report of a run directory (no model, no network)"


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("dev", "eval"), required=True)
    parser.add_argument(
        "--gate-only", action="store_true", help="score the gates, never ask the model"
    )
    parser.add_argument(
        "--narratives", action="store_true", help="also ask the split's narrative questions"
    )
    parser.add_argument("--mode", choices=("record", "replay", "live"), default="record")
    parser.add_argument(
        "--out", type=Path, help="run directory (default: data/eval/<split>-<tier>)"
    )
    parser.add_argument("--tape", type=Path, help="tape directory (default: <out>/tape)")
    parser.add_argument("--index-dir", type=Path, help="default: EDGAR_RAG_INDEX_DIR")
    parser.add_argument("--root", type=Path, default=Path("eval"), help="golden set root")
    parser.add_argument("--limit", type=int, help="only this many cases, chosen by id hash")
    parser.add_argument("--repeat", type=int, default=0, help="generations to repeat live")
    parser.add_argument("--protocol", type=Path, help="the committed protocol (eval split only)")
    parser.add_argument("--generation-model", help="default: EDGAR_RAG_GENERATION_MODEL")


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="a run directory")
    parser.add_argument("--out", type=Path, help="write the report here instead of stdout")
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIRMATORY_CONFIDENCE,
        help="level of the confirmatory intervals (default: %(default)s)",
    )


def run_report_command(args: argparse.Namespace) -> int:
    try:
        run = read_run(args.run)
    except (OSError, ValueError) as error:
        return _fail(f"run: {error}")
    text = build_report(run, ReportOptions(confirmatory_confidence=args.confidence))
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text, encoding="utf-8")
    return 0


def run_run_command(args: argparse.Namespace) -> int:
    try:
        settings = EvalSettings()
    except ValidationError as error:
        return _fail(f"configuration: {error}")
    if args.split == "eval" and (args.protocol is None or not args.protocol.is_file()):
        return _fail(
            "the eval split is held out for the pre-registered round: "
            "pass --protocol with the committed protocol"
        )
    if args.mode == "replay" and args.repeat:
        return _fail("--repeat measures the live model; it cannot run in replay")
    try:
        return _run(args, settings)
    except (TapeMiss, ModelError, IndexFormatError, OSError, ValueError) as error:
        return _fail(f"{type(error).__name__}: {error}")


@dataclass(frozen=True, slots=True)
class _Wiring:
    embedder: Embedder
    generator: Generator
    gates: RelevanceGate
    live_generator: Generator | None
    embedding: ModelInfo
    generation: ModelInfo
    ollama_version: str | None
    brier: BrierInfo | None
    gate_names: tuple[str, ...]


def _run(args: argparse.Namespace, settings: EvalSettings) -> int:
    started = time.perf_counter()
    paths = EvalPaths(args.root)
    # Read before anything runs: the code that ran is the code checked out now,
    # whatever is committed while the run takes its hours.
    code = provenance.repo_state(paths.root)
    questions = _questions(paths, args.split, args.narratives, args.limit)
    tier = "gate-only" if args.gate_only else "e2e"
    out: Path = args.out or Path("data/eval") / f"{args.split}-{tier}"
    tape_dir: Path = args.tape or out / "tape"
    spec = EmbedderSpec(model=settings.embedding_model, lowercase=LOWERCASE_INPUT)
    index = CorpusIndex.load(args.index_dir or settings.index_dir, expect=spec)
    tape = Tape.open(tape_dir) if args.mode != "live" else None
    out.mkdir(parents=True, exist_ok=True)
    with _ollama_client(args.mode) as ollama:
        wiring = _wire(args, settings, spec, index, tape, ollama)
        harness = Harness(
            Answerer(
                index=CapturingIndex(index),
                embedder=wiring.embedder,
                generator=wiring.generator,
                gate=wiring.gates,
            ),
            gates=wiring.gates,
        )
        records = _ask_all(harness, questions, out, generate=not args.gate_only)
        repeats: tuple[RepeatRecord, ...] = ()
        if args.repeat and wiring.live_generator is not None:
            repeats = repeat_generations(
                harness, questions, records, wiring.live_generator, args.repeat
            )
            write_jsonl(out / DETERMINISM_FILE, repeats)
    if tape is not None and args.mode == "record":
        tape.save(_tape_meta(wiring))
    manifest = _manifest(args, code, paths, index, wiring, records, repeats, started)
    write_manifest(out, manifest)
    _summarise(records.values(), manifest, out)
    return 0


def _questions(
    paths: EvalPaths, split: Split, narratives: bool, limit: int | None
) -> tuple[EvalQuestion, ...]:
    golden = [case for case in load_cases(paths.golden, GoldenCase) if case.split == split]
    questions = [from_golden(case) for case in golden]
    if narratives:
        folds = folds_by_ticker(golden)
        questions += [
            from_narrative(case, split, folds[case.ticker])
            for case in load_cases(paths.narrative, NarrativeCase)
            if case.ticker in folds
        ]
    if limit is not None:
        questions = sorted(questions, key=lambda q: repeat_order(q.id))[:limit]
    return tuple(sorted(questions, key=lambda q: q.id))


@contextmanager
def _ollama_client(mode: str) -> Iterator[httpx.Client | None]:
    """The client Ollama is asked through; none in replay."""
    if mode == "replay":
        yield None
        return
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        yield client


def _wire(
    args: argparse.Namespace,
    settings: EvalSettings,
    spec: EmbedderSpec,
    index: CorpusIndex,
    tape: Tape | None,
    ollama: httpx.Client | None,
) -> _Wiring:
    gate_list: list[RelevanceGate] = [PeriodGuard(), CosineGate(min_score=0.0)]
    names = ["period", "cosine"]
    brier = None
    if ollama is None and tape is not None and tape.has(BRIER):
        gate_list.append(TapedBrierScores(tape))
        names.append("brier")
        brier = BrierInfo.model_validate(tape.meta.get("brier") or {"sha": None, "ready": None})
    gates = AllOf(*gate_list)
    if ollama is None:
        return _replay_wiring(args, spec, cast(Tape, tape), gates, brier, tuple(names))
    base = str(settings.ollama_base_url)
    model = args.generation_model or settings.generation_model
    embedding = provenance.ollama_model(ollama, base, spec.model)
    generation = provenance.ollama_model(ollama, base, model)
    live_embedder = OllamaEmbedder(
        base, spec.model, dimensions=index.fingerprint.dimensions, client=ollama
    )
    live_generator = OllamaGenerator(base, model, client=ollama)
    embedder: Embedder = live_embedder
    generator: Generator = live_generator
    if tape is not None:
        _check_tape_models(tape, embedding, generation)
        embedder = TapedEmbedder(tape, spec, live=live_embedder)
        generator = TapedGenerator(tape, model, GENERATION_KEY_OPTIONS, live=live_generator)
    return _Wiring(
        embedder=embedder,
        generator=generator,
        gates=gates,
        live_generator=live_generator,
        embedding=embedding,
        generation=generation,
        ollama_version=provenance.ollama_version(ollama, base),
        brier=brier,
        gate_names=tuple(names),
    )


def _replay_wiring(
    args: argparse.Namespace,
    spec: EmbedderSpec,
    tape: Tape,
    gates: RelevanceGate,
    brier: BrierInfo | None,
    names: tuple[str, ...],
) -> _Wiring:
    meta = tape.meta
    if not meta:
        raise TapeMiss(f"{tape.root} holds no tape; record one with --mode record")
    generation = ModelInfo.model_validate(meta["generation"])
    model = args.generation_model or generation.name
    return _Wiring(
        embedder=TapedEmbedder(tape, spec),
        generator=TapedGenerator(tape, model, GENERATION_KEY_OPTIONS),
        gates=gates,
        live_generator=None,
        embedding=ModelInfo.model_validate(meta["embedding"]),
        generation=generation,
        ollama_version=meta.get("ollama_version"),
        brier=brier,
        gate_names=names,
    )


def _check_tape_models(tape: Tape, embedding: ModelInfo, generation: ModelInfo) -> None:
    """Refuse to add to a tape another build of the same model name recorded."""
    for kind, live in (("embedding", embedding), ("generation", generation)):
        recorded = tape.meta.get(kind)
        if not recorded or recorded.get("name") != live.name:
            continue
        if recorded.get("digest") != live.digest:
            raise ValueError(
                f"{tape.root} was recorded with {live.name} {recorded.get('digest')}, "
                f"and Ollama now serves {live.digest}; record into a fresh tape"
            )


def _tape_meta(wiring: _Wiring) -> dict[str, Any]:
    return {
        "embedding": wiring.embedding.model_dump(mode="json"),
        "generation": wiring.generation.model_dump(mode="json"),
        "generation_options": GENERATION_KEY_OPTIONS,
        "ollama_version": wiring.ollama_version,
        "brier": None if wiring.brier is None else wiring.brier.model_dump(mode="json"),
    }


def _ask_all(
    harness: Harness, questions: tuple[EvalQuestion, ...], out: Path, *, generate: bool
) -> dict[str, CaseRecord]:
    """Ask every question, appending each record as made so a stopped run keeps its work."""
    partial = out / f"{CASES_FILE}.partial"
    partial.unlink(missing_ok=True)
    records: dict[str, CaseRecord] = {}
    with partial.open("a", encoding="utf-8") as handle:
        for record in run_cases(harness, questions, generate=generate):
            records[record.id] = record
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
    write_jsonl(out / CASES_FILE, records.values())
    partial.unlink()
    return records


def _manifest(
    args: argparse.Namespace,
    code: tuple[str, bool],
    paths: EvalPaths,
    index: CorpusIndex,
    wiring: _Wiring,
    records: dict[str, CaseRecord],
    repeats: tuple[RepeatRecord, ...],
    started: float,
) -> RunManifest:
    sha, dirty = code
    generated = any(record.generated for record in records.values())
    return RunManifest(
        split=args.split,
        tier="gate-only" if args.gate_only else "e2e",
        mode=cast(Mode, args.mode),
        limit=args.limit,
        repo_sha=sha,
        repo_dirty=dirty,
        golden_sha256=provenance.file_sha256(paths.golden),
        narrative_sha256=provenance.file_sha256(paths.narrative) if args.narratives else None,
        protocol_sha256=provenance.file_sha256(args.protocol) if args.protocol else None,
        index_digest=index.digest(),
        index_filings=len(index.filings),
        embedding=wiring.embedding,
        generation=wiring.generation if generated else None,
        generation_options=GENERATION_KEY_OPTIONS,
        ollama_version=wiring.ollama_version,
        top_k=DEFAULT_TOP_K,
        gates=wiring.gate_names,
        brier=wiring.brier,
        cases=len(records),
        repeats=len(repeats),
        date=datetime.datetime.now(datetime.UTC).date().isoformat(),
        hardware=provenance.hardware(),
        wall_seconds=round(time.perf_counter() - started, 1),
    )


def _summarise(records: Iterable[CaseRecord], manifest: RunManifest, out: Path) -> None:
    listed = list(records)
    generated = [record for record in listed if record.generated]
    per_case = manifest.wall_seconds / len(listed) if listed else 0.0
    print(
        f"{len(listed)} cases ({len(generated)} generated) in {manifest.wall_seconds:.0f} s, "
        f"{per_case:.2f} s per case; run written to {out}"
    )


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return FAILED
