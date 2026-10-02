"""The ``edgar-rag`` command: ingest a filing, serve answers, run the evaluation.

Each command reads only its own settings (see ``config``), so serving never
asks for the EDGAR contact and ingesting never needs a generation model. A
failure the user can act on (a missing variable, EDGAR or Ollama refusing) is
one line on stderr and exit code 1, not a traceback.
"""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import ValidationError

from edgar_rag.config import IngestSettings, ServiceSettings, SnapshotIngestSettings
from edgar_rag.domain import EmbedderSpec
from edgar_rag.edgar.client import EdgarClient, EdgarError, Filing
from edgar_rag.edgar.fetch import REQUEST_TIMEOUT_SECONDS as EDGAR_TIMEOUT_SECONDS
from edgar_rag.eval import build, commands, power
from edgar_rag.eval.build import EvalPaths
from edgar_rag.eval.corpus import index_pinned, pinned_filings
from edgar_rag.eval.replay_serving import open_replay
from edgar_rag.eval.snapshot import read_lock
from edgar_rag.index import IndexFormatError, write_shard
from edgar_rag.ingest import index_filing
from edgar_rag.models import LOWERCASE_INPUT, ModelError, OllamaEmbedder
from edgar_rag.models import REQUEST_TIMEOUT_SECONDS as MODEL_TIMEOUT_SECONDS
from edgar_rag.service.app import Replay, serve

FAILED = 1
DESCRIPTION = "Question answering over SEC filings that cites its sources and abstains."


@dataclass(frozen=True, slots=True)
class EvalCommand:
    """One ``edgar-rag eval`` subcommand: its arguments and what it runs."""

    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


EVAL_COMMANDS = (
    EvalCommand("power", power.DESCRIPTION, power.add_arguments, power.run),
    EvalCommand("build", build.BUILD_HELP, build.add_build_arguments, build.run_build_command),
    EvalCommand("golden-stats", build.STATS_HELP, build.add_root_argument, build.run_stats_command),
    EvalCommand(
        "golden-fetch", build.FETCH_HELP, build.add_fetch_arguments, build.run_fetch_command
    ),
    EvalCommand("run", commands.RUN_HELP, commands.add_run_arguments, commands.run_run_command),
    EvalCommand(
        "report", commands.REPORT_HELP, commands.add_report_arguments, commands.run_report_command
    ),
)


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    """Run the command in ``argv``; ``transport`` stands in for the network in tests."""
    args = _parser().parse_args(argv)
    command: Callable[[argparse.Namespace, httpx.BaseTransport | None], int] = args.command
    return command(args, transport)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgar-rag", description=DESCRIPTION)
    commands = parser.add_subparsers(title="commands", required=True, metavar="COMMAND")

    ingest = commands.add_parser(
        "ingest",
        help="add filings to the index: a company's latest from EDGAR, or the pinned ones",
    )
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--cik", type=int, help="download this company's latest filing")
    source.add_argument(
        "--lock",
        type=Path,
        help="index the filings this lock pins, from the snapshots beside it, offline",
    )
    ingest.add_argument("--form", default="10-K", help="with --cik: the form (default: 10-K)")
    ingest.add_argument("--split", choices=("dev", "eval"), help="with --lock: only this split")
    ingest.add_argument(
        "--ticker", action="append", default=[], help="with --lock: only this company (repeatable)"
    )
    ingest.add_argument(
        "--index-dir",
        type=Path,
        help="the index to add to (default: EDGAR_RAG_INDEX_DIR, or data/index)",
    )
    ingest.set_defaults(command=_ingest)

    serving = commands.add_parser(
        "serve",
        help="answer questions over HTTP from the index (EDGAR_RAG_MODE=replay: from the tape)",
    )
    serving.set_defaults(command=_serve)

    evaluation = commands.add_parser("eval", help="evaluation tools")
    eval_commands = evaluation.add_subparsers(title="commands", required=True, metavar="COMMAND")
    for eval_command in EVAL_COMMANDS:
        sub = eval_commands.add_parser(eval_command.name, help=eval_command.help)
        eval_command.add_arguments(sub)
        sub.set_defaults(command=_run_eval(eval_command))
    return parser


def _ingest(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    if args.lock is not None:
        return _ingest_pinned(args, transport)
    if args.split is not None or args.ticker:
        return _fail("--split and --ticker choose among the filings of --lock")
    return _ingest_latest(args, transport)


def _ingest_latest(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    """Download, embed and add to the index, writing nothing unless every step succeeded."""
    try:
        settings = IngestSettings()
    except ValidationError as error:
        return _fail(f"configuration: {error}")

    try:
        filing = _latest_filing(settings, args.cik, args.form, transport)
    except EdgarError as error:
        return _fail(f"EDGAR: {error}")

    with httpx.Client(transport=transport, timeout=MODEL_TIMEOUT_SECONDS) as http:
        embedder = OllamaEmbedder(
            str(settings.ollama_base_url), settings.embedding_model, client=http
        )
        try:
            shard = index_filing(filing, embedder)
        except (ModelError, ValueError) as error:
            return _fail(f"embedding: {error}")

    root: Path = args.index_dir or settings.index_dir
    try:
        write_shard(root, shard, embedder.spec)
    except (IndexFormatError, ValueError) as error:
        return _fail(f"index: {error}")
    print(
        f"{filing.company} {filing.form} {filing.filing_date}, "
        f"fiscal {shard.filing.fiscal_year}: {len(shard.chunks)} chunks"
    )
    print(f"{shard.filing.accession} added to the index in {root}")
    return 0


def _ingest_pinned(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    """Embed every chosen pinned filing from its snapshot, adding each as it is done.

    A failure stops the run; the filings added before it stay in the index.
    """
    try:
        settings = SnapshotIngestSettings()
    except ValidationError as error:
        return _fail(f"configuration: {error}")
    try:
        lock = read_lock(args.lock)
        filings = pinned_filings(lock, split=args.split, tickers=args.ticker)
    except (OSError, ValueError) as error:
        return _fail(f"lock: {error}")

    root: Path = args.index_dir or settings.index_dir
    snapshots = EvalPaths(args.lock.parent).snapshots
    started = time.perf_counter()
    chunks = 0
    with httpx.Client(transport=transport, timeout=MODEL_TIMEOUT_SECONDS) as http:
        embedder = OllamaEmbedder(
            str(settings.ollama_base_url), settings.embedding_model, client=http
        )
        for filing in filings:
            filing_started = time.perf_counter()
            name = f"{filing.ticker} fiscal {filing.fiscal_year} {filing.accession}"
            try:
                shard = index_pinned(snapshots, lock, filing, embedder)
            except (ModelError, ValueError) as error:
                return _fail(f"{name}: {error}")
            try:
                write_shard(root, shard, embedder.spec)
            except (IndexFormatError, ValueError) as error:
                return _fail(f"index: {error}")
            chunks += len(shard.chunks)
            seconds = time.perf_counter() - filing_started
            print(f"{name}: {len(shard.chunks)} chunks in {seconds:.1f} s", flush=True)
    total = time.perf_counter() - started
    print(f"{len(filings)} filings, {chunks} chunks added to {root} in {total:.0f} s")
    return 0


def _latest_filing(
    settings: IngestSettings, cik: int, form: str, transport: httpx.BaseTransport | None
) -> Filing:
    # The client EdgarClient would open for itself, with the transport given.
    with (
        httpx.Client(
            transport=transport, timeout=EDGAR_TIMEOUT_SECONDS, follow_redirects=False
        ) as http,
        EdgarClient(settings.edgar_user_agent, http=http) as edgar,
    ):
        return edgar.latest_filing(cik, form)


def _serve(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    try:
        settings = ServiceSettings()
    except ValidationError as error:
        return _fail(f"configuration: {error}")
    if settings.mode == "live":
        serve(settings)
        return 0
    # Replay is checked before the server starts, so a missing tape or an
    # index checked out as Git LFS pointers is one line, not a traceback.
    spec = EmbedderSpec(model=settings.embedding_model, lowercase=LOWERCASE_INPUT)
    try:
        parts = open_replay(
            settings.index_dir, settings.replay_tape, settings.replay_questions, spec
        )
    except (IndexFormatError, OSError, ValueError) as error:
        return _fail(f"replay: {error}")
    print(
        f"replaying {parts.generation.name} from {settings.replay_tape} over "
        f"{len(parts.index.filings)} filings in {settings.index_dir}",
        file=sys.stderr,
    )
    replay = Replay(
        index=parts.index,
        embedder=parts.embedder,
        generator=parts.generator,
        questions=parts.questions,
    )
    serve(settings, replay=replay)
    return 0


def _run_eval(
    command: EvalCommand,
) -> Callable[[argparse.Namespace, httpx.BaseTransport | None], int]:
    def run(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
        return command.run(args)

    return run


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return FAILED
