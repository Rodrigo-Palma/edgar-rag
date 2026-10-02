"""The ``edgar-rag`` command: ingest a filing, serve answers, run the evaluation.

Each command reads only its own settings (see ``config``), so serving never
asks for the EDGAR contact and ingesting never needs a generation model. A
failure the user can act on (a missing variable, EDGAR or Ollama refusing) is
one line on stderr and exit code 1, not a traceback.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from edgar_rag.config import IngestSettings, ServiceSettings
from edgar_rag.edgar.client import EdgarClient, EdgarError, Filing
from edgar_rag.edgar.fetch import REQUEST_TIMEOUT_SECONDS as EDGAR_TIMEOUT_SECONDS
from edgar_rag.eval import build, power
from edgar_rag.ingest import index_filing
from edgar_rag.models import REQUEST_TIMEOUT_SECONDS as MODEL_TIMEOUT_SECONDS
from edgar_rag.models import ModelError, OllamaEmbedder
from edgar_rag.service.app import serve

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
)


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    """Run the command in ``argv``; ``transport`` stands in for the network in tests."""
    args = _parser().parse_args(argv)
    command: Callable[[argparse.Namespace, httpx.BaseTransport | None], int] = args.command
    return command(args, transport)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgar-rag", description=DESCRIPTION)
    commands = parser.add_subparsers(title="commands", required=True, metavar="COMMAND")

    ingest = commands.add_parser("ingest", help="download a company's latest filing and index it")
    ingest.add_argument("--cik", type=int, required=True, help="company CIK, 320193 for Apple")
    ingest.add_argument("--form", default="10-K", help="the form to index (default: 10-K)")
    ingest.set_defaults(command=_ingest)

    serving = commands.add_parser("serve", help="answer questions over HTTP from the index")
    serving.set_defaults(command=_serve)

    evaluation = commands.add_parser("eval", help="evaluation tools")
    eval_commands = evaluation.add_subparsers(title="commands", required=True, metavar="COMMAND")
    for eval_command in EVAL_COMMANDS:
        sub = eval_commands.add_parser(eval_command.name, help=eval_command.help)
        eval_command.add_arguments(sub)
        sub.set_defaults(command=_run_eval(eval_command))
    return parser


def _ingest(args: argparse.Namespace, transport: httpx.BaseTransport | None) -> int:
    """Download, embed and save, writing nothing unless every step succeeded."""
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
            index = index_filing(filing, embedder)
        except (ModelError, ValueError) as error:
            return _fail(f"embedding: {error}")

    index.save(settings.index_dir)
    print(f"{filing.company} {filing.form} {filing.filing_date}: {len(index.chunks)} chunks")
    print(f"index written to {settings.index_dir}")
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
    serve(settings)
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
