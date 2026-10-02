"""``edgar-rag eval ci``: judge a replayed dev run against the committed baseline.

The cosine threshold is read from the service's own default, so the gate
judges the arms as the service would answer, and changing that default
without a new baseline fails here.
"""

import argparse
import sys
from pathlib import Path

from edgar_rag.config import ServiceSettings
from edgar_rag.eval.records import read_run
from edgar_rag.eval.regression import UPDATE, Baseline, compare, dumps, judge_all, loads, render

FAILED = 1
CI_HELP = "judge a replayed run against eval/ci/baseline.json (or rewrite it with --write)"
DEFAULT_BASELINE = Path("eval/ci/baseline.json")


def served_cosine_threshold() -> float:
    default = ServiceSettings.model_fields["min_retrieval_score"].default
    return float(default)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="a replayed run directory")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--write", action="store_true", help="rewrite the baseline from this run instead"
    )


def run(args: argparse.Namespace) -> int:
    try:
        replayed = read_run(args.run)
    except (OSError, ValueError) as error:
        return _fail(f"run: {error}")
    manifest = replayed.manifest
    if manifest.mode != "replay":
        return _fail(f"the CI gate judges a replay; {args.run} was run in {manifest.mode} mode")
    current = Baseline(
        cosine_threshold=served_cosine_threshold(),
        golden_sha256=manifest.golden_sha256,
        index_digest=manifest.index_digest,
        generation=manifest.generation,
        cases=judge_all(replayed.cases, served_cosine_threshold()),
    )
    if args.write:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(dumps(current), encoding="utf-8")
        print(f"{len(current.cases)} cases judged into {args.baseline}")
        return 0
    try:
        baseline = loads(args.baseline.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fail(f"no baseline at {args.baseline}: {UPDATE}")
    except ValueError as error:
        return _fail(f"{args.baseline} is not a baseline: {error}")
    verdict = compare(baseline, current)
    sys.stdout.write(render(verdict, current))
    if verdict.silent_transitions:
        # A GitHub Actions workflow command: a yellow annotation on the run.
        print(
            f"::warning title=eval gate::{verdict.silent_transitions} cases changed outcome "
            "and are judged as before; see the transition matrix in the summary"
        )
    return 0 if verdict.passed else FAILED


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return FAILED
