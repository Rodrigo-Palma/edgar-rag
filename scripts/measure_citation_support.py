"""The v1.1 citation measurement, from the frozen v1 run: ``make eval`` runs it.

    uv run python scripts/measure_citation_support.py

Reads ``eval/runs/v1/cases.jsonl`` and the prompts recorded in
``eval/runs/v1/tape/generate.jsonl``, with no model and no network, and
writes ``docs/eval/report-v1.1-citations.md``. What it counts is fixed by
``docs/eval/protocol-v1.1.md``, committed before it ran.

Exit status 1, writing nothing, when an input is not the one the protocol
pins, or when the passages read back from the tape are not the ones the run
judged (the v1.0.0 rules must reproduce every recorded outcome).
"""

import sys
from pathlib import Path

from edgar_rag.eval.citation_replay import (
    ReplayError,
    arm_masks,
    pinned_sha256,
    read_generations,
    reconstruct,
    sha256_of,
    verify_v1_replay,
)
from edgar_rag.eval.citation_report import Pins, arm_counts, case_results, render
from edgar_rag.eval.records import CASES_FILE, read_run

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "eval" / "runs" / "v1"
TAPE = RUN / "tape" / "generate.jsonl"
PROTOCOL = ROOT / "docs" / "eval" / "protocol-v1.1.md"
REPORT = ROOT / "docs" / "eval" / "report-v1.1-citations.md"


def check_pins(protocol: str, cases: Path, tape: Path) -> Pins:
    """The inputs' sha256, refused unless they are the ones the protocol pins."""
    for path in (cases, tape):
        found, pinned = sha256_of(path), pinned_sha256(protocol, path.name)
        if found != pinned:
            raise ReplayError(f"{path.name} has sha256 {found}; the protocol pins {pinned}")
    return Pins(sha256_of(cases), sha256_of(tape), protocol=PROTOCOL.name)


def build(run_dir: Path = RUN, tape: Path = TAPE, protocol: Path = PROTOCOL) -> str:
    pins = check_pins(protocol.read_text("utf-8"), run_dir / CASES_FILE, tape)
    run = read_run(run_dir)
    replayable = reconstruct(run.cases, read_generations(tape))
    verify_v1_replay(replayable)
    results = case_results(replayable)
    masks = arm_masks(run)
    return render(arm_counts(results, run.cases, masks), results, run.cases, masks, pins)


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else REPORT
    try:
        report = build()
    except ReplayError as error:
        print(f"measure_citation_support: {error}", file=sys.stderr)
        return 1
    out.write_text(report, encoding="utf-8")
    print(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
