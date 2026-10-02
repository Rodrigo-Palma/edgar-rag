"""Print the service's default cosine threshold, fitted on every golden company.

The threshold is R90: the highest cosine score that still admits 90% of the
answerable golden questions. It is fitted after the headline run (ADR-0014)
on the cosine scores of both splits, which together cover all the companies
of the golden set: the eval split from the frozen run, the dev split from the
CI replay. Cosine needs no model to score, so neither run asks one.

Usage: ``make cosine-threshold``, or
``uv run python scripts/cosine_threshold.py eval/runs/v1 data/eval/ci``.
"""

import sys
from pathlib import Path

from edgar_rag.eval.metrics import DEFAULT_TARGET_RECALL, threshold_at_recall
from edgar_rag.eval.records import CaseRecord, read_run
from edgar_rag.gate import COSINE

SPLITS = ("eval", "dev")
DECIMALS = 4  # the precision every recorded score is rounded to


def answerable_golden(run_dir: Path, split: str) -> list[CaseRecord]:
    """The answerable golden cases of a run, which must be of ``split``.

    Raises:
        ValueError: when the run is of another split or has a case without cosine.
    """
    run = read_run(run_dir)
    if run.manifest.split != split:
        raise ValueError(f"{run_dir} is a {run.manifest.split}-split run, expected {split}")
    positives = [case for case in run.cases if case.kind == "golden" and case.answerable]
    unscored = [case.id for case in positives if COSINE not in case.scores]
    if unscored:
        raise ValueError(f"{run_dir}: {len(unscored)} cases have no cosine score: {unscored[:3]}")
    return positives


def main(arguments: list[str]) -> int:
    if len(arguments) != len(SPLITS):
        print(__doc__, file=sys.stderr)
        return 2
    positives = [
        case
        for run_dir, split in zip(arguments, SPLITS, strict=True)
        for case in answerable_golden(Path(run_dir), split)
    ]
    threshold = threshold_at_recall([case.scores[COSINE] for case in positives])
    companies = len({case.ticker for case in positives})
    print(
        f"cosine R{DEFAULT_TARGET_RECALL * 100:.0f} over {len(positives)} answerable golden "
        f"questions, {companies} companies: {threshold:.{DECIMALS}f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
