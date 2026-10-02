"""The headline result, printed from the frozen report: ``make result``.

    uv run python scripts/print_result.py                # docs/eval/report-v1.md
    uv run python scripts/print_result.py REPORT.md      # any report make eval wrote

Prints the six arms (false answers, recall cost, generator seconds per
question) and the confirmatory reading of H1 and H2. Every value is a cell or
a sentence of the report, copied as written; nothing is recomputed, so this
can only show what ``make eval`` reproduces.

Exit status 1, naming what is missing, when the report has no such section.
"""

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "eval" / "report-v1.md"
ARMS = "## Six arms, end-to-end tier"
CONFIRMATORY = "## Confirmatory comparisons"
# Report header -> the shorter label printed above the column.
COLUMNS = {
    "arm": "arm",
    "false answers on unanswerable (Wilson 95%)": "false answers (Wilson 95%)",
    "recall cost": "recall cost",
    "generator s / question": "generator s / question",
}
WIDTH = 104


class ReportError(ValueError):
    """The report lacks a section or a column this summary copies."""


def section(report: str, heading: str) -> list[str]:
    """The lines under ``heading``, up to the next heading of the same level."""
    lines = report.splitlines()
    if heading not in lines:
        raise ReportError(f"no section '{heading}' in the report")
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start:end]


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def arm_rows(lines: list[str]) -> list[list[str]]:
    """The copied columns of the arms table, header row first."""
    table = [line for line in lines if line.startswith("|")]
    if len(table) < 3:
        raise ReportError(f"no arms table under '{ARMS}'")
    header = _cells(table[0])
    missing = [name for name in COLUMNS if name not in header]
    if missing:
        raise ReportError(f"the arms table has no column {', '.join(missing)}")
    picked = [header.index(name) for name in COLUMNS]
    body = [_cells(row) for row in table[2:]]
    return [list(COLUMNS.values()), *([row[i] for i in picked] for row in body)]


def aligned(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]


def summary(report: str, name: str) -> str:
    arms = section(report, ARMS)
    intro = next((line for line in arms if line and not line.startswith("|")), "")
    readings = [line for line in section(report, CONFIRMATORY) if line.strip()]
    if not readings:
        raise ReportError(f"nothing under '{CONFIRMATORY}'")
    out = [f"{name}, as make eval wrote it; nothing recomputed", "", intro, ""]
    out += aligned(arm_rows(arms))
    for paragraph in readings:
        out += ["", textwrap.fill(paragraph, WIDTH)]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else REPORT
    try:
        print(summary(path.read_text("utf-8"), path.name))
    except (OSError, ReportError) as error:
        print(f"print_result: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
