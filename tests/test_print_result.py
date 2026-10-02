"""``make result`` prints the frozen report's cells and sentences as written.

The script lives in ``scripts/print_result.py``; these tests hold it to the
report it reads, so the demo can only show what ``make eval`` reproduces.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "print_result.py"
REPORT = ROOT / "docs" / "eval" / "report-v1.md"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("print_result", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


result = _script()


def _printed(capsys) -> str:
    assert result.main(["print_result.py", str(REPORT)]) == 0
    return capsys.readouterr().out


def test_every_arm_row_is_cells_of_the_report_row(capsys):
    report_rows = [line for line in REPORT.read_text("utf-8").splitlines() if line.startswith("| ")]
    rows = result.arm_rows(result.section(REPORT.read_text("utf-8"), result.ARMS))

    assert [row[0][0] for row in rows[1:]] == list("ABCDEF")
    for row in rows[1:]:
        source = next(line for line in report_rows if line.startswith(f"| {row[0]} |"))
        assert all(f"| {cell} |" in source for cell in row)
    assert "F period guard + cosine        3/300 = 1.0% [0.3%, 2.9%]" in _printed(capsys)


def test_the_h1_and_h2_readings_are_the_report_sentences(capsys):
    printed = " ".join(_printed(capsys).split())
    readings = [
        line
        for line in result.section(REPORT.read_text("utf-8"), result.CONFIRMATORY)
        if line.strip()
    ]

    assert any(line.startswith("H1,") for line in readings)
    assert any(line.startswith("H2,") for line in readings)
    for line in readings:
        assert " ".join(line.split()) in printed


def test_a_report_without_the_arms_section_fails_naming_it(tmp_path, capsys):
    report = tmp_path / "report.md"
    report.write_text("# Evaluation report\n\n## Confirmatory comparisons\n\nH1, ...\n", "utf-8")

    assert result.main(["print_result.py", str(report)]) == 1
    assert "no section '## Six arms, end-to-end tier'" in capsys.readouterr().err


def test_a_renamed_column_fails_naming_it(tmp_path, capsys):
    text = REPORT.read_text("utf-8").replace("| recall cost |", "| recall loss |")
    report = tmp_path / "report.md"
    report.write_text(text, "utf-8")

    assert result.main(["print_result.py", str(report)]) == 1
    assert "no column recall cost" in capsys.readouterr().err
