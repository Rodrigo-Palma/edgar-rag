"""Every number the README states is copied from the evaluation reports.

The checker lives in ``scripts/check_readme_numbers.py`` so ``make check``
can run it on its own; these tests hold its rules and run it on the README.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_readme_numbers.py"
REPORT = "| A | 13/300 = 4.3% [2.5%, 7.3%] | 13.21 | 6,192 cases | +0.6 p.p. | -0.101 |"


def _checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_readme_numbers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _checker()


def test_the_readme_states_no_number_the_reports_do_not_print():
    assert check.main(["check_readme_numbers.py"]) == 0


def test_numbers_copied_from_the_report_pass():
    readme = "A: 13/300 = 4.3% [2.5%, 7.3%], 13.21 s, 6192 cases, 0.6 p.p.; -0.101."

    assert check.missing(readme, [REPORT]) == []


def test_a_changed_rate_fails_with_its_line():
    readme = "# Result\n\nArm A answered 13/300 = 4.4% of them."

    assert check.missing(readme, [REPORT]) == [(3, "4.4%")]


def test_a_fraction_must_be_the_reported_fraction():
    assert check.missing("14/300 answered", [REPORT]) == [(1, "14/300")]


def test_a_percentage_must_appear_as_a_percentage():
    assert check.missing("13.21% of them", [REPORT]) == [(1, "13.21%")]


def test_a_sign_flip_is_caught_but_a_leading_plus_is_not_a_change():
    assert check.missing("+0.6 and 0.101", [REPORT]) == [(1, "0.101")]


def test_code_links_and_references_are_not_results():
    readme = "\n".join(
        [
            "```bash",
            "curl -d '{\"cik\": 320193}'",
            "```",
            "set `EDGAR_RAG_MAX_CONCURRENT_GENERATIONS=2`, see [ADR-0016](docs/adr/0016-x.md)",
            "ollama#15609, pull request #4, a 10-K, https://example.org/v/77",
        ]
    )

    assert check.missing(readme, [REPORT]) == []


def test_a_number_inside_a_mermaid_block_is_checked():
    readme = "\n".join(
        [
            "```mermaid",
            "flowchart LR",
            '  L1["13/300 answered"] --> L2["arm A, 13.21 s; arm F, 6.95 s"]',
            "```",
        ]
    )

    assert check.missing(readme, [REPORT]) == [(3, "6.95")]


def test_other_fenced_blocks_are_still_skipped():
    readme = "\n".join(
        [
            "```json",
            '{"score": 0.7566}',
            "```",
            "```",
            "port 8077",
            "```",
            "```mermaid",
            "  bar [13.21]",
            "```",
            "4.4% after the diagram",
        ]
    )

    assert check.missing(readme, [REPORT]) == [(10, "4.4%")]


def test_every_file_given_is_checked(tmp_path, capsys):
    clean = tmp_path / "clean.md"
    clean.write_text("13/300 = 4.3%", "utf-8")
    invented = tmp_path / "invented.md"
    invented.write_text('```mermaid\nflowchart LR\n  A["99.9% recall"]\n```\n', "utf-8")

    status = check.main(["check_readme_numbers.py", str(clean), str(invented)])

    assert status == 1
    assert f"{invented}:3: 99.9%" in capsys.readouterr().out
