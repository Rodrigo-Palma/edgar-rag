"""The demo's brief view prints fields of the response and its log line, nothing else."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_brief.py"
SOURCE = {"cik": 320193, "fiscal_year": 2025}
ANSWER = {
    "question": "How much revenue did Apple report for fiscal year 2025?",
    "text": "Apple reported total net sales of $416,161 million [2].",
    "citations": [
        {
            "marker": 2,
            "item": "Item 8",
            "title": "Statements",
            "quote": "net sales",
            "score": 0.7566,
        }
    ],
    "abstained": False,
    "reason": None,
    "detail": "best passage scored 0.758",
    "source": SOURCE,
}
DECLINE = {
    "question": "What was Apple's investment banking revenue in fiscal 2024?",
    "text": None,
    "citations": [],
    "abstained": True,
    "reason": "model_declined",
    "detail": "The model read the closest passages and found no answer in them.",
    "source": SOURCE,
}


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_brief", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


brief = _script()


def _log(tmp_path: Path, *stages: list[str]) -> Path:
    log = tmp_path / "service.log"
    lines = ["INFO: Uvicorn running"] + [
        json.dumps(
            {"method": "POST", "path": "/ask", "stages": dict.fromkeys(s, 0.1)},
            separators=(",", ":"),
        )
        for s in stages
    ]
    log.write_text("\n".join(lines) + "\n", "utf-8")
    return log


def test_an_answer_shows_its_text_citation_and_stages():
    shown = brief.render(ANSWER, ["embed", "search", "gate", "generate"])

    assert "answered    Apple reported total net sales of $416,161 million [2]." in shown
    assert "[2] Item 8, Statements, score 0.7566" in shown
    assert shown.endswith("stages: embed, search, gate, generate")


def test_a_decline_shows_its_reason_and_detail():
    shown = brief.render(DECLINE, None)

    assert "abstained   model_declined" in shown
    assert "found no answer in them" in shown
    assert "stages" not in shown


def test_the_stages_are_those_of_the_nth_request(tmp_path, monkeypatch, capsys):
    log = _log(tmp_path, ["embed", "search", "gate", "generate"], ["embed", "search", "gate"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(DECLINE)))

    assert brief.main(["demo_brief.py", str(log), "2"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("stages: embed, search, gate")


def test_an_error_body_is_not_shown_as_an_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"detail": "not recorded"}'))

    assert brief.main(["demo_brief.py", str(_log(tmp_path)), "1"]) == 1
    assert "not an answer" in capsys.readouterr().err


def test_a_bad_request_number_is_a_usage_error(tmp_path):
    assert brief.main(["demo_brief.py", str(_log(tmp_path)), "0"]) == 2
