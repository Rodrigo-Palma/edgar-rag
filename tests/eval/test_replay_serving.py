"""The replayed service: its question book, its tape, and the committed CI pair behind make demo."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edgar_rag import cli
from edgar_rag.config import ServiceSettings
from edgar_rag.domain import Scope
from edgar_rag.eval.golden import GoldenCase, load_cases
from edgar_rag.eval.replay import Tape
from edgar_rag.eval.replay_serving import RecordedGolden, open_replay
from edgar_rag.index import IndexFormatError
from edgar_rag.prompt import case_nonce
from edgar_rag.service.app import Replay, create_app
from tests.eval import harness_fakes as world
from tests.fakes import CIK, SPEC

REPO = Path(__file__).parents[2]
DEMO_SCRIPT = REPO / "scripts" / "demo.sh"
README = REPO / "README.md"
DEMO_BODY = re.compile(r"^ask '(\{.*\})'$", re.MULTILINE)


def _golden(root: Path):
    return load_cases(root / "golden" / "v1.jsonl", GoldenCase)


def test_a_golden_question_gets_the_nonce_of_its_case(tmp_path):
    cases = _golden(world.write_root(tmp_path))
    book = RecordedGolden.of(cases)
    case = cases[0]

    nonce = book.nonce_for(case.question, Scope(case.cik, case.fiscal_year))

    assert nonce is not None
    assert nonce() == case_nonce(case.id)()


@pytest.mark.parametrize(
    "scope", [Scope(CIK, None), Scope(CIK, 2019), Scope(7, 2024)], ids=["no-year", "year", "cik"]
)
def test_the_question_of_another_scope_is_not_recorded(tmp_path, scope):
    book = RecordedGolden.of(_golden(world.write_root(tmp_path)))

    assert book.nonce_for(world.REVENUE, scope) is None


def test_two_cases_asking_one_filing_the_same_question_are_refused(tmp_path):
    case = _golden(world.write_root(tmp_path))[0]
    twin = case.model_copy(update={"id": f"{case.id}:twin"})

    with pytest.raises(ValueError, match="ask the same question"):
        RecordedGolden.of([case, twin])


def test_without_a_tape_the_replay_says_how_to_record_one(tmp_path):
    with pytest.raises(ValueError, match="no tape.*make eval-ci-record"):
        open_replay(tmp_path / "index", tmp_path / "tape", tmp_path / "golden.jsonl", SPEC)


def _recorded_tape(root: Path) -> Path:
    tape = Tape(root=root, meta={}, stores=Tape.open(root).stores)
    tape.save({"generation": {"name": "qwen3:8b", "digest": "500a"}})
    return root


def test_the_replay_reads_the_model_it_was_recorded_with(tmp_path):
    golden = world.write_root(tmp_path / "eval") / "golden" / "v1.jsonl"
    world.index().save(tmp_path / "index")

    parts = open_replay(tmp_path / "index", _recorded_tape(tmp_path / "tape"), golden, SPEC)

    assert parts.generation.name == "qwen3:8b"
    assert parts.generator.model == "qwen3:8b"
    assert parts.index.filings == world.index().filings


def _demo_bodies() -> list[dict[str, object]]:
    return [json.loads(body) for body in DEMO_BODY.findall(DEMO_SCRIPT.read_text("utf-8"))]


def test_the_demo_asks_the_two_questions_the_readme_shows():
    readme = README.read_text("utf-8")
    bodies = DEMO_BODY.findall(DEMO_SCRIPT.read_text("utf-8"))

    assert len(bodies) == 2
    for body in bodies:
        assert f"-d '{body}'" in readme


def test_the_committed_tape_answers_one_demo_question_and_declines_the_other():
    """What `make demo` shows, from the committed CI index and qwen3:8b tape."""
    parts = open_replay(
        REPO / "eval" / "ci" / "index",
        REPO / "eval" / "ci" / "tape",
        REPO / "eval" / "golden" / "v1.jsonl",
        SPEC,
    )
    replay = Replay(parts.index, parts.embedder, parts.generator, parts.questions)
    with TestClient(create_app(ServiceSettings(mode="replay"), replay=replay)) as client:
        answer, decline = (client.post("/ask", json=body).json() for body in _demo_bodies())

    assert answer["replayed"] is True
    assert answer["abstained"] is False
    assert answer["citations"]
    assert "416,161" in answer["text"]
    assert decline["reason"] == "out_of_period"


def test_serve_in_replay_hands_the_tape_to_the_service(monkeypatch, tmp_path):
    golden = world.write_root(tmp_path / "eval") / "golden" / "v1.jsonl"
    world.index().save(tmp_path / "index")
    monkeypatch.setenv("EDGAR_RAG_MODE", "replay")
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("EDGAR_RAG_REPLAY_TAPE", str(_recorded_tape(tmp_path / "tape")))
    monkeypatch.setenv("EDGAR_RAG_REPLAY_QUESTIONS", str(golden))
    served: list[Replay | None] = []
    monkeypatch.setattr(cli, "serve", lambda settings, replay=None: served.append(replay))

    assert cli.main(["serve"]) == 0
    assert served[0] is not None
    assert served[0].index.filings == world.index().filings


def test_serve_in_replay_over_lfs_pointers_says_to_pull_them(monkeypatch, tmp_path, capsys):
    world.index().save(tmp_path / "index")
    for vectors in (tmp_path / "index").glob("*/vectors.npy"):
        vectors.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n")
    monkeypatch.setenv("EDGAR_RAG_MODE", "replay")
    monkeypatch.setenv("EDGAR_RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("EDGAR_RAG_REPLAY_TAPE", str(_recorded_tape(tmp_path / "tape")))
    monkeypatch.setattr(cli, "serve", lambda *args, **kwargs: pytest.fail("served pointers"))

    assert cli.main(["serve"]) == 1
    assert "git lfs pull" in capsys.readouterr().err


def test_an_index_of_pointers_is_an_index_format_error(tmp_path):
    world.index().save(tmp_path / "index")
    for vectors in (tmp_path / "index").glob("*/vectors.npy"):
        vectors.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n")

    with pytest.raises(IndexFormatError, match="git lfs pull"):
        open_replay(tmp_path / "index", _recorded_tape(tmp_path / "tape"), README, SPEC)
