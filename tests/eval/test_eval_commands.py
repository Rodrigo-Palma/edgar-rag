import json

import httpx
import pytest

from edgar_rag import cli
from edgar_rag.eval import provenance
from edgar_rag.eval.records import read_run
from tests.eval import harness_fakes as world

MODELS = {
    "models": [
        {"name": "nomic-embed-text:latest", "digest": "0a109f422b47aaaa"},
        {"name": "qwen3:8b", "digest": "500a1f067a9fbbbb"},
    ]
}


class FakeOllama:
    """Ollama's four endpoints the harness calls, counting generations."""

    def __init__(self, digest: str = "500a1f067a9fbbbb") -> None:
        self.generations = 0
        self.digest = digest

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.18.0"})
        if path == "/ready":
            return httpx.Response(200, json={"status": "ready", "weights": "test"})
        if path == "/api/tags":
            models = json.loads(json.dumps(MODELS))
            models["models"][1]["digest"] = self.digest
            return httpx.Response(200, json=models)
        body = json.loads(request.content)
        if path == "/api/embed":
            table = {question.lower(): vector for question, vector in world.VECTORS.items()}
            return httpx.Response(200, json={"embeddings": [table[t] for t in body["input"]]})
        if path == "/decide":
            # Brier: confident only about the passage that holds the revenue.
            yes = 0.9 if "391,035" in body["state"] else 0.1
            return httpx.Response(200, json={"answers": [{"probabilities": [1 - yes, yes]}]})
        if path == "/api/generate":
            self.generations += 1
            reply = {"response": world.ANSWER, "prompt_eval_count": 50, "eval_count": 9}
            return httpx.Response(200, json=reply)
        return httpx.Response(404)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A golden root, an index and a clean environment, in a temporary directory."""
    monkeypatch.chdir(tmp_path)
    for name in ("EDGAR_RAG_BRIER_URL", "EDGAR_RAG_INDEX_DIR", "EDGAR_RAG_GENERATION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    root = world.write_root(tmp_path / "eval")
    world.index().save(tmp_path / "index")
    return tmp_path, root


def _serve(monkeypatch, handler) -> None:
    """Answer every real HTTP request with ``handler``, as if it were Ollama."""

    def handle_request(transport: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        return handler(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)


def _run(tmp_path, root, *extra: str) -> int:
    return cli.main(
        [
            "eval",
            "run",
            "--split",
            "dev",
            "--root",
            str(root),
            "--index-dir",
            str(tmp_path / "index"),
            "--generation-model",
            "qwen3:8b",
            *extra,
        ]
    )


def test_eval_lists_run_and_report(capsys):
    with pytest.raises(SystemExit):
        cli.main(["eval", "--help"])

    out = capsys.readouterr().out
    assert "run" in out and "report" in out


def test_a_recorded_run_replays_without_ollama_to_the_same_outcomes(setup, monkeypatch):
    tmp_path, root = setup
    ollama = FakeOllama()
    _serve(monkeypatch, ollama)

    assert _run(tmp_path, root, "--out", "recorded", "--tape", "tape", "--repeat", "1") == 0
    recorded = read_run(tmp_path / "recorded")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"replay reached {request.url}")

    _serve(monkeypatch, refuse)
    assert _run(tmp_path, root, "--out", "replayed", "--tape", "tape", "--mode", "replay") == 0
    replayed = read_run(tmp_path / "replayed")

    def outcome(run):
        return [(r.id, r.outcome, r.scores, r.generation, r.retrieved) for r in run.cases]

    assert outcome(replayed) == outcome(recorded)
    assert ollama.generations == 3  # two e2e cases and one repeat; replay asked nothing
    assert recorded.manifest.generation is not None
    assert recorded.manifest.generation.digest == "500a1f067a9fbbbb"
    assert replayed.manifest.mode == "replay"
    assert replayed.manifest.generation == recorded.manifest.generation
    assert [r.identical for r in recorded.repeats] == [True]
    assert recorded.manifest.gates == ("period", "cosine")
    assert recorded.manifest.brier is None


def test_a_resumed_recording_does_not_pay_twice(setup, monkeypatch):
    tmp_path, root = setup
    ollama = FakeOllama()
    _serve(monkeypatch, ollama)

    _run(tmp_path, root, "--out", "first", "--tape", "tape")
    _run(tmp_path, root, "--out", "second", "--tape", "tape")

    assert ollama.generations == 2


def test_a_tape_is_not_extended_by_another_build_of_the_model(setup, monkeypatch, capsys):
    tmp_path, root = setup
    _serve(monkeypatch, FakeOllama())
    _run(tmp_path, root, "--out", "first", "--tape", "tape")

    _serve(monkeypatch, FakeOllama(digest="ffffffffffff"))
    assert _run(tmp_path, root, "--out", "second", "--tape", "tape") == 1
    assert "record into a fresh tape" in capsys.readouterr().err


def test_a_replay_miss_fails_with_the_re_record_hint(setup, monkeypatch, capsys):
    tmp_path, root = setup
    _serve(monkeypatch, FakeOllama())
    _run(tmp_path, root, "--out", "gate", "--tape", "tape", "--gate-only")

    # The tape holds embeddings only: an e2e replay needs generations.
    assert _run(tmp_path, root, "--out", "e2e", "--tape", "tape", "--mode", "replay") == 1
    assert "re-record locally" in capsys.readouterr().err


def test_a_gate_only_run_asks_no_model_and_reports(setup, monkeypatch, capsys):
    tmp_path, root = setup
    ollama = FakeOllama()
    _serve(monkeypatch, ollama)

    assert _run(tmp_path, root, "--out", "gate", "--gate-only") == 0
    run = read_run(tmp_path / "gate")

    assert ollama.generations == 0
    assert run.manifest.tier == "gate-only" and run.manifest.generation is None
    assert {r.outcome for r in run.cases} == {"scored_only"}


def test_the_report_command_gives_the_same_bytes_twice(setup, monkeypatch, capsys):
    tmp_path, root = setup
    _serve(monkeypatch, FakeOllama())
    _run(tmp_path, root, "--out", "run")
    capsys.readouterr()

    assert cli.main(["eval", "report", "--run", "run"]) == 0
    first = capsys.readouterr().out
    assert cli.main(["eval", "report", "--run", "run", "--out", "report.md"]) == 0

    assert first.startswith("# Evaluation report: dev split, e2e tier")
    assert (tmp_path / "report.md").read_text() == first


def test_the_eval_split_is_held_out_without_the_protocol(setup, capsys):
    _, root = setup

    code = cli.main(["eval", "run", "--split", "eval", "--root", str(root)])

    assert code == 1
    assert "pre-registered" in capsys.readouterr().err


def test_repeats_cannot_be_measured_from_a_replay(setup, capsys):
    tmp_path, root = setup

    assert _run(tmp_path, root, "--mode", "replay", "--repeat", "3") == 1
    assert "live model" in capsys.readouterr().err


def test_a_report_of_a_missing_run_fails_cleanly(setup, capsys):
    assert cli.main(["eval", "report", "--run", "nowhere"]) == 1
    assert "run:" in capsys.readouterr().err


def test_brier_scores_are_recorded_and_replayed_when_the_plugin_is_set(setup, monkeypatch, capsys):
    tmp_path, root = setup
    _serve(monkeypatch, FakeOllama())
    monkeypatch.setenv("EDGAR_RAG_BRIER_URL", "http://127.0.0.1:8100")
    monkeypatch.setenv("EDGAR_RAG_BRIER_SHA", "d70e7df")

    assert _run(tmp_path, root, "--out", "recorded", "--tape", "tape") == 0
    monkeypatch.delenv("EDGAR_RAG_BRIER_URL")
    _serve(monkeypatch, lambda request: httpx.Response(599))
    assert _run(tmp_path, root, "--out", "replayed", "--tape", "tape", "--mode", "replay") == 0

    recorded, replayed = read_run(tmp_path / "recorded"), read_run(tmp_path / "replayed")
    assert recorded.manifest.gates == ("period", "cosine", "brier")
    assert recorded.manifest.brier is not None and recorded.manifest.brier.sha == "d70e7df"
    assert recorded.manifest.brier.ready == {"status": "ready", "weights": "test"}
    assert [r.scores for r in replayed.cases] == [r.scores for r in recorded.cases]
    assert recorded.cases[0].scores["brier"] == 0.9
    assert replayed.manifest.brier == recorded.manifest.brier
    capsys.readouterr()
    cli.main(["eval", "report", "--run", "replayed"])
    assert "| C brier | not run" not in capsys.readouterr().out


def test_the_commit_is_read_before_the_run_starts_not_after(setup, monkeypatch):
    tmp_path, root = setup
    ollama = FakeOllama()
    _serve(monkeypatch, ollama)
    # A commit made while a run takes hours must not be credited with it.
    monkeypatch.setattr(
        provenance, "repo_state", lambda root: (f"after-{ollama.generations}-generations", False)
    )

    _run(tmp_path, root, "--out", "run")

    assert read_run(tmp_path / "run").manifest.repo_sha == "after-0-generations"
