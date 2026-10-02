import subprocess

import httpx

from edgar_rag.eval import provenance

GIT = ["git", "-c", "user.name=Test", "-c", "user.email=test@ledgerworks.io"]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_outside_a_repository_the_commit_is_unknown(tmp_path):
    assert provenance.repo_state(tmp_path) == ("unknown", False)


def test_a_git_dir_exported_by_a_rebase_does_not_redirect_the_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "elsewhere" / ".git"))

    assert provenance.repo_state(tmp_path) == ("unknown", False)


def _git(*args: str) -> None:
    # Without the overrides: under `git rebase --exec` GIT_DIR names the
    # repository being rebased, and these commands would write to it.
    subprocess.run([*GIT, *args], check=True, env=provenance.without_git_overrides())


def test_the_commit_and_whether_the_tree_differs_untracked_files_included(tmp_path):
    _git("init", "-q", str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1\n")
    _git("-C", str(tmp_path), "add", "a.py")
    _git("-C", str(tmp_path), "commit", "-q", "-m", "a")

    sha, dirty = provenance.repo_state(tmp_path)
    assert len(sha) == 40 and not dirty

    (tmp_path / "new_module.py").write_text("y = 2\n")
    assert provenance.repo_state(tmp_path) == (sha, True)


def test_a_model_name_without_a_tag_is_its_latest_digest():
    def tags(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "nomic:latest", "digest": "abc"}]})

    with _client(tags) as client:
        assert provenance.ollama_model(client, "http://o", "nomic").digest == "abc"
        assert provenance.ollama_model(client, "http://o", "other:8b").digest is None


def test_an_unreachable_server_is_reported_as_unknown_not_raised():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with _client(down) as client:
        assert provenance.ollama_model(client, "http://o", "m").digest is None
        assert provenance.ollama_version(client, "http://o") is None
        assert provenance.brier_ready(client, "http://b") is None


def test_ollama_version_and_brier_readiness_are_read_from_their_endpoints():
    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.18.0"})
        return httpx.Response(200, json=["not", "an", "object"])

    with _client(serve) as client:
        assert provenance.ollama_version(client, "http://o/") == "0.18.0"
        assert provenance.brier_ready(client, "http://b") is None


def test_the_hardware_is_named():
    assert provenance.hardware()
