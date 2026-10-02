"""Where a run came from: the commit, the files, the models and the machine.

Each probe answers ``None`` (or "unknown") rather than failing: a manifest
that says "unknown" is honest, and a run should not die because ``git`` is
not on the path. What is unknown is printed as such in the report.
"""

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import httpx

from edgar_rag.eval.records import ModelInfo

PROBE_TIMEOUT_SECONDS = 5.0


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_git_overrides() -> dict[str, str]:
    """The environment minus ``GIT_DIR`` and the like.

    ``git rebase --exec`` and hooks export them, and with them set ``git -C
    root`` reads (or, for a write, changes) the repository they name instead
    of the one under ``root``.
    """
    return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            env=without_git_overrides(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def repo_state(root: Path) -> tuple[str, bool]:
    """The commit checked out under ``root``, and whether the tree differs from it.

    A new module nobody has added yet is a difference too, so untracked files
    count; ignored ones (data, caches) do not.
    """
    sha = _git(root, "rev-parse", "HEAD") or "unknown"
    status = _git(root, "status", "--porcelain")
    return sha, bool(status)


def hardware() -> str:
    """The CPU brand where the OS reports one (``Apple M3 Max``), else the machine type."""
    if platform.system() == "Darwin":
        try:
            done = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            if done.stdout.strip():
                return f"{done.stdout.strip()}, {platform.system()} {platform.release()}"
        except (OSError, subprocess.SubprocessError):
            pass
    return f"{platform.machine()}, {platform.system()} {platform.release()}"


def ollama_version(client: httpx.Client, base_url: str) -> str | None:
    try:
        reply = client.get(f"{base_url.rstrip('/')}/api/version", timeout=PROBE_TIMEOUT_SECONDS)
        version = reply.raise_for_status().json().get("version")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    return str(version) if version else None


def ollama_model(client: httpx.Client, base_url: str, name: str) -> ModelInfo:
    """The digest Ollama resolves ``name`` to; ``nomic-embed-text`` means ``:latest``."""
    wanted = name if ":" in name else f"{name}:latest"
    try:
        reply = client.get(f"{base_url.rstrip('/')}/api/tags", timeout=PROBE_TIMEOUT_SECONDS)
        models = reply.raise_for_status().json().get("models", [])
    except (httpx.HTTPError, ValueError, AttributeError):
        return ModelInfo(name=name, digest=None)
    for model in models:
        if isinstance(model, dict) and model.get("name") == wanted:
            digest = model.get("digest")
            return ModelInfo(name=name, digest=str(digest) if digest else None)
    return ModelInfo(name=name, digest=None)
