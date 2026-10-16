"""The CI's own tools are pinned like everything else it runs.

Actions are pinned by SHA, base images by digest and dependencies by uv.lock;
an unpinned uv or auditor would be the one input a compromised release could
slip through. Dependabot bumps the uv image in the Dockerfile, so the
workflows follow it from there.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def _dockerfile_uv() -> str:
    match = re.search(r"ghcr\.io/astral-sh/uv:([\d.]+)@", (ROOT / "Dockerfile").read_text())
    assert match, "the Dockerfile no longer copies uv from a pinned image"
    return match.group(1)


def test_every_setup_uv_step_installs_the_uv_the_image_uses():
    uv = _dockerfile_uv()
    steps = [
        step
        for workflow in WORKFLOWS
        for step in re.split(r"\n\s+- ", workflow.read_text())
        if "astral-sh/setup-uv@" in step
    ]

    assert steps
    for step in steps:
        assert re.search(rf'\n\s+version: "{re.escape(uv)}"', step), step


def test_the_auditor_is_an_exact_version():
    audit = (ROOT / "Makefile").read_text()

    assert re.search(r"--from 'pip-audit==\d+\.\d+\.\d+'", audit)
