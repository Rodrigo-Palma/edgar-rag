"""The demo GIF the README opens with stays light, and stays in plain git.

``docs/media/demo.gif`` is recorded by ``make demo-record`` from
``docs/media/demo.tape``. It is committed as a regular blob: a GIF in Git LFS
would show as a pointer to anyone who clones without pulling it.
"""

from pathlib import Path

MEDIA = Path(__file__).resolve().parents[1] / "docs" / "media"
GIF = MEDIA / "demo.gif"
TAPE = MEDIA / "demo.tape"
MAX_GIF_BYTES = 2 * 1024 * 1024


def test_the_demo_gif_stays_under_two_megabytes():
    assert GIF.stat().st_size <= MAX_GIF_BYTES


def test_the_demo_gif_is_a_gif_and_not_an_lfs_pointer():
    assert GIF.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_the_tape_writes_the_gif_the_readme_shows():
    assert "Output docs/media/demo.gif" in TAPE.read_text("utf-8").splitlines()
