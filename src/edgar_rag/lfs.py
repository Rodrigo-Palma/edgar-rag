"""Tell a Git LFS pointer from the file it stands for.

The index and the tape the CI evaluates against keep their largest files in
Git LFS. A clone made without ``git lfs`` installed checks out a three-line
text pointer in place of each one, which would otherwise surface as a SHA-256
mismatch or a malformed row, an error that names the wrong cause. Readers of
those files check for a pointer first and say what to run.
"""

from pathlib import Path

POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
PULL = "run `git lfs install && git lfs pull` in the clone"


def is_pointer(content: bytes) -> bool:
    """Whether ``content`` is a Git LFS pointer rather than the file it points to."""
    return content.startswith(POINTER_PREFIX)


def pointer_message(path: Path) -> str:
    return f"{path} is a Git LFS pointer, not the file it stands for: {PULL}"
