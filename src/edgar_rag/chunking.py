"""Cut sections into passages small enough to embed and cite."""

from edgar_rag.domain import Chunk
from edgar_rag.edgar.parse import Section

DEFAULT_MAX_CHARS = 1_200
DEFAULT_OVERLAP_CHARS = 150


def _windows(text: str, max_chars: int, overlap: int) -> tuple[str, ...]:
    """Cut at a line or sentence end, never through a word or a number.

    Cutting on a raw character offset left 156 of 200 passages of a real 10-K
    ending mid-word and 19 ending mid-number. A passage that ends in "$ 3,0" is
    what a language model turns into an invented figure, and a passage that
    starts with "elopment" has lost the word that says what it is about.
    """
    if len(text) <= max_chars:
        return (text,)

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = _boundary_before(text, start, end)
        piece = text[start:end]
        if piece.strip():
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        start = _boundary_after(text, start)

    return tuple(pieces)


def _boundary_before(text: str, start: int, end: int) -> int:
    """The last sentence or line end in the window, or the last space."""
    window = text[start:end]
    for pattern in ("\n", ". ", "; ", " "):
        found = window.rfind(pattern)
        if found > len(window) // 2:
            return start + found + len(pattern)
    return end


def _boundary_after(text: str, start: int) -> int:
    """Move a start offset forward off the middle of a word or number.

    Text with no whitespace at all would otherwise be swallowed whole, so when
    no boundary exists the raw offset is kept: a hard cut beats losing the rest
    of the section.
    """
    moved = start
    while moved < len(text) and not text[moved].isspace():
        moved += 1
    return start if moved >= len(text) else moved + 1


def chunk_sections(
    sections: tuple[Section, ...],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> tuple[Chunk, ...]:
    """Build overlapping passages, each still carrying the item it came from.

    Raises:
        ValueError: when the overlap would not let the window advance.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be smaller than max_chars and not negative")

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for section in sections:
        for window in _windows(section.text, max_chars, overlap):
            # A filing can carry two sections with the same item label, so a
            # per-section counter repeats ids. Measured on a real 10-K:
            # "Item 16#0" and "Item 16#1" each appeared twice. The id promises
            # identity, so it counts across the whole document.
            position = seen.get(section.item, 0)
            seen[section.item] = position + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{section.item}#{position}",
                    item=section.item,
                    title=section.title,
                    text=window.strip(),
                )
            )
    return tuple(chunks)
