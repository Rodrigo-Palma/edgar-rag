"""Cut sections into passages small enough to embed and cite."""

from dataclasses import dataclass

from edgar_rag.edgar.parse import Section

DEFAULT_MAX_CHARS = 1_200
DEFAULT_OVERLAP_CHARS = 150


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    item: str
    title: str
    text: str


def _windows(text: str, max_chars: int, overlap: int) -> tuple[str, ...]:
    if len(text) <= max_chars:
        return (text,)

    step = max_chars - overlap
    pieces = [text[start : start + max_chars] for start in range(0, len(text), step)]
    return tuple(piece for piece in pieces if piece.strip())


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
    for section in sections:
        for position, window in enumerate(_windows(section.text, max_chars, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{section.item}#{position}",
                    item=section.item,
                    title=section.title,
                    text=window.strip(),
                )
            )
    return tuple(chunks)
