"""Which passages an answer cites, and the part of each a reader should see."""

import re

from edgar_rag.domain import Citation, ScoredChunk

STOPWORDS = frozenset(
    {
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "have",
        "were",
        "been",
        "they",
        "their",
        "which",
        "what",
        "when",
        "does",
        "much",
        "about",
        "company",
        "than",
        "then",
        "there",
        "here",
        "shall",
        "will",
        "would",
        "could",
        "other",
    }
)
MARKER_PATTERN = re.compile(r"\[(\d+)\]")


def quote_for(text: str, question: str, limit: int = 400) -> str:
    """The part of the passage a reader should look at, not just its opening.

    Quoting the opening of the passage was measured to miss the cited fact in
    5 of 5 gold passages of a real 10-K: the numbers sit 350 to 930 characters
    in, because the section opens with prose and the figures follow. This
    returns a window of ``limit`` characters (400 by default) around the
    sentence that shares the most content words with the question, and falls
    back to the opening when nothing overlaps.
    """
    sentences = _sentences(text)
    if not sentences:
        return text[:limit].strip()

    wanted = _content_words(question)
    best = max(sentences, key=lambda s: len(wanted & _content_words(s)))
    if not wanted & _content_words(best):
        best = sentences[0]

    start = max(0, text.find(best) - limit // 4)
    window = text[start : start + limit].strip()
    return ("..." if start else "") + window + ("..." if start + limit < len(text) else "")


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.;:])\s+|\n+", text)]
    return [part for part in parts if len(part) > 20]


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", text.lower())} - STOPWORDS


def as_citations(
    passages: tuple[ScoredChunk, ...], question: str, cited: frozenset[int]
) -> tuple[Citation, ...]:
    """Only the passages the answer actually pointed at.

    Returning all of top_k regardless meant the caller received passages the
    generated text never referred to, which makes "cites the passage it used"
    untrue by construction.
    """
    return tuple(
        Citation(
            marker=position,
            item=scored.chunk.item,
            title=scored.chunk.title,
            quote=quote_for(scored.chunk.text, question),
            score=round(scored.score, 4),
        )
        for position, scored in enumerate(passages, start=1)
        if position in cited
    )


def markers_in(text: str, available: int) -> frozenset[int]:
    """Which passage numbers the answer refers to, ignoring invented ones.

    A marker outside 1..available is a citation of something that was not
    retrieved, so it is dropped rather than shown as a source.
    """
    return frozenset(
        number
        for number in (int(match) for match in MARKER_PATTERN.findall(text))
        if 1 <= number <= available
    )
