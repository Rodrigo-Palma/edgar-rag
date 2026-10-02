"""Which passages an answer cites, whether they back it, and what a reader should see."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from edgar_rag.amounts import Amount, amount_matches, amounts_in
from edgar_rag.domain import AbstentionReason, Citation, ScoredChunk

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
MAX_MARKER_DIGITS = 6
MAX_UNCITED_CONTENT_WORDS = 8
_SCALES = (Decimal(1), Decimal(10) ** 3, Decimal(10) ** 6, Decimal(10) ** 9)
_SCALE_WORD = re.compile(r"thousand|million|billion|trillion", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_LIST_LEAD = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+")
_ONLY_MARKERS = re.compile(r"^[\s.,;:]*(?:\[\s*-?\d*\s*\][\s.,;:]*)+$")
_ANY_MARKER = re.compile(r"\[\s*-?\d*\s*\]")
# Numbers that name a document or its parts rather than state a figure.
_NOT_A_FIGURE = re.compile(
    r"\b(?:10-K|10-Q|8-K|20-F|40-F|6-K|S-\d+|DEF\s*14A)(?:/A)?\b"
    r"|\bitems?\s+\d+[a-c]?\b"
    r"|\b\d+(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)


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
    retrieved, so it is dropped rather than shown as a source. A marker longer
    than ``MAX_MARKER_DIGITS`` is dropped before ``int()`` sees it: past 4300
    digits ``int()`` raises, and a reply is model output, not trusted input.
    """
    return frozenset(
        number
        for number in (
            int(digits)
            for digits in MARKER_PATTERN.findall(text)
            if len(digits) <= MAX_MARKER_DIGITS
        )
        if 1 <= number <= available
    )


@dataclass(frozen=True, slots=True)
class ClaimFailure:
    """Why a cited answer is still withheld, and the sentence that failed."""

    reason: AbstentionReason
    finding: str


def check_claims(text: str, passages: tuple[ScoredChunk, ...]) -> ClaimFailure | None:
    """Hold each sentence of an answer to the passages it cites, or say why not.

    A single valid marker used to approve the whole answer, so a second
    sentence could state any figure, and a marker could point at a passage
    that never wrote it. Two rules, both lexical and free:

    1. A sentence that states a figure (has a digit) or says enough to be a
       claim (more than ``MAX_UNCITED_CONTENT_WORDS`` content words) carries
       at least one valid marker, else ``no_valid_citation``.
    2. Every amount in that sentence appears in one of the passages it cites,
       else ``unsupported_claim``. "Appears" means the passage writes the same
       value at the precision the answer shows: ``$31.4 billion`` is backed by
       ``31,370`` under an "(in millions)" header, ``$31,371 million`` is not.

    What this does not check: wording without digits (a cited sentence can
    still paraphrase wrongly), figures the model computed (a growth rate the
    passage does not print is withheld, correct or not), and the magnitude of
    a passage figure printed without a scale word, which is read at any scale
    from ones to billions because tables state it once in a header.
    """
    support = tuple(_amounts_supporting(scored) for scored in passages)
    for position, sentence in enumerate(_answer_sentences(text), start=1):
        claim = _claim_text(sentence)
        cited = markers_in(sentence, len(passages))
        if not cited:
            if _needs_marker(claim):
                return ClaimFailure(
                    AbstentionReason.NO_VALID_CITATION,
                    f"sentence {position} states a fact without a valid marker",
                )
            continue
        for amount in amounts_in(claim):
            if not any(_supports(amount, found) for n in cited for found in support[n - 1]):
                where = ", ".join(f"[{n}]" for n in sorted(cited))
                return ClaimFailure(
                    AbstentionReason.UNSUPPORTED_CLAIM,
                    f"sentence {position} states {amount.text}, which {where} does not contain",
                )
    return None


def _answer_sentences(text: str) -> list[str]:
    """Sentences of an answer, with list numbering removed and stray markers reattached.

    Lines split first, so a numbered list item is one sentence and its "1."
    is dropped instead of being read as a figure. A marker the model put after
    the full stop ("... $9 billion. [1]") belongs to the sentence before it.
    """
    sentences: list[str] = []
    for line in text.splitlines():
        for part in _SENTENCE_BREAK.split(_LIST_LEAD.sub("", line)):
            if not part.strip():
                continue
            if sentences and _ONLY_MARKERS.match(part):
                sentences[-1] = f"{sentences[-1]} {part}"
            else:
                sentences.append(part)
    return sentences


def _claim_text(sentence: str) -> str:
    """The sentence as a statement: no markers, form names, item labels or ordinals."""
    folded = unicodedata.normalize("NFKC", sentence)
    return _NOT_A_FIGURE.sub(" ", _ANY_MARKER.sub(" ", folded))


def _needs_marker(claim: str) -> bool:
    return any(char.isdigit() for char in claim) or (
        len(_content_words(claim)) > MAX_UNCITED_CONTENT_WORDS
    )


def _amounts_supporting(scored: ScoredChunk) -> tuple[Amount, ...]:
    """Every amount the model was shown for this passage, its item label included."""
    chunk = scored.chunk
    return amounts_in(unicodedata.normalize("NFKC", f"{chunk.item} {chunk.text}"))


def _scales(amount: Amount) -> tuple[Decimal, ...]:
    """Unscaled numbers could be in thousands, millions or billions; scaled ones cannot."""
    return (Decimal(1),) if _SCALE_WORD.search(amount.text) else _SCALES


def _supports(claimed: Amount, found: Amount) -> bool:
    """Whether ``found`` rounds to ``claimed`` at the precision ``claimed`` shows.

    Signs are compared as magnitudes, because "a loss of $1,234 million" and
    "(1,234)" are the same figure in words and in accounting notation.
    """
    return any(
        amount_matches(
            Amount(value=abs(claimed.value) * mine, unit=claimed.unit * mine, text=claimed.text),
            abs(found.value) * theirs,
            relative_tolerance=Decimal(0),
        )
        for mine in _scales(claimed)
        for theirs in _scales(found)
    )
