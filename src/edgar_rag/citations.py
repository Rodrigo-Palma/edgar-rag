"""Which passages an answer cites, whether they back it, and what a reader should see."""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from edgar_rag.amounts import Amount, amount_matches, amounts_in, spelled_amounts_in
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
_SENTENCE_BREAK = re.compile(r"[.!?]\s+")
# A full stop that ends one of these is not the end of a sentence: "U.S.",
# "e.g.", "Inc." split a cited sentence and left its first half uncited.
_ABBREVIATION = re.compile(
    r"(?:\b(?:[A-Za-z]\.){2,}"
    r"|\b(?:inc|corp|co|ltd|no|nos|vs|approx|mr|ms|dr|st|jan|feb|mar|apr|jun|jul|aug"
    r"|sep|sept|oct|nov|dec|fig|ref)\.)$",
    re.IGNORECASE,
)
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
# The same, and the references a filing makes to its own parts: "See Note 3 on
# page 21", "Exhibit 10.1", "Section 404". A note, page or item number is an
# integer not followed by a decimal or a percent sign, so "Notes 2.5%" stays.
_REFERENCE = re.compile(
    _NOT_A_FIGURE.pattern + r"|\b(?:notes?|pages?)\s+\d{1,4}[a-z]?\b(?![.,]\d|\s*%)"
    r"|\b(?:exhibits?|sections?)\s+\d+(?:\.\d+)*[a-z]?\b",
    re.IGNORECASE,
)
MIN_DIGITS_TO_RESCALE = 3


@dataclass(frozen=True, slots=True)
class SupportRules:
    """What counts as a passage backing a figure.

    ``CURRENT`` is what the service runs. ``V1_0`` is the check v1.0.0 shipped,
    kept only so the frozen v1 run can be replayed under it and the change
    measured (``docs/eval/protocol-v1.1.md``); nothing serves answers with it.
    """

    read_item_label: bool
    strip_passage_references: bool
    strip_claim_references: bool
    percent_backs_only_percent: bool
    min_digits_to_rescale: int
    read_spelled: bool


V1_0 = SupportRules(
    read_item_label=True,
    strip_passage_references=False,
    strip_claim_references=False,
    percent_backs_only_percent=False,
    min_digits_to_rescale=0,
    read_spelled=False,
)
CURRENT = SupportRules(
    read_item_label=False,
    strip_passage_references=True,
    strip_claim_references=True,
    percent_backs_only_percent=True,
    min_digits_to_rescale=MIN_DIGITS_TO_RESCALE,
    read_spelled=True,
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


def check_claims(
    text: str, passages: tuple[ScoredChunk, ...], rules: SupportRules = CURRENT
) -> ClaimFailure | None:
    """Hold each sentence of an answer to the passages it cites, or say why not.

    A single valid marker used to approve the whole answer, so a second
    sentence could state any figure, and a marker could point at a passage
    that never wrote it. Two rules, both lexical and free:

    1. A sentence that states a figure (a digit, or a figure in words) or says
       enough to be a claim (more than ``MAX_UNCITED_CONTENT_WORDS`` content
       words) carries at least one valid marker, else ``no_valid_citation``.
    2. Every amount in that sentence appears in one of the passages it cites,
       else ``unsupported_claim``. "Appears" means the passage writes the same
       value at the precision the answer shows: ``$31.4 billion`` is backed by
       ``31,370`` under an "(in millions)" header, ``$31,371 million`` is not.

    What a passage offers as support: the numbers of its text, not of its item
    label, with references to its own parts (``Note 3``, ``page 21``,
    ``Exhibit 10.1``) removed; a percentage backs only a percentage; a number
    printed with no scale word is read at another scale only when it has at
    least ``MIN_DIGITS_TO_RESCALE`` significant digits, because a table states
    its unit once in a header but ``12 members`` is not twelve billion.

    The finding names the sentence and its markers, never the figure: the
    figure is the part of the text being withheld.

    What this does not check: wording without figures (a cited sentence can
    still paraphrase wrongly), and figures the model computed (a growth rate
    the passage does not print is withheld, correct or not).
    """
    support = tuple(_amounts_supporting(scored, rules) for scored in passages)
    for position, sentence in enumerate(_answer_sentences(text), start=1):
        claim = _claim_text(sentence, rules)
        cited = markers_in(sentence, len(passages))
        if not cited:
            if _needs_marker(claim, rules):
                return ClaimFailure(
                    AbstentionReason.NO_VALID_CITATION,
                    f"sentence {position} states a fact without a valid marker",
                )
            continue
        for amount in _amounts(claim, rules):
            if not any(_supports(amount, found, rules) for n in cited for found in support[n - 1]):
                where = ", ".join(f"[{n}]" for n in sorted(cited))
                return ClaimFailure(
                    AbstentionReason.UNSUPPORTED_CLAIM,
                    f"sentence {position} states a figure that {where} does not contain",
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
        for part in _split_line(_LIST_LEAD.sub("", line)):
            if not part.strip():
                continue
            if sentences and _ONLY_MARKERS.match(part):
                sentences[-1] = f"{sentences[-1]} {part}"
            else:
                sentences.append(part)
    return sentences


def _split_line(line: str) -> list[str]:
    """Split at a full stop, question or exclamation mark, except after an abbreviation."""
    parts: list[str] = []
    start = 0
    for found in _SENTENCE_BREAK.finditer(line):
        head = line[start : found.start() + 1]
        if _ABBREVIATION.search(head):
            continue
        parts.append(head)
        start = found.end()
    parts.append(line[start:])
    return parts


def _claim_text(sentence: str, rules: SupportRules = CURRENT) -> str:
    """The sentence as a statement: no markers, form names, references or ordinals."""
    folded = unicodedata.normalize("NFKC", sentence)
    pattern = _REFERENCE if rules.strip_claim_references else _NOT_A_FIGURE
    return pattern.sub(" ", _ANY_MARKER.sub(" ", folded))


def _amounts(text: str, rules: SupportRules) -> tuple[Amount, ...]:
    spelled = spelled_amounts_in(text) if rules.read_spelled else ()
    return amounts_in(text) + spelled


def _needs_marker(claim: str, rules: SupportRules = CURRENT) -> bool:
    return (
        any(char.isdigit() for char in claim)
        or (rules.read_spelled and bool(spelled_amounts_in(claim)))
        or len(_content_words(claim)) > MAX_UNCITED_CONTENT_WORDS
    )


def _amounts_supporting(scored: ScoredChunk, rules: SupportRules = CURRENT) -> tuple[Amount, ...]:
    """Every amount the passage states, under ``rules``."""
    chunk = scored.chunk
    shown = f"{chunk.item} {chunk.text}" if rules.read_item_label else chunk.text
    folded = unicodedata.normalize("NFKC", shown)
    if rules.strip_passage_references:
        folded = _REFERENCE.sub(" ", folded)
    return _amounts(folded, rules)


def _is_scaled(amount: Amount) -> bool:
    return bool(_SCALE_WORD.search(amount.text))


def _scales(amount: Amount, rules: SupportRules) -> tuple[Decimal, ...]:
    """Unscaled numbers could be in thousands, millions or billions; scaled ones cannot.

    Nor can a percentage, once percentages are their own kind.
    """
    if _is_scaled(amount) or (rules.percent_backs_only_percent and amount.is_percent):
        return (Decimal(1),)
    return _SCALES


def _significant_digits(amount: Amount) -> int:
    return len(re.sub(r"\D", "", amount.text).lstrip("0"))


def _supports(claimed: Amount, found: Amount, rules: SupportRules = CURRENT) -> bool:
    """Whether ``found`` rounds to ``claimed`` at the precision ``claimed`` shows.

    Signs are compared as magnitudes, because "a loss of $1,234 million" and
    "(1,234)" are the same figure in words and in accounting notation.
    """
    if rules.percent_backs_only_percent and claimed.is_percent != found.is_percent:
        return False
    rescalable = _significant_digits(found) >= rules.min_digits_to_rescale
    return any(
        amount_matches(
            Amount(value=abs(claimed.value) * mine, unit=claimed.unit * mine, text=claimed.text),
            abs(found.value) * theirs,
            relative_tolerance=Decimal(0),
        )
        for mine in _scales(claimed, rules)
        for theirs in _scales(found, rules)
        if theirs == 1 or rescalable
    )
