"""Turn the HTML of a 10-K into the sections a reader would cite."""

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# "Item 1A." and "ITEM 7 -" both appear in filings; the trailing separator varies.
ITEM_PATTERN = re.compile(r"\bItem\s+(\d{1,2}[A-C]?)\s*[.:\-—]", re.IGNORECASE)
MIN_SECTION_CHARS = 200


@dataclass(frozen=True, slots=True)
class Section:
    item: str
    title: str
    text: str


def html_to_text(html: str) -> str:
    """Strip markup, scripts and tables of contents down to readable text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _title_of(text: str, start: int) -> str:
    """Read the words right after an item marker, which name the section."""
    tail = text[start : start + 120].split("\n", 1)[0]
    return tail.strip(" .:-—") or "Untitled"


def split_into_sections(text: str) -> tuple[Section, ...]:
    """Split filing text on its ``Item N`` headings.

    A filing repeats every heading in its table of contents, so a match is only
    treated as a section start when it is followed by enough prose to be one.
    """
    matches = list(ITEM_PATTERN.finditer(text))
    if not matches:
        return (Section(item="Full filing", title="Full filing", text=text),)

    sections: list[Section] = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        sections.append(
            Section(
                item=f"Item {match.group(1).upper()}",
                title=_title_of(text, match.end()),
                text=body,
            )
        )

    if not sections:
        return (Section(item="Full filing", title="Full filing", text=text),)
    return tuple(sections)
