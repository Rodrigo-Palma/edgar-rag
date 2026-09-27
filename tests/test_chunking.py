import pytest

from edgar_rag.chunking import chunk_sections
from edgar_rag.edgar.parse import Section


def test_keeps_a_short_section_whole():
    sections = (Section(item="Item 1", title="Business", text="short body"),)

    chunks = chunk_sections(sections, max_chars=100, overlap=10)

    assert [chunk.text for chunk in chunks] == ["short body"]
    assert chunks[0].chunk_id == "Item 1#0"


def test_windows_overlap_so_a_sentence_is_not_cut_out_of_both():
    body = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    sections = (Section(item="Item 7", title="MD&A", text=body),)

    chunks = chunk_sections(sections, max_chars=24, overlap=8)

    assert len(chunks) > 1
    joined = " ".join(chunk.text for chunk in chunks)
    for word in body.split():
        assert word in joined


def test_no_passage_ends_in_the_middle_of_a_word_or_a_number():
    """156 of 200 passages of a real 10-K used to end mid-word, 19 mid-number."""
    body = "Research and development expense was $ 34,550 million in 2025. " * 12
    sections = (Section(item="Item 7", title="MD&A", text=body),)

    chunks = chunk_sections(sections, max_chars=120, overlap=20)

    for chunk in chunks:
        remainder = body[body.find(chunk.text) + len(chunk.text) :]
        assert not remainder or remainder[0].isspace() or chunk.text[-1] in ".;:"


def test_text_with_no_whitespace_is_still_cut_rather_than_lost():
    sections = (Section(item="Item 7", title="MD&A", text="abcdefghij"),)

    chunks = chunk_sections(sections, max_chars=6, overlap=2)

    assert "".join(chunk.text for chunk in chunks).startswith("abcdef")
    assert len(chunks) > 1


def test_chunk_ids_are_unique_even_when_two_sections_share_a_label():
    """A real 10-K produced "Item 16#0" twice, and the id promises identity."""
    sections = (
        Section(item="Item 16", title="Summary", text="first section body"),
        Section(item="Item 16", title="Summary", text="second section body"),
    )

    chunks = chunk_sections(sections)

    identifiers = [chunk.chunk_id for chunk in chunks]
    assert len(identifiers) == len(set(identifiers))


def test_every_chunk_carries_the_item_it_came_from():
    sections = (
        Section(item="Item 1", title="Business", text="a" * 10),
        Section(item="Item 1A", title="Risk Factors", text="b" * 10),
    )

    chunks = chunk_sections(sections, max_chars=5, overlap=0)

    assert {chunk.item for chunk in chunks} == {"Item 1", "Item 1A"}


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_rejects_a_window_that_would_not_advance(max_chars, overlap):
    sections = (Section(item="Item 1", title="Business", text="body"),)

    with pytest.raises(ValueError):
        chunk_sections(sections, max_chars=max_chars, overlap=overlap)
