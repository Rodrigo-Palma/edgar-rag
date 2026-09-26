import pytest

from edgar_rag.chunking import chunk_sections
from edgar_rag.edgar.parse import Section


def test_keeps_a_short_section_whole():
    sections = (Section(item="Item 1", title="Business", text="short body"),)

    chunks = chunk_sections(sections, max_chars=100, overlap=10)

    assert [chunk.text for chunk in chunks] == ["short body"]
    assert chunks[0].chunk_id == "Item 1#0"


def test_windows_overlap_so_a_sentence_is_not_cut_out_of_both():
    sections = (Section(item="Item 7", title="MD&A", text="abcdefghij"),)

    chunks = chunk_sections(sections, max_chars=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij", "ij"]


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
