import numpy as np
import pytest

from edgar_rag.domain import Chunk
from edgar_rag.index import FilingIndex, build_index


def test_search_ranks_the_closest_chunk_first(index):
    results = index.search(np.asarray([0.0, 1.0], dtype=np.float32), top_k=2)

    assert [scored.chunk.item for scored in results] == ["Item 1A", "Item 1"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


def test_scores_are_cosine_so_length_does_not_decide(index):
    short = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)
    long = index.search(np.asarray([50.0, 0.0], dtype=np.float32), top_k=1)

    assert short[0].score == pytest.approx(long[0].score)


def test_a_round_trip_through_disk_keeps_chunks_and_scores(index, tmp_path):
    index.save(tmp_path)

    loaded = FilingIndex.load(tmp_path)

    assert loaded.source == index.source
    assert loaded.chunks == index.chunks
    assert loaded.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)[
        0
    ].score == pytest.approx(1.0)


def test_loading_from_an_empty_directory_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        FilingIndex.load(tmp_path)


def test_building_rejects_a_mismatch_between_chunks_and_vectors():
    chunks = (Chunk(chunk_id="a", item="Item 1", title="Business", text="text"),)

    with pytest.raises(ValueError, match="1 chunks"):
        build_index({}, chunks, np.zeros((2, 3), dtype=np.float32))


def test_building_rejects_an_index_with_no_chunks():
    with pytest.raises(ValueError, match="at least one chunk"):
        build_index({}, (), np.zeros((0, 2), dtype=np.float32))
