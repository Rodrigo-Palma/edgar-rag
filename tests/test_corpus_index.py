"""Index format 2: one shard per filing, a manifest, and the embedder it was built with."""

import hashlib
import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from edgar_rag.domain import Chunk, EmbedderSpec, IndexedFiling, Scope
from edgar_rag.index import (
    CorpusIndex,
    IndexFormatError,
    build_shard,
    write_shard,
)

SPEC = EmbedderSpec(model="nomic-embed-text", lowercase=True)
APPLE, COKE = 320193, 21344


def _filing(cik: int = APPLE, fiscal_year: int = 2024, accession: str | None = None):
    return IndexedFiling(
        accession=accession or f"{cik:010d}-{fiscal_year % 100:02d}-000001",
        cik=cik,
        fiscal_year=fiscal_year,
        form="10-K",
        company=f"Company {cik}",
        filing_date=date(fiscal_year, 11, 1),
        url=f"https://www.sec.gov/Archives/edgar/data/{cik}/doc-{fiscal_year}.htm",
    )


def _chunks(*texts: str) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(chunk_id=f"Item 1#{n}", item="Item 1", title="Business", text=text)
        for n, text in enumerate(texts)
    )


def _shard(cik: int = APPLE, fiscal_year: int = 2024, vectors=((1.0, 0.0), (0.0, 1.0))):
    texts = tuple(f"{cik} {fiscal_year} passage {n}" for n in range(len(vectors)))
    return build_shard(
        _filing(cik, fiscal_year), _chunks(*texts), np.asarray(vectors, dtype=np.float32)
    )


def _corpus(*shards) -> CorpusIndex:
    return CorpusIndex.of(SPEC, shards or (_shard(),))


def _query(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def _rewrite_manifest(root: Path, change) -> None:
    payload = _manifest(root)
    change(payload)
    (root / "manifest.json").write_text(json.dumps(payload))


def _replace_shard_file(root: Path, accession: str, name: str, content: bytes) -> None:
    """Swap a shard file and fix its hash, so only the content check can catch it."""
    (root / accession / name).write_bytes(content)
    key = "vectors_sha256" if name == "vectors.npy" else "chunks_sha256"
    digest = hashlib.sha256(content).hexdigest()

    def update(payload: dict) -> None:
        for entry in payload["filings"]:
            if entry["accession"] == accession:
                entry[key] = digest

    _rewrite_manifest(root, update)


def _npy(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


# Scope


def test_a_search_never_returns_a_passage_of_another_filing():
    """Coca-Cola holds the exact match; a question scoped to Apple must not see it."""
    apple = _shard(APPLE, 2024, vectors=((0.6, 0.8), (0.0, 1.0)))
    coke = _shard(COKE, 2024, vectors=((1.0, 0.0), (1.0, 0.0)))
    index = _corpus(apple, coke)

    results = index.search(_query(1.0, 0.0), Scope(APPLE, 2024), top_k=4)

    assert [scored.chunk for scored in results] == [apple.chunks[0], apple.chunks[1]]
    assert all(scored.score < 0.99 for scored in results)


def test_a_scope_without_a_fiscal_year_is_the_latest_one_indexed():
    index = _corpus(_shard(APPLE, 2024), _shard(APPLE, 2025), _shard(COKE, 2026))

    assert index.resolve(Scope(APPLE)) == _filing(APPLE, 2025)
    assert index.search(_query(1.0, 0.0), Scope(APPLE), top_k=1)[0].chunk.chunk_id.startswith(
        _filing(APPLE, 2025).accession
    )


def test_a_scope_with_a_fiscal_year_is_that_year():
    index = _corpus(_shard(APPLE, 2024), _shard(APPLE, 2025))

    assert index.resolve(Scope(APPLE, 2024)) == _filing(APPLE, 2024)


@pytest.mark.parametrize("scope", [Scope(COKE), Scope(APPLE, 2019)], ids=["company", "year"])
def test_a_scope_with_no_indexed_filing_resolves_to_none_and_finds_nothing(scope):
    index = _corpus(_shard(APPLE, 2024))

    assert index.resolve(scope) is None
    assert index.search(_query(1.0, 0.0), scope) == ()


def test_chunk_ids_carry_the_accession_so_they_stay_unique_across_filings():
    index = _corpus(_shard(APPLE, 2024), _shard(COKE, 2024))

    ids = [chunk.chunk_id for shard in index.shards for chunk in shard.chunks]

    assert len(set(ids)) == len(ids) == 4
    for shard in index.shards:
        assert all(
            chunk.chunk_id.startswith(f"{shard.filing.accession}:Item 1#") for chunk in shard.chunks
        )


@pytest.mark.parametrize(
    ("cik", "fiscal_year"), [(0, None), (-1, None), (True, None), (1, 0), (1, False)]
)
def test_a_scope_refuses_what_cannot_be_a_cik_or_a_year(cik, fiscal_year):
    with pytest.raises(ValueError):
        Scope(cik, fiscal_year)


def test_a_scope_reads_as_the_company_and_year_it_names():
    assert str(Scope(APPLE, 2024)) == "CIK 320193, fiscal 2024"
    assert str(Scope(APPLE)) == "CIK 320193, the latest fiscal year"


# Search


def test_scores_are_cosine_so_length_does_not_decide():
    index = _corpus()

    short = index.search(_query(1.0, 0.0), Scope(APPLE), top_k=1)
    long = index.search(_query(50.0, 0.0), Scope(APPLE), top_k=1)

    assert short[0].score == pytest.approx(long[0].score) == pytest.approx(1.0)


def test_a_top_k_larger_than_the_filing_returns_all_of_it_best_first():
    results = _corpus().search(_query(0.0, 1.0), Scope(APPLE), top_k=10)

    assert [round(scored.score, 3) for scored in results] == [1.0, 0.0]


def test_a_top_k_below_one_is_refused():
    with pytest.raises(ValueError, match="top_k"):
        _corpus().search(_query(1.0, 0.0), Scope(APPLE), top_k=0)


def test_a_query_of_another_size_is_refused_instead_of_broadcast():
    with pytest.raises(ValueError, match="3 dimensions"):
        _corpus().search(_query(1.0, 0.0, 0.0), Scope(APPLE))


def test_the_digest_changes_with_the_vectors():
    first = _corpus(_shard(vectors=((1.0, 0.0), (0.0, 1.0))))
    second = _corpus(_shard(vectors=((0.0, 1.0), (1.0, 0.0))))

    assert len(first.digest()) == 16
    assert first.digest() != second.digest()


# Building


def test_a_shard_stores_unit_length_vectors():
    shard = _shard(vectors=((3.0, 4.0), (0.0, 2.0)))

    assert np.allclose(np.linalg.norm(shard.vectors, axis=1), 1.0)


@pytest.mark.parametrize(
    ("filing", "chunks", "vectors", "message"),
    [
        (_filing(accession="../escape"), _chunks("a"), [[1.0, 0.0]], "accession"),
        (_filing(), (), np.zeros((0, 2)), "at least one chunk"),
        (_filing(), _chunks("a"), [[1.0, 0.0], [0.0, 1.0]], "1 chunks against 2"),
        (_filing(), _chunks("a"), [[np.nan, 0.0]], "not finite"),
    ],
    ids=["accession", "empty", "misaligned", "nan"],
)
def test_building_a_shard_refuses_what_could_not_be_searched(filing, chunks, vectors, message):
    with pytest.raises(ValueError, match=message):
        build_shard(filing, chunks, np.asarray(vectors, dtype=np.float32))


def test_an_index_needs_a_filing():
    with pytest.raises(ValueError, match="at least one filing"):
        CorpusIndex.of(SPEC, ())


def test_an_index_refuses_shards_of_different_sizes():
    with pytest.raises(ValueError, match="different sizes"):
        _corpus(_shard(APPLE), _shard(COKE, vectors=((1.0, 0.0, 0.0),)))


def test_an_index_refuses_two_filings_for_one_scope():
    twin = build_shard(
        _filing(APPLE, 2024, accession="0000320193-24-000999"),
        _chunks("a"),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        _corpus(_shard(APPLE, 2024), twin)


# Disk


def test_a_round_trip_through_disk_keeps_filings_chunks_and_scores(tmp_path):
    index = _corpus(_shard(APPLE, 2024), _shard(COKE, 2025))
    index.save(tmp_path)

    loaded = CorpusIndex.load(tmp_path, SPEC)

    assert loaded.filings == index.filings
    assert loaded.shards[0].chunks == index.shards[0].chunks
    assert loaded.fingerprint == index.fingerprint
    assert loaded.digest() == index.digest()
    assert loaded.search(_query(1.0, 0.0), Scope(COKE), top_k=1)[0].score == pytest.approx(1.0)


def test_the_manifest_records_the_format_the_fingerprint_and_every_filing(tmp_path):
    _corpus(_shard(APPLE, 2024), _shard(COKE, 2025)).save(tmp_path)

    manifest = _manifest(tmp_path)

    assert manifest["format_version"] == 2
    assert manifest["fingerprint"] == {
        "model": "nomic-embed-text",
        "lowercase": True,
        "dimensions": 2,
    }
    entry = manifest["filings"][0]
    assert {key: entry[key] for key in ("accession", "cik", "fiscal_year", "chunks")} == {
        "accession": "0000021344-25-000001",
        "cik": COKE,
        "fiscal_year": 2025,
        "chunks": 2,
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "0000021344-25-000001",
        "0000320193-24-000001",
        "manifest.json",
    ]


@pytest.mark.parametrize(
    "expect",
    [
        EmbedderSpec(model="mxbai-embed-large", lowercase=True),
        EmbedderSpec(model="nomic-embed-text", lowercase=False),
    ],
    ids=["model", "casing"],
)
def test_an_index_built_by_another_embedder_is_refused(tmp_path, expect):
    _corpus().save(tmp_path)

    with pytest.raises(IndexFormatError, match="was built with nomic-embed-text"):
        CorpusIndex.load(tmp_path, expect)


def test_loading_an_empty_directory_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest"):
        CorpusIndex.load(tmp_path, SPEC)


def test_a_format_one_index_is_refused_with_what_to_do(tmp_path):
    (tmp_path / "vectors.npy").write_bytes(_npy(np.zeros((1, 2), dtype=np.float32)))
    (tmp_path / "chunks.json").write_text('{"source": {}, "chunks": []}')

    with pytest.raises(IndexFormatError, match="format 1.*re-ingest"):
        CorpusIndex.load(tmp_path, SPEC)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.update(format_version=3), "format 3"),
        (lambda m: m.pop("fingerprint"), "not a valid manifest"),
        (lambda m: m.update(owner="someone"), "not a valid manifest"),
        (lambda m: m.update(filings=[]), "not a valid manifest"),
        (lambda m: m["filings"][0].update(accession="../../etc"), "not a valid manifest"),
        (lambda m: m["filings"][0].update(chunks=3), r"\(3, 2\)"),
        (lambda m: m["fingerprint"].update(dimensions=3), r"\(2, 3\)"),
        (lambda m: m["filings"].append(dict(m["filings"][0])), "share an accession"),
    ],
    ids=[
        "version",
        "missing-key",
        "unknown-key",
        "no-filing",
        "traversal",
        "count",
        "dimensions",
        "duplicate",
    ],
)
def test_a_manifest_that_does_not_describe_its_shards_is_refused(tmp_path, change, message):
    _corpus().save(tmp_path)
    _rewrite_manifest(tmp_path, change)

    with pytest.raises(IndexFormatError, match=message):
        CorpusIndex.load(tmp_path, SPEC)


def test_a_manifest_that_is_not_json_is_refused(tmp_path):
    _corpus().save(tmp_path)
    (tmp_path / "manifest.json").write_text("{not json")

    with pytest.raises(IndexFormatError, match="JSON"):
        CorpusIndex.load(tmp_path, SPEC)


def test_a_shard_changed_after_it_was_written_is_refused(tmp_path):
    index = _corpus()
    index.save(tmp_path)
    vectors = tmp_path / index.filings[0].accession / "vectors.npy"
    vectors.write_bytes(vectors.read_bytes()[:-4] + b"\x00\x00\x00\x40")

    with pytest.raises(IndexFormatError, match="SHA-256"):
        CorpusIndex.load(tmp_path, SPEC)


LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
    "size 1234567\n"
)


def test_a_git_lfs_pointer_in_place_of_the_vectors_says_to_pull_them(tmp_path):
    index = _corpus()
    index.save(tmp_path)
    (tmp_path / index.filings[0].accession / "vectors.npy").write_text(LFS_POINTER)

    with pytest.raises(IndexFormatError, match=r"Git LFS pointer.*git lfs pull"):
        CorpusIndex.load(tmp_path, SPEC)


def test_a_shard_listed_but_missing_is_refused(tmp_path):
    index = _corpus()
    index.save(tmp_path)
    (tmp_path / index.filings[0].accession / "chunks.json").unlink()

    with pytest.raises(IndexFormatError, match="cannot be read"):
        CorpusIndex.load(tmp_path, SPEC)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("vectors.npy", _npy(np.zeros((2, 2), dtype=np.float64)), "float64"),
        ("vectors.npy", _npy(np.zeros((3, 2), dtype=np.float32)), r"\(3, 2\)"),
        ("vectors.npy", _npy(np.full((2, 2), np.inf, dtype=np.float32)), "not finite"),
        ("vectors.npy", b"not an array", "cannot be read"),
        ("chunks.json", b'[{"chunk_id": "x"}]', "cannot be read"),
        ("chunks.json", b"[]", "holds 0 chunks"),
        ("chunks.json", b"7", "cannot be read"),
    ],
    ids=["dtype", "shape", "infinite", "not-npy", "chunk-fields", "chunk-count", "not-a-list"],
)
def test_a_shard_that_does_not_match_its_manifest_entry_is_refused(
    tmp_path, name, content, message
):
    index = _corpus()
    index.save(tmp_path)
    _replace_shard_file(tmp_path, index.filings[0].accession, name, content)

    with pytest.raises(IndexFormatError, match=message):
        CorpusIndex.load(tmp_path, SPEC)


def test_pickled_vectors_are_refused_not_unpickled(tmp_path):
    """allow_pickle=False: loading an index never runs code that came with it."""
    index = _corpus()
    index.save(tmp_path)
    pickled = _npy(np.asarray([object(), object()], dtype=object), allow_pickle=True)
    _replace_shard_file(tmp_path, index.filings[0].accession, "vectors.npy", pickled)

    with pytest.raises(IndexFormatError, match="cannot be read"):
        CorpusIndex.load(tmp_path, SPEC)


def test_a_chunk_of_another_filing_in_a_shard_is_refused(tmp_path):
    index = _corpus()
    index.save(tmp_path)
    accession = index.filings[0].accession
    stored = json.loads((tmp_path / accession / "chunks.json").read_text())
    stored[0]["chunk_id"] = "0000021344-24-000001:Item 1#0"
    _replace_shard_file(tmp_path, accession, "chunks.json", json.dumps(stored).encode())

    with pytest.raises(IndexFormatError, match="another filing"):
        CorpusIndex.load(tmp_path, SPEC)


# Adding filings


def test_adding_a_filing_keeps_the_ones_already_indexed(tmp_path):
    write_shard(tmp_path, _shard(APPLE, 2024), SPEC)
    write_shard(tmp_path, _shard(COKE, 2024), SPEC)

    loaded = CorpusIndex.load(tmp_path, SPEC)

    assert [filing.cik for filing in loaded.filings] == [COKE, APPLE]


def test_ingesting_the_same_filing_again_replaces_it(tmp_path):
    write_shard(tmp_path, _shard(vectors=((1.0, 0.0), (0.0, 1.0))), SPEC)
    write_shard(tmp_path, _shard(vectors=((1.0, 0.0),)), SPEC)

    loaded = CorpusIndex.load(tmp_path, SPEC)

    assert loaded.chunk_count == 1
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".partial-")]


def test_a_second_filing_for_the_same_scope_is_refused(tmp_path):
    write_shard(tmp_path, _shard(APPLE, 2024), SPEC)
    amended = build_shard(
        _filing(APPLE, 2024, accession="0000320193-24-000999"),
        _chunks("a"),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="already indexed as 0000320193-24-000001"):
        write_shard(tmp_path, amended, SPEC)


def test_a_filing_embedded_by_another_model_cannot_join_the_index(tmp_path):
    write_shard(tmp_path, _shard(APPLE), SPEC)

    with pytest.raises(IndexFormatError, match="embedded with other-model"):
        write_shard(tmp_path, _shard(COKE), EmbedderSpec("other-model", lowercase=True))


def test_a_filing_with_vectors_of_another_size_cannot_join_the_index(tmp_path):
    write_shard(tmp_path, _shard(APPLE), SPEC)

    with pytest.raises(IndexFormatError, match="2-dimension"):
        write_shard(tmp_path, _shard(COKE, vectors=((1.0, 0.0, 0.0),)), SPEC)


def test_a_filing_cannot_be_added_beside_a_format_one_index(tmp_path):
    (tmp_path / "chunks.json").write_text("{}")

    with pytest.raises(IndexFormatError, match="format 1"):
        write_shard(tmp_path, _shard(), SPEC)


def test_a_failed_shard_write_leaves_neither_a_partial_directory_nor_a_manifest(
    tmp_path, monkeypatch
):
    def refuse(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "rename", refuse)

    with pytest.raises(OSError, match="disk full"):
        write_shard(tmp_path, _shard(), SPEC)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_manifest_write_keeps_the_previous_manifest(tmp_path, monkeypatch):
    write_shard(tmp_path, _shard(APPLE), SPEC)
    before = _manifest(tmp_path)

    def refuse(source, target):
        raise OSError("disk full")

    monkeypatch.setattr("edgar_rag.index.os.replace", refuse)

    with pytest.raises(OSError, match="disk full"):
        write_shard(tmp_path, _shard(COKE), SPEC)
    assert _manifest(tmp_path) == before
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".partial-")]
