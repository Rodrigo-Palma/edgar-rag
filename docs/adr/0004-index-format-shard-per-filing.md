# 0004. Index format 2: one shard per filing under a manifest that pins the embedder

- Status: Accepted (retroactive, decided in `11bd002` on 2026-10-01)
- Date: 2026-10-02

## Context

The first index held one filing: chunk ids such as `Item 7#3` collided across
filings, ingesting a second filing overwrote the first, and the files recorded
nothing about how the vectors were made. The golden set needs 48 filings in
one index.

Vectors from another embedding model, or from text cased differently
([ADR-0003](0003-lowercase-embedding-input.md)), still load and still return
passages; they are just not the closest ones. Nothing in a wrong answer would
show it.

## Decision

`src/edgar_rag/index.py` writes format 2:

- `<root>/<accession>/vectors.npy` and `chunks.json`, one shard per filing;
  every `chunk_id` carries the accession;
- `<root>/manifest.json` with `format_version`, the embedding fingerprint
  (model, lower-cased input, vector size) and one entry per filing, including
  the SHA-256 of both files of its shard;
- the manifest is written last, through a temporary file, so it never lists a
  shard that was not fully written.

`CorpusIndex.load(root, expect)` refuses, with `IndexFormatError` and what to
run, an index of another format, another embedder, a shard whose hash changed,
a shard listed but missing, pickled arrays, and a Git LFS pointer in place of
the vectors. A format 1 index is not migrated: it is ingested again.

## Consequences

- Adding a filing adds a shard; ingesting the same filing again replaces it.
- The service refuses to start on a mismatched index instead of answering from
  the wrong passages ([ADR-0011](0011-composition-root-in-an-app-factory.md)).
- The index digest identifies an index, and the evaluation records
  it in each run's manifest (`bbe4812361d6c95e` for the headline run).
- A change of embedder or of the lower-case flag means a full re-ingest; that
  is the point.

## Enforced by

- [`test_the_manifest_records_the_format_the_fingerprint_and_every_filing`](../../tests/test_corpus_index.py)
- [`test_an_index_built_by_another_embedder_is_refused`](../../tests/test_corpus_index.py) and [`test_a_filing_embedded_by_another_model_cannot_join_the_index`](../../tests/test_corpus_index.py)
- [`test_a_shard_changed_after_it_was_written_is_refused`](../../tests/test_corpus_index.py) and [`test_a_shard_listed_but_missing_is_refused`](../../tests/test_corpus_index.py)
- [`test_a_format_one_index_is_refused_with_what_to_do`](../../tests/test_corpus_index.py)
- [`test_a_failed_shard_write_leaves_neither_a_partial_directory_nor_a_manifest`](../../tests/test_corpus_index.py)
- [`test_chunk_ids_carry_the_accession_so_they_stay_unique_across_filings`](../../tests/test_corpus_index.py)
- [`test_the_service_refuses_to_start_on_an_index_from_another_embedder`](../../tests/test_api.py)
