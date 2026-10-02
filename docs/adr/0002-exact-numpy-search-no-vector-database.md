# 0002. Exact NumPy search on disk, no vector database

- Status: Accepted (retroactive, decided in `2949077` on 2026-09-26)
- Date: 2026-10-02

## Context

Every question is scoped to one filing ([ADR-0005](0005-retrieval-scope-in-the-request.md)),
so a search runs over one shard, a few hundred passages of 768 float32 values.
The 48 filings of the golden set make 23,904 passages and index in 5 min 37 s
on an Apple M3 Max (`make ingest`). On the headline run the search stage took
0.000 s at P50 and P95 over 6192 questions ([report](../eval/report-v1.md),
"Operating cost"), against 12.913 s at P50 for generation.

An approximate index (HNSW, IVF) trades exactness for speed. That trade would
change which passages a question retrieves, and so what the evaluation
measures, for a stage that costs nothing measurable today.

## Decision

`CorpusIndex.search` in `src/edgar_rag/index.py` is an exact dot product of
the unit-length query against the unit-length rows of the scoped shard, held
in memory as a NumPy array and loaded once from `.npy` files. No vector
database, no ANN library, no external service.

## Consequences

- Retrieval is deterministic and exact, which the replayed CI gate
  ([ADR-0009](0009-ci-eval-gate-replayed.md)) depends on: the same index and
  question always give the same passages and scores.
- The whole corpus sits in memory: about 70 MB of vectors for the 48 filings.
  That bounds the corpus, not the request.
- Revisit when a scoped search reaches about 50 ms at P95, or the corpus no
  longer fits comfortably in memory (on the order of a million passages).
  Whatever replaces it has to be evaluated against the exact search first,
  since any recall it loses shows up as a recall cost of every arm.

## Enforced by

- [`test_scores_are_cosine_so_length_does_not_decide`](../../tests/test_corpus_index.py) and [`test_a_shard_stores_unit_length_vectors`](../../tests/test_corpus_index.py): the scores are exact cosine.
- [`test_a_top_k_larger_than_the_filing_returns_all_of_it_best_first`](../../tests/test_corpus_index.py): the ranking is complete, not approximate.
- [`test_pickled_vectors_are_refused_not_unpickled`](../../tests/test_corpus_index.py): the files on disk are plain arrays.
