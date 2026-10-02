# 0003. Lower-case embedding input to work around ollama#15609

- Status: Accepted (retroactive, decided in `7b0d9ae` on 2026-09-26)
- Date: 2026-10-01

## Context

On Ollama 0.18.0 with `nomic-embed-text`, every capitalised token collapses onto
one vector. Measured (README, "The embedder was discarding every proper noun"):

- `cos("Apple", "Cat") = 1.0000` and `cos("Apple", "Zebra") = 1.0000`, while
  `cos("apple", "petrobras") = 0.4075`.
- Two sentences differing only in a company name came back byte for byte
  identical; 287 distinct texts produced 119 distinct vectors.
- In the Apple 10-K, 18.3% of words start with a capital, against 2.9% non-ASCII.

This is [ollama/ollama#15609](https://github.com/ollama/ollama/issues/15609), a
regression at v0.14.0 attributed to `BasicTokenizer` preprocessing lost in the
HF to gguf conversion. The upstream issue frames it as non-ASCII; the
measurements above show it is wider. A filing identifies what a passage is about
mostly through proper nouns, so this discards the most informative tokens.

## Decision

`OllamaEmbedder` in `src/edgar_rag/models.py` lower-cases every text before
sending it, by default. The flag is fixed at construction (`lowercase=True`) and
applies to every call, so passages at ingest time and questions at query time go
through the same transformation. `lowercase=False` exists for the day upstream
fixes the tokenizer.

## Consequences

- Re-indexing the same filing raised every measured question's best score and
  moved the R&D question's top passage from Item 1A (0.574) to Item 7 (0.635),
  where the figures are. Measured on four questions and one filing.
- Case is lost for every token, not just the broken ones: "Apple" and "apple",
  "US" and "us" become indistinguishable to retrieval. On 10-K text this costs
  less than the collapse it replaces, but it is a cost.
- The index and the query must agree on the flag and on the model, and nothing
  checks that today. `FilingIndex.save` writes no record of how the vectors were
  produced, so an index built with `lowercase=False` or another model loads
  without error and returns plausible but meaningless scores. Recording an
  embedding fingerprint (model, lowercase, dimension) in the index and refusing
  to serve on a mismatch is the planned fix, in the index format ADR.
- Revisit when ollama#15609 is closed: turn the flag off, re-index, and keep it
  off only if the evaluation does not get worse.

## Enforced by

- [`test_passages_are_lower_cased_before_they_are_sent`](../../tests/test_models.py): the request body carries lower-cased text by default.
- [`test_lowercasing_can_be_turned_off_when_the_tokenizer_is_fixed`](../../tests/test_models.py): the escape hatch sends text as written.
- Not yet enforced: index and query agreeing on the flag. No test can fail on a
  mismatch until the fingerprint exists.
