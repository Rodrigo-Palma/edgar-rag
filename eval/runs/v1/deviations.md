# Deviations from protocol v1

Protocol `docs/eval/protocol.md`, sha256
`f004e4c2cfdc9f45354bf6e52004a6d809e4925e1cf4762ddffda945feb11b7c`, added in
`ad4174b`. Recorded before the run in the commit that adds this file; parent
`b6be0a5`. The run's manifest gives the run commit.

## DV-01, frozen-file diff outside the allowed PT-15 change

Found by pre-flight item 2 of section 11 on 2026-10-02, before any eval-split
call. `git diff --stat ad4174b HEAD` over the frozen files prints two files:

- `src/edgar_rag/index.py`, 3 lines (`61565c5`): the Git LFS pointer check of
  PT-15, the one change section 11 allows in advance.
- `src/edgar_rag/domain.py`, 4 lines (`a888090`): a new exception class
  `NotRecorded(LookupError)`, which `TapeMiss` in `eval/replay.py` (not a
  frozen file) now subclasses instead of `LookupError` directly, and which the
  replay service raises for a question its tape never recorded.

Why it does not change what the run measures: the class is added, nothing in
`domain.py` is edited or removed; `TapeMiss` keeps `LookupError` as an
ancestor, so every `except` that caught it still does; no frozen module
raises or catches it. Evidence: the pilot tape replays on this commit with no
miss and the same outcome, generation, retrieved passages and scores for all
1,344 dev cases (200 end to end), and CI run 36960591688 on `b6be0a5` passes
the replayed dev-split gate.

Not reverted: reverting would be a code change between protocol and run, which
section 11 treats as a deviation in its own right.

## DV-02, post-run: the fixed headline sentence and the exploratory labels

Recorded after the run, on 2026-10-02, in the commit that freezes it. DV-01
above was recorded before the run, in `d8d23b9`, whose version of this file
the git log keeps.

Section 7 fixes a headline sentence for the case H1 is not attainable, and
section 6 asks that everything outside H1 and H2 be labelled exploratory. The
report prints neither, and `src/edgar_rag/eval/report.py` is frozen, so both
live in `docs/eval/report-v1-addendum.md`. It changes no number: every figure
in the addendum is copied from `report-v1.md`.
