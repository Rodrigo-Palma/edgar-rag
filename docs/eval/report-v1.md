# Evaluation report: eval split, e2e tier

| field | value |
|---|---|
| date, hardware | 2026-10-02, Apple M3 Max, Darwin 27.0.0 |
| code | `d8d23b9f9610` |
| mode | record |
| golden set | sha256 `00419f11876cacad` |
| index | digest `bbe4812361d6c95e`, 48 filings |
| embedder | nomic-embed-text `0a109f422b47` |
| generator | qwen3:32b `030ee887880f`, options num_ctx=8192, seed=0, temperature=0, think=False |
| Ollama | 0.18.0 |
| gates scored | period, cosine, brier; top_k 4 |
| brier | commit d70e7df9b0b4f765e2d533c7bc9dd532155916e5 |
| protocol | sha256 `f004e4c2cfdc9f45` |
| cases | 6192 (20 companies); wall time 8669 s |

## Thresholds

| gate | rule | threshold by fold |
|---|---|---|
| cosine | cross-fitted R90, target recall 0.90 | fold 0: 0.7376, fold 1: 0.7247 |
| brier | cross-fitted R90, target recall 0.90 | fold 0: 0.5921, fold 1: 0.5445 |

## Six arms, end-to-end tier

300 unanswerable and 180 answerable questions, 20 companies. One generation per question serves every arm.

| arm | false answers on unanswerable (Wilson 95%) | company bootstrap 95% | recall cost | answered correct | numeric accuracy (answered) | citation support (answered) | gate recall (gate-only) | generator s / question |
|---|---|---|---|---|---|---|---|---|
| A no gate, model refusal only | 13/300 = 4.3% [2.5%, 7.3%] | [2.3%, 6.7%] | 0/180 = 0.0% [0.0%, 2.1%] | 48/180 = 26.7% [20.7%, 33.6%] | 51/88 = 58.0% [47.5%, 67.7%] | 61/88 = 69.3% [59.0%, 78.0%] | 100.0% | 13.21 |
| B cosine | 3/300 = 1.0% [0.3%, 2.9%] | [0.0%, 2.3%] | 1/180 = 0.6% [0.1%, 3.1%] | 47/180 = 26.1% [20.2%, 33.0%] | 48/80 = 60.0% [49.0%, 70.0%] | 58/80 = 72.5% [61.9%, 81.1%] | 88.0% | 8.24 |
| C brier | 5/300 = 1.7% [0.7%, 3.8%] | [0.0%, 3.7%] | 3/180 = 1.7% [0.6%, 4.8%] | 45/180 = 25.0% [19.2%, 31.8%] | 48/80 = 60.0% [49.0%, 70.0%] | 56/80 = 70.0% [59.2%, 78.9%] | 88.1% | 9.58 |
| D period guard | 13/300 = 4.3% [2.5%, 7.3%] | [2.3%, 6.7%] | 0/180 = 0.0% [0.0%, 2.1%] | 48/180 = 26.7% [20.7%, 33.6%] | 51/88 = 58.0% [47.5%, 67.7%] | 61/88 = 69.3% [59.0%, 78.0%] | 100.0% | 11.08 |
| E period guard + brier | 5/300 = 1.7% [0.7%, 3.8%] | [0.0%, 3.7%] | 3/180 = 1.7% [0.6%, 4.8%] | 45/180 = 25.0% [19.2%, 31.8%] | 48/80 = 60.0% [49.0%, 70.0%] | 56/80 = 70.0% [59.2%, 78.9%] | 88.1% | 7.62 |
| F period guard + cosine | 3/300 = 1.0% [0.3%, 2.9%] | [0.0%, 2.3%] | 1/180 = 0.6% [0.1%, 3.1%] | 47/180 = 26.1% [20.2%, 33.0%] | 48/80 = 60.0% [49.0%, 70.0%] | 58/80 = 72.5% [61.9%, 81.1%] | 88.0% | 6.59 |

## Confirmatory comparisons

H1, FAR(A) - FAR(F) on 300 negatives: +3.3 p.p., 99.0% CI [+1.3, +5.7] p.p.; MDE at 80% power about 3.0 p.p. Reading: not attainable: arm A answers 4.3% of unanswerable questions, under the 5.0% margin, and a gate cannot remove answers the model does not give. Recall cost of F: +0.6 p.p., 99.0% CI [+0.0, +2.2] p.p.; MDE at 80% power about 1.8 p.p. Criterion (lower bound above 5.0%, recall cost upper bound at most 12.0%): not met.

H2, FAR(E) - FAR(F), paired: +0.7 p.p., 99.0% CI [+0.0, +2.7] p.p.; MDE at 80% power about 2.2 p.p. Discordant: E alone answers 2, F alone answers 0 (exact McNemar p = 0.500). Criterion (upper bound below 0): not met.

A difference smaller than the MDE printed beside it is not distinguishable at this n.

## False answers by subtype (descriptive)

| arm | off_domain (n=75) | other_company (n=75) | unreported_concept (n=75) | wrong_year (n=75) |
|---|---|---|---|---|
| A | 1/75 = 1.3% [0.2%, 7.2%] | 9/75 = 12.0% [6.4%, 21.3%] | 3/75 = 4.0% [1.4%, 11.1%] | 0/75 = 0.0% [0.0%, 4.9%] |
| B | 0/75 = 0.0% [0.0%, 4.9%] | 1/75 = 1.3% [0.2%, 7.2%] | 2/75 = 2.7% [0.7%, 9.2%] | 0/75 = 0.0% [0.0%, 4.9%] |
| C | 1/75 = 1.3% [0.2%, 7.2%] | 1/75 = 1.3% [0.2%, 7.2%] | 3/75 = 4.0% [1.4%, 11.1%] | 0/75 = 0.0% [0.0%, 4.9%] |
| D | 1/75 = 1.3% [0.2%, 7.2%] | 9/75 = 12.0% [6.4%, 21.3%] | 3/75 = 4.0% [1.4%, 11.1%] | 0/75 = 0.0% [0.0%, 4.9%] |
| E | 1/75 = 1.3% [0.2%, 7.2%] | 1/75 = 1.3% [0.2%, 7.2%] | 3/75 = 4.0% [1.4%, 11.1%] | 0/75 = 0.0% [0.0%, 4.9%] |
| F | 0/75 = 0.0% [0.0%, 4.9%] | 1/75 = 1.3% [0.2%, 7.2%] | 2/75 = 2.7% [0.7%, 9.2%] | 0/75 = 0.0% [0.0%, 4.9%] |

Arm D on wrong_year is low by construction: the guard and the label share one definition of the period a filing covers.

## Gate only, every golden case of the split

| gate | AUROC (company bootstrap 95%) | FAR at R90 | realised recall | n | companies |
|---|---|---|---|---|---|
| cosine | 0.834 [0.820, 0.849] | 38.8% | 88.0% | 3076/3076 | 20 |
| brier | 0.732 [0.713, 0.755] | 55.7% | 88.1% | 3076/3076 | 20 |
| period guard (a rule, no score) | n/a | 75.0% | 100.0% | 3076/3076 | 20 |

AUROC(brier) - AUROC(cosine): -0.101, 95% CI [-0.124, -0.080]; MDE at 80% power about 0.031.

## Risk and coverage, end-to-end tier

Admitting in decreasing score order, then arm A. An error is an answer to an unanswerable question, or an answerable question not answered correctly.

| gate | AURC | risk at 25.0% coverage | risk at 50.0% coverage | risk at 75.0% coverage | risk at 90.0% coverage |
|---|---|---|---|---|---|
| cosine | 0.418 | 44.2% (at 25.0%) | 42.1% (at 50.0%) | 38.6% (at 75.0%) | 33.6% (at 90.0%) |
| brier | 0.372 | 39.2% (at 25.0%) | 38.3% (at 50.0%) | 34.7% (at 75.0%) | 33.3% (at 90.0%) |

## Narrative questions (written by hand, reported apart, never pooled)

Correct decision (answer when answerable, abstain when not), arm A: 36/40 = 90.0% [76.9%, 96.0%], 18 companies.
Expected item cited, answered answerable questions on filings whose item split is sane: 10/12 = 83.3% [55.2%, 95.3%]. Excluded for an unsound split: CVX, DE, JNJ, UNP.

## Determinism

Same prompt sent twice, text identical: 30/30 = 100.0% [88.6%, 100.0%].

## Operating cost

| stage | n | P50 s | P95 s |
|---|---|---|---|
| embed | 6192 | 0.016 | 0.029 |
| search | 6192 | 0.000 | 0.000 |
| gate | 6192 | 0.213 | 0.249 |
| generate | 520 | 12.913 | 19.412 |

| arm A outcome | n | prompt tokens | completion tokens | generator s |
|---|---|---|---|---|
| answered | 101 | 1648.8 | 25.1 | 15.10 |
| declined after generating | 379 | 1475.6 | 13.3 | 12.71 |

## Deviations from the protocol

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

## What this does not show

- The golden questions are generated from XBRL templates; the hand-written narrative questions are reported apart and never pooled with them.
- The generator may have seen these filings in pre-training. FY2024 and FY2025 filings and the check that a cited passage prints the gold value limit, not remove, that risk.
- Company-clustered percentile intervals undercover with few companies; Wilson intervals assume independent questions.
- 6192 cases from 20 companies; nothing here generalises beyond filings like these.
