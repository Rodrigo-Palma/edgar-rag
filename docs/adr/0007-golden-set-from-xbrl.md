# 0007. Generate the golden set from XBRL companyfacts and the filing text

- Status: Accepted (retroactive, decided in `fcda182` and `411373c` on 2026-10-01)
- Date: 2026-10-02

## Context

The first evidence for the gate was ten hand-picked questions on one Apple
10-K: n=5 per class, where no arrangement could reach significance. A useful
estimate needs hundreds of labelled questions per class, across companies,
with labels nobody chose while looking at the system's answers.

The SEC publishes every reported figure as XBRL (`companyfacts`), with value,
unit, period and the accession of the filing that reported it. That is a
label source with no model and no annotator in the loop. Its risk is the
opposite one: a fact in XBRL is not necessarily printed in the text a parser
extracts from the 10-K, and a question whose answer is not in the indexed text
would be scored as a retrieval failure when it is a parser failure.

## Decision

`edgar-rag eval build` (`src/edgar_rag/eval/build.py`) generates
`eval/golden/v1.jsonl` from committed snapshots (`eval/snapshots/`, each
checked against its SHA-256 in `eval/filings.lock.json`), with no network and
no model:

- the roster is `eval/companies.toml`: 24 companies by ticker, CIK checked
  against the SEC's ticker map, split by company into dev (4) and eval (20);
  filings are the fiscal 2024 and 2025 10-Ks, pinned by accession;
- answerable questions come from twelve us-gaap concepts in
  `src/edgar_rag/eval/templates.py`, and are kept only when the value the
  filing's XBRL reports is printed in the filing's indexed text;
- unanswerable questions are of four kinds, each verified against the filing
  (`src/edgar_rag/eval/candidates.py`): `wrong_year` (a year four to seven
  back, its value not printed), `other_company` (another company's value, not
  printed), `unreported_concept` (a concept the company never reported, its
  phrases absent), `off_domain`;
- two tiers per split: gate-only (every candidate the quotas allow) and end to
  end (on the eval split, 9 answerable and 15 unanswerable per company);
- 40 narrative questions are written by hand in
  `eval/golden/narrative-v1.jsonl`, validated against their filing, and
  reported apart, never pooled;
- no question uses a company name or the question wrapper of brier's training
  data (leakage check, protocol section 12).

The same inputs give the same bytes; `eval build --check` compares them.

## Consequences

- Labels are reproducible by anyone from the committed snapshots, and the
  drop rate per concept and reason is printed (`eval golden-stats`).
- Questions read like templates. That inflates any result that rewards
  vocabulary overlap, and is stated under the README's result table; the
  narratives are the only check against it.
- Answering from XBRL is ruled out for the service: the labels come from it,
  so it would make the evaluation circular.
- A roster change follows a mechanical rule (decision D-01 in protocol
  section 13), not a choice by hand.

## Enforced by

- [`test_the_committed_golden_set_rebuilds_byte_for_byte_from_the_snapshots`](../../tests/eval/test_golden_v1.py) and [`test_the_committed_golden_set_matches_the_plan`](../../tests/eval/test_golden_v1.py)
- [`test_a_tampered_snapshot_fails_the_build`](../../tests/eval/test_build.py)
- [`test_a_cik_the_sec_maps_elsewhere_is_named`](../../tests/eval/test_snapshot.py)
- [`test_an_eval_question_naming_a_brier_company_is_refused`](../../tests/eval/test_build.py) and [`test_the_brier_template_is_refused_in_any_split`](../../tests/eval/test_build.py)
- [`test_narratives_are_checked_against_their_filing`](../../tests/eval/test_build.py)
