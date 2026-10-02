# 0009. CI eval gate: a replayed dev split against a committed baseline

- Status: Accepted (retroactive, decided in `5b5a27f` and `5d78b73` on 2026-10-01; amended 2026-10-02)
- Date: 2026-10-02

## Context

Unit tests accepted changes that made the system answer worse: a one-line
change to the period guard's year pattern passed every test and stopped the
guard from reading a year. Catching that needs the golden set run on every
pull request.

CI has no GPU and no Ollama. A live run with `qwen3:32b` takes hours; even the
embedder would add a download and its own nondeterminism. An earlier plan put
the real embedder in CI and failed a pull request on a bootstrap ΔAUROC; with
200 end-to-end cases that has an MDE of several points and lets real
regressions through.

## Decision

- `make eval-ci` replays the dev split (4 companies, 8 filings) through the
  production pipeline ([ADR-0008](0008-eval-runs-the-production-pipeline.md))
  from a tape: question embeddings and `qwen3:8b` generations recorded
  locally, keyed by the SHA-256 of the exact input. The index
  (`eval/ci/index`) is committed; search, gates, citation check and
  abstention run for real. A call missing from the tape fails the replay with
  "re-record locally", never falls through to a live model.
- `edgar-rag eval ci` (`src/edgar_rag/eval/regression.py`) judges each case
  right or wrong under each arm against `eval/ci/baseline.json`, grouped by
  tier and class plus an `unanswerable` group per tier, and fails on a net
  worsening of `NET_CHANGE_LIMIT` (3) cases in any group and arm. A net
  improvement as large fails until `make eval-ci-baseline` writes it in the
  same pull request. A change of `outcome` that keeps a case right is warned
  about, not failed.
- Vectors and tape embeddings live in Git LFS (`.gitattributes`); text stays
  in git. Frozen runs' tapes are excluded from every download (`.lfsconfig`).

## Consequences

- Replay is deterministic, so any changed case changed because the code did:
  a fixed limit of 3 cases is a regression test, not a significance test.
  The CI tier is small, and passing means these cases did not get worse, not
  that nothing did.
- A change to the prompt, the retrieval or the embedder input misses the tape
  and requires `make eval-ci-record` (index, tape and baseline in one change).
- The gate decisions are judged at the service's default cosine threshold; a
  CI step checks that threshold against `make cosine-threshold`.
- Rewriting the baseline can hide a regression; the diff of
  `eval/ci/baseline.json` is visible in review, not in the checks.
- [Pull request #4](https://github.com/Rodrigo-Palma/edgar-rag/pull/4), closed
  unmerged, is the gate failing the period guard change above.

## Enforced by

- [`test_a_net_worsening_at_the_limit_fails_naming_the_group`](../../tests/eval/test_regression.py) and [`test_a_worsening_spread_over_unanswerable_kinds_fails_on_their_aggregate`](../../tests/eval/test_regression.py)
- [`test_an_improvement_at_the_limit_fails_until_the_baseline_records_it`](../../tests/eval/test_regression.py)
- [`test_an_end_to_end_case_whose_generation_disappears_is_judged_end_to_end_as_worse`](../../tests/eval/test_regression.py)
- [`test_an_outcome_change_that_keeps_the_case_right_is_reported_not_failed`](../../tests/eval/test_regression.py)
- [`test_a_miss_in_replay_says_to_re_record_locally`](../../tests/eval/test_replay.py) and [`test_a_replay_miss_fails_with_the_re_record_hint`](../../tests/eval/test_eval_commands.py)
- [`test_a_git_lfs_pointer_in_place_of_a_tape_file_says_to_pull_it`](../../tests/eval/test_replay.py)
- The CI job `eval` runs `make eval-ci` on every pull request.
