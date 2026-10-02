# 0015. One generation per question serves every evaluation arm

- Status: Accepted
- Date: 2026-10-01

## Context

The pre-registered evaluation ([protocol](../eval/protocol.md)) compares six
arms: no gate (A), cosine (B), brier (C), the period guard (D), and the guard
in front of brier (E) or cosine (F). Run naively, each arm asks the generator
about every question, so the end-to-end tier costs six times its generations.

Generation is the whole budget. On the M3 Max with `qwen3:32b`
(`030ee887880f`, Ollama 0.18.0, `num_ctx=8192`) the dev-split pilot measured
a median and a mean of 14.2 s per generation (n = 200, protocol section 10),
while embedding and search take 0.015 s median per question and the period
guard and cosine gate under 0.001 s. The end-to-end tier is 480 golden
questions, 40 narratives and 30 determinism repeats, 550 generations: about
2 h 10 min once, about 13 h six times.

A gate in this code decides only whether the model is called. `Answerer.ask`
(`src/edgar_rag/answer.py`) asks the gate, and on admission sends the same
retrieved passages and the same prompt it would send with no gate at all. No
gate rewrites, reorders or drops passages.

## Decision

The harness generates once per question, with every gate scored and none
allowed to stop the call (`Harness` in `src/edgar_rag/eval/runner.py`), and
derives each arm as a mask over that one record (`src/edgar_rag/eval/arms.py`):
a gated arm answers a question exactly when arm A answered it and the arm's
gate admits it. Learned gates admit at the cross-fitted R90 threshold; the
period guard is a rule with no threshold.

## Consequences

- Six arms cost one arm. This is what lets the protocol fit in about four
  hours instead of a day.
- Arms are paired at the level of the generated text, so differences between
  arms carry no sampling noise from the generator, only from which questions
  each gate admits. This also means FAR(A) - FAR(F) is non-negative per
  question by construction, so H1 is tested against a margin of 5 p.p., never
  against zero.
- A gated arm's generator time is computed (the sum over the questions it
  admits), not measured in a run of its own. Gate time is measured.
- The equivalence holds only while a gate is a pure filter. A gate that changes
  the passages (a reranker, a passage filter, a query rewrite) breaks it and
  needs its own generations; this ADR must be revisited before adding one.
- Brier is optional. When it is not plugged in, arms C and E have no scores and
  the report prints them as not run, rather than dropping the rows.

## Enforced by

- [`test_every_gated_arm_answers_a_subset_of_what_arm_a_answers`](../../tests/eval/test_arms.py) and [`test_arm_a_answers_exactly_what_was_recorded_as_answered`](../../tests/eval/test_arms.py): the mask semantics.
- [`test_every_gate_score_is_recorded_whatever_the_gate_would_have_decided`](../../tests/eval/test_runner.py) and [`test_score_every_keeps_the_scores_and_overrides_the_decision`](../../tests/eval/test_runner.py): no gate stops the recorded generation.
- [`test_the_prompt_of_a_case_is_the_same_on_every_run`](../../tests/eval/test_runner.py): the prompt does not depend on the gate.
- [`test_an_arm_whose_gate_did_not_run_is_not_runnable`](../../tests/eval/test_arms.py) and [`test_without_brier_its_arms_are_printed_as_not_run`](../../tests/eval/test_report.py): arms without scores are reported as not run.
- Not enforced by a test: that no gate alters the passages. It holds because
  `RelevanceGate.admits` returns a decision and no passages; a gate that needed
  to change them would have to change the protocol type first.
