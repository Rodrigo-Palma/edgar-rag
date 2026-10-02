# 0008. The evaluation runs the production pipeline

- Status: Accepted (retroactive, decided in `c2087b2` on 2026-10-01)
- Date: 2026-10-02

## Context

The first evaluation, `scripts/evaluate_gates.py`, embedded, searched and
gated questions with its own code: its own `TOP_K`, no citation check, no
prompt. It measured a path the service never executes, so a change to the
service could not move its numbers, and a number it printed said nothing
certain about the service.

## Decision

The harness (`src/edgar_rag/eval/runner.py`) asks every golden question
through the same `Answerer` the service builds (`src/edgar_rag/answer.py`),
with the same prompt, citation check and default `top_k`. The only
difference is the gate: `ScoreEvery` asks every gate under evaluation for its
score and then admits (end-to-end tier) or declines (gate-only tier)
regardless, so each record holds every score and arm A's outcome
([ADR-0015](0015-one-generation-serves-every-arm.md)). The nonce comes from
the case id instead of a random draw, so a case's prompt is the same on every
run and can be replayed. `scripts/evaluate_gates.py` is deleted.

## Consequences

- A change to the service changes the evaluation, which is what lets the CI
  gate ([ADR-0009](0009-ci-eval-gate-replayed.md)) catch a regression.
- The evaluation cannot test a gate the service could not run, and a new gate
  enters the service before it can be measured.
- The deterministic nonce is the one difference from serving; the injection
  tests cover both nonce sources.

## Enforced by

- [`test_an_e2e_case_is_generated_and_a_gate_only_case_is_only_scored`](../../tests/eval/test_runner.py): the harness is built on `Answerer`.
- [`test_the_prompt_of_a_case_is_the_same_on_every_run`](../../tests/eval/test_runner.py) and [`test_a_case_nonce_is_stable_per_case_and_differs_between_cases`](../../tests/test_injection.py)
- [`test_a_recorded_run_replays_without_ollama_to_the_same_outcomes`](../../tests/eval/test_eval_commands.py)
- Not enforced by a test: that nobody reimplements the pipeline in the
  evaluation again. `src/edgar_rag/eval/runner.py` imports `Answerer`, and a
  reviewer has to keep it that way.
