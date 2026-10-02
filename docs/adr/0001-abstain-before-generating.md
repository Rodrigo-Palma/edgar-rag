# 0001. Abstain before generating; the relevance gate is a port

- Status: Accepted (retroactive, decided in `7b0d9ae` on 2026-09-26)
- Date: 2026-10-01

## Context

A model given passages that do not contain the answer still produces one, and
nothing in the generated text tells the caller it was invented. Retrieval always
returns `top_k` passages, so "something was retrieved" carries no information
about whether the filing answers the question.

How relevance should be judged is the least settled part of the pipeline. Two
judges exist today, both in `src/edgar_rag/gate.py`:

- `CosineGate` compares the best retrieval score to a threshold. It is free and
  needs nothing beyond the index, but it measures that the words are close, so a
  question about the wrong year or the wrong company that reuses the filing's
  vocabulary scores high.
- `BrierGate` asks an external calibrated model, one call per passage, stopping
  at the first that clears `min_confidence`. It depends on a service that can be
  down.

Measured on ten hand-picked questions against one Apple 10-K (README, "What the
model gate buys"): at the operating point that admits all 5 answerable
questions, cosine wrongly admits 4 of 5 unanswerable ones and the model gate 1
of 5. That is n=5 per class and a direction, not an inference.

## Decision

`Answerer.ask` in `src/edgar_rag/answer.py` asks the gate first and calls the
generator only when the gate admits the passages. A refusal returns
`abstained=True` with the gate's `reason` and the best retrieval score, and the
model is never called.

The gate is a `typing.Protocol`, `RelevanceGate`, taking the question and the
retrieved passages and returning a `GateDecision(admitted, confidence, reason,
degraded)`. The pipeline depends only on the protocol. `build_gate` in
`src/edgar_rag/service/app.py` is the single place that picks an adapter: cosine alone
when `brier_url` is unset, otherwise `BrierGate` with `CosineGate` as fallback.

A fallback is never reported as the primary judge: when `BrierGate` cannot reach
its service and falls back, the decision carries `degraded=True` and the reason
says so. Without a fallback the failure is raised as `GateError`.

## Consequences

- A refused question costs one embedding and one vector search (0.06 s measured
  on the off-topic question in the README), against 6.0 s and 19.6 s for the two
  questions that reached generation.
  This is what makes large gate-only evaluations affordable.
- The gate sees passages, not the answer, so it cannot catch a model that
  answers wrongly from relevant passages. That is covered after generation by
  the model's own refusal and by the citation check, which are separate
  abstention paths with their own reasons.
- Swapping the judge is a new adapter and one line in `provide_gate`; nothing in
  `answer.py` changes. The cost is that thresholds live per adapter
  (`min_retrieval_score`, `brier_min_confidence`) and are not comparable across
  them.
- Known gap: `degraded` stops at `GateDecision`. `Answer` has no field for it, so
  the HTTP response does not yet tell the caller that the fallback judged the
  question, even though the README shows the field. Until it is propagated the
  only signal is the text of `reason`.
- `BrierGate` is plugged in over plain HTTP and is optional by design; the
  default configuration runs with no external service.

## Enforced by

- [`test_abstains_before_generating_when_retrieval_is_weak`](../../tests/test_answer.py): the fake generator records no prompt when the gate refuses.
- [`test_abstains_when_the_model_says_the_filing_does_not_cover_it`](../../tests/test_answer.py) and [`test_an_answer_that_cites_nothing_is_an_abstention`](../../tests/test_answer.py): the post-generation paths are distinct from the gate's.
- [`test_an_unreachable_brier_falls_back_to_cosine_and_says_it_did`](../../tests/test_gate.py) and [`test_an_unreachable_brier_without_a_fallback_is_an_error`](../../tests/test_gate.py): `degraded=True` on fallback, `GateError` without one.
- [`test_the_gate_is_cosine_only_while_no_brier_url_is_configured`](../../tests/test_api.py) and [`test_configuring_a_brier_url_puts_the_model_in_front_with_cosine_behind_it`](../../tests/test_api.py): the wiring in `build_gate`.
- Not yet enforced: a test that `degraded` reaches the HTTP response.
