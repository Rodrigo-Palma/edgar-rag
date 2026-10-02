# 0006. Abstention reasons are a closed set, and the answer carries the gate score

- Status: Accepted (retroactive, decided in `73f7c25` on 2026-10-01)
- Date: 2026-10-02

## Context

The first API returned `asdict(Answer)`: a free-text `reason`, no gate score
and no `degraded`, while the README showed `answer`, `confidence` and
`degraded`. A client could not tell a gate rejection from a model refusal
without parsing prose, and the evaluation could not compute a gate's AUROC
through the production pipeline because the score never left the gate.

Which check withheld an answer matters to a caller: a gate rejection means
the filing had nothing close; a model refusal means it had something close
that did not hold the answer.

## Decision

`AbstentionReason` in `src/edgar_rag/domain.py` is a `StrEnum` whose values
are part of the HTTP contract: `gate_rejected`, `out_of_period`,
`model_declined`, `no_valid_citation`, `unsupported_claim`, `out_of_scope`.
Prose lives in `detail`. `Answer` carries `gate_score`, `degraded`,
`retrieval_score` and a `Trace` (seconds per stage, tokens). `AskResponse` in
`src/edgar_rag/service/schemas.py` is the endpoint's `response_model`; the
trace goes to the request log, not to the client.

## Consequences

- A client branches on `reason`; adding a reason is a contract change.
- The evaluation reads every gate's score from the same `Answer` the service
  returns ([ADR-0008](0008-eval-runs-the-production-pipeline.md)).
- The README's JSON examples are tested against a served response, so the
  README cannot drift from the API again.
- `degraded` is false with the built-in gates; it stays because a replay of
  the headline run sets it ([ADR-0014](0014-remove-brier-default-to-no-gate.md)).

## Enforced by

- [`test_the_reasons_are_the_published_wire_values`](../../tests/test_answer_contract.py) and [`test_each_abstention_reason_has_its_own_message`](../../tests/test_answer_contract.py)
- [`test_a_model_refusal_is_not_reported_as_missing_passages`](../../tests/test_answer_contract.py)
- [`test_a_healthy_answer_carries_every_field_of_the_contract`](../../tests/test_api.py) and [`test_the_ask_response_is_published_in_the_openapi_schema`](../../tests/test_api.py)
- [`tests/test_readme_contract.py`](../../tests/test_readme_contract.py): the README examples have the shape of a served response.
