# 0012. Bound concurrent generations and log one JSON line per request

- Status: Accepted (retroactive, decided in `39e98f7` on 2026-10-01)
- Date: 2026-10-02

## Context

A generation holds the GPU: 12.913 s at P50 and 19.412 s at P95 on the
headline run (`qwen3:32b`, Apple M3 Max, n = 520). Concurrent generations
share the GPU and all get slower, so a queue in front of it only grows the
latency of everyone in it. Embedding, search and the gates are cheap by
comparison, and a question a gate declines never needs the GPU.

The service had no logging, so a fallback, a timeout or a slow stage was
invisible to whoever ran it.

## Decision

- The pipeline is synchronous and runs in a worker thread per request
  (`anyio.to_thread.run_sync`), with no async rewrite.
- `GenerationSlots` in `src/edgar_rag/service/app.py` wraps the generator in
  a bounded semaphore of `EDGAR_RAG_MAX_CONCURRENT_GENERATIONS` (default 2).
  A request that needs a generation when none is free gets `503` with
  `Retry-After` at once, without queueing. A request whose gate declines
  never takes a slot.
- Each request has a budget, `EDGAR_RAG_REQUEST_TIMEOUT_SECONDS` (default 90);
  past it the client gets `504`, and the worker keeps its slot until the
  model's own timeout ends it, so abandoned work cannot pile up on the GPU.
- Every request writes one line of JSON to stderr
  (`src/edgar_rag/telemetry.py`): status, total and per-stage seconds,
  `reason`, `degraded`, every gate score, and prompt and completion tokens.
  The question is not logged. Money cost is derived offline from the tokens,
  not computed in the service.

## Consequences

- Under load the service sheds work instead of hanging, and a caller can
  retry on `503`.
- Per-stage latency and token counts are measured where they happen; the
  evaluation report's operating cost section comes from the same timers.
- No queue and no fairness: a burst gets as many `503`s as it exceeds the
  slots by. Acceptable for a local service ([ADR-0016](0016-local-only-no-hosted-instance.md)).
- Two slots is a guess for one GPU, not a measurement; revisit with a load
  test on the target hardware.

## Enforced by

- [`test_four_simultaneous_requests_serve_two_and_turn_two_away_at_once`](../../tests/test_service_load.py) and [`test_a_slot_is_given_back_when_its_generation_ends`](../../tests/test_service_load.py)
- [`test_a_request_past_its_time_budget_is_answered_with_504`](../../tests/test_service_load.py)
- [`test_every_request_is_logged_as_one_line_of_json`](../../tests/test_service_load.py) and [`test_a_request_turned_away_or_timed_out_is_still_logged`](../../tests/test_service_load.py)
- [`test_the_question_itself_is_not_logged`](../../tests/test_service_load.py)
- [`test_the_service_limits_generations_and_request_time_by_default`](../../tests/test_config.py)
