# 0011. Compose the service once in an app factory; no DI framework

- Status: Accepted (retroactive, decided in `5194c56` and `cd016f5` on 2026-10-01)
- Date: 2026-10-02

## Context

The first service was a module-level FastAPI app that built its collaborators
inside each request: it read the index from disk on every `/ask` and every
`/health`, and tests swapped them by mutating the global
`app.dependency_overrides`. One `Settings` class served the service and the
ingestion, so the service demanded the SEC contact it never uses.

## Decision

`create_app(settings, answerer=None, ...)` in `src/edgar_rag/service/app.py`
is the composition root. Its lifespan loads the index once (refusing a
mismatched one, [ADR-0004](0004-index-format-shard-per-filing.md)), opens one
HTTP client, builds the gate named by `EDGAR_RAG_GATE` and the `Answerer`, and
puts them in `app.state`; endpoints only read from there. Tests build their
own app with a fake answerer. Settings are split by entry point in
`src/edgar_rag/config.py`: `ServiceSettings`, `IngestSettings`,
`EvalSettings`, all under the `EDGAR_RAG_` prefix. `cli.py` and `create_app`
are the only places that instantiate adapters. No DI container.

## Consequences

- A request costs no disk read for the index; `/health` reports the loaded
  index and its fingerprint.
- Each test owns its app, so no test leaks state into another.
- The service starts without `EDGAR_RAG_EDGAR_USER_AGENT`; ingestion refuses
  to start without a real one.
- Wiring is explicit Python, so a new collaborator is a change to
  `create_app`; with a handful of collaborators that is cheaper than a
  container.

## Enforced by

- [`test_the_index_is_read_once_for_ten_requests`](../../tests/test_service_load.py)
- [`test_the_service_starts_without_an_edgar_user_agent`](../../tests/test_config.py) and [`test_ingestion_still_requires_the_sec_contact`](../../tests/test_config.py)
- [`test_every_variable_carries_the_project_prefix`](../../tests/test_config.py)
- [`test_health_reports_a_missing_index_instead_of_failing`](../../tests/test_api.py) and [`test_ask_without_an_index_is_unavailable_not_a_crash`](../../tests/test_api.py)
- The import-linter layers contract (`pyproject.toml`, `make imports`): only
  the entry points sit above the pipeline.
