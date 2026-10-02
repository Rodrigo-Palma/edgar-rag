# 0010. Reach models through injected HTTP clients, without a framework

- Status: Accepted (retroactive, decided in `22651bd` on 2026-10-01)
- Date: 2026-10-02

## Context

The adapters called `httpx.post` at module level: no connection pool, and
testable only by monkeypatching the module. A test that forgot the patch
would quietly call a local Ollama when one happened to be running.

The pipeline makes two kinds of model call, an embedding and a generation,
both to Ollama's HTTP API. A framework (LangChain, LlamaIndex) would add a
layer of abstraction over two POST requests and hide the request the
evaluation needs to record byte for byte.

## Decision

`OllamaEmbedder` and `OllamaGenerator` in `src/edgar_rag/models.py` take an
optional `httpx.Client` at construction and send every request through it.
The generator's options are fixed (`temperature=0`, `seed=0`, thinking off,
`num_ctx=8192`), and it returns a `Generation` with the text, prompt and
completion tokens, and seconds. Both implement the `Embedder` and `Generator`
ports in `src/edgar_rag/domain.py`; the answering core never imports
`models.py`. No LLM framework.

## Consequences

- The service opens one client in its lifespan and closes it at shutdown
  ([ADR-0011](0011-composition-root-in-an-app-factory.md)).
- Tests use `httpx.MockTransport`, and an autouse fixture fails any test that
  reaches a real socket, so no test depends on a model being up.
- Recording and replay ([ADR-0009](0009-ci-eval-gate-replayed.md)) wrap the
  ports, not HTTP, so they key on what the model receives.
- Another provider is another adapter of the same ports. None is needed yet.

## Enforced by

- [`test_an_embedder_given_a_client_sends_through_it`](../../tests/test_models.py) and [`test_a_generator_given_a_client_sends_through_it`](../../tests/test_models.py)
- [`test_the_generator_asks_for_the_same_answer_every_time`](../../tests/test_models.py) and [`test_the_generator_reports_its_tokens_and_the_seconds_it_took`](../../tests/test_models.py)
- [`test_the_models_are_called_through_one_client_closed_at_shutdown`](../../tests/test_service_load.py)
- [`no_network`](../../tests/conftest.py): the autouse fixture.
- The import-linter contract "The answering core sees adapters only through
  the ports in domain" (`pyproject.toml`, `make imports`).
