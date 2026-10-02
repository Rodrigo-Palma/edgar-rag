# 0016. Local only, no hosted instance

- Status: Accepted
- Date: 2026-10-02

## Context

The numbers in the README describe one configuration: `qwen3:32b` on Ollama
`0.18.0`, an Apple M3 Max GPU, `nomic-embed-text` with lower-cased input.
A generation takes 12.913 s at P50 there ([report](../eval/report-v1.md)).

Hosting that configuration needs a GPU with enough memory for a 32B model,
paid by the hour, and an account with a cloud provider. The project runs at
no cost and with no cloud account. Hosting a smaller model on a CPU instead
would put a different system online than the one the evaluation measured,
and its answers would not be the ones the report describes.

The service also has no authentication and no rate limit.

## Decision

There is no hosted instance. The service binds to `127.0.0.1` by default
(`ServiceSettings.host`). What stands in for a URL:

- `make demo` serves the dev split in replay mode (`EDGAR_RAG_MODE=replay`)
  with no model: the index, search, gates and citation check run for real,
  and embeddings and generations come from the committed tape. It answers
  only the golden set's recorded questions.
- `make up` runs the service in a container, published on the host's
  `127.0.0.1`, with Ollama on the host where the GPU is.
- `make eval` rebuilds the report from the frozen run without a model.

## Consequences

- A reader can check every claim locally without a GPU, but cannot type an
  arbitrary question at a public URL. The README says so on its first screen.
- Nothing is exposed, so the missing authentication and rate limit are not a
  risk as shipped. Binding to another interface requires both first.
- Revisit if the measured configuration becomes cheap to host, or if a
  smaller model is evaluated under the same protocol; then the hosted system
  and the measured one would be the same.

## Enforced by

- [`test_the_service_binds_to_the_local_machine_by_default`](../../tests/test_config.py) and [`test_the_service_is_served_on_the_local_machine_unless_configured_otherwise`](../../tests/test_api.py)
- [`test_the_committed_tape_answers_one_demo_question_and_declines_the_other`](../../tests/eval/test_replay_serving.py) and [`test_a_question_the_tape_does_not_hold_is_404_and_reaches_no_model`](../../tests/test_replay_service.py)
- The CI step "make demo answers one question and declines the other without
  a model" in the `eval` job, and the `image` job, which starts the container
  and reads `/health` with no index and no Ollama.
