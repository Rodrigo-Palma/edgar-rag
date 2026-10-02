# Operations

Running the service beyond `make demo`: its limits, what it logs, what it
costs per question, the container, and the headline run's tape. Every number
here is copied from the [report](eval/report-v1.md), like the README's.

## Limits

The service reads the index once at startup and shares one HTTP client. At
most `EDGAR_RAG_MAX_CONCURRENT_GENERATIONS` (default `2`) generations run at
once; a request that needs another gets `503` with `Retry-After` at once
instead of queueing, and a question the gate declines is answered whatever the
load. A request past `EDGAR_RAG_REQUEST_TIMEOUT_SECONDS` (default `90`) gets
`504`.

There is no authentication and no rate limit, so the service binds to
`127.0.0.1` by default and logs a warning when `EDGAR_RAG_HOST` is not a
loopback address.

## Logs and health

Every request writes one line of JSON to stderr: status, total and per-stage
seconds, `reason`, `degraded`, every gate score, and the tokens of the
generation. The question itself is not logged. `/health` reports the indexed
filings, a fingerprint of the index and whether Ollama answers.

## Cost per question

Arm A, the default, by outcome on the headline run (means per question):

| outcome | n | prompt tokens | completion tokens | generator s |
|---|---|---|---|---|
| answered | 101 | 1648.8 | 25.1 | 15.10 |
| declined after generating | 379 | 1475.6 | 13.3 | 12.71 |

A decline costs nearly as much as an answer when the model is the one that
declines; a gate decline costs the embedding, the search and the gate.

## Container

With Ollama on the host where the GPU is (a container on a Mac cannot reach
Metal): `make up` builds the image, mounts `data/index` read-only and waits for
`/health`; `make down` stops it. The image holds the locked runtime
dependencies and the package, runs as an unprivileged user, and is published
on the host's `127.0.0.1` only.

## The headline run's tape

`make eval` reads `eval/runs/v1/cases.jsonl`, which is plain git. The run's
tape (every embedding, generation and brier reply it recorded) is in Git LFS
and excluded from every download by `.lfsconfig`; to fetch it:

```bash
git lfs pull --include="eval/runs/v1/tape/**" --exclude=""
```
