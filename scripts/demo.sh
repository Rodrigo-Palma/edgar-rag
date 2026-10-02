#!/usr/bin/env bash
# `make demo`: the service in replay mode, asked the two questions the README
# shows, with no model running. The index is eval/ci/index and the replies come
# from the qwen3:8b tape in eval/ci/tape; both need `git lfs pull` in the clone.
set -euo pipefail

port="${EDGAR_RAG_DEMO_PORT:-8077}"
log="$(mktemp)"
trap 'kill "${pid:-}" 2>/dev/null || true; rm -f "$log"' EXIT

EDGAR_RAG_MODE=replay \
EDGAR_RAG_INDEX_DIR=eval/ci/index \
EDGAR_RAG_PORT="$port" \
    uv run --frozen edgar-rag serve >"$log" 2>&1 &
pid=$!

for _ in $(seq 1 60); do
    if curl --silent --fail "localhost:$port/health" >/dev/null; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then cat "$log" >&2; exit 1; fi
    sleep 0.5
done
curl --silent --fail "localhost:$port/health" >/dev/null || { cat "$log" >&2; exit 1; }

ask() {
    echo "\$ curl -s localhost:$port/ask -d '$1'"
    curl --silent --show-error --fail-with-body "localhost:$port/ask" \
        -H 'content-type: application/json' -d "$1" | python3 -m json.tool
    echo
}

ask '{"cik": 320193, "fiscal_year": 2025, "question": "How much revenue did Apple report for fiscal year 2025?"}'
ask '{"cik": 320193, "fiscal_year": 2025, "question": "What cash dividends did Apple pay to shareholders in fiscal 2020?"}'
