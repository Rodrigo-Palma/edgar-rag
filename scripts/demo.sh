#!/usr/bin/env bash
# `make demo`: the service in replay mode, asked the two questions the README
# shows, with no model running. It runs the period+cosine gate, not the
# default none, so the second question shows a decline before generation.
# The index is eval/ci/index and the replies come from the qwen3:8b tape in
# eval/ci/tape; both need `git lfs pull` in the clone.
#
# EDGAR_RAG_DEMO_VIEW=brief prints, per question, the outcome, the citation
# and the stages the request's log line timed (scripts/demo_brief.py), instead
# of the full JSON. The checks at the end run in both views.
set -euo pipefail

port="${EDGAR_RAG_DEMO_PORT:-8077}"
view="${EDGAR_RAG_DEMO_VIEW:-full}"
case "$view" in
    full | brief) ;;
    *) echo "EDGAR_RAG_DEMO_VIEW is full or brief, not '$view'" >&2; exit 2 ;;
esac
log="$(mktemp)"
# Stop the server and wait for it to free the port, keeping the script's status.
trap 'status=$?; kill "${pid:-}" 2>/dev/null; wait "${pid:-}" 2>/dev/null || true; rm -f "$log"; exit "$status"' EXIT

EDGAR_RAG_MODE=replay \
EDGAR_RAG_GATE=period+cosine \
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
    if [ "$view" = full ]; then echo "\$ curl -s localhost:$port/ask -d '$1'" >&2; fi
    curl --silent --show-error --fail-with-body "localhost:$port/ask" \
        -H 'content-type: application/json' -d "$1"
}

answer=$(ask '{"cik": 320193, "fiscal_year": 2025, "question": "How much revenue did Apple report for fiscal year 2025?"}')
if [ "$view" = full ]; then echo "$answer" | python3 -m json.tool; fi
decline=$(ask '{"cik": 320193, "fiscal_year": 2025, "question": "What cash dividends did Apple pay to shareholders in fiscal 2020?"}')
if [ "$view" = full ]; then echo "$decline" | python3 -m json.tool; fi

if [ "$view" = brief ]; then
    echo "$answer" | python3 scripts/demo_brief.py "$log" 1 && echo
    echo "$decline" | python3 scripts/demo_brief.py "$log" 2 && echo
fi

# The demo shows what it promises, or fails: a guard that never declines, or
# a relevance gate that admits the wrong side, must not exit 0.
python3 - "$answer" "$decline" <<'CHECK'
import json
import sys

answer, decline = (json.loads(body) for body in sys.argv[1:3])
problems = []
if answer["abstained"] or not answer["citations"]:
    problems.append(f"the first question was not answered with a citation: {answer['reason']}")
if not decline["abstained"] or decline["reason"] != "out_of_period":
    problems.append(f"the second question was not declined as out_of_period: {decline['reason']}")
if problems:
    sys.exit("make demo: " + "; ".join(problems))
print("demo: one answer with a citation, one out_of_period decline, no model")
CHECK
