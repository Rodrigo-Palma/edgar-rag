# Sourced, not run, by docs/media/demo.tape, from the repo root:
#   source scripts/demo_tour.sh
# Starts the service `make demo` starts (replay of eval/ci/tape over
# eval/ci/index, gate period+cosine, no model) in the background, and defines
#   ask YEAR "QUESTION"   POST /ask about Apple's (CIK 320193) filing for
#                         fiscal YEAR, shown by scripts/demo_brief.py
#   tour_stop             stop the service
# Replay answers only questions recorded in the tape; any other gets 404.

tour_port="${EDGAR_RAG_DEMO_PORT:-8077}"
tour_log="$(mktemp)"
tour_asked=0

EDGAR_RAG_MODE=replay \
EDGAR_RAG_GATE=period+cosine \
EDGAR_RAG_INDEX_DIR=eval/ci/index \
EDGAR_RAG_PORT="$tour_port" \
    uv run --frozen edgar-rag serve >"$tour_log" 2>&1 &
tour_pid=$!

for _ in $(seq 1 60); do
    curl --silent --fail "localhost:$tour_port/health" >/dev/null && break
    if ! kill -0 "$tour_pid" 2>/dev/null; then cat "$tour_log" >&2; break; fi
    sleep 0.5
done

ask() {
    local body
    body=$(python3 -c 'import json, sys; print(json.dumps({"cik": 320193, "fiscal_year": int(sys.argv[1]), "question": sys.argv[2]}))' "$1" "$2") || return
    tour_asked=$((tour_asked + 1))
    curl --silent --show-error "localhost:$tour_port/ask" \
        -H 'content-type: application/json' -d "$body" |
        python3 scripts/demo_brief.py "$tour_log" "$tour_asked"
}

tour_stop() {
    kill "$tour_pid" 2>/dev/null
    wait "$tour_pid" 2>/dev/null
    rm -f "$tour_log"
}
