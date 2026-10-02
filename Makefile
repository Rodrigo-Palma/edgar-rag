# Local gate. `make check` is what CI runs; run it before every commit.

UV_RUN := uv run --frozen
SRC := src tests

.DEFAULT_GOAL := help
.PHONY: help sync check lint format typecheck imports test audit serve demo image up down \
	ingest eval eval-full

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-17s %s\n", $$1, $$2}'

sync: ## Install the locked dependencies, dev tools included
	uv sync --frozen

check: lint typecheck imports test ## Everything CI checks except the dependency audit

lint: ## Lint and check formatting
	$(UV_RUN) ruff check $(SRC)
	$(UV_RUN) ruff format --check $(SRC)

format: ## Apply lint fixes and formatting
	$(UV_RUN) ruff check --fix $(SRC)
	$(UV_RUN) ruff format $(SRC)

typecheck: ## mypy --strict over the package
	$(UV_RUN) mypy --strict src

imports: ## Layer contracts between modules (import-linter)
	$(UV_RUN) lint-imports

test: ## Tests with branch coverage, failing under 85%
	$(UV_RUN) pytest --cov=edgar_rag --cov-report=term --cov-fail-under=85

audit: ## Known vulnerabilities in the locked dependencies (needs network)
	@tmp=$$(mktemp) && trap 'rm -f "$$tmp"' EXIT && \
	uv export --frozen --format requirements-txt --no-emit-project --all-groups -o "$$tmp" >/dev/null && \
	uvx --from 'pip-audit>=2.7,<3' pip-audit --strict --disable-pip --require-hashes -r "$$tmp"

serve: ## Serve answers from the index on 127.0.0.1:8000 (needs Ollama)
	$(UV_RUN) edgar-rag serve

demo: ## Replay two recorded questions through the service: one answer, one decline (no model)
	./scripts/demo.sh

# --- evaluation --------------------------------------------------------------

LOCK := eval/filings.lock.json
RUNS := eval/runs
REPORTS := docs/eval
HEADLINE_RUN := $(RUNS)/v1
HEADLINE_MODEL := qwen3:32b

ingest: ## Index the golden set's 48 pinned filings from their snapshots (needs Ollama)
	$(UV_RUN) edgar-rag ingest --lock $(LOCK)

eval: ## Rebuild docs/eval/ from the frozen runs in eval/runs/ (no model, no network)
	@runs=$$(find $(RUNS) -mindepth 2 -maxdepth 2 -name manifest.json -exec dirname {} \; 2>/dev/null | sort); \
	if [ -z "$$runs" ]; then \
		echo "no frozen run in $(RUNS)/ yet: make eval-full writes $(HEADLINE_RUN)"; exit 0; \
	fi; \
	mkdir -p $(REPORTS); \
	for run in $$runs; do \
		out=$(REPORTS)/report-$$(basename $$run).md; \
		$(UV_RUN) edgar-rag eval report --run $$run --out $$out || exit 1; \
		echo "$$out"; \
	done

eval-full: ## The pre-registered round on the eval split, into eval/runs/v1 (needs Ollama, hours)
	$(UV_RUN) edgar-rag eval run --split eval --narratives --repeat 30 \
		--protocol $(REPORTS)/protocol.md --generation-model $(HEADLINE_MODEL) --out $(HEADLINE_RUN)
	$(MAKE) eval

IMAGE := edgar-rag:local

image: ## Build the service image and print its size
	docker build -t $(IMAGE) .
	@docker run --rm --entrypoint sh $(IMAGE) -c 'du -sxh / 2>/dev/null' | awk '{print "image filesystem: " $$1}'

up: ## Serve from the container, index from EDGAR_RAG_INDEX (default data/index), Ollama on the host
	docker compose up --build --detach --wait

down: ## Stop the container
	docker compose down
