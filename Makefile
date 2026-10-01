# Local gate. `make check` is what CI runs; run it before every commit.

UV_RUN := uv run --frozen
SRC := src tests scripts

.DEFAULT_GOAL := help
.PHONY: help sync check lint format typecheck imports test audit

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

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
