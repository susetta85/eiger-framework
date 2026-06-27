.PHONY: help setup install dev-install up down test test-unit test-integration lint type-check format clean

PYTHON := python3
PIP    := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Environment ─────────────────────────────────────────────────────────────
setup: ## Create virtualenv and install all dependencies
	$(PYTHON) -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e ".[dev]"
	@echo "✅  Run: source venv/bin/activate"

install: ## Install package (production)
	$(PIP) install -e .

dev-install: ## Install package with dev dependencies
	$(PIP) install -e ".[dev]"

env: ## Copy .env.example to .env (first-time setup)
	@test -f .env || (cp .env.example .env && echo "✅  Created .env — fill in your values.")

# ─── Infrastructure ───────────────────────────────────────────────────────────
up: ## Start Qdrant + Ollama via Docker Compose
	docker compose up -d
	@echo "⏳  Waiting for Qdrant..."
	@sleep 3
	@curl -sf http://localhost:6333/healthz && echo "✅  Qdrant is up" || echo "❌  Qdrant not ready"

down: ## Stop all services
	docker compose down

# ─── Tests ───────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage
	$(PYTEST) tests/ -v

test-unit: ## Run unit tests only (no external services)
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests (requires: make up)
	$(PYTEST) tests/integration/ -v

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint: ## Lint with ruff
	$(PYTHON) -m ruff check eiger/ tests/

format: ## Auto-format with ruff
	$(PYTHON) -m ruff format eiger/ tests/

type-check: ## Type-check with mypy
	$(PYTHON) -m mypy eiger/

security: ## Security scan with bandit
	$(PYTHON) -m bandit -r eiger/ -ll

audit: ## Audit dependencies with pip-audit
	$(PYTHON) -m pip_audit

# ─── Experiments ─────────────────────────────────────────────────────────────
run: ## Run an experiment: make run CFG=experiments/baseline.yaml
	$(PYTHON) -m eiger run $(CFG)

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
