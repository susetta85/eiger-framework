.PHONY: help bootstrap setup install dev-install up up-docker-ollama ollama-pull down test test-unit test-integration lint type-check format clean

PYTHON := python3
PIP    := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Environment ─────────────────────────────────────────────────────────────
bootstrap: ## Fresh machine, nothing installed yet: installs Python/pip, Docker, Ollama (+ pulls its default model), then runs 'setup' (macOS via Homebrew, or Debian/Ubuntu via apt)
	./setup.sh

setup: ## Create virtualenv and install all dependencies (assumes python3.10+ already installed)
	$(PYTHON) -m venv venv
	@# Some Debian/Ubuntu python3-venv builds create a venv WITHOUT pip
	@# (pip is stripped from the distro's ensurepip bundle on some
	@# versions). Bootstrap it explicitly rather than assuming
	@# `python3 -m venv` always includes it — this is exactly what
	@# produced a confusing "pip: command not found" for a contributor
	@# on Debian even though setup.sh had already apt-installed
	@# python3-pip system-wide.
	@test -x ./venv/bin/pip || ./venv/bin/python -m ensurepip --upgrade
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -e ".[dev,data-import]"
	@echo "✅  Run: source venv/bin/activate"

install: ## Install package (production)
	$(PIP) install -e .

dev-install: ## Install package with dev dependencies
	$(PIP) install -e ".[dev]"

env: ## Copy .env.example to .env (first-time setup)
	@test -f .env || (cp .env.example .env && echo "✅  Created .env — fill in your values.")

# ─── Infrastructure ───────────────────────────────────────────────────────────
# Default: Qdrant via Docker, Ollama installed+run NATIVELY (by setup.sh /
# `make bootstrap`) rather than in Docker. Two reasons: native Ollama gets
# straightforward GPU access without container passthrough config, and it
# avoids a port-11434 clash with a Dockerized Ollama container (a native
# install and `docker compose`'s ollama service both bind the host's 11434
# — starting both at once fails). The Dockerized Ollama service still
# exists in docker-compose.yml, gated behind the "docker-ollama" Compose
# profile, for anyone who wants the exact-pinned-image reproducibility
# path instead — see docker-compose.yml's comments.
up: ## Start Qdrant via Docker (Ollama should already be running natively — see 'make bootstrap')
	@docker info >/dev/null 2>&1 || ( \
		echo "❌  Docker daemon not reachable."; \
		echo "    If you're using OrbStack: open it once from Applications/Launchpad"; \
		echo "    (or run: open -a OrbStack), wait a few seconds for it to finish"; \
		echo "    starting, then re-run 'make up'. Same idea for Docker Desktop."; \
		exit 1 \
	)
	docker compose up -d qdrant
	@echo "⏳  Waiting for Qdrant..."
	@sleep 3
	@curl -sf http://localhost:6333/healthz && echo "✅  Qdrant is up" || echo "❌  Qdrant not ready"
	@curl -sf http://localhost:11434/api/version >/dev/null 2>&1 \
		&& echo "✅  Ollama is up (native)" \
		|| echo "⚠️   Ollama not reachable on :11434 — see 'make bootstrap' or docker-compose.yml's docker-ollama profile"

up-docker-ollama: ## Alternative to native Ollama: start the pinned ollama/ollama:0.2.8 container too (do NOT run alongside native Ollama — port 11434 clash)
	docker compose --profile docker-ollama up -d

ollama-pull: ## Pull the default LLM into a NATIVE Ollama install (already done automatically by 'make bootstrap' — use this to re-pull or after a manual Ollama install)
	ollama pull llama3.1:8b

down: ## Stop all Docker services (does not touch a natively-installed Ollama)
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
