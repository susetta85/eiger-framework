#!/usr/bin/env bash
#
# setup.sh — bootstrap a fresh macOS machine for EIGER, from zero.
#
# For machines that already have Python 3.10+ and Docker/OrbStack, `make
# setup` alone is enough — this script exists for the "I have nothing
# installed yet" case (no Homebrew, no Python, no Docker), and simply
# installs each missing piece before handing off to `make setup`.
#
# What this script does, in order:
#   1. Installs Homebrew if missing (https://brew.sh — the standard
#      macOS package manager; everything else is installed through it).
#   2. Installs Python 3.12 via Homebrew if python3 is missing or older
#      than the 3.10 minimum in pyproject.toml's requires-python.
#   3. Offers to install OrbStack (a lightweight Docker Desktop
#      alternative for macOS) via Homebrew Cask if no `docker` command
#      is found. Docker Desktop works exactly as well — OrbStack is
#      just lighter/faster on a Mac; either one provides the same
#      `docker`/`docker compose` CLI this project's Makefile calls.
#   4. Runs `make setup` (creates venv/, installs the package with the
#      dev + data-import extras).
#
# What this script does NOT do:
#   - It does not start Docker services (`make up`) or pull the Ollama
#     model — see the printed "Next steps" at the end.
#   - It does not touch ContainerLab (infra/containerlab/) — that stack
#     is entirely optional, only for distributed network-topology
#     research, and is not needed to run experiments. Docker Compose
#     (via OrbStack/Docker Desktop) is all that's required.
#   - It is macOS-only (uses Homebrew). On Linux, install python3.10+,
#     pip, and Docker via your distribution's package manager, then run
#     `make setup` directly.
#
# Usage:
#   ./setup.sh
#   make bootstrap   # equivalent, via the Makefile

set -euo pipefail

_GREEN='\033[0;32m'
_YELLOW='\033[0;33m'
_RESET='\033[0m'

info()  { echo -e "${_GREEN}==>${_RESET} $1"; }
warn()  { echo -e "${_YELLOW}!!${_RESET} $1"; }

if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "This script is macOS-only (it uses Homebrew)."
  warn "On Linux: install python3.10+, pip, and Docker via your package"
  warn "manager, then run 'make setup' directly."
  exit 1
fi

# ─── 1. Homebrew ───────────────────────────────────────────────────────────

if ! command -v brew >/dev/null 2>&1; then
  info "Homebrew not found — installing (https://brew.sh)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Homebrew installs to different prefixes on Apple Silicon vs Intel Macs.
  eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null \
    || eval "$(/usr/local/bin/brew shellenv)" 2>/dev/null \
    || true
else
  info "Homebrew found ($(brew --version | head -1))"
fi

# ─── 2. Python 3.10+ ────────────────────────────────────────────────────────

_python_ok() {
  command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
}

if _python_ok; then
  info "Python found: $(python3 --version)"
else
  info "Installing Python 3.12 via Homebrew..."
  brew install python@3.12
  if ! _python_ok; then
    warn "python3 still not on PATH after install. Open a new terminal"
    warn "(or run 'eval \"\$(brew shellenv)\"') and re-run this script."
    exit 1
  fi
fi

# pip ships with Homebrew's python3, but ensure it's present/up to date.
python3 -m ensurepip --upgrade >/dev/null 2>&1 || true

# ─── 3. Docker (OrbStack or Docker Desktop) ────────────────────────────────

if command -v docker >/dev/null 2>&1; then
  info "Docker found: $(docker --version)"
else
  warn "No 'docker' command found. EIGER needs it to run Qdrant + Ollama"
  warn "(via 'make up' — plain docker compose, nothing more exotic)."
  read -r -p "Install OrbStack now via Homebrew Cask? [Y/n] " reply
  reply=${reply:-Y}
  if [[ "$reply" =~ ^[Yy] ]]; then
    brew install --cask orbstack
    warn "Open OrbStack once from Applications to finish its first-run"
    warn "setup, then re-run this script (or just 'make setup' directly —"
    warn "Python is already installed at this point)."
    exit 0
  else
    warn "Skipping. Install Docker Desktop or OrbStack manually before"
    warn "running 'make up'."
  fi
fi

# ─── 4. Project virtualenv + dependencies ──────────────────────────────────

info "Running 'make setup' (creates venv/, installs the package)..."
make setup

echo ""
info "Bootstrap complete. Next steps:"
echo "    1. source venv/bin/activate"
echo "    2. make env      # creates .env from .env.example"
echo "    3. make up       # starts Qdrant + Ollama (Docker/OrbStack must be running)"
echo "    4. docker exec eiger-ollama ollama pull llama3.1:8b"
echo "    5. make test-unit"
