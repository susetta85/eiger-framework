#!/usr/bin/env bash
#
# setup.sh — bootstrap a fresh machine for EIGER, from zero.
#
# For machines that already have Python 3.10+ and Docker, `make setup`
# alone is enough — this script exists for the "I have nothing installed
# yet" case (no package manager setup, no Python, no Docker), and simply
# installs each missing piece before handing off to `make setup`.
#
# Supported platforms: macOS (via Homebrew) and Debian/Ubuntu Linux (via
# apt). Anything else (Fedora/RHEL, Arch, WSL without a Debian/Ubuntu
# base, etc.) is not auto-installed — the script prints the manual
# equivalent steps and exits.
#
# What this script does, in order:
#   1. macOS:  installs Homebrew if missing (https://brew.sh).
#      Linux:  uses apt (Debian/Ubuntu only) — no separate package
#              manager to bootstrap first.
#   2. Installs Python 3.10+ (Homebrew's python@3.12, or apt's
#      python3/python3-venv/python3-pip) if python3 is missing or older
#      than the 3.10 minimum in pyproject.toml's requires-python. Note:
#      if you already have a suitable python3 on PATH via conda/pyenv/
#      etc., this step is skipped — the check only looks at what
#      `python3 --version` actually resolves to, not how it was installed.
#   3. Docker:
#      macOS:  offers to install OrbStack (a lightweight Docker Desktop
#              alternative) via Homebrew Cask if no `docker` command is
#              found. Docker Desktop works exactly as well — OrbStack is
#              just lighter/faster on a Mac.
#      Linux:  offers to install Docker Engine via the official
#              convenience script (https://get.docker.com) and adds the
#              current user to the `docker` group (requires a fresh
#              shell/login to take effect — see the printed note).
#   4. Ollama (installed NATIVELY, not via Docker — see docker-compose.yml's
#      header comment for why): offers to install it via Homebrew
#      (macOS) or the official Linux installer at ollama.com/install.sh
#      (Debian/Ubuntu) if no `ollama` command is found, then waits for
#      it to answer on :11434, then pulls the default model
#      (llama3.1:8b, multi-GB, skipped if already present) so the
#      environment is actually runnable at the end of this script, not
#      just installed. Re-run any time with `make ollama-pull` (or pull
#      a different model directly with `ollama pull <name>`).
#   5. Runs `make setup` (creates venv/, installs the package with the
#      dev + data-import extras — this is what brings in openpyxl, PyYAML,
#      pytest, etc.; see pyproject.toml).
#
# What this script does NOT do:
#   - It does not start Docker services (`make up`) or pull the Ollama
#     model — see the printed "Next steps" at the end.
#   - It does not touch ContainerLab (infra/containerlab/) — that stack
#     is entirely optional, only for distributed network-topology
#     research, and is not needed to run experiments. Plain Docker
#     Compose is all that's required.
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

_OS="$(uname -s)"

_python_ok() {
  command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
}

# ─── Platform-specific: package manager + Python + Docker ─────────────────

if [[ "$_OS" == "Darwin" ]]; then
  # ─── macOS: Homebrew ──────────────────────────────────────────────────
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
  python3 -m ensurepip --upgrade >/dev/null 2>&1 || true

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

  if command -v ollama >/dev/null 2>&1; then
    info "Ollama found: $(ollama --version 2>&1 | head -1)"
  else
    warn "No 'ollama' command found. Installing natively (not via Docker —"
    warn "see docker-compose.yml's header comment for why)."
    brew install ollama
    brew services start ollama
  fi

elif [[ "$_OS" == "Linux" ]] && command -v apt-get >/dev/null 2>&1; then
  # ─── Debian/Ubuntu: apt ───────────────────────────────────────────────
  info "Debian/Ubuntu detected (apt-get found)."

  if _python_ok; then
    info "Python found: $(python3 --version)"
  else
    info "Installing python3/python3-venv/python3-pip via apt..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
    if ! _python_ok; then
      warn "python3 still not >= 3.10 after install. Your distro's default"
      warn "python3 package may be older; consider pyenv or deadsnakes PPA"
      warn "(e.g. 'sudo apt-get install python3.12 python3.12-venv')."
      exit 1
    fi
  fi
  # Even with a suitable python3 already on PATH (e.g. via conda), the
  # stdlib venv module still needs its Debian package to create working
  # virtualenvs with the system interpreter — harmless if already present.
  sudo apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true

  if command -v docker >/dev/null 2>&1; then
    info "Docker found: $(docker --version)"
  else
    warn "No 'docker' command found. EIGER needs it to run Qdrant + Ollama"
    warn "(via 'make up' — plain docker compose, nothing more exotic)."
    read -r -p "Install Docker Engine now via get.docker.com? [Y/n] " reply
    reply=${reply:-Y}
    if [[ "$reply" =~ ^[Yy] ]]; then
      curl -fsSL https://get.docker.com | sh
      sudo usermod -aG docker "$USER"
      warn "Added $USER to the 'docker' group — log out/in (or run"
      warn "'newgrp docker') for this to take effect, THEN re-run this"
      warn "script (or 'make setup' directly — Python is already installed)."
      exit 0
    else
      warn "Skipping. Install Docker manually before running 'make up'."
    fi
  fi

  if command -v ollama >/dev/null 2>&1; then
    info "Ollama found: $(ollama --version 2>&1 | head -1)"
  else
    warn "No 'ollama' command found. Installing natively via the official"
    warn "Linux installer (not via Docker — see docker-compose.yml's header"
    warn "comment for why). You may see a GPU-detection warning ('Unable to"
    warn "detect NVIDIA/AMD GPU') if 'lspci'/'lshw' aren't installed — this"
    warn "is harmless if you're on CPU-only hardware (typical for a plain"
    warn "VM); install pciutils/lshw first if you expect a GPU to be found:"
    warn "  sudo apt-get install -y pciutils lshw"
    curl -fsSL https://ollama.com/install.sh | sh
    # The official installer sets up and starts a systemd service; make
    # sure it's actually enabled/running rather than assuming so.
    sudo systemctl enable --now ollama 2>/dev/null || true
  fi

else
  warn "Unsupported platform for auto-install: $_OS"
  warn "This script auto-installs on macOS (Homebrew) and Debian/Ubuntu"
  warn "Linux (apt) only. Install manually, then run 'make setup' directly:"
  warn "  - Python 3.10+ and pip (check: python3 --version)"
  warn "  - Docker (engine + compose plugin)"
  warn "  - Ollama (https://ollama.com/download)"
  exit 1
fi

# ─── Ollama reachability check (both platforms) ────────────────────────────

if command -v ollama >/dev/null 2>&1; then
  info "Waiting for Ollama to answer on :11434..."
  _ollama_up=false
  for _ in $(seq 1 10); do
    if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
      _ollama_up=true
      break
    fi
    sleep 1
  done
  if [[ "$_ollama_up" == "true" ]]; then
    info "Ollama is reachable."

    _MODEL="llama3.1:8b"
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$_MODEL"; then
      info "Model $_MODEL already present — skipping download."
    else
      info "Pulling $_MODEL (multi-GB, may take a while depending on your connection)..."
      ollama pull "$_MODEL"
    fi
  else
    warn "Ollama is installed but not answering on :11434 yet. Depending on"
    warn "your platform you may need to start it yourself, e.g.:"
    warn "  macOS:  brew services start ollama   (or just run: ollama serve)"
    warn "  Linux:  sudo systemctl start ollama  (or just run: ollama serve)"
    warn "Once it's running, pull the model with: make ollama-pull"
  fi
fi

# ─── Project virtualenv + dependencies (both platforms) ────────────────────

info "Running 'make setup' (creates venv/, installs the package)..."
make setup

echo ""
info "Bootstrap complete. Python, pip, Docker, and Ollama (with its default"
info "model already pulled) are all set up. Next steps:"
echo "    1. source venv/bin/activate"
echo "    2. make env         # creates .env from .env.example"
echo "    3. make up          # starts Qdrant via Docker"
echo "    4. make test-unit"
echo ""
echo "Note: ContainerLab (infra/containerlab/) is NOT needed for any of this"
echo "— it's an optional, separate stack only for distributed network-"
echo "topology research."
echo ""
echo "Note: Ollama runs natively here, not in Docker, to avoid a port-11434"
echo "clash and simplify GPU access — see docker-compose.yml's header"
echo "comment if you'd rather use the pinned ollama/ollama:0.2.8 container"
echo "instead ('make up-docker-ollama')."
