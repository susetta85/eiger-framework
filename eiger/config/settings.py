"""
Global runtime settings for EIGER.

Values are resolved from (highest priority first):
  1. Environment variables prefixed with EIGER_
  2. .env file in the working directory
  3. Defaults defined here

Why pydantic-settings:
  - Provides the same validation guarantees as Pydantic BaseModel (type
    coercion, constraint checking) but automatically reads values from
    environment variables and .env files.
  - env_prefix="EIGER_" namespaces all environment variables to avoid
    collisions with other tools (e.g. EIGER_QDRANT_HOST rather than
    the generic QDRANT_HOST).
  - case_sensitive=False allows EIGER_QDRANT_HOST and eiger_qdrant_host
    to be treated identically, which is convenient for CI environments
    that normalize variable names.

Why lru_cache on get_settings():
  - Settings are read once and cached as a module-level singleton. This
    avoids redundant .env file reads on every call and ensures that all
    parts of the application share the same settings object.
  - maxsize=1 is sufficient because only one settings instance is ever
    created; the function takes no arguments.
  - To override settings in tests, use EigerSettings(qdrant_host="test-host")
    directly rather than patching get_settings().

Usage:
    from eiger.config import get_settings
    cfg = get_settings()
    print(cfg.qdrant_host)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EigerSettings(BaseSettings):
    """
    Validated runtime settings for the EIGER framework.

    All fields have sensible defaults for local development. In production
    or CI, override individual fields via EIGER_-prefixed environment variables
    or a .env file in the working directory.

    What this class does NOT do:
      - It does not validate that the configured services are reachable;
        connectivity checks are the responsibility of infrastructure modules.
      - It does not persist settings; it reads from the environment on
        construction and is then immutable (Pydantic BaseModel semantics).
    """

    # SettingsConfigDict controls how pydantic-settings resolves values.
    # env_file=".env" means a .env in the CWD is auto-loaded if present;
    # this is intentionally not a hard requirement (missing .env is silently
    # ignored), so the framework works without a .env file in CI.
    model_config = SettingsConfigDict(
        env_prefix="EIGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─── Vector store ─────────────────────────────────────────────────────

    # Default "localhost" assumes a locally-running Qdrant instance,
    # which is the standard setup for local development and Docker Compose.
    qdrant_host: str = Field(default="localhost")
    # 6333 is Qdrant's default HTTP REST port.
    qdrant_port: int = Field(default=6333)

    # ─── LLM backend ──────────────────────────────────────────────────────

    # Ollama runs locally at localhost:11434 by default; no API key needed.
    ollama_host: str = Field(default="localhost")
    ollama_port: int = Field(default=11434)

    # ─── Embedding model ──────────────────────────────────────────────────

    # all-MiniLM-L6-v2 is a good default: 384-dim vectors, fast on CPU,
    # strong enough for benchmark-scale corpora, and freely available via
    # the Hugging Face hub without authentication.
    default_embedder: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ─── Experiment defaults ───────────────────────────────────────────────

    # results/ is a relative path; the experiment runner resolves it to
    # an absolute path at startup based on the working directory.
    results_dir: str = Field(default="results/")
    # "INFO" is the default log level; set EIGER_LOG_LEVEL=DEBUG for
    # verbose output during development.
    log_level: str = Field(default="INFO")
    # Seed 42 is the EIGER project convention — all experiments that do
    # not specify a seed will use this value, making cross-run comparisons
    # meaningful by default.
    default_seed: int = Field(default=42)

    @property
    def qdrant_url(self) -> str:
        """
        Fully-qualified HTTP URL for the Qdrant REST API.

        Computed from qdrant_host and qdrant_port so callers don't have
        to assemble the URL themselves.

        Returns:
            str: e.g. "http://localhost:6333"
        """
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def ollama_url(self) -> str:
        """
        Fully-qualified HTTP URL for the Ollama API.

        Returns:
            str: e.g. "http://localhost:11434"
        """
        return f"http://{self.ollama_host}:{self.ollama_port}"


@lru_cache(maxsize=1)
def get_settings() -> EigerSettings:
    """
    Return a cached singleton EigerSettings instance.

    The first call constructs the object (reading environment variables
    and the .env file); subsequent calls return the cached instance.

    Why singleton: settings are immutable after construction, and re-reading
    the environment on every call would be redundant and potentially fragile
    (if the environment changes mid-run, we want the original values).

    Returns:
        EigerSettings: The validated, cached settings object.
    """
    return EigerSettings()
