"""
Global runtime settings for EIGER.

Values are resolved from (highest priority first):
  1. Environment variables prefixed with EIGER_
  2. .env file in the working directory
  3. Defaults defined here

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
    model_config = SettingsConfigDict(
        env_prefix="EIGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─── Vector store ─────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)

    # ─── LLM backend ──────────────────────────────────────────────────────
    ollama_host: str = Field(default="localhost")
    ollama_port: int = Field(default=11434)

    # ─── Embedding model ──────────────────────────────────────────────────
    default_embedder: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ─── Experiment defaults ───────────────────────────────────────────────
    results_dir: str = Field(default="results/")
    log_level: str = Field(default="INFO")
    default_seed: int = Field(default=42)

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def ollama_url(self) -> str:
        return f"http://{self.ollama_host}:{self.ollama_port}"


@lru_cache(maxsize=1)
def get_settings() -> EigerSettings:
    """Return a cached singleton settings instance."""
    return EigerSettings()
