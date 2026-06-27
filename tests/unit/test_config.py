"""
Unit tests for EigerSettings (eiger.config.settings).

Tests verify:
  - Default field values are correct for local-development use
  - Computed URL properties assemble the correct scheme/host/port strings
  - get_settings() returns an EigerSettings instance

What these tests do NOT cover:
  - Reading from environment variables (requires process-level env patching;
    tested in integration tests).
  - .env file loading (same reason).
  - lru_cache singleton behaviour; tested implicitly by calling get_settings()
    twice and comparing identity.
"""

from __future__ import annotations

import pytest

from eiger.config.settings import EigerSettings, get_settings


class TestEigerSettingsDefaults:
    """Verify that out-of-the-box defaults match the documented values."""

    def test_default_qdrant_host(self) -> None:
        """qdrant_host must default to 'localhost' for local Docker Compose usage."""
        s = EigerSettings()
        assert s.qdrant_host == "localhost"

    def test_default_qdrant_port(self) -> None:
        """qdrant_port must default to 6333 (Qdrant's standard REST port)."""
        s = EigerSettings()
        assert s.qdrant_port == 6333

    def test_default_ollama_host(self) -> None:
        """ollama_host must default to 'localhost'."""
        s = EigerSettings()
        assert s.ollama_host == "localhost"

    def test_default_ollama_port(self) -> None:
        """ollama_port must default to 11434 (Ollama's default port)."""
        s = EigerSettings()
        assert s.ollama_port == 11434

    def test_default_seed(self) -> None:
        """default_seed must be 42 — the EIGER project-wide convention."""
        s = EigerSettings()
        assert s.default_seed == 42

    def test_default_log_level(self) -> None:
        """log_level must default to 'INFO' for production-safe verbosity."""
        s = EigerSettings()
        assert s.log_level == "INFO"

    def test_default_results_dir(self) -> None:
        """results_dir must default to 'results/'."""
        s = EigerSettings()
        assert s.results_dir == "results/"


class TestEigerSettingsURLProperties:
    """Verify that computed URL properties produce correct HTTP URLs."""

    def test_qdrant_url_default(self) -> None:
        """qdrant_url must combine qdrant_host and qdrant_port into an HTTP URL."""
        s = EigerSettings()
        assert s.qdrant_url == "http://localhost:6333"

    def test_qdrant_url_custom(self) -> None:
        """qdrant_url must reflect overridden host and port values."""
        s = EigerSettings(qdrant_host="qdrant-server", qdrant_port=1234)
        assert s.qdrant_url == "http://qdrant-server:1234"

    def test_ollama_url_default(self) -> None:
        """ollama_url must combine ollama_host and ollama_port into an HTTP URL."""
        s = EigerSettings()
        assert s.ollama_url == "http://localhost:11434"

    def test_ollama_url_custom(self) -> None:
        """ollama_url must reflect overridden host and port values."""
        s = EigerSettings(ollama_host="remote-gpu", ollama_port=9999)
        assert s.ollama_url == "http://remote-gpu:9999"


class TestGetSettings:
    """Verify the cached factory function get_settings()."""

    def test_returns_eiger_settings_instance(self) -> None:
        """get_settings() must return an EigerSettings object."""
        cfg = get_settings()
        assert isinstance(cfg, EigerSettings)

    def test_returns_same_instance_on_repeated_calls(self) -> None:
        """
        get_settings() is decorated with lru_cache(maxsize=1), so two calls
        must return the exact same object (identity check, not equality).
        """
        cfg1 = get_settings()
        cfg2 = get_settings()
        assert cfg1 is cfg2
