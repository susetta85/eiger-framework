"""
Unit tests for structured logging utilities (eiger.utils.logging).

Tests verify:
  - configure_logging() does not raise for standard log levels ("INFO", "DEBUG")
  - configure_logging() degrades gracefully for invalid level strings, falling
    back to INFO via the getattr(logging, level.upper(), logging.INFO) default
  - get_logger() returns a non-None structlog logger for any module name
  - Different logger names do not cause errors (smoke test for name binding)

What these tests do NOT cover:
  - The exact format of log output (console vs. JSON renderer; tested in
    integration/snapshot tests once the snapshot suite is added in Sprint 3).
  - Thread-safety of structlog's context variable mechanism.
  - Log level filtering behaviour (structlog handles this internally).
"""

from __future__ import annotations

from eiger.utils.logging import configure_logging, get_logger


class TestConfigureLogging:
    """Tests for the configure_logging() setup function."""

    def test_configure_with_info_level_does_not_raise(self) -> None:
        """
        configure_logging("INFO") must complete without raising any exception.

        This is the standard startup call in all experiment runners and
        the pipeline quickstart script.
        """
        configure_logging("INFO")  # must not raise

    def test_configure_with_debug_level_does_not_raise(self) -> None:
        """
        configure_logging("DEBUG") must complete without raising any exception.

        DEBUG is used during development and reproducing experiments locally
        with verbose output enabled.
        """
        configure_logging("DEBUG")  # must not raise

    def test_configure_with_warning_level_does_not_raise(self) -> None:
        """configure_logging("WARNING") must complete without raising any exception."""
        configure_logging("WARNING")  # must not raise

    def test_configure_with_invalid_level_falls_back_gracefully(self) -> None:
        """
        configure_logging() with an unrecognised level string must NOT raise.

        The implementation uses ``getattr(logging, level.upper(), logging.INFO)``
        which returns ``logging.INFO`` for any unrecognised string, so invalid
        levels silently fall back to INFO. This prevents configuration errors
        from crashing the experiment runner at startup.
        """
        configure_logging("NOT_A_REAL_LEVEL")  # must not raise


class TestGetLogger:
    """Tests for the get_logger() factory function."""

    def test_returns_non_none_logger(self) -> None:
        """get_logger() must return a non-None structlog logger object."""
        log = get_logger("eiger.test_module")
        assert log is not None

    def test_different_names_both_succeed(self) -> None:
        """
        Calling get_logger() with two different module names must both succeed.

        This is a smoke test that the structlog factory does not raise for
        any valid name string.
        """
        log_a = get_logger("eiger.module_a")
        log_b = get_logger("eiger.module_b")
        assert log_a is not None
        assert log_b is not None

    def test_module_dunder_name_style_works(self) -> None:
        """
        The conventional usage ``get_logger(__name__)`` must not raise.

        Most EIGER modules call ``log = get_logger(__name__)`` at module level,
        so this verifies the exact usage pattern is safe.
        """
        log = get_logger(__name__)
        assert log is not None
