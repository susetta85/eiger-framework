"""
Structured logging setup for EIGER.

Uses structlog for machine-readable, leveled log output. All log output
includes the experiment_id and module name automatically via structlog's
context variable mechanism.

Why structlog (rather than the stdlib logging module alone):
  - Key-value pairs in log events (e.g. n_documents=42) make log output
    grep-able and ingestible by log aggregators (Loki, Splunk, etc.) without
    regex parsing.
  - merge_contextvars automatically injects experiment_id (set once via
    structlog.contextvars.bind_contextvars) into every subsequent log call
    in the same thread, avoiding the need to pass it explicitly.
  - ConsoleRenderer produces human-readable output locally while allowing
    a JSON renderer to be swapped in for production deployments by changing
    only configure_logging().

Usage:
    from eiger.utils import get_logger
    log = get_logger(__name__)
    log.info("ingestion.complete", n_documents=42, collection="eiger_corpus")
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    """
    Configure structlog and the stdlib logging bridge. Call once at startup.

    This function must be called before any log.info()/log.debug() calls
    in the application, typically at the top of the experiment runner's
    main() function. Calling it multiple times is safe but redundant —
    subsequent calls overwrite the previous configuration.

    The processor pipeline (in order):
      1. merge_contextvars:  Injects thread-local context (e.g. experiment_id).
      2. add_log_level:      Adds the level name ("info", "debug", etc.) to events.
      3. add_logger_name:    Adds the logger name (passed to get_logger()) to events.
      4. TimeStamper:        Adds an ISO-8601 UTC timestamp to each event.
      5. ConsoleRenderer:    Formats the event dict as a human-readable line.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
               Case-insensitive. Invalid values fall back to INFO.
    """
    # Configure the stdlib root logger so that third-party libraries that
    # use stdlib logging (e.g. httpx, sentence-transformers) emit to stdout
    # at the same level as structlog, keeping log output unified.
    logging.basicConfig(
        format="%(message)s",  # structlog handles formatting; this avoids double-formatting
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            # Step 1: Pull thread-local context vars (e.g. experiment_id) into each event.
            # Callers set these once via structlog.contextvars.bind_contextvars(experiment_id=...).
            structlog.contextvars.merge_contextvars,
            # Step 2: Add the log level as a string key ("level": "info").
            structlog.stdlib.add_log_level,
            # Step 3: Add the logger name set in get_logger(__name__).
            structlog.stdlib.add_logger_name,
            # Step 4: Add ISO-8601 timestamp. fmt="iso" produces e.g. "2024-01-15T12:34:56.789Z".
            structlog.processors.TimeStamper(fmt="iso"),
            # Step 5: Render the final event dict as a human-readable console line.
            # To switch to JSON output for production, replace with JSONRenderer().
            structlog.dev.ConsoleRenderer(),
        ],
        # make_filtering_bound_logger creates a logger class that skips
        # log calls below the configured level without entering the
        # processor pipeline at all, making filtered-out calls near-zero cost.
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        # Use a plain dict as the context class — no thread-local magic here;
        # merge_contextvars handles that via the processor above.
        context_class=dict,
        # PrintLoggerFactory writes to stdout via print(). For production,
        # swap to WritelnLogger or stdlib integration.
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a named structlog logger bound to the given module name.

    The name is typically __name__ of the calling module, which produces
    log lines like:
        [info] corpus_builder.start  logger=eiger.ingestion.corpus_builder  ...

    Args:
        name: Logger name, conventionally the module's __name__.

    Returns:
        A structlog BoundLogger instance. Thread-safe; can be stored as a
        module-level variable without issues.
    """
    return structlog.get_logger(name)
