"""
Domain-specific exceptions for the EIGER framework.

Design principles:
  - All exceptions inherit from EigerError so callers can catch the
    entire hierarchy with a single `except EigerError` clause.
  - Exceptions that carry structured context (e.g. which name failed
    and what was available) format a helpful message in __init__ so
    stack traces are immediately informative without a debugger.
  - Each exception maps to exactly one failure domain (attack registry,
    metric registry, configuration, ingestion, retrieval, generation,
    reproducibility). This makes error routing unambiguous.

What this file does NOT do:
  - It does not log anything — logging is the caller's responsibility.
  - It does not import any EIGER modules (zero intra-package deps) so
    the exceptions module can always be imported safely, even before
    settings or models are initialized.
"""


class EigerError(Exception):
    """
    Base exception for all EIGER framework errors.

    Catching EigerError catches every framework-defined error.
    Catching a sub-class catches only that specific failure domain.
    Standard library errors (ValueError, IOError, etc.) are NOT
    wrapped here — they bubble up as-is unless a caller re-raises.
    """


# ─── Registry errors ──────────────────────────────────────────────────────────

class AttackNotFoundError(EigerError):
    """
    Raised when an attack name is not found in the attack registry.

    Typically triggered when a YAML experiment config references an
    attack that has not been registered via the @register_attack decorator.

    Args:
        name:      The requested attack identifier.
        available: List of currently registered attack names.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        # Include the full list of available attacks so the error message
        # is self-contained — no need to consult the registry separately.
        super().__init__(
            f"Attack '{name}' not found. Available: {available}"
        )


class MetricNotFoundError(EigerError):
    """
    Raised when a metric name is not found in the metric registry.

    Mirrors AttackNotFoundError in structure for consistency.

    Args:
        name:      The requested metric identifier.
        available: List of currently registered metric names.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        # Same pattern as AttackNotFoundError: surface actionable info
        # directly in the exception message.
        super().__init__(
            f"Metric '{name}' not found. Available: {available}"
        )


# ─── Lifecycle errors ─────────────────────────────────────────────────────────

class ConfigurationError(EigerError):
    """
    Raised when experiment configuration is invalid.

    Examples:
      - A YAML field has an out-of-range value (e.g. poison_rate > 1.0).
      - Required fields are missing after environment resolution.
      - Conflicting options are set simultaneously.

    Pydantic validation errors are typically caught and re-raised as
    ConfigurationError so the experiment runner sees a single error type.
    """


class IngestionError(EigerError):
    """
    Raised when corpus ingestion fails.

    Examples:
      - The vector store is unreachable during upsert.
      - An embedder returns vectors of the wrong dimension.
      - A dataset file is corrupt or missing.
    """


class RetrievalError(EigerError):
    """
    Raised when retrieval fails.

    Examples:
      - The vector store returns an unexpected response format.
      - A query produces zero results when at least one is required.
      - The embedder fails to encode the query at retrieval time.
    """


class GenerationError(EigerError):
    """
    Raised when LLM generation fails.

    Examples:
      - The Ollama or OpenAI backend returns an error status.
      - The response cannot be parsed into the expected structure.
      - A timeout occurs during a synchronous generation call.
    """


class ReproducibilityError(EigerError):
    """
    Raised when reproducibility checks fail (seed mismatch, metric drift).

    EIGER treats reproducibility as a first-class concern. This exception
    is raised when:
      - The config_hash of a reloaded result differs from the stored hash.
      - Metric values deviate beyond tolerance when re-running an experiment
        with the same seed.
      - A dataset content_hash differs from the recorded provenance hash.
    """
