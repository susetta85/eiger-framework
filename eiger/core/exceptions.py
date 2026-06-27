"""Domain-specific exceptions for the EIGER framework."""


class EigerError(Exception):
    """Base exception for all EIGER errors."""


class AttackNotFoundError(EigerError):
    """Raised when an attack name is not found in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            f"Attack '{name}' not found. Available: {available}"
        )


class MetricNotFoundError(EigerError):
    """Raised when a metric name is not found in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(
            f"Metric '{name}' not found. Available: {available}"
        )


class ConfigurationError(EigerError):
    """Raised when experiment configuration is invalid."""


class IngestionError(EigerError):
    """Raised when corpus ingestion fails."""


class RetrievalError(EigerError):
    """Raised when retrieval fails."""


class GenerationError(EigerError):
    """Raised when LLM generation fails."""


class ReproducibilityError(EigerError):
    """Raised when reproducibility checks fail (seed mismatch, metric drift)."""
