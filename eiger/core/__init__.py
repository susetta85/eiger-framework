"""
Core domain models, abstract interfaces, and shared exceptions.

This package is the stable heart of the EIGER framework. Everything
exported here is infrastructure-free: no database connections, no HTTP
clients, no ML model weights. Keeping the core clean means it can be
imported in tests, notebooks, and downstream analysis code without
pulling in heavy dependencies.

What this file does NOT do:
  - It does not import interfaces (eiger.core.interfaces) into the
    top-level namespace to avoid circular imports — interfaces depend
    on models, so models must be importable independently.
  - It does not define any concrete implementations; those live in
    eiger.attacks, eiger.retrieval, eiger.generation, etc.
"""

# Re-export the central domain models so callers can write:
#   from eiger.core import Claim, Document, ...
# instead of reaching into sub-modules directly.
from eiger.core.models import Claim, Document, PoisonedDocument, RetrievalResult, ExperimentResult

# Re-export all framework-specific exceptions so error handling can be
# done uniformly via `from eiger.core import EigerError`.
from eiger.core.exceptions import (
    EigerError,
    AttackNotFoundError,
    MetricNotFoundError,
    ConfigurationError,
    IngestionError,
)

# __all__ controls what `from eiger.core import *` exposes and also
# serves as the authoritative list of public symbols for documentation
# generators (Sphinx autodoc, mkdocs-gen-files, etc.).
__all__ = [
    # Domain models
    "Claim",
    "Document",
    "PoisonedDocument",
    "RetrievalResult",
    "ExperimentResult",
    # Exceptions
    "EigerError",
    "AttackNotFoundError",
    "MetricNotFoundError",
    "ConfigurationError",
    "IngestionError",
]
