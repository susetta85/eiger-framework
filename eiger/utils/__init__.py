"""
Shared utilities: logging, seeding, hashing.

This package provides lightweight helpers used across the entire EIGER
codebase. All utilities are infrastructure-free (no DB, no HTTP, no ML
model weights) and can be imported at any point in the initialization
sequence without side effects.

Re-exported symbols:
  - get_logger:      Returns a named structlog logger for a module.
  - make_rng:        Creates an isolated Random instance from a seed.
  - seed_everything: Seeds all global RNGs for full reproducibility.
  - embed_cosine_similarity_01: Cosine similarity between two embedded
                     strings, rescaled to [0, 1] (shared by
                     EmbeddingFaithfulnessScorer and PCSMetric).
"""

# Re-export logging helper so callers can write:
#   from eiger.utils import get_logger
# rather than reaching into the sub-module.
from eiger.utils.logging import get_logger

# Re-export seeding helpers for the same ergonomic reason.
from eiger.utils.seeding import make_rng, seed_everything

# Re-export the shared cosine-similarity helper (Sprint 5+, factored out
# when eiger.metrics.pcs needed the same computation as
# eiger.metrics.heuristic_scorer).
from eiger.utils.similarity import embed_cosine_similarity_01

# Explicit __all__ prevents internal helpers (configure_logging,
# derive_seed, etc.) from leaking into `from eiger.utils import *`.
__all__ = [
    "get_logger",
    "make_rng",
    "seed_everything",
    "embed_cosine_similarity_01",
]
