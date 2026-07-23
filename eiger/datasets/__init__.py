"""
Dataset loaders for EIBench experiments.

This package exposes the dataset registry (register_dataset, get_dataset,
list_datasets) and every concrete BaseDataset implementation, mirroring
the pattern already established by eiger.attacks and eiger.metrics.

Currently registered datasets
------------------------------
  - JSONFixtureDataset ("json_fixture") — bundled development/CI fixture.
    See eiger.datasets.json_fixture for details and docs/DATASETS.md for
    the full dataset roadmap (AVeriTeC, PolitiFact, FactCheck.org — not
    yet implemented).

Responsibilities of this module
--------------------------------
- Re-export the registry API so callers never import
  eiger.datasets.registry directly.
- Re-export every built-in dataset class for direct import convenience.
- Trigger auto-registration of all built-in datasets at import time.

What this module does NOT do
-----------------------------
- It does not call .load() or .download() on anything; construction and
  invocation are the caller's responsibility (typically resolved from a
  DatasetConfig.name via get_dataset()).
- It does not implement AVeriTeC/PolitiFact/FactCheck.org loaders yet —
  see docs/DATASETS.md for the documented roadmap and expected formats.
"""

# ─── Registry helpers ───────────────────────────────────────────────────────

# ─── Built-in dataset classes ────────────────────────────────────────────────
from eiger.datasets.json_fixture import JSONFixtureDataset
from eiger.datasets.registry import get_dataset, list_datasets, register_dataset

# ─── Auto-registration ───────────────────────────────────────────────────────

# Registered at import time so that `import eiger.datasets` immediately
# enables `get_dataset("json_fixture")` with no further setup. Idempotent,
# so repeated imports (e.g. across test modules) are harmless.
register_dataset(JSONFixtureDataset)

# ─── Public API declaration ───────────────────────────────────────────────────

__all__ = [
    # Registry interface
    "register_dataset",
    "get_dataset",
    "list_datasets",
    # Concrete dataset classes
    "JSONFixtureDataset",
]
