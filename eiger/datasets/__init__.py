"""
Dataset loaders for EIBench experiments.

This package exposes the dataset registry (register_dataset, get_dataset,
list_datasets) and every concrete BaseDataset implementation, mirroring
the pattern already established by eiger.attacks and eiger.metrics.

Currently registered datasets
------------------------------
  - JSONFixtureDataset ("json_fixture") — bundled development/CI fixture.
  - SnopesDataset ("snopes") — LLM-enriched Snopes fact-checks (True-rated
    subset only). Requires running scripts/enrich_snopes_claims.py first;
    see eiger.datasets.snopes for the full pipeline and scripts/README.md.
  - AVeriTecDataset ("averitec") — AVeriTeC fact-checks (Supported-label
    subset only), using each record's own evidence questions as
    context_query (no LLM enrichment needed). Requires pre-downloaded
    *.jsonl split files under data/averitec/ — see eiger.datasets.averitec
    and docs/DATASETS.md section 3 for manual download steps (automated
    download() is not yet implemented).

See docs/DATASETS.md for the full dataset roadmap (PolitiFact,
FactCheck.org — not yet implemented).

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
- It does not implement PolitiFact/FactCheck.org loaders yet — see
  docs/DATASETS.md for the documented roadmap and expected formats.
- It does not implement AVeriTecDataset's automated download(); that
  loader's download() is a guard (raises if data is missing) rather than
  a fetcher — see eiger.datasets.averitec's module docstring.
"""

# ─── Registry helpers ───────────────────────────────────────────────────────

# ─── Built-in dataset classes ────────────────────────────────────────────────
from eiger.datasets.averitec import AVeriTecDataset
from eiger.datasets.json_fixture import JSONFixtureDataset
from eiger.datasets.registry import get_dataset, list_datasets, register_dataset
from eiger.datasets.snopes import SnopesDataset

# ─── Auto-registration ───────────────────────────────────────────────────────

# Registered at import time so that `import eiger.datasets` immediately
# enables `get_dataset("json_fixture")` / `get_dataset("snopes")` /
# `get_dataset("averitec")` with no further setup. Idempotent, so
# repeated imports (e.g. across test modules) are harmless.
register_dataset(JSONFixtureDataset)
register_dataset(SnopesDataset)
register_dataset(AVeriTecDataset)

# ─── Public API declaration ───────────────────────────────────────────────────

__all__ = [
    # Registry interface
    "register_dataset",
    "get_dataset",
    "list_datasets",
    # Concrete dataset classes
    "JSONFixtureDataset",
    "SnopesDataset",
    "AVeriTecDataset",
]
