"""
SnopesDataset — BaseDataset implementation over LLM-enriched Snopes
fact-checks.

Where the data comes from
--------------------------
A colleague provided ``Snopes.xlsx``: a 19,631-row export of Snopes fact
-checks (1995-2025), each row already carrying a ``normalised_rating``
(True/False/"partially true"/"misleading"/"unverifiable") and a real
source ``url``. This class does NOT read that raw file directly — two
things are missing from it that ``Claim`` requires:
  1. Only the ``normalised_rating == True`` subset (4,832 of 19,631 rows,
     verified as of this writing) is usable: ``Claim.original_fact`` must
     be a verified TRUE statement — EIGER generates its own falsehoods via
     the attack registry, it does not import externally-sourced false
     claims as ground truth.
  2. None of the 19,631 rows have a natural-language question
     (``context_query``) — Snopes fact-checks a claim, it doesn't index
     one by the question a person would ask to surface it.

``scripts/enrich_snopes_claims.py`` does both: it filters and deduplicates
the raw export, generates a ``context_query`` for each surviving claim via
a local Ollama LLM, and writes a JSON file in exactly the schema
``JSONFixtureDataset`` (this class's parent) already knows how to load.
This class only overrides ``name``/``description``/the default ``path``
(pointing at the enrichment script's output rather than the bundled
fixture), so that ``Claim.source_dataset`` correctly reports ``"snopes"``
rather than ``"json_fixture"``. Every other behavior — parsing, the
optional ``source``/``domain``/``notes``/``verified`` provenance
passthrough, ``content_hash``, error handling — is inherited unchanged.

What this class does NOT do
----------------------------
- It does not read Snopes.xlsx, call Ollama, filter, or deduplicate
  anything itself — see scripts/enrich_snopes_claims.py.
- It does not independently re-verify Snopes' own "True" rating; claims
  loaded this way are tagged ``metadata["verified"] = False`` by the
  enrichment script until a member of the research team spot-checks them,
  even though Snopes itself already rated them True.
- It does not support multiple splits, matching JSONFixtureDataset (the
  enriched export has no train/dev/test partitioning).
"""

from __future__ import annotations

from pathlib import Path

from eiger.datasets.json_fixture import JSONFixtureDataset
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# Matches docs/DATASETS.md's established data/<name>/ convention for
# externally-sourced dataset files that are too large to commit to Git
# (see .gitignore's "Data & storage" section). Path(__file__) is
# eiger/datasets/snopes.py, so parents[0]=datasets, [1]=eiger,
# [2]=repository root — same resolution as JSONFixtureDataset's own
# _DEFAULT_FIXTURE_PATH.
_DEFAULT_ENRICHED_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "snopes" / "snopes_enriched.json"
)


class SnopesDataset(JSONFixtureDataset):
    """
    Loads Claim objects from an LLM-enriched Snopes export.

    See the module docstring for the full collect -> filter -> enrich ->
    load pipeline. This class exists (rather than reusing
    JSONFixtureDataset directly) purely so that ``Claim.source_dataset``
    reports ``"snopes"`` and so ``list_datasets()``/``get_dataset()``
    expose it as its own, honestly-labeled entry.
    """

    name: str = "snopes"
    description: str = (
        "LLM-enriched Snopes fact-checks (normalised_rating == True subset "
        "only), with generated context_query. Requires running "
        "scripts/enrich_snopes_claims.py first — see that script's "
        "docstring and scripts/README.md."
    )

    def __init__(self, path: str | Path | None = None) -> None:
        """
        Args:
            path: Optional override of the enriched JSON file location.
                  Defaults to ``data/snopes/snopes_enriched.json`` at the
                  repository root (the enrichment script's default output
                  path), matching DatasetConfig.path's "local path
                  override" semantics when set explicitly.
        """
        super().__init__(path=path if path is not None else _DEFAULT_ENRICHED_PATH)

    def download(self, target_dir: str) -> None:
        """
        No-op, for a different reason than JSONFixtureDataset's: preparing
        Snopes data is a slow, Ollama-dependent offline enrichment step,
        not a quick fetch that belongs in BaseDataset.download()'s
        "fetch raw files" contract. Run scripts/enrich_snopes_claims.py
        separately; this override exists only to give SnopesDataset its
        own log key rather than inheriting JSONFixtureDataset's literal
        "json_fixture.download_noop" message unchanged.

        Args:
            target_dir: Accepted but unused — see above.
        """
        log.debug("snopes.download_noop", target_dir=target_dir)
