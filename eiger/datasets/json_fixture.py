"""
JSONFixtureDataset — BaseDataset implementation backed by the bundled
eibench_raw_claims.json fixture.

This is the first concrete BaseDataset subclass in the project (Sprint 3).
It intentionally targets the lowest-risk, already-available data source:
a small, hand-written JSON fixture that ships at the repository root and
is already used informally throughout development and the integration
tests (as inline Claim(...) construction, not yet through this class).

Why start here rather than with AVeriTeC/PolitiFact/FactCheck.org?
--------------------------------------------------------------------
Those loaders require new runtime dependencies (the HuggingFace
``datasets`` library, TSV parsing for LIAR, or the CheckThat! corpus
tooling) and network access at download() time. JSONFixtureDataset needs
neither: it validates the BaseDataset contract and the new dataset
registry end-to-end with a fully deterministic, offline, dependency-free
implementation, ahead of the heavier real-corpus loaders.

What this class does NOT do
----------------------------
- It does not download anything: the fixture is committed to the
  repository, so download() is a documented no-op.
- It does not support multiple splits: the fixture has no train/dev/test
  partitioning, so ``split`` is accepted (to satisfy the BaseDataset
  signature) but otherwise ignored.
- It is not a real fact-checking benchmark: EIBench taxonomy documents
  (docs/DATASETS.md) are explicit that this fixture exists for tests,
  development, and CI, not for reportable research results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# The fixture lives at the repository root (eibench_raw_claims.json),
# NOT inside the installable `eiger` package directory — it is a
# development/test fixture, not packaged data. Path(__file__) is
# eiger/datasets/json_fixture.py, so parents[0]=datasets, [1]=eiger,
# [2]=repository root. This mirrors how ExperimentRunner._REPO_ROOT
# resolves the repo root in eiger/experiments/runner.py.
_DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "eibench_raw_claims.json"


class JSONFixtureDataset(BaseDataset):
    """
    Loads Claim objects from the bundled eibench_raw_claims.json fixture.

    Expected raw JSON shape (a list of objects):
        {
          "claim_id": str,
          "original_fact": str,
          "context_query": str,
          "adversarial_variants": {attack_name: poisoned_text, ...}  # optional
        }

    Field mapping to Claim (see docs/DATASETS.md, section "JSON Fixture"):
        claim_id       -> Claim.claim_id
        original_fact  -> Claim.original_fact
        context_query  -> Claim.context_query
        (this class)   -> Claim.source_dataset = "json_fixture"
        adversarial_variants -> Claim.metadata["adversarial_variants"]

    ``adversarial_variants`` is informational provenance only (pre-authored
    example poisoned texts for documentation/manual inspection); it is not
    consumed by CorpusBuilder, which generates its own poisoned documents
    via the attack registry at ingestion time.
    """

    name: str = "json_fixture"
    description: str = (
        "Bundled fixture (eibench_raw_claims.json) for tests, development, "
        "and CI. Not a real fact-checking corpus — see docs/DATASETS.md."
    )

    def __init__(self, path: str | Path | None = None) -> None:
        """
        Args:
            path: Optional override of the fixture file location. Defaults
                  to the bundled repository-root eibench_raw_claims.json,
                  matching DatasetConfig.path's "local path override"
                  semantics when set explicitly.
        """
        self.path: Path = Path(path) if path is not None else _DEFAULT_FIXTURE_PATH
        # Populated by load(); used by content_hash so the hash reflects
        # whatever was actually loaded rather than re-reading the file.
        self._loaded_claims: list[Claim] = []

    # ─── BaseDataset interface ─────────────────────────────────────────────

    def download(self, target_dir: str) -> None:
        """
        No-op: the fixture is committed to the repository, not downloaded.

        Accepts ``target_dir`` to satisfy the BaseDataset signature (other
        loaders use it as a real destination directory); logged at debug
        level so callers can see the no-op happened without it being
        mistaken for a silent failure.
        """
        log.debug("json_fixture.download_noop", target_dir=target_dir)

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """
        Parse the fixture file and return Claim objects.

        Args:
            split:      Accepted but ignored — the fixture has a single,
                        unpartitioned set of claims (see class docstring).
            max_claims: If set, return at most this many claims, taking
                        the first N in file order (the fixture is small
                        and hand-curated, so file order is already stable
                        and deterministic — no re-sorting is needed).

        Returns:
            List of Claim objects parsed from the fixture.

        Raises:
            IngestionError: If the file is missing, unreadable, not valid
                            JSON, or an entry is missing a required field.
        """
        log.debug("json_fixture.load_start", path=str(self.path), split=split)
        raw_items = self._read_raw()

        try:
            claims = [self._to_claim(item) for item in raw_items]
        except KeyError as exc:
            raise IngestionError(
                f"JSON fixture entry at '{self.path}' is missing required "
                f"field {exc}. Expected keys: claim_id, original_fact, "
                "context_query."
            ) from exc

        if max_claims is not None:
            claims = claims[:max_claims]

        self._loaded_claims = claims
        log.debug("json_fixture.load_complete", n_claims=len(claims))
        return claims

    @property
    def content_hash(self) -> str:
        """
        SHA-256 (truncated to 16 hex chars) over the concatenated
        original_fact text of every claim loaded by the most recent
        load() call, mirroring Claim.content_hash's own truncation
        convention (eiger/core/models.py).

        Returns "0" * 16 if load() has not been called yet, so accessing
        the property early fails loudly-but-safely rather than raising.
        """
        if not self._loaded_claims:
            return "0" * 16
        combined = "".join(claim.original_fact for claim in self._loaded_claims)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # ─── Internal helpers ──────────────────────────────────────────────────

    def _read_raw(self) -> list[dict[str, Any]]:
        """Read and JSON-decode the fixture file, wrapping failures."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(
                f"Could not read JSON fixture at '{self.path}': {exc}"
            ) from exc

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise IngestionError(
                f"JSON fixture at '{self.path}' is not valid JSON: {exc}"
            ) from exc

        if not isinstance(data, list):
            raise IngestionError(
                f"JSON fixture at '{self.path}' must contain a top-level "
                f"list of claim objects, got {type(data).__name__}."
            )
        return data

    def _to_claim(self, item: dict[str, Any]) -> Claim:
        """Map one raw fixture entry to a Claim, per the class docstring."""
        return Claim(
            claim_id=item["claim_id"],
            original_fact=item["original_fact"],
            context_query=item["context_query"],
            source_dataset=self.name,
            metadata={"adversarial_variants": item.get("adversarial_variants", {})},
        )
