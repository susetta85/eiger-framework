"""
Unit tests for SnopesDataset (eiger.datasets.snopes).

SnopesDataset subclasses JSONFixtureDataset and overrides only
name/description/the default path/download() — see the class's own
docstring. These tests therefore focus on exactly what's overridden
(identity, default path, source_dataset correctness, download()'s log
key) rather than re-testing inherited parsing/error-handling logic
already covered exhaustively by test_datasets.py's JSONFixtureDataset
tests.

What these tests do NOT cover:
  - scripts/enrich_snopes_claims.py (not part of the eiger package; see
    scripts/README.md — not covered by the 100% coverage gate).
  - Parsing/error-handling edge cases already covered by
    TestJSONFixtureDatasetLoad et al. in test_datasets.py (inherited
    unchanged, verified once there).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from eiger.datasets import SnopesDataset, get_dataset, list_datasets
from eiger.datasets.json_fixture import JSONFixtureDataset

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """
    Patch both module-level structlog loggers SnopesDataset can touch:
    eiger.datasets.json_fixture.log (inherited load()) and
    eiger.datasets.snopes.log (overridden download()).
    """
    with patch("eiger.datasets.json_fixture.log"), patch("eiger.datasets.snopes.log"):
        yield


@pytest.fixture()
def enriched_path(tmp_path: Path) -> Path:
    return tmp_path / "snopes_enriched.json"


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _valid_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "claim_id": "SNOPES_80214",
        "original_fact": (
            "A photo authentically shows a bright baby-blue, retro Volkswagen "
            "van parked on a street that survived the Pacific Palisades "
            "Wildfire unscathed in Los Angeles in early 2025."
        ),
        "context_query": "Did a blue Volkswagen van survive the Palisades wildfire unscathed?",
        "source": "https://www.snopes.com/fact-check/vw-van-palisades-fire/",
        "notes": "original_verdict=True; date_published=2025-01-16T21:19:56",
        "verified": False,
    }
    entry.update(overrides)
    return entry


# ─── Identity & defaults ────────────────────────────────────────────────────

class TestSnopesDatasetIdentity:
    def test_is_a_json_fixture_dataset_subclass(self) -> None:
        """Reuses all parsing/error-handling logic via inheritance."""
        assert issubclass(SnopesDataset, JSONFixtureDataset)

    def test_name_and_description(self) -> None:
        dataset = SnopesDataset(path="/tmp/whatever.json")
        assert dataset.name == "snopes"
        assert "snopes" in dataset.description.lower()

    def test_default_path_is_data_snopes_enriched_json(self) -> None:
        dataset = SnopesDataset()
        assert dataset.path.parts[-3:] == ("data", "snopes", "snopes_enriched.json")

    def test_custom_path_override(self, enriched_path: Path) -> None:
        dataset = SnopesDataset(path=enriched_path)
        assert dataset.path == enriched_path

    def test_registered_under_snopes_name(self) -> None:
        assert "snopes" in list_datasets()
        assert isinstance(get_dataset("snopes"), SnopesDataset)


# ─── load() / source_dataset correctness ───────────────────────────────────

class TestSnopesDatasetLoad:
    def test_load_reports_snopes_as_source_dataset(self, enriched_path: Path) -> None:
        """
        The one behavior JSONFixtureDataset could NOT provide unmodified:
        source_dataset must say "snopes", not "json_fixture".
        """
        _write(enriched_path, [_valid_entry()])
        claim = SnopesDataset(path=enriched_path).load()[0]
        assert claim.source_dataset == "snopes"

    def test_load_carries_provenance_fields(self, enriched_path: Path) -> None:
        _write(enriched_path, [_valid_entry()])
        claim = SnopesDataset(path=enriched_path).load()[0]
        assert claim.metadata["source"] == "https://www.snopes.com/fact-check/vw-van-palisades-fire/"
        assert claim.metadata["verified"] is False
        assert "original_verdict=True" in claim.metadata["notes"]

    def test_load_maps_claim_id_and_context_query(self, enriched_path: Path) -> None:
        _write(enriched_path, [_valid_entry()])
        claim = SnopesDataset(path=enriched_path).load()[0]
        assert claim.claim_id == "SNOPES_80214"
        assert claim.context_query == (
            "Did a blue Volkswagen van survive the Palisades wildfire unscathed?"
        )


# ─── download() ─────────────────────────────────────────────────────────────

class TestSnopesDatasetDownload:
    def test_download_is_a_noop_and_does_not_raise(self, tmp_path: Path) -> None:
        dataset = SnopesDataset(path="/tmp/whatever.json")
        dataset.download(str(tmp_path))  # must not raise
