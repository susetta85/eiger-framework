"""
Unit tests for eiger.datasets: the dataset registry and JSONFixtureDataset.

Tests verify:
  - register_dataset / get_dataset / list_datasets behave like the
    attack/metric registries they mirror (registration, lookup,
    DatasetNotFoundError on unknown names).
  - JSONFixtureDataset.load() correctly parses eibench_raw_claims.json
    (or an injected custom path) into Claim objects with the documented
    field mapping (docs/DATASETS.md).
  - max_claims truncation, split being accepted-but-ignored, download()
    being a logged no-op, and content_hash's before/after-load behavior.
  - IngestionError is raised (with a chained cause where applicable) for
    a missing file, invalid JSON, a non-list top-level value, and a
    missing required field in an entry.

What these tests do NOT cover:
  - AVeriTeC/PolitiFact/FactCheck.org loaders (not yet implemented).
  - CorpusBuilder's consumption of Claim objects (see test_corpus_builder.py).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from eiger.core.exceptions import DatasetNotFoundError, IngestionError
from eiger.core.models import Claim
from eiger.datasets import JSONFixtureDataset, get_dataset, list_datasets, register_dataset
from eiger.datasets import registry as dataset_registry

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """
    Patch the module-level structlog logger for every test in this file.

    Matches the convention used across all Sprint 2 test files (see
    test_pipeline.py) to avoid a structlog PrintLogger quirk when
    configure_logging() has not run in this process.
    """
    with patch("eiger.datasets.json_fixture.log"):
        yield


@pytest.fixture()
def fixture_path(tmp_path: Path) -> Path:
    """A writable temp-path fixture file, populated per-test by helpers below."""
    return tmp_path / "claims.json"


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _valid_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "claim_id": "EIB_CLAIM_001",
        "original_fact": "L'inflazione core nel 2024 e' stabile al 2.1%.",
        "context_query": "Quali sono i dati ufficiali sull'inflazione core nel 2024?",
        "adversarial_variants": {"numerical_shift": "L'inflazione core e' al 21.1%."},
    }
    entry.update(overrides)
    return entry


# ─── JSONFixtureDataset: metadata ─────────────────────────────────────────────

class TestJSONFixtureDatasetMetadata:
    def test_name_and_description(self) -> None:
        dataset = JSONFixtureDataset()
        assert dataset.name == "json_fixture"
        assert "fixture" in dataset.description.lower()

    def test_default_path_points_at_repo_root_fixture(self) -> None:
        """
        With no override, the default path must resolve to the real,
        already-existing eibench_raw_claims.json bundled at the repo root.
        """
        dataset = JSONFixtureDataset()
        assert dataset.path.name == "eibench_raw_claims.json"
        assert dataset.path.exists()

    def test_custom_path_override(self, fixture_path: Path) -> None:
        dataset = JSONFixtureDataset(path=fixture_path)
        assert dataset.path == fixture_path

    def test_custom_path_accepts_str(self, fixture_path: Path) -> None:
        dataset = JSONFixtureDataset(path=str(fixture_path))
        assert dataset.path == fixture_path


# ─── JSONFixtureDataset: download ──────────────────────────────────────────────

class TestJSONFixtureDatasetDownload:
    def test_download_is_a_noop_and_does_not_raise(self, tmp_path: Path) -> None:
        dataset = JSONFixtureDataset()
        dataset.download(str(tmp_path))  # must not raise


# ─── JSONFixtureDataset: load() ────────────────────────────────────────────────

class TestJSONFixtureDatasetLoad:
    def test_load_parses_real_bundled_fixture(self) -> None:
        """The real repo-root eibench_raw_claims.json must load successfully."""
        claims = JSONFixtureDataset().load()
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)
        assert claims[0].source_dataset == "json_fixture"

    def test_load_maps_fields_correctly(self, fixture_path: Path) -> None:
        _write(fixture_path, [_valid_entry()])
        claims = JSONFixtureDataset(path=fixture_path).load()

        assert len(claims) == 1
        claim = claims[0]
        assert claim.claim_id == "EIB_CLAIM_001"
        assert claim.original_fact.startswith("L'inflazione")
        assert claim.context_query.startswith("Quali sono")
        assert claim.source_dataset == "json_fixture"
        assert claim.metadata["adversarial_variants"] == {
            "numerical_shift": "L'inflazione core e' al 21.1%."
        }

    def test_load_defaults_adversarial_variants_to_empty_dict_when_absent(
        self, fixture_path: Path
    ) -> None:
        entry = _valid_entry()
        del entry["adversarial_variants"]
        _write(fixture_path, [entry])

        claim = JSONFixtureDataset(path=fixture_path).load()[0]
        assert claim.metadata["adversarial_variants"] == {}

    def test_load_multiple_entries_preserves_file_order(self, fixture_path: Path) -> None:
        _write(
            fixture_path,
            [_valid_entry(claim_id="A"), _valid_entry(claim_id="B"), _valid_entry(claim_id="C")],
        )
        claims = JSONFixtureDataset(path=fixture_path).load()
        assert [c.claim_id for c in claims] == ["A", "B", "C"]

    def test_load_respects_max_claims(self, fixture_path: Path) -> None:
        _write(
            fixture_path,
            [_valid_entry(claim_id="A"), _valid_entry(claim_id="B"), _valid_entry(claim_id="C")],
        )
        claims = JSONFixtureDataset(path=fixture_path).load(max_claims=2)
        assert [c.claim_id for c in claims] == ["A", "B"]

    def test_load_max_claims_none_returns_all(self, fixture_path: Path) -> None:
        _write(fixture_path, [_valid_entry(claim_id="A"), _valid_entry(claim_id="B")])
        claims = JSONFixtureDataset(path=fixture_path).load(max_claims=None)
        assert len(claims) == 2

    def test_load_split_argument_is_accepted_but_ignored(self, fixture_path: Path) -> None:
        _write(fixture_path, [_valid_entry()])
        dataset = JSONFixtureDataset(path=fixture_path)
        assert dataset.load(split="train") == dataset.load(split="test")

    def test_load_empty_list_returns_empty_claims(self, fixture_path: Path) -> None:
        _write(fixture_path, [])
        assert JSONFixtureDataset(path=fixture_path).load() == []

    def test_load_missing_file_raises_ingestion_error_with_cause(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(IngestionError) as exc_info:
            JSONFixtureDataset(path=missing).load()
        assert "does_not_exist.json" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_load_invalid_json_raises_ingestion_error_with_cause(
        self, fixture_path: Path
    ) -> None:
        fixture_path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(IngestionError) as exc_info:
            JSONFixtureDataset(path=fixture_path).load()
        assert exc_info.value.__cause__ is not None

    def test_load_non_list_top_level_raises_ingestion_error(self, fixture_path: Path) -> None:
        _write(fixture_path, {"not": "a list"})
        with pytest.raises(IngestionError, match="top-level list"):
            JSONFixtureDataset(path=fixture_path).load()

    def test_load_missing_required_field_raises_ingestion_error(
        self, fixture_path: Path
    ) -> None:
        entry = _valid_entry()
        del entry["original_fact"]
        _write(fixture_path, [entry])
        with pytest.raises(IngestionError, match="original_fact"):
            JSONFixtureDataset(path=fixture_path).load()


# ─── JSONFixtureDataset: content_hash ──────────────────────────────────────────

class TestJSONFixtureDatasetContentHash:
    def test_content_hash_before_load_is_all_zeros(self) -> None:
        assert JSONFixtureDataset().content_hash == "0" * 16

    def test_content_hash_after_load_is_deterministic(self, fixture_path: Path) -> None:
        _write(fixture_path, [_valid_entry()])
        dataset_a = JSONFixtureDataset(path=fixture_path)
        dataset_b = JSONFixtureDataset(path=fixture_path)
        dataset_a.load()
        dataset_b.load()
        assert dataset_a.content_hash == dataset_b.content_hash
        assert len(dataset_a.content_hash) == 16

    def test_content_hash_changes_when_content_changes(self, fixture_path: Path) -> None:
        _write(fixture_path, [_valid_entry(original_fact="Fact one.")])
        dataset = JSONFixtureDataset(path=fixture_path)
        dataset.load()
        hash_one = dataset.content_hash

        _write(fixture_path, [_valid_entry(original_fact="A completely different fact.")])
        dataset.load()
        hash_two = dataset.content_hash

        assert hash_one != hash_two


# ─── Registry ─────────────────────────────────────────────────────────────────

class TestDatasetRegistry:
    """Tests for the dataset registry lookup functions."""

    def test_json_fixture_registered_on_package_import(self) -> None:
        """
        Guards against accidental removal of json_fixture from the
        auto-registration block in eiger.datasets.__init__.
        """
        assert "json_fixture" in list_datasets()

    def test_get_dataset_returns_instance(self) -> None:
        dataset = get_dataset("json_fixture")
        assert isinstance(dataset, JSONFixtureDataset)

    def test_get_dataset_returns_fresh_instance_each_call(self) -> None:
        """Matches get_attack/get_metric: no shared state between calls."""
        assert get_dataset("json_fixture") is not get_dataset("json_fixture")

    def test_get_unknown_dataset_raises(self) -> None:
        with pytest.raises(DatasetNotFoundError):
            get_dataset("nonexistent_dataset_xyz")

    def test_get_unknown_dataset_error_lists_available_names(self) -> None:
        with pytest.raises(DatasetNotFoundError, match="json_fixture"):
            get_dataset("nonexistent_dataset_xyz")

    def test_list_datasets_is_sorted(self) -> None:
        assert list_datasets() == sorted(list_datasets())

    def test_register_dataset_is_idempotent(self) -> None:
        before = dict(dataset_registry._REGISTRY)
        register_dataset(JSONFixtureDataset)
        register_dataset(JSONFixtureDataset)
        assert before == dataset_registry._REGISTRY

    def test_register_dataset_returns_class_unchanged(self) -> None:
        """Enables usage as a class decorator, matching register_attack/register_metric."""
        assert register_dataset(JSONFixtureDataset) is JSONFixtureDataset
