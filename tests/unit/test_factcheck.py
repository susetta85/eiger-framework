"""
Unit tests for FactCheckDataset (eiger.datasets.factcheck).

Covers everything specific to this loader: the assumed JSONL raw format
(see the class's module docstring for why this is an unverified
assumption, unlike PolitiFact's documented TSV or AVeriTeC's documented
JSONL), the "true"-verdict-only filter (consistent with every other
loader's verified-true-only philosophy), the templated context_query
fallback (no evidence Q&A documented for this source), and direct reuse
of the raw claim_id (unlike AVeriTeC, which had none in its documented
fields).

What these tests do NOT cover:
  - Real network downloads of the CheckThat! mirror — not a runtime
    dependency of this loader (download() is a guard, not a fetcher; see
    TestFactCheckDatasetDownload).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from eiger.core.exceptions import IngestionError
from eiger.datasets import FactCheckDataset, get_dataset, list_datasets

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    with patch("eiger.datasets.factcheck.log"):
        yield


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "factcheck"


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def _true_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "claim_id": "FC-2024-0091",
        "claim": "The federal minimum wage has not increased since 2009.",
        "verdict": "true",
        "article_url": "https://www.factcheck.org/2024/minimum-wage-2009/",
        "date": "2024-03-12",
    }
    record.update(overrides)
    return record


# ─── Identity ─────────────────────────────────────────────────────────────────

class TestFactCheckDatasetIdentity:
    def test_name_and_description(self) -> None:
        dataset = FactCheckDataset(data_dir="/tmp/whatever")
        assert dataset.name == "factcheck_org"
        assert "factcheck" in dataset.description.lower()

    def test_default_data_dir_is_data_factcheck(self) -> None:
        dataset = FactCheckDataset()
        assert dataset.data_dir.parts[-2:] == ("data", "factcheck")

    def test_custom_data_dir_override(self, data_dir: Path) -> None:
        dataset = FactCheckDataset(data_dir=data_dir)
        assert dataset.data_dir == data_dir

    def test_registered_under_factcheck_org_name(self) -> None:
        assert "factcheck_org" in list_datasets()
        assert isinstance(get_dataset("factcheck_org"), FactCheckDataset)


# ─── load() ───────────────────────────────────────────────────────────────────

class TestFactCheckDatasetLoad:
    def test_load_reports_factcheck_org_as_source_dataset(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record()])
        claim = FactCheckDataset(data_dir=data_dir).load()[0]
        assert claim.source_dataset == "factcheck_org"

    def test_load_maps_claim_text(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record()])
        claim = FactCheckDataset(data_dir=data_dir).load()[0]
        assert claim.original_fact == (
            "The federal minimum wage has not increased since 2009."
        )

    def test_load_prefixes_raw_claim_id(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record(claim_id="FC-2024-0091")])
        claim = FactCheckDataset(data_dir=data_dir).load()[0]
        assert claim.claim_id == "FACTCHECK_FC-2024-0091"

    def test_load_uses_templated_context_query(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record()])
        claim = FactCheckDataset(data_dir=data_dir).load()[0]
        assert claim.context_query == (
            "Is it true that The federal minimum wage has not increased "
            "since 2009.?"
        )

    def test_load_keeps_only_true_verdict(self, data_dir: Path) -> None:
        records = [
            _true_record(claim_id="1", claim="Claim A", verdict="true"),
            _true_record(claim_id="2", claim="Claim B", verdict="false"),
            _true_record(claim_id="3", claim="Claim C", verdict="mixture"),
            _true_record(claim_id="4", claim="Claim D", verdict="unverifiable"),
        ]
        _write_jsonl(data_dir / "test.jsonl", records)
        claims = FactCheckDataset(data_dir=data_dir).load()
        assert [c.original_fact for c in claims] == ["Claim A"]

    def test_load_verdict_match_is_case_insensitive(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record(verdict="True")])
        claims = FactCheckDataset(data_dir=data_dir).load()
        assert len(claims) == 1

    def test_load_carries_optional_metadata_fields(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record()])
        claim = FactCheckDataset(data_dir=data_dir).load()[0]
        assert claim.metadata["verdict"] == "true"
        assert claim.metadata["article_url"] == (
            "https://www.factcheck.org/2024/minimum-wage-2009/"
        )
        assert claim.metadata["date"] == "2024-03-12"
        assert claim.metadata["verified"] is False

    def test_load_omits_optional_metadata_fields_when_absent(self, data_dir: Path) -> None:
        record = _true_record()
        del record["article_url"]
        del record["date"]
        _write_jsonl(data_dir / "test.jsonl", [record])
        claim = FactCheckDataset(data_dir=data_dir).load()[0]
        assert "article_url" not in claim.metadata
        assert "date" not in claim.metadata

    def test_load_ignores_blank_lines(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(_true_record()) + "\n\n   \n"
        (data_dir / "test.jsonl").write_text(text, encoding="utf-8")
        claims = FactCheckDataset(data_dir=data_dir).load()
        assert len(claims) == 1

    def test_load_respects_max_claims(self, data_dir: Path) -> None:
        records = [_true_record(claim_id=str(i), claim=f"Claim {i}") for i in range(5)]
        _write_jsonl(data_dir / "test.jsonl", records)
        claims = FactCheckDataset(data_dir=data_dir).load(max_claims=2)
        assert [c.original_fact for c in claims] == ["Claim 0", "Claim 1"]

    def test_load_selects_file_by_split(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "dev.jsonl", [_true_record(claim="Dev claim")])
        _write_jsonl(data_dir / "test.jsonl", [_true_record(claim="Test claim")])
        claims = FactCheckDataset(data_dir=data_dir).load(split="dev")
        assert [c.original_fact for c in claims] == ["Dev claim"]

    def test_load_raises_on_missing_file(self, data_dir: Path) -> None:
        with pytest.raises(IngestionError, match="Could not read"):
            FactCheckDataset(data_dir=data_dir).load()

    def test_load_raises_on_invalid_json_line(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "test.jsonl").write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(IngestionError, match="invalid JSON"):
            FactCheckDataset(data_dir=data_dir).load()

    def test_load_raises_when_line_is_not_a_json_object(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "test.jsonl").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(IngestionError, match="must be a JSON object"):
            FactCheckDataset(data_dir=data_dir).load()

    def test_load_raises_on_true_verdict_record_missing_claim_field(self, data_dir: Path) -> None:
        record = _true_record()
        del record["claim"]
        _write_jsonl(data_dir / "test.jsonl", [record])
        with pytest.raises(IngestionError, match="missing required field"):
            FactCheckDataset(data_dir=data_dir).load()


# ─── content_hash ─────────────────────────────────────────────────────────────

class TestFactCheckDatasetContentHash:
    def test_content_hash_before_load_is_all_zeros(self, data_dir: Path) -> None:
        assert FactCheckDataset(data_dir=data_dir).content_hash == "0" * 16

    def test_content_hash_after_load_is_deterministic(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record()])
        first = FactCheckDataset(data_dir=data_dir)
        first.load()
        second = FactCheckDataset(data_dir=data_dir)
        second.load()
        assert first.content_hash == second.content_hash
        assert len(first.content_hash) == 16

    def test_content_hash_changes_with_different_content(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record(claim="Claim A")])
        dataset_a = FactCheckDataset(data_dir=data_dir)
        dataset_a.load()

        _write_jsonl(data_dir / "test.jsonl", [_true_record(claim="Claim B")])
        dataset_b = FactCheckDataset(data_dir=data_dir)
        dataset_b.load()

        assert dataset_a.content_hash != dataset_b.content_hash


# ─── download() ─────────────────────────────────────────────────────────────

class TestFactCheckDatasetDownload:
    def test_download_noops_when_jsonl_files_already_present(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_true_record()])
        FactCheckDataset(data_dir=data_dir).download(str(data_dir))  # must not raise

    def test_download_raises_when_directory_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(IngestionError, match="Automated download is not implemented"):
            FactCheckDataset().download(str(missing))

    def test_download_raises_when_directory_empty(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(IngestionError, match="Automated download is not implemented"):
            FactCheckDataset().download(str(data_dir))
