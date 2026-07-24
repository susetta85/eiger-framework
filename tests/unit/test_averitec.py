"""
Unit tests for AVeriTecDataset (eiger.datasets.averitec).

Covers everything NOT already exercised by test_datasets.py/test_snopes.py's
shared patterns, since AVeriTecDataset implements BaseDataset directly
(it does not subclass JSONFixtureDataset — the raw format is JSONL, not a
JSON array, and filtering/claim_id/context_query derivation are all
AVeriTeC-specific). See eiger/datasets/averitec.py's module docstring for
the full rationale.

What these tests do NOT cover:
  - Real network downloads or the HuggingFace `datasets` library — not a
    runtime dependency of this loader (download() is a guard, not a
    fetcher; see TestAVeriTecDatasetDownload).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from eiger.core.exceptions import IngestionError
from eiger.datasets import AVeriTecDataset, get_dataset, list_datasets

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    with patch("eiger.datasets.averitec.log"):
        yield


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "averitec"


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )


def _supported_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "claim": "The Amazon rainforest produces roughly 20% of the world's oxygen.",
        "label": "Supported",
        "evidence": [
            {
                "question": "What share of global oxygen production is attributed to the Amazon rainforest?",
                "answer": "Estimates vary, but commonly cited figures are around 6-9%, not 20%.",
                "url": "https://example.org/amazon-oxygen-fact-check",
            }
        ],
        "claim_date": "2023-05-01",
        "speaker": "Anonymous social media post",
    }
    record.update(overrides)
    return record


# ─── Identity ─────────────────────────────────────────────────────────────────

class TestAVeriTecDatasetIdentity:
    def test_name_and_description(self) -> None:
        dataset = AVeriTecDataset(data_dir="/tmp/whatever")
        assert dataset.name == "averitec"
        assert "averitec" in dataset.description.lower()

    def test_default_data_dir_is_data_averitec(self) -> None:
        dataset = AVeriTecDataset()
        assert dataset.data_dir.parts[-2:] == ("data", "averitec")

    def test_custom_data_dir_override(self, data_dir: Path) -> None:
        dataset = AVeriTecDataset(data_dir=data_dir)
        assert dataset.data_dir == data_dir

    def test_registered_under_averitec_name(self) -> None:
        assert "averitec" in list_datasets()
        assert isinstance(get_dataset("averitec"), AVeriTecDataset)


# ─── load() ───────────────────────────────────────────────────────────────────

class TestAVeriTecDatasetLoad:
    def test_load_reports_averitec_as_source_dataset(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record()])
        claim = AVeriTecDataset(data_dir=data_dir).load()[0]
        assert claim.source_dataset == "averitec"

    def test_load_maps_claim_text(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record()])
        claim = AVeriTecDataset(data_dir=data_dir).load()[0]
        assert claim.original_fact == (
            "The Amazon rainforest produces roughly 20% of the world's oxygen."
        )

    def test_load_uses_first_evidence_question_as_context_query(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record()])
        claim = AVeriTecDataset(data_dir=data_dir).load()[0]
        assert claim.context_query == (
            "What share of global oxygen production is attributed to the "
            "Amazon rainforest?"
        )

    def test_load_falls_back_to_templated_question_when_no_evidence(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record(evidence=[])])
        claim = AVeriTecDataset(data_dir=data_dir).load()[0]
        assert claim.context_query == (
            "Is it true that The Amazon rainforest produces roughly 20% of "
            "the world's oxygen.?"
        )

    def test_load_falls_back_when_evidence_missing_question_key(self, data_dir: Path) -> None:
        record = _supported_record(evidence=[{"answer": "no question here", "url": "https://x"}])
        _write_jsonl(data_dir / "test.jsonl", [record])
        claim = AVeriTecDataset(data_dir=data_dir).load()[0]
        assert claim.context_query.startswith("Is it true that")

    def test_load_skips_non_supported_labels(self, data_dir: Path) -> None:
        records = [
            _supported_record(claim="Claim A"),
            _supported_record(claim="Claim B", label="Refuted"),
            _supported_record(claim="Claim C", label="Not Enough Evidence"),
            _supported_record(claim="Claim D", label="Conflicting"),
        ]
        _write_jsonl(data_dir / "test.jsonl", records)
        claims = AVeriTecDataset(data_dir=data_dir).load()
        assert [c.original_fact for c in claims] == ["Claim A"]

    def test_load_claim_ids_reflect_raw_file_position_not_filtered_position(
        self, data_dir: Path
    ) -> None:
        """
        Claim B (index 1) is dropped by the label filter, so the surviving
        Claim C should be AVERITEC_00002, not AVERITEC_00001 — see the
        module docstring's stability rationale.
        """
        records = [
            _supported_record(claim="Claim A"),
            _supported_record(claim="Claim B", label="Refuted"),
            _supported_record(claim="Claim C"),
        ]
        _write_jsonl(data_dir / "test.jsonl", records)
        claims = AVeriTecDataset(data_dir=data_dir).load()
        assert [c.claim_id for c in claims] == ["AVERITEC_00000", "AVERITEC_00002"]

    def test_load_carries_optional_metadata_fields(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record()])
        claim = AVeriTecDataset(data_dir=data_dir).load()[0]
        assert claim.metadata["label"] == "Supported"
        assert claim.metadata["claim_date"] == "2023-05-01"
        assert claim.metadata["speaker"] == "Anonymous social media post"
        assert claim.metadata["evidence_urls"] == [
            "https://example.org/amazon-oxygen-fact-check"
        ]
        assert claim.metadata["verified"] is False

    def test_load_omits_optional_metadata_fields_when_absent(self, data_dir: Path) -> None:
        record = _supported_record(evidence=[])
        del record["claim_date"]
        del record["speaker"]
        _write_jsonl(data_dir / "test.jsonl", [record])
        claim = AVeriTecDataset(data_dir=data_dir).load()[0]
        assert "claim_date" not in claim.metadata
        assert "speaker" not in claim.metadata
        assert "evidence_urls" not in claim.metadata

    def test_load_ignores_blank_lines(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(_supported_record()) + "\n\n   \n"
        (data_dir / "test.jsonl").write_text(text, encoding="utf-8")
        claims = AVeriTecDataset(data_dir=data_dir).load()
        assert len(claims) == 1

    def test_load_respects_max_claims(self, data_dir: Path) -> None:
        records = [_supported_record(claim=f"Claim {i}") for i in range(5)]
        _write_jsonl(data_dir / "test.jsonl", records)
        claims = AVeriTecDataset(data_dir=data_dir).load(max_claims=2)
        assert [c.original_fact for c in claims] == ["Claim 0", "Claim 1"]

    def test_load_selects_file_by_split(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "dev.jsonl", [_supported_record(claim="Dev claim")])
        _write_jsonl(data_dir / "test.jsonl", [_supported_record(claim="Test claim")])
        claims = AVeriTecDataset(data_dir=data_dir).load(split="dev")
        assert [c.original_fact for c in claims] == ["Dev claim"]

    def test_load_raises_on_missing_file(self, data_dir: Path) -> None:
        with pytest.raises(IngestionError, match="Could not read"):
            AVeriTecDataset(data_dir=data_dir).load()

    def test_load_raises_on_invalid_json_line(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "test.jsonl").write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(IngestionError, match="invalid JSON"):
            AVeriTecDataset(data_dir=data_dir).load()

    def test_load_raises_when_line_is_not_a_json_object(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "test.jsonl").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(IngestionError, match="must be a JSON object"):
            AVeriTecDataset(data_dir=data_dir).load()

    def test_load_raises_on_supported_record_missing_claim_field(self, data_dir: Path) -> None:
        record = _supported_record()
        del record["claim"]
        _write_jsonl(data_dir / "test.jsonl", [record])
        with pytest.raises(IngestionError, match="missing required field"):
            AVeriTecDataset(data_dir=data_dir).load()


# ─── content_hash ─────────────────────────────────────────────────────────────

class TestAVeriTecDatasetContentHash:
    def test_content_hash_before_load_is_all_zeros(self, data_dir: Path) -> None:
        assert AVeriTecDataset(data_dir=data_dir).content_hash == "0" * 16

    def test_content_hash_after_load_is_deterministic(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record()])
        first = AVeriTecDataset(data_dir=data_dir)
        first.load()
        second = AVeriTecDataset(data_dir=data_dir)
        second.load()
        assert first.content_hash == second.content_hash
        assert len(first.content_hash) == 16

    def test_content_hash_changes_with_different_content(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record(claim="Claim A")])
        dataset_a = AVeriTecDataset(data_dir=data_dir)
        dataset_a.load()

        _write_jsonl(data_dir / "test.jsonl", [_supported_record(claim="Claim B")])
        dataset_b = AVeriTecDataset(data_dir=data_dir)
        dataset_b.load()

        assert dataset_a.content_hash != dataset_b.content_hash


# ─── download() ─────────────────────────────────────────────────────────────

class TestAVeriTecDatasetDownload:
    def test_download_noops_when_jsonl_files_already_present(self, data_dir: Path) -> None:
        _write_jsonl(data_dir / "test.jsonl", [_supported_record()])
        AVeriTecDataset(data_dir=data_dir).download(str(data_dir))  # must not raise

    def test_download_raises_when_directory_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(IngestionError, match="Automated download is not implemented"):
            AVeriTecDataset().download(str(missing))

    def test_download_raises_when_directory_empty(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(IngestionError, match="Automated download is not implemented"):
            AVeriTecDataset().download(str(data_dir))
