"""
Unit tests for PolitiFactDataset (eiger.datasets.politifact).

Covers everything AVeriTecDataset's tests don't already establish as a
shared pattern, focused on what's specific to the LIAR TSV format: no
header row, no evidence Q&A (templated context_query fallback), a
deliberately strict "true"-label-only filter (see the class's module
docstring for why this differs from docs/DATASETS.md section 4's own
stale example comment), and defensive handling of a raw schema whose
exact column count/positions aren't independently re-verified against a
real downloaded LIAR file.

What these tests do NOT cover:
  - Real network downloads of the LIAR zip archive — not a runtime
    dependency of this loader (download() is a guard, not a fetcher; see
    TestPolitiFactDatasetDownload).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from eiger.core.exceptions import IngestionError
from eiger.datasets import PolitiFactDataset, get_dataset, list_datasets

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    with patch("eiger.datasets.politifact.log"):
        yield


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "politifact"


def _row(
    claim_id: str = "11972.json",
    label: str = "true",
    statement: str = "The unemployment rate fell to a 50-year low in 2019.",
    subject: str = "economy,jobs",
    speaker: str = "jane-doe",
    job_title: str = "Senator",
    context: str = "a press conference",
) -> list[str]:
    # 9 columns (indices 0-8), matching _CONTEXT_COLUMN=8; columns 6-7
    # (state_info/party_affiliation) are left blank since this loader
    # doesn't consume them.
    return [claim_id, label, statement, subject, speaker, job_title, "", "", context]


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join("\t".join(cell for cell in row) for row in rows),
        encoding="utf-8",
    )


# ─── Identity ─────────────────────────────────────────────────────────────────

class TestPolitiFactDatasetIdentity:
    def test_name_and_description(self) -> None:
        dataset = PolitiFactDataset(data_dir="/tmp/whatever")
        assert dataset.name == "politifact"
        assert "politifact" in dataset.description.lower()

    def test_default_data_dir_is_data_politifact(self) -> None:
        dataset = PolitiFactDataset()
        assert dataset.data_dir.parts[-2:] == ("data", "politifact")

    def test_custom_data_dir_override(self, data_dir: Path) -> None:
        dataset = PolitiFactDataset(data_dir=data_dir)
        assert dataset.data_dir == data_dir

    def test_registered_under_politifact_name(self) -> None:
        assert "politifact" in list_datasets()
        assert isinstance(get_dataset("politifact"), PolitiFactDataset)


# ─── load() ───────────────────────────────────────────────────────────────────

class TestPolitiFactDatasetLoad:
    def test_load_reports_politifact_as_source_dataset(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row()])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert claim.source_dataset == "politifact"

    def test_load_maps_statement_to_original_fact(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row()])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert claim.original_fact == "The unemployment rate fell to a 50-year low in 2019."

    def test_load_strips_json_suffix_from_claim_id(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row(claim_id="11972.json")])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert claim.claim_id == "POLITIFACT_11972"

    def test_load_keeps_claim_id_without_json_suffix_unchanged(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row(claim_id="99999")])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert claim.claim_id == "POLITIFACT_99999"

    def test_load_uses_templated_context_query(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row()])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert claim.context_query == (
            "Is it true that The unemployment rate fell to a 50-year low in 2019.?"
        )

    def test_load_keeps_only_true_label(self, data_dir: Path) -> None:
        rows = [
            _row(claim_id="1", label="true", statement="Claim A"),
            _row(claim_id="2", label="false", statement="Claim B"),
            _row(claim_id="3", label="pants-fire", statement="Claim C"),
            _row(claim_id="4", label="barely-true", statement="Claim D"),
            _row(claim_id="5", label="half-true", statement="Claim E"),
            _row(claim_id="6", label="mostly-true", statement="Claim F"),
        ]
        _write_tsv(data_dir / "test.tsv", rows)
        claims = PolitiFactDataset(data_dir=data_dir).load()
        assert [c.original_fact for c in claims] == ["Claim A"]

    def test_load_label_match_is_case_insensitive(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row(label="True")])
        claims = PolitiFactDataset(data_dir=data_dir).load()
        assert len(claims) == 1

    def test_load_carries_optional_metadata_fields(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row()])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert claim.metadata["label"] == "true"
        assert claim.metadata["subject"] == "economy,jobs"
        assert claim.metadata["speaker"] == "jane-doe"
        assert claim.metadata["job_title"] == "Senator"
        assert claim.metadata["context"] == "a press conference"
        assert claim.metadata["verified"] is False

    def test_load_omits_optional_metadata_when_row_too_short(self, data_dir: Path) -> None:
        # Only id/label/statement — no subject/speaker/job_title/context.
        _write_tsv(data_dir / "test.tsv", [["1", "true", "Claim A"]])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert "subject" not in claim.metadata
        assert "speaker" not in claim.metadata
        assert "job_title" not in claim.metadata
        assert "context" not in claim.metadata

    def test_load_omits_optional_metadata_when_blank(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row(subject="", speaker="", job_title="", context="")])
        claim = PolitiFactDataset(data_dir=data_dir).load()[0]
        assert "subject" not in claim.metadata
        assert "speaker" not in claim.metadata
        assert "job_title" not in claim.metadata
        assert "context" not in claim.metadata

    def test_load_ignores_blank_lines(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        text = "\t".join(_row()) + "\n\n\n"
        (data_dir / "test.tsv").write_text(text, encoding="utf-8")
        claims = PolitiFactDataset(data_dir=data_dir).load()
        assert len(claims) == 1

    def test_load_respects_max_claims(self, data_dir: Path) -> None:
        rows = [_row(claim_id=str(i), statement=f"Claim {i}") for i in range(5)]
        _write_tsv(data_dir / "test.tsv", rows)
        claims = PolitiFactDataset(data_dir=data_dir).load(max_claims=2)
        assert [c.original_fact for c in claims] == ["Claim 0", "Claim 1"]

    def test_load_selects_file_by_split(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "valid.tsv", [_row(statement="Valid claim")])
        _write_tsv(data_dir / "test.tsv", [_row(statement="Test claim")])
        claims = PolitiFactDataset(data_dir=data_dir).load(split="valid")
        assert [c.original_fact for c in claims] == ["Valid claim"]

    def test_load_raises_on_missing_file(self, data_dir: Path) -> None:
        with pytest.raises(IngestionError, match="Could not read"):
            PolitiFactDataset(data_dir=data_dir).load()

    def test_load_raises_on_row_with_too_few_columns(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "test.tsv").write_text("only-one-column", encoding="utf-8")
        with pytest.raises(IngestionError, match="expected at least"):
            PolitiFactDataset(data_dir=data_dir).load()


# ─── content_hash ─────────────────────────────────────────────────────────────

class TestPolitiFactDatasetContentHash:
    def test_content_hash_before_load_is_all_zeros(self, data_dir: Path) -> None:
        assert PolitiFactDataset(data_dir=data_dir).content_hash == "0" * 16

    def test_content_hash_after_load_is_deterministic(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row()])
        first = PolitiFactDataset(data_dir=data_dir)
        first.load()
        second = PolitiFactDataset(data_dir=data_dir)
        second.load()
        assert first.content_hash == second.content_hash
        assert len(first.content_hash) == 16

    def test_content_hash_changes_with_different_content(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row(statement="Claim A")])
        dataset_a = PolitiFactDataset(data_dir=data_dir)
        dataset_a.load()

        _write_tsv(data_dir / "test.tsv", [_row(statement="Claim B")])
        dataset_b = PolitiFactDataset(data_dir=data_dir)
        dataset_b.load()

        assert dataset_a.content_hash != dataset_b.content_hash


# ─── download() ─────────────────────────────────────────────────────────────

class TestPolitiFactDatasetDownload:
    def test_download_noops_when_tsv_files_already_present(self, data_dir: Path) -> None:
        _write_tsv(data_dir / "test.tsv", [_row()])
        PolitiFactDataset(data_dir=data_dir).download(str(data_dir))  # must not raise

    def test_download_raises_when_directory_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(IngestionError, match="Automated download is not implemented"):
            PolitiFactDataset().download(str(missing))

    def test_download_raises_when_directory_empty(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(IngestionError, match="Automated download is not implemented"):
            PolitiFactDataset().download(str(data_dir))
