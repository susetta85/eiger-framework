"""
Unit tests for scripts/enrich_snopes_claims.py's filter_and_dedupe().

Not covered by the coverage gate (pyproject.toml omits scripts/*), but
this function has real, non-trivial dedup/filter logic that a Sprint-5+
review found two genuine bugs in (a missing original_verdict allowlist on
the verified_false side, and a dedup-priority bug that could let a false
row shadow a later true row with the same claim_id) — these tests lock in
the fix so a future refactor can't silently reintroduce either one.

scripts/ has no __init__.py (it's not a package), so the module is loaded
directly from its file path rather than via a normal import statement.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "enrich_snopes_claims.py"


@pytest.fixture(scope="module")
def enrich_module():
    spec = importlib.util.spec_from_file_location("enrich_snopes_claims", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(claim_id: str, rating: bool, verdict: str) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim": f"Claim text for {claim_id}",
        "url": "https://example.org",
        "date_published": "2020-01-01",
        "original_verdict": verdict,
        "normalised_rating": rating,
    }


class TestFilterAndDedupeDefault:
    def test_default_keeps_only_true_and_verdict_consistent(self, enrich_module) -> None:
        rows = [
            _row("1", True, "true"),
            _row("2", True, "correct attribution"),
            _row("3", True, "legit"),
            _row("4", True, "no"),  # known contamination, must stay excluded
            _row("5", False, "false"),
        ]
        kept = enrich_module.filter_and_dedupe(rows)
        assert {r["claim_id"] for r in kept} == {"1", "2", "3"}
        assert all(r["_ground_truth_label"] == "verified_true" for r in kept)

    def test_default_preserves_file_order(self, enrich_module) -> None:
        rows = [_row("2", True, "true"), _row("1", True, "true")]
        kept = enrich_module.filter_and_dedupe(rows)
        assert [r["claim_id"] for r in kept] == ["2", "1"]

    def test_default_dedupes_by_claim_id_first_occurrence_wins(self, enrich_module) -> None:
        rows = [
            _row("1", True, "true"),
            {**_row("1", True, "true"), "claim": "A duplicate, different text"},
        ]
        kept = enrich_module.filter_and_dedupe(rows)
        assert len(kept) == 1
        assert kept[0]["claim"] == "Claim text for 1"


class TestFilterAndDedupeIncludeVerifiedFalse:
    def test_keeps_false_verdict_tagged_verified_false(self, enrich_module) -> None:
        rows = [_row("1", True, "true"), _row("2", False, "false")]
        kept = enrich_module.filter_and_dedupe(rows, include_verified_false=True)
        by_id = {r["claim_id"]: r["_ground_truth_label"] for r in kept}
        assert by_id == {"1": "verified_true", "2": "verified_false"}

    @pytest.mark.parametrize(
        "ambiguous_verdict",
        ["satire", "mixture", "miscaptioned", "unproven", "legend", "scam",
         "misattributed", "correct attribution", "no"],
    )
    def test_ambiguous_false_verdicts_stay_excluded(self, enrich_module, ambiguous_verdict) -> None:
        """
        Only original_verdict=="false" corroborates normalised_rating=False
        (see _VERIFIED_FALSE_ORIGINAL_VERDICTS's docstring) — a real Snopes
        export was found to have "correct attribution" (a TRUE-side verdict
        string) on 43 normalised_rating=False rows, which must not be
        admitted just because the flag is on.
        """
        rows = [_row("1", False, ambiguous_verdict)]
        kept = enrich_module.filter_and_dedupe(rows, include_verified_false=True)
        assert kept == []

    def test_true_row_wins_dedup_even_when_false_row_occurs_first(self, enrich_module) -> None:
        """
        Regression test: a false-rated row and a true-rated row sharing the
        same claim_id must always resolve to the true row, regardless of
        which one appears first in the raw file — a verified-true claim
        must never be silently dropped just because include_verified_false
        was turned on.
        """
        rows = [_row("1", False, "false"), _row("1", True, "true")]
        kept = enrich_module.filter_and_dedupe(rows, include_verified_false=True)
        assert len(kept) == 1
        assert kept[0]["_ground_truth_label"] == "verified_true"

    def test_false_row_excluded_when_claim_id_already_claimed_by_true(self, enrich_module) -> None:
        rows = [_row("1", True, "true"), _row("1", False, "false")]
        kept = enrich_module.filter_and_dedupe(rows, include_verified_false=True)
        assert len(kept) == 1
        assert kept[0]["_ground_truth_label"] == "verified_true"

    def test_false_side_dedupes_by_claim_id_first_occurrence_wins(self, enrich_module) -> None:
        rows = [
            _row("1", False, "false"),
            {**_row("1", False, "false"), "claim": "A duplicate false row"},
        ]
        kept = enrich_module.filter_and_dedupe(rows, include_verified_false=True)
        assert len(kept) == 1
        assert kept[0]["claim"] == "Claim text for 1"
