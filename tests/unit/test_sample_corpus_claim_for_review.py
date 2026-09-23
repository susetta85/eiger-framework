"""
Unit tests for scripts/sample_corpus_claim_for_review.py's stratified_sample().

Not covered by the coverage gate (pyproject.toml omits scripts/*), but the
stratification/allocation logic is non-trivial enough to warrant regression
tests. write_review_workbook()/main() (I/O-heavy) are not covered here —
they were verified manually against the real workbook (see the alignment
document / task history) rather than re-tested against a synthetic file,
consistent with test_enrich_snopes_claims.py's own scope note.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import namedtuple
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "sample_corpus_claim_for_review.py"
)

_FakeClaim = namedtuple("_FakeClaim", ["claim_id", "sensitivity_class", "metadata"])


@pytest.fixture(scope="module")
def sample_module():
    spec = importlib.util.spec_from_file_location("sample_corpus_claim_for_review", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _claim(claim_id: str, manip: str | None, sens: str | None) -> _FakeClaim:
    return _FakeClaim(claim_id=claim_id, sensitivity_class=sens, metadata={"manipulation_type_applied": manip})


class TestStratifiedSample:
    def test_respects_min_per_stratum_floor(self, sample_module) -> None:
        claims = [_claim(f"C{i}", "M04", "S1") for i in range(5)] + [
            _claim(f"D{i}", "M06", "S2") for i in range(200)
        ]
        sampled = sample_module.stratified_sample(claims, sample_size=10, seed=42, min_per_stratum=3)
        by_stratum: dict[tuple, int] = {}
        for c in sampled:
            key = (c.metadata["manipulation_type_applied"], c.sensitivity_class)
            by_stratum[key] = by_stratum.get(key, 0) + 1
        assert by_stratum[("M04", "S1")] >= 3

    def test_floor_capped_at_stratum_size(self, sample_module) -> None:
        """A stratum smaller than min_per_stratum contributes all of itself, not more."""
        claims = [_claim(f"C{i}", "M04", "S1") for i in range(2)]
        sampled = sample_module.stratified_sample(claims, sample_size=10, seed=42, min_per_stratum=20)
        assert len(sampled) == 2

    def test_deterministic_for_same_seed(self, sample_module) -> None:
        claims = [_claim(f"C{i}", "M01", "S1") for i in range(50)]
        first = sample_module.stratified_sample(claims, sample_size=10, seed=7, min_per_stratum=2)
        second = sample_module.stratified_sample(claims, sample_size=10, seed=7, min_per_stratum=2)
        assert [c.claim_id for c in first] == [c.claim_id for c in second]

    def test_different_seeds_can_produce_different_samples(self, sample_module) -> None:
        claims = [_claim(f"C{i}", "M01", "S1") for i in range(50)]
        first = sample_module.stratified_sample(claims, sample_size=10, seed=1, min_per_stratum=5)
        second = sample_module.stratified_sample(claims, sample_size=10, seed=2, min_per_stratum=5)
        assert {c.claim_id for c in first} != {c.claim_id for c in second}

    def test_proportional_allocation_favors_larger_stratum(self, sample_module) -> None:
        claims = [_claim(f"A{i}", "M01", "S1") for i in range(900)] + [
            _claim(f"B{i}", "M02", "S2") for i in range(100)
        ]
        sampled = sample_module.stratified_sample(claims, sample_size=100, seed=42, min_per_stratum=5)
        n_a = sum(1 for c in sampled if c.claim_id.startswith("A"))
        n_b = sum(1 for c in sampled if c.claim_id.startswith("B"))
        assert n_a > n_b

    def test_empty_population_returns_empty_sample(self, sample_module) -> None:
        assert sample_module.stratified_sample([], sample_size=10, seed=42, min_per_stratum=5) == []

    def test_no_claim_sampled_twice_within_a_stratum(self, sample_module) -> None:
        claims = [_claim(f"C{i}", "M01", "S1") for i in range(30)]
        sampled = sample_module.stratified_sample(claims, sample_size=15, seed=42, min_per_stratum=15)
        ids = [c.claim_id for c in sampled]
        assert len(ids) == len(set(ids))

    def test_none_manipulation_type_handled_without_raising(self, sample_module) -> None:
        """A claim with no manipulation_type_applied metadata forms its own stratum."""
        claims = [_claim(f"C{i}", None, "S1") for i in range(5)]
        sampled = sample_module.stratified_sample(claims, sample_size=3, seed=42, min_per_stratum=2)
        # Single stratum, so it gets the full proportional share of the
        # population: round(3 * 5 / 5) = 3, not the min_per_stratum floor.
        assert len(sampled) == 3
