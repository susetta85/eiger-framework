"""
Reproducibility tests for the seeding utilities.

These tests are the scientific validity gate for the framework:
same seed must always produce identical output.
"""

from __future__ import annotations

import random

from eiger.utils.seeding import make_rng, derive_seed, seed_everything


class TestMakeRng:
    def test_same_seed_same_sequence(self) -> None:
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]
        assert seq1 == seq2

    def test_different_seeds_different_sequences(self) -> None:
        seq1 = [make_rng(1).random() for _ in range(5)]
        seq2 = [make_rng(2).random() for _ in range(5)]
        assert seq1 != seq2

    def test_does_not_affect_global_state(self) -> None:
        random.seed(0)
        before = random.random()
        random.seed(0)
        rng = make_rng(99)
        _ = [rng.random() for _ in range(100)]
        after = random.random()
        assert before == after


class TestDeriveSeed:
    def test_deterministic(self) -> None:
        s1 = derive_seed(42, "claim_001", "numerical_shift")
        s2 = derive_seed(42, "claim_001", "numerical_shift")
        assert s1 == s2

    def test_different_context_different_seed(self) -> None:
        s1 = derive_seed(42, "claim_001", "attack_a")
        s2 = derive_seed(42, "claim_001", "attack_b")
        assert s1 != s2

    def test_different_parent_different_seed(self) -> None:
        s1 = derive_seed(42, "ctx")
        s2 = derive_seed(99, "ctx")
        assert s1 != s2

    def test_returns_int(self) -> None:
        assert isinstance(derive_seed(42, "ctx"), int)


class TestSeedEverything:
    def test_seeds_random_module(self) -> None:
        seed_everything(42)
        v1 = random.random()
        seed_everything(42)
        v2 = random.random()
        assert v1 == v2
