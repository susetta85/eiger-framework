"""
Reproducibility tests for the seeding utilities.

These tests are the scientific validity gate for the framework:
same seed must always produce identical output.

Tests verify:
  - make_rng() produces the same sequence for the same seed
  - make_rng() does not affect the global random.random() state
  - derive_seed() is deterministic and context-sensitive
  - seed_everything() seeds the Python random module
  - seed_everything() gracefully handles missing numpy (ImportError branch)
  - seed_everything() gracefully handles missing torch (ImportError branch)
  - seed_everything() seeds CUDA devices when torch.cuda.is_available() is True
"""

from __future__ import annotations

import random
import sys
from unittest.mock import MagicMock

import pytest

from eiger.utils.seeding import make_rng, derive_seed, seed_everything


class TestMakeRng:
    """Tests for make_rng() — isolated random.Random instance factory."""

    def test_same_seed_same_sequence(self) -> None:
        """
        Two make_rng() calls with the same seed must produce identical sequences.

        This is the core reproducibility guarantee: given the same seed, every
        stochastic operation that uses make_rng() will produce the same outcome
        across processes, machines, and Python versions.
        """
        rng1 = make_rng(42)
        rng2 = make_rng(42)
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]
        assert seq1 == seq2

    def test_different_seeds_different_sequences(self) -> None:
        """Different seeds must produce different random sequences."""
        seq1 = [make_rng(1).random() for _ in range(5)]
        seq2 = [make_rng(2).random() for _ in range(5)]
        assert seq1 != seq2

    def test_does_not_affect_global_state(self) -> None:
        """
        Using a make_rng() instance must not change the global random.random() state.

        Verification: record the value of random.random() with a fixed global seed,
        reset the global seed, draw 100 values from an isolated rng, then check
        that random.random() still produces the same value. If make_rng() used
        random.seed() internally, the check would fail.
        """
        random.seed(0)
        before = random.random()
        random.seed(0)
        rng = make_rng(99)
        _ = [rng.random() for _ in range(100)]
        after = random.random()
        assert before == after


class TestDeriveSeed:
    """Tests for derive_seed() — SHA-256-based deterministic child seed derivation."""

    def test_deterministic(self) -> None:
        """Same parent seed + same context must always produce the same child seed."""
        s1 = derive_seed(42, "claim_001", "numerical_shift")
        s2 = derive_seed(42, "claim_001", "numerical_shift")
        assert s1 == s2

    def test_different_context_different_seed(self) -> None:
        """Different context strings must produce different child seeds."""
        s1 = derive_seed(42, "claim_001", "attack_a")
        s2 = derive_seed(42, "claim_001", "attack_b")
        assert s1 != s2

    def test_different_parent_different_seed(self) -> None:
        """Different parent seeds (same context) must produce different child seeds."""
        s1 = derive_seed(42, "ctx")
        s2 = derive_seed(99, "ctx")
        assert s1 != s2

    def test_returns_int(self) -> None:
        """derive_seed() must return a plain int suitable for seeding any RNG."""
        assert isinstance(derive_seed(42, "ctx"), int)


class TestSeedEverything:
    """Tests for seed_everything() — global RNG seeding for full reproducibility."""

    def test_seeds_random_module(self) -> None:
        """
        seed_everything() must seed Python's random module so that repeated
        calls with the same seed produce the same random.random() value.
        """
        seed_everything(42)
        v1 = random.random()
        seed_everything(42)
        v2 = random.random()
        assert v1 == v2


class TestSeedEverythingCoverage:
    """
    Branch-coverage tests for the ImportError fallback paths in seed_everything().

    seed_everything() is designed to work in environments without numpy or torch.
    It uses try/except ImportError blocks for both libraries. These tests simulate
    missing packages by injecting None into sys.modules, which causes any subsequent
    ``import <name>`` to raise ImportError — the standard Python mechanism for
    mocking absent packages.
    """

    def test_missing_numpy_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        seed_everything() must complete without error when numpy is not installed.

        The ``except ImportError: pass`` block after the numpy import must
        silently catch the error and continue. No warning or exception must
        propagate to the caller.
        """
        # Setting sys.modules['numpy'] = None makes ``import numpy`` raise ImportError.
        monkeypatch.setitem(sys.modules, "numpy", None)
        seed_everything(42)  # must not raise

    def test_missing_torch_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        seed_everything() must complete without error when torch is not installed.

        The ``except ImportError: pass`` block after the torch import must
        silently catch the error and continue.
        """
        monkeypatch.setitem(sys.modules, "torch", None)
        seed_everything(42)  # must not raise

    def test_cuda_manual_seed_called_when_cuda_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        When torch.cuda.is_available() returns True, seed_everything() must call
        torch.cuda.manual_seed_all() to seed all GPU devices.

        This exercises the ``if torch.cuda.is_available()`` branch that is
        otherwise unreachable in CPU-only CI environments.

        Implementation: replace sys.modules['torch'] with a MagicMock whose
        cuda.is_available() returns True, then verify cuda.manual_seed_all()
        was called with the correct seed.
        """
        mock_torch = MagicMock()
        # Make is_available() return True to enter the CUDA branch
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        seed_everything(99)
        # Verify the CUDA seeding call was made with the correct seed
        mock_torch.cuda.manual_seed_all.assert_called_once_with(99)
        # Also verify the main manual_seed was called
        mock_torch.manual_seed.assert_called_once_with(99)
