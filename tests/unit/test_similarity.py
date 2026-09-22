"""
Unit tests for eiger.utils.similarity (raw_cosine / embed_cosine_similarity_01).

Factored out of test_heuristic_scorer.py when this module was extracted
from eiger.metrics.heuristic_scorer so eiger.metrics.pcs could reuse the
same cosine-similarity computation without duplicating it.

Tests verify:
  - raw_cosine returns 0.0 for a zero vector instead of raising ZeroDivisionError
  - raw_cosine raises on mismatched vector lengths (zip(strict=True)) rather
    than silently truncating
  - raw_cosine returns the correct value for a known pair of vectors
  - embed_cosine_similarity_01 rescales cosine similarity from [-1, 1] to
    [0, 1]: identical embeddings -> 1.0, opposite -> 0.0, orthogonal -> 0.5
"""

from __future__ import annotations

import pytest

from eiger.core.interfaces import BaseEmbedder
from eiger.utils.similarity import embed_cosine_similarity_01, raw_cosine


class _StubEmbedder(BaseEmbedder):
    """Minimal BaseEmbedder returning a fixed vector pair, in call order."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == len(self._vectors)
        return self._vectors

    @property
    def embedding_dim(self) -> int:
        return len(self._vectors[0]) if self._vectors else 0


# ─── raw_cosine ─────────────────────────────────────────────────────────────────

class TestRawCosine:
    def test_zero_vector_returns_zero_not_raising(self) -> None:
        assert raw_cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert raw_cosine([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            raw_cosine([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_known_value(self) -> None:
        # [1, 0] . [1, 1] / (1 * sqrt(2)) = 1/sqrt(2) ~= 0.7071
        assert raw_cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(0.7071067811865476)


# ─── embed_cosine_similarity_01 ─────────────────────────────────────────────────

class TestEmbedCosineSimilarity01:
    def test_identical_embeddings_give_one(self) -> None:
        embedder = _StubEmbedder([[1.0, 0.0], [1.0, 0.0]])
        assert embed_cosine_similarity_01(embedder, "a", "b") == pytest.approx(1.0)

    def test_opposite_embeddings_give_zero(self) -> None:
        embedder = _StubEmbedder([[1.0, 0.0], [-1.0, 0.0]])
        assert embed_cosine_similarity_01(embedder, "a", "b") == pytest.approx(0.0)

    def test_orthogonal_embeddings_give_half(self) -> None:
        embedder = _StubEmbedder([[1.0, 0.0], [0.0, 1.0]])
        assert embed_cosine_similarity_01(embedder, "a", "b") == pytest.approx(0.5)

    def test_zero_vector_rescales_to_half(self) -> None:
        # raw_cosine's degenerate zero-vector fallback is 0.0 (see
        # TestRawCosine.test_zero_vector_returns_zero_not_raising above) —
        # embed_cosine_similarity_01 always rescales [-1, 1] -> [0, 1] via
        # (raw + 1) / 2 with no special-casing for the zero-vector fallback,
        # so a raw value of 0.0 rescales to 0.5, indistinguishable from a
        # genuinely orthogonal pair. This matches the pre-refactor behaviour
        # of EmbeddingFaithfulnessScorer._cosine_similarity (callers guard
        # against blank text upstream instead — see that class's __call__).
        embedder = _StubEmbedder([[0.0, 0.0], [1.0, 1.0]])
        assert embed_cosine_similarity_01(embedder, "a", "b") == 0.5
