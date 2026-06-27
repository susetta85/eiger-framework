"""
Unit tests for SentenceTransformerEmbedder (eiger.retrieval.embedder).

Tests verify:
  - Default and custom model names are stored correctly
  - _model starts as None (lazy-load contract)
  - encode([]) returns [] without loading the model
  - _load_model() raises ImportError when sentence-transformers is missing
  - _load_model() is idempotent (calling twice does not reload)
  - encode() returns correct shape (n_texts × embedding_dim)
  - encode() output is a plain Python list of lists (JSON-serialisable)
  - embedding_dim triggers load and returns a positive integer
  - encode() delegates to the model with the configured batch_size

What these tests do NOT cover:
  - Real model download / inference (covered in integration tests).
  - GPU device placement.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from eiger.retrieval.embedder import SentenceTransformerEmbedder, DEFAULT_MODEL


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_model(dim: int = 384, n_texts: int = 1) -> MagicMock:
    """
    Return a MagicMock that behaves like a SentenceTransformer model.

    Args:
        dim:     Embedding dimensionality to simulate.
        n_texts: Number of texts the mock encode() call will process.

    Returns:
        MagicMock with .encode() and .get_sentence_embedding_dimension() set up.
    """
    import numpy as np

    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim
    # encode() returns a numpy array of shape (n_texts, dim)
    mock_model.encode.return_value = np.zeros((n_texts, dim), dtype="float32")
    return mock_model


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestEmbedderInit:
    """Tests for __init__ defaults and attribute storage."""

    def test_default_model_name(self) -> None:
        """The default model name must match the module-level DEFAULT_MODEL constant."""
        emb = SentenceTransformerEmbedder()
        assert emb.model_name == DEFAULT_MODEL

    def test_custom_model_name_stored(self) -> None:
        """A custom model name must be stored on the instance."""
        emb = SentenceTransformerEmbedder(model_name="my-org/custom-model")
        assert emb.model_name == "my-org/custom-model"

    def test_model_starts_as_none(self) -> None:
        """
        _model must be None at construction (lazy-load contract).

        Eager loading would trigger a ~22 MB download on every import,
        which is unacceptable for CI environments without a model cache.
        """
        emb = SentenceTransformerEmbedder()
        assert emb._model is None


# ─── encode edge cases ────────────────────────────────────────────────────────

class TestEncodeEdgeCases:
    """Tests for encode() with empty input or missing dependency."""

    def test_encode_empty_list_returns_empty(self) -> None:
        """
        encode([]) must return [] immediately, without loading the model.

        This short-circuit prevents unnecessary model initialisation when
        the corpus happens to be empty (e.g. during a dry-run experiment).
        """
        emb = SentenceTransformerEmbedder()
        result = emb.encode([])
        assert result == []
        # Model must still be None — we never loaded it
        assert emb._model is None

    def test_load_model_raises_when_sentence_transformers_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        _load_model() must raise ImportError when sentence-transformers is absent.

        Setting sys.modules['sentence_transformers'] = None forces any subsequent
        ``import sentence_transformers`` or ``from sentence_transformers import ...``
        to raise ImportError, simulating a missing package.
        """
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        emb = SentenceTransformerEmbedder()
        with pytest.raises(ImportError, match="sentence-transformers"):
            emb._load_model()


# ─── _load_model ─────────────────────────────────────────────────────────────

class TestLoadModel:
    """Tests for the lazy model loader."""

    def test_idempotent_when_already_loaded(self) -> None:
        """
        Calling _load_model() when _model is already set must be a no-op.

        This prevents accidental re-downloading in loops that call encode()
        many times.
        """
        emb = SentenceTransformerEmbedder()
        mock_model = _make_mock_model()
        emb._model = mock_model  # pre-load
        emb._load_model()        # must not overwrite
        assert emb._model is mock_model

    def test_load_model_sets_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_load_model() must set _model to a SentenceTransformer instance."""
        mock_model = _make_mock_model()
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st_module)

        emb = SentenceTransformerEmbedder()
        with patch("eiger.retrieval.embedder.log"):
            emb._load_model()

        assert emb._model is mock_model


# ─── encode ──────────────────────────────────────────────────────────────────

class TestEncode:
    """Tests for encode() with a mocked model."""

    def _embedder_with_mock(self, dim: int = 384, n_texts: int = 2) -> SentenceTransformerEmbedder:
        """Return an embedder whose _model is already set to a mock."""
        emb = SentenceTransformerEmbedder()
        emb._model = _make_mock_model(dim=dim, n_texts=n_texts)
        return emb

    def test_returns_list_of_lists(self) -> None:
        """encode() must return a plain list of lists (not numpy arrays)."""
        emb = self._embedder_with_mock(dim=384, n_texts=2)
        with patch("eiger.retrieval.embedder.log"):
            result = emb.encode(["hello", "world"])
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(x, float) for v in result for x in v)

    def test_output_length_matches_input(self) -> None:
        """encode() must return one vector per input text."""
        n = 3
        emb = self._embedder_with_mock(dim=384, n_texts=n)
        with patch("eiger.retrieval.embedder.log"):
            result = emb.encode(["a", "b", "c"])
        assert len(result) == n

    def test_vector_dimension_correct(self) -> None:
        """Each returned vector must have length equal to embedding_dim."""
        dim = 128
        emb = self._embedder_with_mock(dim=dim, n_texts=1)
        with patch("eiger.retrieval.embedder.log"):
            result = emb.encode(["test"])
        assert len(result[0]) == dim

    def test_batch_size_passed_to_model(self) -> None:
        """encode() must pass the configured batch_size to model.encode()."""
        emb = SentenceTransformerEmbedder(batch_size=16)
        emb._model = _make_mock_model(n_texts=1)
        with patch("eiger.retrieval.embedder.log"):
            emb.encode(["hello"])
        call_kwargs = emb._model.encode.call_args.kwargs
        assert call_kwargs.get("batch_size") == 16


# ─── embedding_dim ────────────────────────────────────────────────────────────

class TestEmbeddingDim:
    """Tests for the embedding_dim property."""

    def test_embedding_dim_returns_positive_int(self) -> None:
        """embedding_dim must return a positive integer."""
        emb = SentenceTransformerEmbedder()
        emb._model = _make_mock_model(dim=384)
        with patch("eiger.retrieval.embedder.log"):
            dim = emb.embedding_dim
        assert isinstance(dim, int)
        assert dim > 0

    def test_embedding_dim_matches_mock(self) -> None:
        """embedding_dim must return the value reported by the underlying model."""
        emb = SentenceTransformerEmbedder()
        emb._model = _make_mock_model(dim=768)
        with patch("eiger.retrieval.embedder.log"):
            assert emb.embedding_dim == 768
