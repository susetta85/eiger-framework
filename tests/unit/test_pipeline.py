"""
Unit tests for IngestionPipeline (eiger.ingestion.pipeline).

Tests verify:
  - __init__ stores embedder, vector_store, collection correctly
  - IngestionResult.n_documents sums n_ground_truth and n_poisoned
  - ingest() calls embedder.embedding_dim (needed for collection sizing)
  - ingest() resets the collection with the correct name/dim by default
  - ingest(reset=False) skips the reset_collection call
  - ingest() encodes all document texts (ground-truth + poisoned, combined
    in CorpusBuilderResult.all_documents order)
  - ingest() calls vector_store.upsert with the correct collection,
    documents, and vectors
  - ingest() returns an IngestionResult with correct counts and dimension
  - an empty corpus resets the collection (if requested) but skips encode/upsert
  - IngestionError is raised (with chained cause) when reset_collection fails
  - IngestionError is raised (with chained cause) when embedder.encode fails
  - IngestionError is raised (with chained cause) when vector_store.upsert fails

What these tests do NOT cover:
  - Real embedding models or a real Qdrant server (covered by embedder/
    qdrant_store unit tests and integration tests).
  - CorpusBuilder's own logic (covered in test_corpus_builder.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseEmbedder, BaseVectorStore
from eiger.core.models import Document, PoisonedDocument
from eiger.ingestion.corpus_builder import CorpusBuilderResult
from eiger.ingestion.pipeline import IngestionPipeline, IngestionResult

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """
    Patch the module-level structlog logger for every test in this file.

    IngestionPipeline.ingest() calls log.info()/log.warning() unconditionally.
    Patching it out here (once, for the whole module) avoids a structlog
    version quirk (PrintLogger has no .name) that surfaces whenever
    configure_logging() has not been called, matching the pattern used in
    test_qdrant_store.py, test_embedder.py, and test_retriever.py.
    """
    with patch("eiger.ingestion.pipeline.log"):
        yield


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_gt(claim_id: str = "C1", text: str = "ground truth text") -> Document:
    """Return a minimal ground-truth Document."""
    return Document(claim_id=claim_id, text=text, doc_type="ground_truth")


def _make_poisoned(claim_id: str = "C1", text: str = "poisoned text") -> PoisonedDocument:
    """Return a minimal PoisonedDocument."""
    return PoisonedDocument(
        claim_id=claim_id,
        text=text,
        attack_name="numerical_shift",
        original_text="original text",
    )


def _make_pipeline(embedding_dim: int = 384) -> tuple[IngestionPipeline, MagicMock, MagicMock]:
    """
    Return an IngestionPipeline wired to mock embedder and vector_store.

    Returns:
        (pipeline, mock_embedder, mock_vector_store)
    """
    mock_embedder = MagicMock(spec=BaseEmbedder)
    mock_embedder.embedding_dim = embedding_dim
    mock_embedder.encode.side_effect = lambda texts: [[0.1] * embedding_dim for _ in texts]

    mock_vector_store = MagicMock(spec=BaseVectorStore)

    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        collection="eiger_corpus",
    )
    return pipeline, mock_embedder, mock_vector_store


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestIngestionPipelineInit:
    """Tests for __init__ attribute storage."""

    def test_stores_embedder(self) -> None:
        pipeline, mock_embedder, _ = _make_pipeline()
        assert pipeline.embedder is mock_embedder

    def test_stores_vector_store(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        assert pipeline.vector_store is mock_vector_store

    def test_stores_collection(self) -> None:
        pipeline, _, _ = _make_pipeline()
        assert pipeline.collection == "eiger_corpus"


# ─── IngestionResult ──────────────────────────────────────────────────────────

class TestIngestionResult:
    """Tests for the IngestionResult dataclass."""

    def test_n_documents_sums_ground_truth_and_poisoned(self) -> None:
        result = IngestionResult(
            collection="c", embedding_dim=384, n_ground_truth=3, n_poisoned=2
        )
        assert result.n_documents == 5

    def test_n_documents_zero_when_empty(self) -> None:
        result = IngestionResult(
            collection="c", embedding_dim=384, n_ground_truth=0, n_poisoned=0
        )
        assert result.n_documents == 0


# ─── ingest() — collection reset ──────────────────────────────────────────────

class TestIngestResetCollection:
    """Tests for the reset_collection step of ingest()."""

    def test_reset_called_by_default(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        pipeline.ingest(corpus)
        mock_vector_store.reset_collection.assert_called_once()

    def test_reset_called_with_correct_collection_and_dim(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline(embedding_dim=128)
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        pipeline.ingest(corpus)
        args, kwargs = mock_vector_store.reset_collection.call_args
        assert args[0] == "eiger_corpus"
        assert kwargs["dim"] == 128

    def test_reset_skipped_when_reset_false(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        pipeline.ingest(corpus, reset=False)
        mock_vector_store.reset_collection.assert_not_called()

    def test_raises_ingestion_error_when_reset_fails(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        mock_vector_store.reset_collection.side_effect = RuntimeError("qdrant down")
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        with pytest.raises(IngestionError, match="Failed to reset collection"):
            pipeline.ingest(corpus)

    def test_reset_failure_chains_original_exception(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        original = RuntimeError("qdrant down")
        mock_vector_store.reset_collection.side_effect = original
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        with pytest.raises(IngestionError) as exc_info:
            pipeline.ingest(corpus)
        assert exc_info.value.__cause__ is original


# ─── ingest() — encoding ──────────────────────────────────────────────────────

class TestIngestEncoding:
    """Tests for the document-encoding step of ingest()."""

    def test_encode_called_with_all_document_texts_in_order(self) -> None:
        gt = _make_gt(text="gt text")
        pois = _make_poisoned(text="poisoned text")
        pipeline, mock_embedder, _ = _make_pipeline()
        corpus = CorpusBuilderResult(ground_truth_docs=[gt], poisoned_docs=[pois])
        pipeline.ingest(corpus)
        mock_embedder.encode.assert_called_once_with(["gt text", "poisoned text"])

    def test_raises_ingestion_error_when_encode_fails(self) -> None:
        pipeline, mock_embedder, _ = _make_pipeline()
        mock_embedder.encode.side_effect = RuntimeError("model crashed")
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        with pytest.raises(IngestionError, match="Failed to encode"):
            pipeline.ingest(corpus)

    def test_encode_failure_chains_original_exception(self) -> None:
        pipeline, mock_embedder, _ = _make_pipeline()
        original = RuntimeError("model crashed")
        mock_embedder.encode.side_effect = original
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        with pytest.raises(IngestionError) as exc_info:
            pipeline.ingest(corpus)
        assert exc_info.value.__cause__ is original


# ─── ingest() — upsert ────────────────────────────────────────────────────────

class TestIngestUpsert:
    """Tests for the vector-store-upsert step of ingest()."""

    def test_upsert_called_with_correct_collection(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        pipeline.ingest(corpus)
        args, _ = mock_vector_store.upsert.call_args
        assert args[0] == "eiger_corpus"

    def test_upsert_called_with_all_documents(self) -> None:
        gt = _make_gt()
        pois = _make_poisoned()
        pipeline, _, mock_vector_store = _make_pipeline()
        corpus = CorpusBuilderResult(ground_truth_docs=[gt], poisoned_docs=[pois])
        pipeline.ingest(corpus)
        args, _ = mock_vector_store.upsert.call_args
        assert args[1] == [gt, pois]

    def test_upsert_called_with_vectors_matching_document_count(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline(embedding_dim=16)
        corpus = CorpusBuilderResult(
            ground_truth_docs=[_make_gt("C1"), _make_gt("C2")]
        )
        pipeline.ingest(corpus)
        args, _ = mock_vector_store.upsert.call_args
        vectors = args[2]
        assert len(vectors) == 2
        assert all(len(v) == 16 for v in vectors)

    def test_raises_ingestion_error_when_upsert_fails(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        mock_vector_store.upsert.side_effect = RuntimeError("connection refused")
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        with pytest.raises(IngestionError, match="Failed to upsert"):
            pipeline.ingest(corpus)

    def test_upsert_failure_chains_original_exception(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        original = RuntimeError("connection refused")
        mock_vector_store.upsert.side_effect = original
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        with pytest.raises(IngestionError) as exc_info:
            pipeline.ingest(corpus)
        assert exc_info.value.__cause__ is original


# ─── ingest() — result & empty corpus ─────────────────────────────────────────

class TestIngestResultAndEmptyCorpus:
    """Tests for the returned IngestionResult and the empty-corpus edge case."""

    def test_returns_ingestion_result(self) -> None:
        pipeline, _, _ = _make_pipeline()
        corpus = CorpusBuilderResult(ground_truth_docs=[_make_gt()])
        result = pipeline.ingest(corpus)
        assert isinstance(result, IngestionResult)

    def test_result_counts_and_dimension_correct(self) -> None:
        pipeline, _, _ = _make_pipeline(embedding_dim=64)
        corpus = CorpusBuilderResult(
            ground_truth_docs=[_make_gt("C1"), _make_gt("C2")],
            poisoned_docs=[_make_poisoned("C1")],
        )
        result = pipeline.ingest(corpus)
        assert result.collection == "eiger_corpus"
        assert result.embedding_dim == 64
        assert result.n_ground_truth == 2
        assert result.n_poisoned == 1
        assert result.n_documents == 3

    def test_empty_corpus_skips_encode_and_upsert(self) -> None:
        pipeline, mock_embedder, mock_vector_store = _make_pipeline()
        corpus = CorpusBuilderResult()
        result = pipeline.ingest(corpus)
        mock_embedder.encode.assert_not_called()
        mock_vector_store.upsert.assert_not_called()
        assert result.n_documents == 0

    def test_empty_corpus_still_resets_collection_by_default(self) -> None:
        pipeline, _, mock_vector_store = _make_pipeline()
        corpus = CorpusBuilderResult()
        pipeline.ingest(corpus)
        mock_vector_store.reset_collection.assert_called_once()

    def test_empty_corpus_with_reset_false_calls_nothing(self) -> None:
        pipeline, mock_embedder, mock_vector_store = _make_pipeline()
        corpus = CorpusBuilderResult()
        pipeline.ingest(corpus, reset=False)
        mock_vector_store.reset_collection.assert_not_called()
        mock_embedder.encode.assert_not_called()
        mock_vector_store.upsert.assert_not_called()
