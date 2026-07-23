"""
Unit tests for DenseRetriever (eiger.retrieval.retriever).

Tests verify:
  - __init__ stores embedder, vector_store, collection correctly
  - retrieve() encodes the query via embedder.encode([query])
  - retrieve() calls vector_store.search with correct collection/vector/top_k
  - retrieve() returns a RetrievalResult with correct query/claim_id/top_k
  - hits are built in rank order (1, 2, 3, …)
  - each hit's Document is correctly reconstructed from the payload
  - score normalization maps raw cosine similarity [-1, 1] into [0, 1]
  - score normalization clamps out-of-range floating point values
  - empty search results produce an empty hits list
  - RetrievalError is raised (with chained cause) when embedder.encode fails
  - RetrievalError is raised when the embedder returns no vector
  - RetrievalError is raised (with chained cause) when vector_store.search fails

What these tests do NOT cover:
  - Real embedding models or a real Qdrant server (covered by embedder/
    qdrant_store unit tests and integration tests).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.exceptions import RetrievalError
from eiger.core.interfaces import BaseEmbedder, BaseVectorStore
from eiger.core.models import RetrievalResult
from eiger.retrieval.retriever import DenseRetriever

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """
    Patch the module-level structlog logger for every test in this file.

    DenseRetriever.retrieve() calls log.debug()/log.info() unconditionally.
    Patching it out here (once, for the whole module) keeps every test body
    focused on retrieval behavior rather than logging, mirroring the intent
    behind the `with patch("eiger.<module>.log")` blocks used in
    test_qdrant_store.py and test_embedder.py.
    """
    with patch("eiger.retrieval.retriever.log"):
        yield


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_raw_hit(
    doc_id: str,
    score: float,
    claim_id: str = "C1",
    text: str = "some text",
) -> dict[str, Any]:
    """Build a raw hit dict in the shape returned by BaseVectorStore.search()."""
    return {
        "doc_id": doc_id,
        "score": score,
        "payload": {
            "doc_id": doc_id,
            "claim_id": claim_id,
            "text": text,
            "doc_type": "ground_truth",
        },
    }


def _make_retriever(
    query_vector: list[float] | None = None,
    search_results: list[dict[str, Any]] | None = None,
) -> tuple[DenseRetriever, MagicMock, MagicMock]:
    """
    Return a DenseRetriever wired to mock embedder and vector_store.

    Returns:
        (retriever, mock_embedder, mock_vector_store)
    """
    mock_embedder = MagicMock(spec=BaseEmbedder)
    default_vector = [0.1, 0.2, 0.3]
    resolved_vector = query_vector if query_vector is not None else default_vector
    mock_embedder.encode.return_value = [resolved_vector]

    mock_vector_store = MagicMock(spec=BaseVectorStore)
    mock_vector_store.search.return_value = search_results if search_results is not None else []

    retriever = DenseRetriever(
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        collection="eiger_corpus",
    )
    return retriever, mock_embedder, mock_vector_store


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestDenseRetrieverInit:
    """Tests for __init__ attribute storage."""

    def test_stores_embedder(self) -> None:
        retriever, mock_embedder, _ = _make_retriever()
        assert retriever.embedder is mock_embedder

    def test_stores_vector_store(self) -> None:
        retriever, _, mock_vector_store = _make_retriever()
        assert retriever.vector_store is mock_vector_store

    def test_stores_collection(self) -> None:
        retriever, _, _ = _make_retriever()
        assert retriever.collection == "eiger_corpus"


# ─── retrieve() — query encoding ──────────────────────────────────────────────

class TestRetrieveEncoding:
    """Tests for the query-encoding step of retrieve()."""

    def test_calls_encode_with_query_in_list(self) -> None:
        retriever, mock_embedder, _ = _make_retriever()
        retriever.retrieve("What is the inflation rate?", claim_id="C1", top_k=5)
        mock_embedder.encode.assert_called_once_with(["What is the inflation rate?"])

    def test_raises_retrieval_error_when_encode_fails(self) -> None:
        retriever, mock_embedder, _ = _make_retriever()
        mock_embedder.encode.side_effect = RuntimeError("model crashed")
        with pytest.raises(RetrievalError, match="Failed to encode query"):
            retriever.retrieve("query", claim_id="C1", top_k=5)

    def test_encode_failure_chains_original_exception(self) -> None:
        retriever, mock_embedder, _ = _make_retriever()
        original = RuntimeError("model crashed")
        mock_embedder.encode.side_effect = original
        with pytest.raises(RetrievalError) as exc_info:
            retriever.retrieve("query", claim_id="C1", top_k=5)
        assert exc_info.value.__cause__ is original

    def test_raises_retrieval_error_when_encoder_returns_empty(self) -> None:
        retriever, mock_embedder, _ = _make_retriever()
        mock_embedder.encode.return_value = []
        with pytest.raises(RetrievalError, match="no vector"):
            retriever.retrieve("query", claim_id="C1", top_k=5)


# ─── retrieve() — vector store search ─────────────────────────────────────────

class TestRetrieveSearch:
    """Tests for the vector-store-search step of retrieve()."""

    def test_calls_search_with_correct_collection(self) -> None:
        retriever, _, mock_vector_store = _make_retriever()
        retriever.retrieve("query", claim_id="C1", top_k=5)
        kwargs = mock_vector_store.search.call_args.kwargs
        assert kwargs["collection"] == "eiger_corpus"

    def test_calls_search_with_query_vector(self) -> None:
        retriever, _, mock_vector_store = _make_retriever(query_vector=[0.9, 0.8])
        retriever.retrieve("query", claim_id="C1", top_k=5)
        kwargs = mock_vector_store.search.call_args.kwargs
        assert kwargs["query_vector"] == [0.9, 0.8]

    def test_calls_search_with_top_k(self) -> None:
        retriever, _, mock_vector_store = _make_retriever()
        retriever.retrieve("query", claim_id="C1", top_k=7)
        kwargs = mock_vector_store.search.call_args.kwargs
        assert kwargs["top_k"] == 7

    def test_raises_retrieval_error_when_search_fails(self) -> None:
        retriever, _, mock_vector_store = _make_retriever()
        mock_vector_store.search.side_effect = RuntimeError("connection refused")
        with pytest.raises(RetrievalError, match="Vector store search failed"):
            retriever.retrieve("query", claim_id="C1", top_k=5)

    def test_search_failure_chains_original_exception(self) -> None:
        retriever, _, mock_vector_store = _make_retriever()
        original = RuntimeError("connection refused")
        mock_vector_store.search.side_effect = original
        with pytest.raises(RetrievalError) as exc_info:
            retriever.retrieve("query", claim_id="C1", top_k=5)
        assert exc_info.value.__cause__ is original


# ─── retrieve() — RetrievalResult assembly ────────────────────────────────────

class TestRetrieveResultAssembly:
    """Tests for the RetrievalResult built by retrieve()."""

    def test_returns_retrieval_result(self) -> None:
        retriever, _, _ = _make_retriever()
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert isinstance(result, RetrievalResult)

    def test_result_query_matches_input(self) -> None:
        retriever, _, _ = _make_retriever()
        result = retriever.retrieve("what is x?", claim_id="C1", top_k=5)
        assert result.query == "what is x?"

    def test_result_claim_id_matches_input(self) -> None:
        retriever, _, _ = _make_retriever()
        result = retriever.retrieve("query", claim_id="TEST_042", top_k=5)
        assert result.claim_id == "TEST_042"

    def test_result_top_k_matches_input(self) -> None:
        retriever, _, _ = _make_retriever()
        result = retriever.retrieve("query", claim_id="C1", top_k=3)
        assert result.top_k == 3

    def test_empty_search_results_produce_empty_hits(self) -> None:
        retriever, _, _ = _make_retriever(search_results=[])
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert result.hits == []

    def test_hits_length_matches_search_results(self) -> None:
        raw_hits = [_make_raw_hit("doc-1", 0.9), _make_raw_hit("doc-2", 0.5)]
        retriever, _, _ = _make_retriever(search_results=raw_hits)
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert len(result.hits) == 2

    def test_hits_are_ranked_in_order(self) -> None:
        raw_hits = [
            _make_raw_hit("doc-1", 0.9),
            _make_raw_hit("doc-2", 0.5),
            _make_raw_hit("doc-3", 0.1),
        ]
        retriever, _, _ = _make_retriever(search_results=raw_hits)
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert [h.rank for h in result.hits] == [1, 2, 3]

    def test_hit_document_fields_reconstructed_from_payload(self) -> None:
        raw_hits = [_make_raw_hit("doc-99", 0.7, claim_id="C7", text="hello world")]
        retriever, _, _ = _make_retriever(search_results=raw_hits)
        result = retriever.retrieve("query", claim_id="C7", top_k=5)
        doc = result.hits[0].document
        assert doc.doc_id == "doc-99"
        assert doc.claim_id == "C7"
        assert doc.text == "hello world"
        assert doc.doc_type == "ground_truth"

    def test_hits_preserve_search_order(self) -> None:
        """Ranks follow the order returned by the vector store, not a re-sort."""
        raw_hits = [_make_raw_hit("first", 0.5), _make_raw_hit("second", 0.9)]
        retriever, _, _ = _make_retriever(search_results=raw_hits)
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert [h.document.doc_id for h in result.hits] == ["first", "second"]


# ─── Score normalization ──────────────────────────────────────────────────────

class TestScoreNormalization:
    """Tests for DenseRetriever._normalize_score() and its use in retrieve()."""

    @pytest.mark.parametrize(
        "raw_score, expected",
        [
            (1.0, 1.0),
            (-1.0, 0.0),
            (0.0, 0.5),
            (0.5, 0.75),
            (-0.5, 0.25),
        ],
    )
    def test_normalize_score_maps_cosine_range(self, raw_score: float, expected: float) -> None:
        assert DenseRetriever._normalize_score(raw_score) == pytest.approx(expected)

    @pytest.mark.parametrize("raw_score", [1.0000000002, 5.0])
    def test_normalize_score_clamps_above_one(self, raw_score: float) -> None:
        assert DenseRetriever._normalize_score(raw_score) == 1.0

    @pytest.mark.parametrize("raw_score", [-1.0000000002, -5.0])
    def test_normalize_score_clamps_below_zero(self, raw_score: float) -> None:
        assert DenseRetriever._normalize_score(raw_score) == 0.0

    def test_retrieve_applies_normalization_to_hit_scores(self) -> None:
        raw_hits = [_make_raw_hit("doc-1", 1.0), _make_raw_hit("doc-2", -1.0)]
        retriever, _, _ = _make_retriever(search_results=raw_hits)
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert result.hits[0].score == pytest.approx(1.0)
        assert result.hits[1].score == pytest.approx(0.0)

    def test_hit_scores_stay_within_pydantic_bounds(self) -> None:
        """RetrievedDocument.score has ge=0.0, le=1.0 — must never raise."""
        raw_hits = [_make_raw_hit("doc-1", 1.0000000005)]
        retriever, _, _ = _make_retriever(search_results=raw_hits)
        # Should not raise a pydantic ValidationError.
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert 0.0 <= result.hits[0].score <= 1.0
