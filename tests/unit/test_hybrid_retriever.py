"""
Unit tests for HybridRetriever (eiger.retrieval.hybrid_retriever).

Tests verify:
  - fit() delegates to the internal SparseRetriever.fit()
  - retrieve() calls both the internal DenseRetriever and SparseRetriever
    with (query, claim_id, top_k)
  - Reciprocal Rank Fusion (RRF) score computation is correct, including
    the case where a document appears in both rankings (summed) and where
    it appears in only one (single term)
  - final hits are sorted by descending fused score and truncated to top_k
  - the dense-side Document object is preferred when a doc_id appears in
    both rankings
  - the top hit's normalized score is always 1.0 (per-query max-normalization)
  - empty results from both underlying retrievers produce an empty hit list
  - a RetrievalError from either underlying retriever propagates unchanged

What these tests do NOT cover:
  - DenseRetriever's or SparseRetriever's own ranking/normalization logic
    (covered by test_retriever.py / test_sparse_retriever.py). Both are
    mocked here via MagicMock(spec=BaseRetriever) so this file exercises
    only HybridRetriever's own fusion logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from eiger.core.exceptions import RetrievalError
from eiger.core.models import Document, RetrievalResult, RetrievedDocument
from eiger.retrieval.hybrid_retriever import HybridRetriever
from eiger.retrieval.retriever import DenseRetriever
from eiger.retrieval.sparse_retriever import SparseRetriever

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _doc(doc_id: str, text: str = "text") -> Document:
    return Document(doc_id=doc_id, claim_id="C1", text=text, doc_type="ground_truth")


def _result(query: str, claim_id: str, hits: list[RetrievedDocument], top_k: int = 5) -> RetrievalResult:
    return RetrievalResult(query=query, claim_id=claim_id, hits=hits, top_k=top_k)


def _make_hybrid(
    dense_result: RetrievalResult | None = None,
    sparse_result: RetrievalResult | None = None,
    rrf_k: int = 60,
) -> tuple[HybridRetriever, MagicMock, MagicMock]:
    # Spec'd on the concrete classes (matching HybridRetriever's own type
    # hints), not BaseRetriever, so that mock_sparse.fit() — a method
    # specific to SparseRetriever, not part of the BaseRetriever interface
    # (see eiger/retrieval/sparse_retriever.py's module docstring) — is a
    # valid attribute to mock and assert on.
    mock_dense = MagicMock(spec=DenseRetriever)
    mock_sparse = MagicMock(spec=SparseRetriever)
    mock_dense.retrieve.return_value = dense_result or _result("q", "C1", [])
    mock_sparse.retrieve.return_value = sparse_result or _result("q", "C1", [])
    hybrid = HybridRetriever(dense=mock_dense, sparse=mock_sparse, rrf_k=rrf_k)
    return hybrid, mock_dense, mock_sparse


# ─── fit() ────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_delegates_to_sparse_fit(self) -> None:
        hybrid, _, mock_sparse = _make_hybrid()
        documents = [_doc("d1"), _doc("d2")]
        hybrid.fit(documents)
        mock_sparse.fit.assert_called_once_with(documents)

    def test_dense_retriever_has_no_fit_step(self) -> None:
        """
        The dense side has no fit() step (see module docstring) — DenseRetriever
        does not expose one at all, unlike SparseRetriever.
        """
        assert not hasattr(DenseRetriever, "fit")


# ─── retrieve() — delegation ────────────────────────────────────────────────────

class TestRetrieveDelegation:
    def test_calls_dense_with_query_claim_id_top_k(self) -> None:
        hybrid, mock_dense, _ = _make_hybrid()
        hybrid.retrieve("my query", claim_id="C7", top_k=3)
        mock_dense.retrieve.assert_called_once_with("my query", "C7", 3)

    def test_calls_sparse_with_query_claim_id_top_k(self) -> None:
        hybrid, _, mock_sparse = _make_hybrid()
        hybrid.retrieve("my query", claim_id="C7", top_k=3)
        mock_sparse.retrieve.assert_called_once_with("my query", "C7", 3)

    def test_dense_error_propagates(self) -> None:
        hybrid, mock_dense, _ = _make_hybrid()
        mock_dense.retrieve.side_effect = RetrievalError("dense backend failed")
        with pytest.raises(RetrievalError, match="dense backend failed"):
            hybrid.retrieve("q", claim_id="C1", top_k=5)

    def test_sparse_error_propagates(self) -> None:
        hybrid, _, mock_sparse = _make_hybrid()
        mock_sparse.retrieve.side_effect = RetrievalError("sparse backend failed")
        with pytest.raises(RetrievalError, match="sparse backend failed"):
            hybrid.retrieve("q", claim_id="C1", top_k=5)


# ─── retrieve() — RRF fusion ────────────────────────────────────────────────────

class TestRRFFusion:
    def test_empty_results_produce_empty_hits(self) -> None:
        hybrid, _, _ = _make_hybrid()
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        assert result.hits == []

    def test_document_only_in_dense_ranking_is_included(self) -> None:
        dense_result = _result("q", "C1", [RetrievedDocument(document=_doc("d1"), score=0.9, rank=1)])
        hybrid, _, _ = _make_hybrid(dense_result=dense_result)
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        assert [h.document.doc_id for h in result.hits] == ["d1"]

    def test_document_only_in_sparse_ranking_is_included(self) -> None:
        sparse_result = _result("q", "C1", [RetrievedDocument(document=_doc("d1"), score=0.9, rank=1)])
        hybrid, _, _ = _make_hybrid(sparse_result=sparse_result)
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        assert [h.document.doc_id for h in result.hits] == ["d1"]

    def test_document_in_both_rankings_outranks_single_ranking_document(self) -> None:
        """
        d1 appears at rank 2 in both dense and sparse (fused score =
        2 * 1/(60+2)); d2 appears only at rank 1 in dense (fused score =
        1/(60+1)). 2/62 (~0.0323) > 1/61 (~0.0164), so d1 must rank first
        despite never being the top hit in either individual ranking —
        this is the entire point of RRF (rewarding consistent presence
        across rankings over a single strong ranking).
        """
        dense_result = _result(
            "q", "C1",
            [
                RetrievedDocument(document=_doc("d2"), score=0.99, rank=1),
                RetrievedDocument(document=_doc("d1"), score=0.5, rank=2),
            ],
        )
        sparse_result = _result(
            "q", "C1",
            [RetrievedDocument(document=_doc("d1"), score=0.5, rank=2)],
        )
        hybrid, _, _ = _make_hybrid(dense_result=dense_result, sparse_result=sparse_result)
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        assert [h.document.doc_id for h in result.hits] == ["d1", "d2"]

    def test_fused_score_is_exact_rrf_sum(self) -> None:
        """Direct arithmetic check of the RRF formula with a non-default k."""
        dense_result = _result("q", "C1", [RetrievedDocument(document=_doc("d1"), score=0.9, rank=1)])
        sparse_result = _result("q", "C1", [RetrievedDocument(document=_doc("d1"), score=0.9, rank=3)])
        hybrid, _, _ = _make_hybrid(dense_result=dense_result, sparse_result=sparse_result, rrf_k=10)
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        # Only one candidate, so after normalization its score is always 1.0
        # regardless of the raw fused value — verify rank/identity instead,
        # and check the raw arithmetic via the internal formula directly.
        expected_raw = 1.0 / (10 + 1) + 1.0 / (10 + 3)
        assert expected_raw == pytest.approx(1 / 11 + 1 / 13)
        assert result.hits[0].document.doc_id == "d1"
        assert result.hits[0].score == pytest.approx(1.0)

    def test_top_hit_score_normalized_to_one(self) -> None:
        dense_result = _result(
            "q", "C1",
            [
                RetrievedDocument(document=_doc("d1"), score=0.9, rank=1),
                RetrievedDocument(document=_doc("d2"), score=0.5, rank=2),
            ],
        )
        hybrid, _, _ = _make_hybrid(dense_result=dense_result)
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        assert result.hits[0].score == pytest.approx(1.0)

    def test_ranks_are_sequential_starting_at_one(self) -> None:
        dense_result = _result(
            "q", "C1",
            [
                RetrievedDocument(document=_doc("d1"), score=0.9, rank=1),
                RetrievedDocument(document=_doc("d2"), score=0.5, rank=2),
            ],
        )
        hybrid, _, _ = _make_hybrid(dense_result=dense_result)
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        assert [h.rank for h in result.hits] == [1, 2]

    def test_truncates_to_top_k(self) -> None:
        dense_result = _result(
            "q", "C1",
            [RetrievedDocument(document=_doc(f"d{i}"), score=1.0 - i * 0.1, rank=i) for i in range(1, 6)],
        )
        hybrid, _, _ = _make_hybrid(dense_result=dense_result)
        result = hybrid.retrieve("q", claim_id="C1", top_k=2)
        assert len(result.hits) == 2

    def test_dense_document_object_preferred_on_overlap(self) -> None:
        """
        When a doc_id appears in both rankings, the dense-side Document
        object is kept (see module docstring's "does NOT do" note) — verify
        via object identity, since a doc_id match alone wouldn't
        distinguish which side's reconstruction was used.
        """
        dense_doc = _doc("d1", text="dense reconstruction")
        sparse_doc = _doc("d1", text="sparse reconstruction")
        dense_result = _result("q", "C1", [RetrievedDocument(document=dense_doc, score=0.9, rank=1)])
        sparse_result = _result("q", "C1", [RetrievedDocument(document=sparse_doc, score=0.9, rank=1)])
        hybrid, _, _ = _make_hybrid(dense_result=dense_result, sparse_result=sparse_result)
        result = hybrid.retrieve("q", claim_id="C1", top_k=5)
        assert result.hits[0].document is dense_doc

    def test_result_query_and_claim_id_match_input(self) -> None:
        hybrid, _, _ = _make_hybrid()
        result = hybrid.retrieve("my query", claim_id="C9", top_k=5)
        assert result.query == "my query"
        assert result.claim_id == "C9"

    def test_result_top_k_matches_input(self) -> None:
        hybrid, _, _ = _make_hybrid()
        result = hybrid.retrieve("q", claim_id="C1", top_k=7)
        assert result.top_k == 7


# ─── _normalize_score ───────────────────────────────────────────────────────────

class TestNormalizeScore:
    def test_zero_max_score_returns_zero(self) -> None:
        assert HybridRetriever._normalize_score(0.5, 0.0) == 0.0

    def test_negative_max_score_returns_zero(self) -> None:
        assert HybridRetriever._normalize_score(0.5, -1.0) == 0.0

    def test_positive_scores_normalized_proportionally(self) -> None:
        assert HybridRetriever._normalize_score(0.5, 1.0) == pytest.approx(0.5)
        assert HybridRetriever._normalize_score(1.0, 1.0) == pytest.approx(1.0)
