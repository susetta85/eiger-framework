"""
Unit tests for SparseRetriever (eiger.retrieval.sparse_retriever).

Tests verify:
  - __init__ starts with empty internal state
  - fit() builds an internal BM25 index over the given documents
  - fit([]) (or never calling fit()) leaves retrieve() returning empty,
    valid results rather than raising
  - fit() raises an actionable ImportError if rank-bm25 is not installed
  - retrieve() ranks documents by lexical overlap with the query
  - retrieve() respects top_k
  - score normalization: top hit is always 1.0, scores stay within [0, 1],
    and an all-zero-overlap query produces all-zero scores rather than a
    division error
  - RetrievalResult.query/claim_id/top_k are passed through correctly
  - RetrievalError is raised (with chained cause) if BM25 scoring itself
    fails
  - _tokenize() lowercases and strips punctuation

What these tests do NOT cover:
  - ExperimentRunner's wiring of fit()/retrieve() into a full experiment
    run (covered by test_runner.py's TestRunSparseRetriever).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.exceptions import RetrievalError
from eiger.core.models import Document, RetrievalResult
from eiger.retrieval.sparse_retriever import SparseRetriever, _tokenize

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """Patch the module-level structlog logger, matching test_retriever.py."""
    with patch("eiger.retrieval.sparse_retriever.log"):
        yield


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_doc(doc_id: str, text: str, claim_id: str = "C1") -> Document:
    return Document(doc_id=doc_id, claim_id=claim_id, text=text, doc_type="ground_truth")


_CORPUS = [
    _make_doc("d1", "The central bank raised interest rates to fight inflation."),
    _make_doc("d2", "The football team won the championship match yesterday."),
    _make_doc("d3", "Inflation and interest rates dominated economic headlines."),
]


# ─── __init__ ───────────────────────────────────────────────────────────────────

class TestInit:
    def test_starts_with_no_documents(self) -> None:
        retriever = SparseRetriever()
        assert retriever._documents == []

    def test_starts_with_no_bm25_index(self) -> None:
        retriever = SparseRetriever()
        assert retriever._bm25 is None


# ─── fit() ──────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_stores_documents(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        assert retriever._documents == _CORPUS

    def test_fit_builds_bm25_index(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        assert retriever._bm25 is not None

    def test_fit_empty_list_leaves_index_none(self) -> None:
        retriever = SparseRetriever()
        retriever.fit([])
        assert retriever._documents == []
        assert retriever._bm25 is None

    def test_refit_replaces_previous_index(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        retriever.fit([_CORPUS[0]])
        assert retriever._documents == [_CORPUS[0]]

    def test_fit_raises_actionable_import_error_when_rank_bm25_missing(self) -> None:
        retriever = SparseRetriever()
        with (
            patch.dict(sys.modules, {"rank_bm25": None}),
            pytest.raises(ImportError, match="pip install rank-bm25"),
        ):
            retriever.fit(_CORPUS)


# ─── retrieve() — empty / unfitted state ──────────────────────────────────────

class TestRetrieveEmptyState:
    def test_retrieve_before_fit_returns_empty_result(self) -> None:
        retriever = SparseRetriever()
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert isinstance(result, RetrievalResult)
        assert result.hits == []

    def test_retrieve_after_fitting_empty_corpus_returns_empty_result(self) -> None:
        retriever = SparseRetriever()
        retriever.fit([])
        result = retriever.retrieve("query", claim_id="C1", top_k=5)
        assert result.hits == []

    def test_empty_result_still_carries_query_claim_id_top_k(self) -> None:
        retriever = SparseRetriever()
        result = retriever.retrieve("what is x?", claim_id="C42", top_k=3)
        assert result.query == "what is x?"
        assert result.claim_id == "C42"
        assert result.top_k == 3


# ─── retrieve() — ranking behavior ────────────────────────────────────────────

class TestRetrieveRanking:
    def test_retrieves_documents_with_lexical_overlap_first(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        result = retriever.retrieve("What happened to inflation?", claim_id="C1", top_k=3)
        ranked_ids = [hit.document.doc_id for hit in result.hits]
        # d1 and d3 both mention inflation/interest rates; d2 (football) does not.
        assert ranked_ids[0] in {"d1", "d3"}
        assert "d2" in ranked_ids  # still present (small corpus), just ranked lower
        assert ranked_ids.index("d2") > 0

    def test_top_k_limits_number_of_hits(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        result = retriever.retrieve("inflation interest rates", claim_id="C1", top_k=1)
        assert len(result.hits) == 1

    def test_hits_are_ranked_starting_at_one(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        result = retriever.retrieve("inflation", claim_id="C1", top_k=3)
        assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))

    def test_returned_document_is_the_original_object(self) -> None:
        """SparseRetriever returns the exact Document instances passed to
        fit() rather than reconstructing them (unlike DenseRetriever, which
        must rebuild Documents from a vector store payload)."""
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        result = retriever.retrieve("inflation", claim_id="C1", top_k=3)
        for hit in result.hits:
            original = next(d for d in _CORPUS if d.doc_id == hit.document.doc_id)
            assert hit.document is original

    def test_single_document_corpus_is_always_returned(self) -> None:
        retriever = SparseRetriever()
        retriever.fit([_CORPUS[0]])
        result = retriever.retrieve("completely unrelated query text", claim_id="C1", top_k=5)
        assert len(result.hits) == 1
        assert result.hits[0].document.doc_id == "d1"


# ─── retrieve() — score normalization ─────────────────────────────────────────

class TestScoreNormalization:
    def test_top_hit_score_is_one(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        result = retriever.retrieve("inflation interest rates", claim_id="C1", top_k=3)
        assert result.hits[0].score == pytest.approx(1.0)

    def test_all_scores_within_unit_range(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        result = retriever.retrieve("inflation interest rates", claim_id="C1", top_k=3)
        assert all(0.0 <= hit.score <= 1.0 for hit in result.hits)

    def test_no_lexical_overlap_produces_zero_scores(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        result = retriever.retrieve(
            "zzzznonexistenttermzzzz qwertyuiopasdfgh", claim_id="C1", top_k=3
        )
        assert all(hit.score == 0.0 for hit in result.hits)

    @pytest.mark.parametrize(
        "raw_score, max_score, expected",
        [
            (10.0, 10.0, 1.0),
            (5.0, 10.0, 0.5),
            (0.0, 10.0, 0.0),
            (-3.0, 10.0, 0.0),   # clamped
            (5.0, 0.0, 0.0),      # non-positive max_score guard
            (5.0, -1.0, 0.0),     # negative max_score guard
        ],
    )
    def test_normalize_score_static_method(
        self, raw_score: float, max_score: float, expected: float
    ) -> None:
        assert SparseRetriever._normalize_score(raw_score, max_score) == pytest.approx(expected)


# ─── retrieve() — error handling ──────────────────────────────────────────────

class TestRetrieveErrors:
    def test_raises_retrieval_error_when_bm25_scoring_fails(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        retriever._bm25 = MagicMock()
        retriever._bm25.get_scores.side_effect = RuntimeError("boom")
        with pytest.raises(RetrievalError, match="BM25 scoring failed"):
            retriever.retrieve("query", claim_id="C1", top_k=3)

    def test_scoring_failure_chains_original_exception(self) -> None:
        retriever = SparseRetriever()
        retriever.fit(_CORPUS)
        retriever._bm25 = MagicMock()
        original = RuntimeError("boom")
        retriever._bm25.get_scores.side_effect = original
        with pytest.raises(RetrievalError) as exc_info:
            retriever.retrieve("query", claim_id="C1", top_k=3)
        assert exc_info.value.__cause__ is original


# ─── _tokenize() ────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_lowercases(self) -> None:
        assert _tokenize("INFLATION Rose") == ["inflation", "rose"]

    def test_strips_punctuation(self) -> None:
        assert _tokenize("inflation, rose! (a lot)") == ["inflation", "rose", "a", "lot"]

    def test_empty_string_returns_empty_list(self) -> None:
        assert _tokenize("") == []

    def test_no_word_characters_returns_empty_list(self) -> None:
        assert _tokenize("!!! ??? ---") == []
