"""
Unit tests for QdrantVectorStore (eiger.vector_stores.qdrant_store).

Tests verify:
  - Default attributes stored correctly at construction
  - _client starts as None (lazy-load contract)
  - _get_client() raises ImportError when qdrant-client is missing
  - _get_client() is idempotent (calling twice reuses the same client)
  - _get_client() creates a QdrantClient with correct host/port/timeout
  - create_collection() calls client.create_collection with correct params
  - reset_collection() calls client.recreate_collection with correct params
  - upsert() builds correct PointStruct list and calls client.upsert
  - upsert() stores doc_id, claim_id, text, doc_type in the payload
  - search() calls client.search with correct params and maps results to dicts
  - search() returns list of dicts with doc_id, score, payload keys

What these tests do NOT cover:
  - Real Qdrant server interaction (covered in integration tests).
  - Network timeouts or authentication.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from eiger.core.models import Document
from eiger.vector_stores.qdrant_store import QdrantVectorStore


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_doc(doc_id: str = "doc-001", claim_id: str = "C1") -> Document:
    """Return a minimal Document for upsert/search tests."""
    return Document(
        doc_id=doc_id,
        claim_id=claim_id,
        text="The inflation rate rose to 3.5% in 2023.",
        doc_type="ground_truth",
    )


def _make_store_with_mock_client() -> tuple[QdrantVectorStore, MagicMock]:
    """
    Return a QdrantVectorStore whose _client is already set to a MagicMock.

    This bypasses _get_client() so tests can focus on the operation under
    test rather than the connection setup.
    """
    store = QdrantVectorStore()
    mock_client = MagicMock()
    store._client = mock_client
    return store, mock_client


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestQdrantVectorStoreInit:
    """Tests for __init__ defaults and attribute storage."""

    def test_default_host(self) -> None:
        assert QdrantVectorStore().host == "localhost"

    def test_default_port(self) -> None:
        assert QdrantVectorStore().port == 6333

    def test_default_timeout(self) -> None:
        assert QdrantVectorStore().timeout == 10.0

    def test_custom_params_stored(self) -> None:
        store = QdrantVectorStore(host="qdrant-server", port=6334, timeout=30.0)
        assert store.host == "qdrant-server"
        assert store.port == 6334
        assert store.timeout == 30.0

    def test_client_starts_as_none(self) -> None:
        """_client must be None at construction (lazy-load contract)."""
        assert QdrantVectorStore()._client is None


# ─── _get_client ──────────────────────────────────────────────────────────────

class TestGetClient:
    """Tests for the lazy Qdrant client initialiser."""

    def test_raises_when_qdrant_client_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_get_client() must raise ImportError if qdrant-client is not installed."""
        monkeypatch.setitem(sys.modules, "qdrant_client", None)
        store = QdrantVectorStore()
        with pytest.raises(ImportError, match="qdrant-client"):
            store._get_client()

    def test_idempotent_returns_same_client(self) -> None:
        """Calling _get_client() twice must return the same client object."""
        store, mock_client = _make_store_with_mock_client()
        assert store._get_client() is mock_client
        assert store._get_client() is mock_client

    def test_creates_client_with_correct_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_get_client() must pass host, port, timeout to QdrantClient."""
        mock_qdrant_module = MagicMock()
        mock_client_instance = MagicMock()
        mock_qdrant_module.QdrantClient.return_value = mock_client_instance
        monkeypatch.setitem(sys.modules, "qdrant_client", mock_qdrant_module)

        store = QdrantVectorStore(host="myhost", port=9999, timeout=5.0)
        with patch("eiger.vector_stores.qdrant_store.log"):
            client = store._get_client()

        mock_qdrant_module.QdrantClient.assert_called_once_with(
            host="myhost", port=9999, timeout=5.0
        )
        assert client is mock_client_instance


# ─── create_collection ────────────────────────────────────────────────────────

class TestCreateCollection:
    """Tests for create_collection()."""

    def test_calls_create_collection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_collection() must delegate to client.create_collection."""
        store, mock_client = _make_store_with_mock_client()

        # Mock the qdrant_client.models module so VectorParams/Distance resolve
        mock_models = MagicMock()
        monkeypatch.setitem(sys.modules, "qdrant_client.models", mock_models)

        with patch("eiger.vector_stores.qdrant_store.log"):
            store.create_collection("test_col", dim=384)

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_col"


# ─── reset_collection ─────────────────────────────────────────────────────────

class TestResetCollection:
    """Tests for reset_collection()."""

    def test_calls_recreate_collection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reset_collection() must call client.recreate_collection."""
        store, mock_client = _make_store_with_mock_client()

        mock_models = MagicMock()
        monkeypatch.setitem(sys.modules, "qdrant_client.models", mock_models)

        with patch("eiger.vector_stores.qdrant_store.log"):
            store.reset_collection("test_col", dim=128)

        mock_client.recreate_collection.assert_called_once()
        call_kwargs = mock_client.recreate_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_col"


# ─── upsert ───────────────────────────────────────────────────────────────────

class TestUpsert:
    """Tests for upsert()."""

    def _run_upsert(
        self,
        monkeypatch: pytest.MonkeyPatch,
        docs: list[Document],
        vectors: list[list[float]],
    ) -> MagicMock:
        """Helper: run upsert() with mocked SDK and return the mock client."""
        store, mock_client = _make_store_with_mock_client()
        mock_models = MagicMock()
        # Make PointStruct a real callable that stores its args as attributes
        mock_models.PointStruct.side_effect = lambda **kw: kw
        monkeypatch.setitem(sys.modules, "qdrant_client.models", mock_models)
        with patch("eiger.vector_stores.qdrant_store.log"):
            store.upsert("corpus", docs, vectors)
        return mock_client

    def test_calls_client_upsert(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """upsert() must call client.upsert exactly once."""
        doc = _make_doc()
        mock_client = self._run_upsert(monkeypatch, [doc], [[0.1] * 384])
        mock_client.upsert.assert_called_once()

    def test_upsert_collection_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """upsert() must pass the correct collection name."""
        doc = _make_doc()
        mock_client = self._run_upsert(monkeypatch, [doc], [[0.1] * 384])
        assert mock_client.upsert.call_args.kwargs["collection_name"] == "corpus"

    def test_payload_contains_required_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each PointStruct payload must include doc_id, claim_id, text, doc_type."""
        doc = _make_doc(doc_id="d1", claim_id="C42")
        store, mock_client = _make_store_with_mock_client()
        captured_points: list[dict] = []

        mock_models = MagicMock()
        def capture_point(**kw):
            captured_points.append(kw)
            return kw
        mock_models.PointStruct.side_effect = capture_point
        monkeypatch.setitem(sys.modules, "qdrant_client.models", mock_models)

        with patch("eiger.vector_stores.qdrant_store.log"):
            store.upsert("corpus", [doc], [[0.0] * 384])

        assert len(captured_points) == 1
        payload = captured_points[0]["payload"]
        assert payload["doc_id"] == "d1"
        assert payload["claim_id"] == "C42"
        assert payload["text"] == doc.text
        assert payload["doc_type"] == "ground_truth"

    def test_point_id_is_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point IDs must be the list indices (0, 1, 2, …) for idempotent upserts."""
        docs = [_make_doc(f"doc-{i}") for i in range(3)]
        vectors = [[float(i)] * 384 for i in range(3)]
        store, mock_client = _make_store_with_mock_client()
        captured_points: list[dict] = []

        mock_models = MagicMock()
        mock_models.PointStruct.side_effect = lambda **kw: captured_points.append(kw) or kw
        monkeypatch.setitem(sys.modules, "qdrant_client.models", mock_models)

        with patch("eiger.vector_stores.qdrant_store.log"):
            store.upsert("corpus", docs, vectors)

        ids = [p["id"] for p in captured_points]
        assert ids == [0, 1, 2]


# ─── search ───────────────────────────────────────────────────────────────────

class TestSearch:
    """Tests for search()."""

    def _make_scored_point(self, doc_id: str, score: float) -> MagicMock:
        """Build a mock ScoredPoint as returned by qdrant-client."""
        sp = MagicMock()
        sp.score = score
        sp.payload = {
            "doc_id":   doc_id,
            "claim_id": "C1",
            "text":     "some text",
            "doc_type": "ground_truth",
        }
        return sp

    def test_returns_list_of_dicts(self) -> None:
        """search() must return a list of dicts."""
        store, mock_client = _make_store_with_mock_client()
        mock_client.search.return_value = [
            self._make_scored_point("doc-1", 0.9)
        ]
        with patch("eiger.vector_stores.qdrant_store.log"):
            results = store.search("corpus", [0.1] * 384, top_k=1)
        assert isinstance(results, list)
        assert isinstance(results[0], dict)

    def test_result_contains_required_keys(self) -> None:
        """Each result dict must contain doc_id, score, and payload."""
        store, mock_client = _make_store_with_mock_client()
        mock_client.search.return_value = [
            self._make_scored_point("doc-1", 0.85)
        ]
        with patch("eiger.vector_stores.qdrant_store.log"):
            results = store.search("corpus", [0.0] * 384, top_k=1)
        assert "doc_id" in results[0]
        assert "score" in results[0]
        assert "payload" in results[0]

    def test_score_and_doc_id_mapped_correctly(self) -> None:
        """doc_id and score must be taken from the ScoredPoint fields."""
        store, mock_client = _make_store_with_mock_client()
        mock_client.search.return_value = [
            self._make_scored_point("my-doc", 0.77)
        ]
        with patch("eiger.vector_stores.qdrant_store.log"):
            results = store.search("corpus", [0.0] * 384, top_k=1)
        assert results[0]["doc_id"] == "my-doc"
        assert results[0]["score"] == pytest.approx(0.77)

    def test_calls_client_search_with_correct_params(self) -> None:
        """search() must pass collection_name, query_vector, limit to client."""
        store, mock_client = _make_store_with_mock_client()
        mock_client.search.return_value = []
        query = [0.5] * 384
        with patch("eiger.vector_stores.qdrant_store.log"):
            store.search("my_collection", query, top_k=10)
        mock_client.search.assert_called_once()
        kwargs = mock_client.search.call_args.kwargs
        assert kwargs["collection_name"] == "my_collection"
        assert kwargs["query_vector"] == query
        assert kwargs["limit"] == 10

    def test_empty_results_returns_empty_list(self) -> None:
        """search() must return [] when Qdrant returns no hits."""
        store, mock_client = _make_store_with_mock_client()
        mock_client.search.return_value = []
        with patch("eiger.vector_stores.qdrant_store.log"):
            results = store.search("corpus", [0.0] * 384, top_k=5)
        assert results == []
