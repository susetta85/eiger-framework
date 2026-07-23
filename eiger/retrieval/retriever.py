"""
DenseRetriever: orchestrates query embedding, vector search, and
RetrievalResult assembly for dense (embedding-based) retrieval.

This module provides the concrete BaseRetriever implementation used by
the default EIGER pipeline. It is deliberately a thin orchestration layer:
it owns no state about the corpus itself and delegates all actual work to
the BaseEmbedder and BaseVectorStore it is constructed with.

Pipeline position
------------------
    query (str)
        │
        ▼  embedder.encode([query])
    query_vector (list[float])
        │
        ▼  vector_store.search(collection, query_vector, top_k)
    raw hits (list[dict])   # {"doc_id", "score", "payload"}
        │
        ▼  _to_retrieved_document() per hit
    RetrievalResult(query, claim_id, hits, top_k)

Design decisions
----------------
- **Same embedder for query and corpus**: DenseRetriever does not create
  its own embedder; the caller (typically the ExperimentRunner) is
  responsible for passing the *same* embedder instance/model used during
  ingestion. Mixing embedders would make similarity scores meaningless
  (see BaseEmbedder docstring in eiger.core.interfaces).
- **Score normalization**: BaseVectorStore.search() returns raw backend
  scores (for Qdrant with COSINE distance, cosine similarity in
  [-1, 1]). RetrievedDocument.score is constrained to [0, 1] by the
  Pydantic model, so DenseRetriever rescales with (score + 1) / 2 and
  clamps defensively in case a backend ever returns an out-of-range
  value due to floating point error.
- **Document reconstruction from payload**: the vector store returns
  raw dicts (not Document objects) so that DenseRetriever controls how
  a Document is rebuilt from the stored payload. This keeps
  BaseVectorStore implementations backend-agnostic (see qdrant_store.py).
- **Fail loud**: any failure while encoding the query or querying the
  vector store is wrapped in RetrievalError so callers only need to
  catch one exception type at the retrieval boundary.

What this module does NOT do:
  - It does not compute embeddings itself; that is BaseEmbedder's job.
  - It does not manage vector store connections or collections; that is
    BaseVectorStore's job (see QdrantVectorStore).
  - It does not implement hybrid or sparse retrieval; a future
    HybridRetriever would compose multiple BaseRetriever/BaseVectorStore
    instances rather than extending this class.
"""

from __future__ import annotations

from typing import Any

from eiger.core.exceptions import RetrievalError
from eiger.core.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore
from eiger.core.models import Document, RetrievalResult, RetrievedDocument
from eiger.utils.logging import get_logger

log = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense (embedding-based) retriever: encode → search → assemble.

    Args:
        embedder:     Any BaseEmbedder implementation. Must be the same
                      model used to embed the corpus at ingestion time.
        vector_store: Any BaseVectorStore implementation, already pointed
                      at a populated collection.
        collection:   Name of the vector store collection to search.

    Example::

        retriever = DenseRetriever(
            embedder=SentenceTransformerEmbedder(),
            vector_store=QdrantVectorStore(),
            collection="eiger_corpus",
        )
        result = retriever.retrieve(
            query="What did the WHO report about 2023 inflation?",
            claim_id="TEST_001",
            top_k=5,
        )
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        collection: str,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection = collection

    # ─── BaseRetriever interface ──────────────────────────────────────────────

    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult:
        """
        Retrieve the top_k most similar documents for a query.

        Steps:
          1. Encode the query into a single embedding vector.
          2. Search the vector store's collection for the closest vectors.
          3. Rebuild each hit into a RetrievedDocument (Document + score + rank).
          4. Wrap everything into a RetrievalResult.

        Args:
            query:    Natural language query (typically Claim.context_query).
            claim_id: ID of the claim this query belongs to, propagated into
                      the RetrievalResult for downstream traceability.
            top_k:    Maximum number of documents to retrieve.

        Returns:
            RetrievalResult with hits ranked 1..N by descending similarity.

        Raises:
            RetrievalError: If query encoding or the vector store search fails,
                             or if the embedder returns no vector for the query.
        """
        log.debug("retriever.encoding_query", claim_id=claim_id, top_k=top_k)
        query_vector = self._encode_query(query, claim_id)

        log.debug(
            "retriever.searching",
            claim_id=claim_id,
            collection=self.collection,
            top_k=top_k,
        )
        raw_hits = self._search(query_vector, claim_id, top_k)

        hits = [
            self._to_retrieved_document(raw_hit, rank)
            for rank, raw_hit in enumerate(raw_hits, start=1)
        ]

        log.info(
            "retriever.retrieved",
            claim_id=claim_id,
            n_hits=len(hits),
            top_k=top_k,
        )
        return RetrievalResult(query=query, claim_id=claim_id, hits=hits, top_k=top_k)

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _encode_query(self, query: str, claim_id: str) -> list[float]:
        """
        Encode a single query string into an embedding vector.

        Wraps embedder.encode() (which operates on lists) for the
        single-query case used at retrieval time.

        Args:
            query:    Query string to encode.
            claim_id: Included in the error message for traceability.

        Returns:
            The embedding vector for the query.

        Raises:
            RetrievalError: If encoding fails, or if the embedder returns
                             an empty result for a non-empty query.
        """
        try:
            vectors = self.embedder.encode([query])
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise RetrievalError(
                f"Failed to encode query for claim '{claim_id}': {exc}"
            ) from exc

        if not vectors:
            raise RetrievalError(
                f"Embedder returned no vector for query (claim_id={claim_id!r})."
            )
        return vectors[0]

    def _search(
        self,
        query_vector: list[float],
        claim_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Query the vector store, wrapping any backend error in RetrievalError.

        Args:
            query_vector: Embedding of the query.
            claim_id:     Included in the error message for traceability.
            top_k:        Maximum number of results requested.

        Returns:
            Raw hit dicts as returned by BaseVectorStore.search()
            ({"doc_id", "score", "payload"}), ordered by descending similarity.

        Raises:
            RetrievalError: If the vector store search fails.
        """
        try:
            return self.vector_store.search(
                collection=self.collection,
                query_vector=query_vector,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise RetrievalError(
                f"Vector store search failed for claim '{claim_id}' "
                f"in collection '{self.collection}': {exc}"
            ) from exc

    @staticmethod
    def _to_retrieved_document(raw_hit: dict[str, Any], rank: int) -> RetrievedDocument:
        """
        Rebuild a RetrievedDocument from a raw vector store hit.

        The Document is reconstructed from the hit's payload rather than
        fetched from a separate store, since QdrantVectorStore.upsert()
        stores doc_id, claim_id, text, and doc_type directly in the payload.

        Args:
            raw_hit: One hit dict from BaseVectorStore.search(), containing
                     at minimum {"doc_id", "score", "payload"}.
            rank:    1-based rank position of this hit within the result list.

        Returns:
            RetrievedDocument with the rebuilt Document, normalized score,
            and rank.
        """
        payload = raw_hit["payload"]
        document = Document(
            doc_id=payload["doc_id"],
            claim_id=payload["claim_id"],
            text=payload["text"],
            doc_type=payload["doc_type"],
        )
        return RetrievedDocument(
            document=document,
            score=DenseRetriever._normalize_score(raw_hit["score"]),
            rank=rank,
        )

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        """
        Rescale a raw similarity score into [0, 1] for RetrievedDocument.

        Assumes the raw score is a cosine similarity in [-1, 1] (true for
        QdrantVectorStore, which uses Distance.COSINE). The linear mapping
        (score + 1) / 2 sends -1 -> 0.0, 0 -> 0.5, 1 -> 1.0. The result is
        clamped to [0, 1] defensively, since floating point error could
        otherwise push a boundary value (e.g. 1.0000000002) outside the
        range Pydantic enforces on RetrievedDocument.score.

        Args:
            raw_score: Raw similarity score from the vector store backend.

        Returns:
            float in [0.0, 1.0].
        """
        normalized = (raw_score + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))
