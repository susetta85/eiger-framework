"""
QdrantVectorStore: vector database backend backed by Qdrant.

Qdrant (https://qdrant.tech) is an open-source, production-ready vector
database with a Docker image pinned in the EIGER infrastructure
(docker-compose.yml: qdrant/qdrant:v1.9.4).

This module provides a concrete implementation of BaseVectorStore that wraps
the official ``qdrant-client`` Python SDK.  The rest of the EIGER pipeline
only interacts with the BaseVectorStore interface, so swapping Qdrant for
another backend (FAISS, Chroma, Weaviate) requires only a new file here.

Design decisions
----------------
- **Lazy client**: the QdrantClient connection is not opened at construction
  time.  The first call to any operation triggers ``_get_client()``, which
  opens the connection once and reuses it for all subsequent calls.  This
  keeps the class importable in tests and config contexts without a running
  Qdrant instance.
- **Integer point IDs**: Qdrant accepts either UUIDs or unsigned integers as
  point IDs.  We use the list index (0, 1, 2, …) as the ID.  This means
  ``upsert()`` is idempotent: re-upserting the same corpus overwrites the
  same point IDs deterministically.
- **COSINE distance**: chosen because sentence-transformer embeddings are
  typically L2-normalised, making cosine similarity equivalent to dot-product
  but more numerically stable across models that are not normalised.
- **recreate_collection**: ``reset_collection`` uses Qdrant's
  ``recreate_collection`` call (drop + create in one atomic operation) to
  ensure a clean slate at the start of each experiment run.

What this module does NOT do:
- It does not compute embeddings; vectors are passed in pre-computed.
- It does not manage experiment state or logging of results.
- It does not support sparse or hybrid (dense+sparse) retrieval; that is
  left for a future HybridRetriever implementation.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from eiger.core.interfaces import BaseVectorStore
from eiger.core.models import Document
from eiger.utils.logging import get_logger

if TYPE_CHECKING:
    # Only used for type annotations; the real import happens lazily in
    # _get_client() so that the module can be imported without qdrant-client
    # installed (e.g. in environments that only run unit tests).
    from qdrant_client import QdrantClient as _QdrantClient

log = get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """
    Vector store implementation backed by a Qdrant server.

    Connects to a running Qdrant instance (local Docker or remote) via the
    official ``qdrant-client`` SDK.  All network I/O is synchronous.

    Args:
        host:    Hostname or IP of the Qdrant server.  Defaults to
                 ``"localhost"`` (suitable for the Docker Compose setup).
        port:    gRPC/HTTP port of the Qdrant server.  Default: 6333.
        timeout: Timeout in seconds for each SDK call.  Default: 10.0.

    Example::

        store = QdrantVectorStore(host="localhost", port=6333)
        store.reset_collection("corpus", dim=384)
        store.upsert("corpus", documents, vectors)
        hits = store.search("corpus", query_vector, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        # Client is not initialised yet — see _get_client().
        self._client: _QdrantClient | None = None

    # ─── Lazy client loader ───────────────────────────────────────────────────

    def _get_client(self) -> _QdrantClient:
        """
        Return the QdrantClient, initialising it on first call.

        Idempotent: subsequent calls return the already-open client without
        reconnecting.

        Returns:
            An open QdrantClient connected to self.host:self.port.

        Raises:
            ImportError: If ``qdrant-client`` is not installed.
                         Install with: pip install qdrant-client
            ConnectionError: If the Qdrant server is unreachable (raised by
                             the SDK on the first actual network call, not here).
        """
        if self._client is not None:
            # Already connected — reuse the existing client.
            return self._client

        try:
            from qdrant_client import QdrantClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install qdrant-client"
            ) from exc

        log.info("qdrant.connecting", host=self.host, port=self.port)
        self._client = QdrantClient(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
        )
        log.info("qdrant.client_ready", host=self.host, port=self.port)
        return self._client

    # ─── BaseVectorStore interface ────────────────────────────────────────────

    def create_collection(self, name: str, dim: int) -> None:
        """
        Create a new Qdrant collection with COSINE distance and the given
        vector dimensionality.

        Args:
            name: Collection name.  Must be unique within the Qdrant instance.
            dim:  Dimensionality of the vectors to be stored (e.g. 384).

        Raises:
            IngestionError: Wraps any SDK exception (e.g. collection already
                            exists, dimension mismatch).
        """
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        client = self._get_client()
        log.info("qdrant.create_collection", name=name, dim=dim)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def reset_collection(self, name: str, dim: int) -> None:
        """
        Drop and recreate a collection, discarding all existing documents.

        Uses Qdrant's ``recreate_collection`` which is atomic — it drops the
        old collection and creates the new one in a single server-side call.
        This is the recommended way to reset state between experiment runs.

        Args:
            name: Collection name to reset.
            dim:  Dimensionality for the recreated collection.

        Raises:
            IngestionError: If the reset fails on the server side.
        """
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        client = self._get_client()
        log.info("qdrant.reset_collection", name=name, dim=dim)
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(
        self,
        collection: str,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        """
        Insert or update documents with their pre-computed embedding vectors.

        Each document is stored as a Qdrant ``PointStruct`` with:
          - ``id``:      the list index (integer), making re-upserts idempotent.
          - ``vector``:  the pre-computed embedding from the embedder.
          - ``payload``: a dict with doc_id, claim_id, text, and doc_type so
                         that search results can be reconstructed into Document
                         objects without a separate database lookup.

        Args:
            collection: Target collection name (must already exist).
            documents:  Document objects to store.  Must be same length as
                        ``vectors`` and in the same order.
            vectors:    Pre-computed embedding vectors, one per document.

        Raises:
            IngestionError: If the upsert fails (e.g. dimension mismatch,
                            collection not found).
        """
        from qdrant_client.models import PointStruct  # type: ignore[import]

        client = self._get_client()

        # Build one PointStruct per document.  The payload stores all fields
        # needed to reconstruct a Document at retrieval time.
        points = [
            PointStruct(
                id=idx,
                vector=vectors[idx],
                payload={
                    "doc_id":   doc.doc_id,
                    "claim_id": doc.claim_id,
                    "text":     doc.text,
                    "doc_type": doc.doc_type,
                },
            )
            for idx, doc in enumerate(documents)
        ]

        log.info("qdrant.upserting", collection=collection, n_points=len(points))
        client.upsert(collection_name=collection, points=points)
        log.info("qdrant.upsert_complete", collection=collection)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Return the top_k most similar documents for a query vector.

        Results are returned as raw dicts rather than Document objects so
        that the calling BaseRetriever can apply its own re-ranking or
        score-normalisation logic without being coupled to Qdrant's internal
        ScoredPoint representation.

        Args:
            collection:   Name of the collection to search.
            query_vector: Embedding of the query, same dimension as stored vectors.
            top_k:        Maximum number of results to return.

        Returns:
            List of dicts ordered by descending similarity score, each with:
              - ``"doc_id"``  — original Document.doc_id string
              - ``"score"``   — cosine similarity in [−1, 1] (higher = more similar)
              - ``"payload"`` — full payload dict (doc_id, claim_id, text, doc_type)

        Raises:
            RetrievalError: If the search fails (collection not found,
                            dimension mismatch, network error).
        """
        client = self._get_client()

        log.debug("qdrant.searching", collection=collection, top_k=top_k)
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            # with_payload=True is the default in qdrant-client >= 1.0,
            # but we set it explicitly for clarity.
            with_payload=True,
        )

        # Normalise ScoredPoint objects into plain dicts for backend-agnostic
        # consumption by DenseRetriever.
        return [
            {
                "doc_id":  r.payload["doc_id"],
                "score":   r.score,
                "payload": r.payload,
            }
            for r in results
        ]
