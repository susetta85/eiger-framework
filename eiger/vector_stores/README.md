# eiger.vector_stores

**Status: Not yet implemented — planned for Sprint 3.**

This module will provide vector database backends for storing and searching
document embeddings. All concrete implementations extend `BaseVectorStore`
from `eiger.core.interfaces`.

---

## Supported Stores

| Store              | Status    | Primary Use Case                              |
|--------------------|-----------|-----------------------------------------------|
| `QdrantVectorStore`  | Planned (primary) | Production experiments, Docker-based deployment |
| `ChromaVectorStore`  | Planned   | Lightweight local development                 |
| `FAISSVectorStore`   | Planned   | Offline / no-Docker environments              |

Qdrant v1.9.4 is the primary target. All integration tests run against Qdrant.

---

## Interface Contract

```python
from eiger.core.interfaces import BaseVectorStore

class BaseVectorStore(ABC):

    @abstractmethod
    def create_collection(self, name: str, dim: int) -> None:
        """Create a new collection. Raises if it already exists."""

    @abstractmethod
    def reset_collection(self, name: str, dim: int) -> None:
        """Drop and recreate a collection."""

    @abstractmethod
    def upsert(
        self,
        collection: str,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        """Insert or update documents with their pre-computed vectors."""

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the top_k most similar documents as raw result dicts."""
```

---

## Infrastructure: Qdrant

Qdrant is started via Docker Compose:

```bash
make up        # docker compose up -d
make down      # docker compose down
```

Connection is configured via environment variables (see `.env.example`):

| Variable            | Default       | Description              |
|---------------------|---------------|--------------------------|
| `EIGER_QDRANT_HOST` | `localhost`   | Qdrant server hostname   |
| `EIGER_QDRANT_PORT` | `6333`        | Qdrant HTTP/gRPC port    |

Health check endpoint: `GET http://localhost:6333/healthz`

---

## Qdrant-Specific Notes

**Collection lifecycle.** Each experiment uses a named collection specified in
`RetrieverConfig.collection_name`. Use `reset_collection` when re-running an
experiment to avoid stale data from prior runs. The `make run` command will
warn if a collection already exists and `--force-reset` is not passed.

**Vector dimensions must match the embedder.** The `dim` argument passed to
`create_collection` must equal `BaseEmbedder.embedding_dim` for the configured
embedder. A mismatch raises `CollectionDimensionError` at upsert time.

**Payload storage.** Documents are stored as Qdrant payload alongside their
vectors. The full `Document` model is serialized to JSON and stored under the
`payload` key so that retrieval results can be reconstructed without a separate
database lookup.

---

## Sprint 3 Milestone

- [ ] `BaseVectorStore` ABC (already defined in `eiger.core.interfaces`)
- [ ] `QdrantVectorStore` — primary implementation, Qdrant v1.9.4
- [ ] `ChromaVectorStore` — local development alternative
- [ ] `FAISSVectorStore` — offline fallback
- [ ] Unit tests: interface conformance, payload round-trip
- [ ] Integration tests: collection lifecycle against live Qdrant
