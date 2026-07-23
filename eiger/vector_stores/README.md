# eiger.vector_stores

**Status: `QdrantVectorStore` implemented (Sprint 2, Step 2). `ChromaVectorStore` / `FAISSVectorStore` remain future work.**

This module provides vector database backends for storing and searching
document embeddings. All concrete implementations extend `BaseVectorStore`
from `eiger.core.interfaces`.

---

## Supported Stores

| Store              | Status    | Primary Use Case                              |
|--------------------|-----------|-----------------------------------------------|
| `QdrantVectorStore`  | ✅ Implemented (primary) | Production experiments, Docker-based deployment |
| `ChromaVectorStore`  | 🔲 Planned   | Lightweight local development                 |
| `FAISSVectorStore`   | 🔲 Planned   | Offline / no-Docker environments              |

Qdrant v1.9.4 is the primary target (pinned in `docker-compose.yml`).

---

## `QdrantVectorStore`

```python
from eiger.vector_stores import QdrantVectorStore

store = QdrantVectorStore(host="localhost", port=6333, timeout=10.0)
store.reset_collection("eiger_corpus", dim=384)
store.upsert("eiger_corpus", documents, vectors)
hits = store.search("eiger_corpus", query_vector, top_k=5)
```

- **Lazy client**: the `QdrantClient` connection opens on first use, not at
  construction time — the class is importable and constructible without a
  running Qdrant instance or `qdrant-client` installed (only raises
  `ImportError` when an operation actually needs the client).
- **Integer point IDs**: documents are stored with the list index (0, 1, 2, …)
  as the Qdrant point ID, so re-upserting the same corpus overwrites the same
  points deterministically (idempotent ingestion).
- **COSINE distance**: chosen because sentence-transformer embeddings are
  typically L2-normalized.
- **`reset_collection`** uses Qdrant's `recreate_collection` (atomic drop +
  create), used by `IngestionPipeline` at the start of every experiment run
  to guarantee a clean corpus.
- **Payload storage**: each point's payload stores `doc_id`, `claim_id`,
  `text`, and `doc_type` — enough for `DenseRetriever` to reconstruct a
  `Document` from a search hit without a separate database lookup.

---

## Interface Contract

```python
from eiger.core.interfaces import BaseVectorStore

class BaseVectorStore(ABC):

    @abstractmethod
    def create_collection(self, name: str, dim: int) -> None: ...

    @abstractmethod
    def reset_collection(self, name: str, dim: int) -> None: ...

    @abstractmethod
    def upsert(
        self, collection: str, documents: list[Document], vectors: list[list[float]],
    ) -> None: ...

    @abstractmethod
    def search(
        self, collection: str, query_vector: list[float], top_k: int,
    ) -> list[dict[str, Any]]: ...
```

`search()` returns raw dicts (`{"doc_id", "score", "payload"}`) rather than
`Document` objects, so callers (`DenseRetriever`) control how a `Document` is
rebuilt without coupling `BaseVectorStore` implementations to a fixed schema.

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
`RetrieverConfig.collection_name`. `IngestionPipeline.ingest()` calls
`reset_collection()` by default at the start of every run (pass `reset=False`
to append to an existing collection instead).

**Vector dimensions must match the embedder.** `IngestionPipeline` reads
`BaseEmbedder.embedding_dim` and passes it as `dim` to `reset_collection()`,
so the collection is always sized correctly for whichever embedder is
injected — no manual dimension bookkeeping needed.

**Payload storage.** Documents are stored as a Qdrant payload alongside their
vectors (`doc_id`, `claim_id`, `text`, `doc_type` — not the full serialized
`Document` model), so retrieval results can be reconstructed without a
separate database lookup.

---

## Test coverage

`tests/unit/test_qdrant_store.py` (19 tests) covers the full interface with
100% line coverage, using a mocked `qdrant-client` SDK — no real Qdrant
server required.

## Remaining work

- [ ] `ChromaVectorStore` — local development alternative
- [ ] `FAISSVectorStore` — offline fallback
- [ ] Integration tests: collection lifecycle against a live Qdrant instance
