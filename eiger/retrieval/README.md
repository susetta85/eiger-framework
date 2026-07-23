# eiger.retrieval

**Status: `SentenceTransformerEmbedder` and `DenseRetriever` implemented (Sprint 2, Steps 1 & 3). `SparseRetriever` / `HybridRetriever` remain future work.**

This module provides the embedder and retrieval strategies used in the EIGER
evaluation pipeline. `DenseRetriever` consumes a query string and returns a
ranked list of documents from the vector corpus; `SentenceTransformerEmbedder`
turns text into the dense vectors that both retrieval and ingestion rely on.

---

## Architecture

| Component | Class | Method | Library | Status |
|-----------|-------|--------|---------|--------|
| Embedder | `SentenceTransformerEmbedder` | Dense embedding | sentence-transformers | ✅ Implemented |
| Dense retriever | `DenseRetriever` | Cosine similarity via Qdrant | `BaseEmbedder` + `BaseVectorStore` | ✅ Implemented |
| Sparse retriever | `SparseRetriever` | BM25 | rank-bm25 | 🔲 Planned |
| Hybrid retriever | `HybridRetriever` | RRF fusion (dense + sparse) | rank-bm25 + Qdrant | 🔲 Planned |

`DenseRetriever` extends `BaseRetriever` from `eiger.core.interfaces`. It is a
thin orchestration layer — it holds no corpus state itself and delegates
entirely to the injected `BaseEmbedder` and `BaseVectorStore`.

---

## `SentenceTransformerEmbedder`

```python
from eiger.retrieval import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder()  # default: all-MiniLM-L6-v2, 384 dims
vectors = embedder.encode(["Hello world", "Another sentence"])
print(embedder.embedding_dim)  # 384
```

- **Lazy loading**: the underlying `SentenceTransformer` model is loaded on the
  first call to `encode()` (or to the `embedding_dim` property), not at
  construction time. Import and instantiation are fast and require no network
  access; only the first real `encode()` call downloads/loads the model.
- **Batching**: `encode()` calls the model once per batch (`batch_size`,
  default 64), not once per text.
- Raises `ImportError` with an actionable install hint if `sentence-transformers`
  is not installed.

---

## `DenseRetriever`

```python
from eiger.retrieval import DenseRetriever, SentenceTransformerEmbedder
from eiger.vector_stores import QdrantVectorStore

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
```

**Same embedder for ingestion and retrieval.** `DenseRetriever` does not
construct its own embedder — the caller must inject the *same* embedder
(same model) used to embed the corpus during ingestion (see
`eiger.ingestion.IngestionPipeline`). Mixing embedders makes similarity
scores meaningless.

**Score normalization.** `BaseVectorStore.search()` returns raw backend
scores — for `QdrantVectorStore` (COSINE distance), cosine similarity in
`[-1, 1]`. `RetrievedDocument.score` is constrained to `[0, 1]` by Pydantic,
so `DenseRetriever` rescales with `(score + 1) / 2`, clamped defensively
against floating-point edge cases.

**Document reconstruction.** `BaseVectorStore.search()` returns raw dicts
(`{"doc_id", "score", "payload"}`), not `Document` objects — `DenseRetriever`
rebuilds a `Document` from each hit's payload (`doc_id`, `claim_id`, `text`,
`doc_type`), keeping vector store backends decoupled from the retrieval layer.

**Error handling.** Any failure encoding the query or querying the vector
store is wrapped in `RetrievalError` (including the case where the embedder
returns no vector for a non-empty query), so callers only need to catch one
exception type at the retrieval boundary.

---

## Interface Contract

```python
from eiger.core.interfaces import BaseRetriever
from eiger.core.models import RetrievalResult

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult:
        """
        Retrieve top_k documents for a query.

        Returns:
            RetrievalResult containing ranked RetrievedDocument objects,
            each with a similarity score in [0, 1] and a rank index.

        Raises:
            RetrievalError: If retrieval fails.
        """
```

`RetrievalResult` exposes two convenience properties:
- `contains_poisoned` — whether any retrieved document is of type `"poisoned"`
- `poison_ratio` — fraction of hits that are poisoned documents

---

## Configuration Reference

Retrievers are configured via `RetrieverConfig` from `eiger.core.models`:

```python
class RetrieverConfig(BaseModel):
    type: str = "dense"           # "dense" | "sparse" | "hybrid" (only "dense" implemented)
    embedder: str                 # HuggingFace model ID — provenance only, see note below
    vector_store: str = "qdrant"  # provenance only, see note below
    top_k: int = 5
    collection_name: str = "eiger_corpus"
```

**Note on `embedder` / `vector_store` fields:** `ExperimentRunner` does not
build an embedder/vector-store instance from these strings — there is no
factory for them yet. They exist for provenance (serialized into every result
file) so a result can always be traced back to the model/backend that
produced it. The caller constructs the actual `SentenceTransformerEmbedder`
and `QdrantVectorStore` instances and injects them into `ExperimentRunner`,
which should match what `RetrieverConfig` declares.

---

## RRF Fusion (planned — `HybridRetriever`)

Reciprocal Rank Fusion combines dense and sparse rankings without requiring
score normalization. Given rank `r` from each retriever, the fused score is:

```
RRF(d) = sum(1 / (k + r_i(d)))   for each retriever i
```

The default constant `k = 60` follows the original RRF paper. Not yet implemented.

---

## Test coverage

`tests/unit/test_embedder.py` (13 tests) and `tests/unit/test_retriever.py`
(32 tests) cover both classes with 100% line coverage, using mocked
`sentence-transformers` / `BaseVectorStore` — no real model download or
Qdrant server required.

## Remaining work

- [ ] `SparseRetriever` — BM25 index via `rank-bm25`
- [ ] `HybridRetriever` — RRF fusion of dense and sparse rankings
- [ ] Integration tests: round-trip against a live Qdrant instance + real embedder
