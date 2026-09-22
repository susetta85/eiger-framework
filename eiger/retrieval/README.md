# eiger.retrieval

**Status: `SentenceTransformerEmbedder`, `DenseRetriever` (Sprint 2, Steps 1 & 3), `SparseRetriever` (Sprint 4), and `HybridRetriever` (Sprint 5) all implemented.**

This module provides the embedder and retrieval strategies used in the EIGER
evaluation pipeline. `DenseRetriever`/`SparseRetriever`/`HybridRetriever` all
consume a query string and return a ranked list of documents — one from the
vector corpus, one from an in-memory BM25 index, and one that fuses both via
RRF; `SentenceTransformerEmbedder` turns text into the dense vectors that
retrieval (dense and hybrid) and ingestion rely on.

---

## Architecture

| Component | Class | Method | Library | Status |
|-----------|-------|--------|---------|--------|
| Embedder | `SentenceTransformerEmbedder` | Dense embedding | sentence-transformers | ✅ Implemented |
| Dense retriever | `DenseRetriever` | Cosine similarity via Qdrant | `BaseEmbedder` + `BaseVectorStore` | ✅ Implemented |
| Sparse retriever | `SparseRetriever` | BM25 (Okapi) | rank-bm25 | ✅ Implemented (Sprint 4) |
| Hybrid retriever | `HybridRetriever` | RRF fusion (dense + sparse) | rank-bm25 + Qdrant | ✅ Implemented (Sprint 5) |

`DenseRetriever` and `SparseRetriever` both extend `BaseRetriever` from
`eiger.core.interfaces`; `HybridRetriever` also extends `BaseRetriever` but
composes one `DenseRetriever` and one `SparseRetriever` internally rather
than implementing its own ranking logic (see its own section below).
`DenseRetriever` is a thin orchestration layer — it
holds no corpus state itself and delegates entirely to the injected
`BaseEmbedder` and `BaseVectorStore`. `SparseRetriever` is different: it
*does* hold corpus state (see its own section below), because BM25 needs the
full tokenized corpus up front and `BaseVectorStore` has no method to
enumerate every stored document.

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

## `SparseRetriever`

```python
from eiger.retrieval import SparseRetriever

retriever = SparseRetriever()
retriever.fit(corpus.all_documents)  # after CorpusBuilder.build() — see below
result = retriever.retrieve(
    query="What did the WHO report about 2023 inflation?",
    claim_id="TEST_001",
    top_k=5,
)
```

**`fit()` is not part of `BaseRetriever`.** Unlike `DenseRetriever`, which is
immediately usable once the vector store is populated,`SparseRetriever` must
be explicitly given the corpus via `fit(documents)` before `retrieve()` can
return anything. This is a consequence of `BaseVectorStore` having no method
to enumerate every document in a collection — BM25 needs the entire tokenized
corpus up front to compute term frequencies, which a similarity-search API
cannot provide. `ExperimentRunner` calls `fit()` automatically when
`config.retriever.type == "sparse"` (see its own docstring); direct callers
must do the same.

**No embedder or vector store involved.** `SparseRetriever()` takes no
constructor arguments — it is entirely self-contained, in-memory, and
independent of `BaseEmbedder`/`BaseVectorStore`. A `config.retriever.type ==
"sparse"` experiment run does not require Qdrant to be reachable at all (see
`ExperimentRunner`'s module docstring — the ingestion step is skipped
entirely in that case).

**Score normalization.** Okapi BM25 scores are non-negative in the typical
case but have no fixed upper bound (unlike cosine similarity's `[-1, 1]`),
and can go negative when a term's IDF is negative (appears in more than half
the corpus). `SparseRetriever` therefore normalizes per query: every hit's
score is divided by the maximum score among that query's own top-`k`
candidates, so the top hit is always `1.0`. If every candidate scores `<= 0`
(no positive lexical overlap at all), every hit's score is `0.0` instead of
dividing by a non-positive number.

**Tokenization.** Text is lowercased and split on `\w+` (word characters) —
no stemming, no stopword removal. This is intentionally simple; BM25's
term-frequency statistics are the point, not a production-grade tokenizer.

**Empty corpus is not an error.** `fit([])` — or never calling `fit()` at
all — leaves `retrieve()` always returning a valid `RetrievalResult` with an
empty hit list, rather than raising. (`rank_bm25.BM25Okapi` itself raises
`ZeroDivisionError` on an empty corpus internally; this is special-cased so
callers never see that.)

**Error handling.** Any failure during BM25 scoring is wrapped in
`RetrievalError`, matching `DenseRetriever`'s convention of a single
exception type at the retrieval boundary. A missing `rank-bm25` install
raises `ImportError` with an actionable `pip install rank-bm25` hint from
`fit()` (mirroring `SentenceTransformerEmbedder`/`QdrantVectorStore`'s lazy
optional-dependency pattern) — though as of Sprint 4, `rank-bm25` is a core
dependency (see `pyproject.toml`), not optional.

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
    type: str = "dense"           # "dense" | "sparse" | "hybrid"
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

## RRF Fusion — `HybridRetriever` (Sprint 5)

Reciprocal Rank Fusion combines dense and sparse rankings without requiring
score normalization. Given rank `r` from each retriever, the fused score is:

```
RRF(d) = sum(1 / (k + r_i(d)))   for each retriever i
```

The default constant `k = 60` follows the original RRF paper (configurable
via `HybridRetriever(dense=..., sparse=..., rrf_k=...)`). `HybridRetriever`
composes a `DenseRetriever` and a `SparseRetriever` directly — neither
underlying retriever needed any change — calling both retrievers' full
`retrieve(query, claim_id, top_k)` and fusing their hit lists by document
`doc_id`. A document need not appear in both rankings: it simply accumulates
one RRF term per ranking it does appear in. The fused score has no fixed
range (like BM25's own raw score), so the final hits are re-normalized into
`[0, 1]` via the same per-query max-normalization technique as
`SparseRetriever._normalize_score()`.

```python
from eiger.retrieval import DenseRetriever, SparseRetriever, HybridRetriever

hybrid = HybridRetriever(
    dense=DenseRetriever(embedder=embedder, vector_store=store, collection="eiger_corpus"),
    sparse=SparseRetriever(),
)
hybrid.fit(corpus.all_documents)  # populates the sparse (BM25) side only —
                                  # the dense side must already have been
                                  # ingested into the shared vector store
result = hybrid.retrieve(query="What happened to inflation?", claim_id="C1", top_k=5)
```

`ExperimentRunner` wires this up automatically for `retriever.type: hybrid`
— see `eiger/experiments/runner.py`'s module docstring — ingesting the
corpus into the vector store (dense side) *and* calling `fit()` (sparse
side) before any claim is evaluated.

---

## Test coverage

`tests/unit/test_embedder.py` (13 tests), `tests/unit/test_retriever.py`
(32 tests), `tests/unit/test_sparse_retriever.py` (30 tests), and
`tests/unit/test_hybrid_retriever.py` cover all four classes with 100% line
coverage, using mocked `sentence-transformers` / `BaseVectorStore` (dense),
a small hand-built in-memory corpus (sparse), or mocked `DenseRetriever`/
`SparseRetriever` instances (hybrid) — no real model download or Qdrant
server required for any of them.

## Remaining work

- [x] `SparseRetriever` — BM25 index via `rank-bm25` (Sprint 4)
- [x] `HybridRetriever` — RRF fusion of dense and sparse rankings (Sprint 5)
- [ ] Integration tests: round-trip against a live Qdrant instance + real embedder
