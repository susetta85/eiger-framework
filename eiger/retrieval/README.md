# eiger.retrieval

**Status: Not yet implemented — planned for Sprint 3.**

This module will provide all document retrieval strategies used in the EIGER
evaluation pipeline. Retrievers consume a query string and return a ranked list
of documents from the vector corpus.

---

## Planned Architecture

| Retriever Type  | Method                  | Library                          | Strengths                          | Limitations                            |
|-----------------|-------------------------|----------------------------------|------------------------------------|----------------------------------------|
| `DenseRetriever`  | Cosine similarity        | Qdrant + sentence-transformers   | Semantic recall, robust to paraphrase | Slower indexing, requires GPU for large models |
| `SparseRetriever` | BM25                    | rank-bm25                        | Fast, interpretable, no embedder needed | Lexical matching only, no semantic understanding |
| `HybridRetriever` | RRF fusion (dense + sparse) | rank-bm25 + Qdrant            | Best of both, generally highest recall | More moving parts, two indices required |

All concrete retrievers will extend `BaseRetriever` from `eiger.core.interfaces`.

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

        Args:
            query:    Natural language query string.
            claim_id: Identifier of the parent claim (for provenance).
            top_k:    Maximum number of documents to return.

        Returns:
            RetrievalResult containing ranked RetrievedDocument objects,
            each with a similarity score in [0, 1] and a rank index.
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
    type: str            # "dense" | "sparse" | "hybrid"
    embedder: str        # HuggingFace model ID (dense and hybrid only)
    vector_store: str    # "qdrant" | "chroma" | "faiss"
    top_k: int           # Number of documents to retrieve
    collection_name: str # Target collection in the vector store
```

Example (from `experiments/baseline_v1.yaml`):

```yaml
retriever:
  type: dense
  embedder: sentence-transformers/all-MiniLM-L6-v2
  vector_store: qdrant
  top_k: 5
  collection_name: eibench_baseline_v1
```

---

## RRF Fusion (HybridRetriever)

Reciprocal Rank Fusion combines dense and sparse rankings without requiring
score normalization. Given rank `r` from each retriever, the fused score is:

```
RRF(d) = sum(1 / (k + r_i(d)))   for each retriever i
```

The default constant `k = 60` follows the original RRF paper.

---

## Sprint 3 Milestone

- [ ] `BaseRetriever` ABC (already defined in `eiger.core.interfaces`)
- [ ] `DenseRetriever` — cosine search via `QdrantVectorStore`
- [ ] `SparseRetriever` — BM25 index via `rank-bm25`
- [ ] `HybridRetriever` — RRF fusion of dense and sparse rankings
- [ ] Unit tests: determinism, score bounds, `RetrievalResult` shape
- [ ] Integration tests: round-trip against a live Qdrant instance
