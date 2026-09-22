"""
HybridRetriever: Reciprocal Rank Fusion (RRF) of dense and sparse retrieval.

This module provides the third concrete BaseRetriever implementation
(alongside DenseRetriever and SparseRetriever), closing the "HybridRetriever
— RRF fusion of dense and sparse rankings" item on the retrieval roadmap
(see eiger/retrieval/README.md's "RRF Fusion" section).

Pipeline position
------------------
    query (str)
        │
        ├──▶ DenseRetriever.retrieve()  ──▶ dense RetrievalResult (ranked 1..N)
        │
        └──▶ SparseRetriever.retrieve() ──▶ sparse RetrievalResult (ranked 1..M)
        │
        ▼  fuse by RRF: score(d) = sum(1 / (k + rank_i(d))) over every
        │  retriever i in which d appears (0 if it does not appear at all)
        ▼  sort by fused score, descending; normalize; truncate to top_k
    RetrievalResult(query, claim_id, hits, top_k)

Design decisions
----------------
- **Composition, not inheritance.** HybridRetriever does not extend
  DenseRetriever or SparseRetriever; it holds one instance of each and calls
  their public `retrieve()` (and, for the sparse side, `fit()`) methods. This
  matches the plan already recorded in eiger/retrieval/README.md: "now that
  both DenseRetriever and SparseRetriever exist, HybridRetriever can compose
  them directly ... rather than needing either underlying retriever to
  change." Neither DenseRetriever nor SparseRetriever needed any
  modification to support this.
- **RRF, not score averaging.** Dense (cosine similarity) and sparse (BM25)
  scores live on different, incomparable scales even after each retriever's
  own [0, 1] normalization — a dense score of 0.9 and a sparse score of 0.9
  do not represent the same strength of match. RRF sidesteps this entirely
  by fusing on *rank position* rather than raw score, which is exactly why
  it is the standard technique for this kind of heterogeneous fusion (see
  Cormack, Clarke & Buettcher, 2009).
- **k=60 default**, per the original RRF paper and already documented in
  eiger/retrieval/README.md's formula. Configurable via the constructor for
  experimentation.
- **A document need not appear in both rankings.** A document retrieved only
  by the dense side (or only by the sparse side) still receives a fused
  score from that one ranking alone — RRF naturally handles partial overlap
  without special-casing it.
- **Result score is re-normalized into [0, 1]**, mirroring
  SparseRetriever._normalize_score()'s per-query max-normalization design:
  raw RRF scores have no fixed range (they depend on k and how many
  retrievers a document appears in), but RetrievedDocument.score is
  Pydantic-constrained to [0, 1].
- **Candidate pool = top_k from each sub-retriever.** Each underlying
  retriever is asked for `top_k` hits (not more), matching the "run both,
  fuse rankings" phrasing in the README. This means the final fused result
  can contain at most `2 * top_k` distinct candidates before truncation to
  `top_k`.

What this module does NOT do:
  - It does not modify DenseRetriever or SparseRetriever's own ranking or
    normalization logic.
  - It does not query the underlying retrievers in parallel; retrieve()
    calls block sequentially (dense, then sparse). Parallelizing is a
    possible future optimization, not a correctness requirement.
  - It does not deduplicate or merge two Document objects that share a
    doc_id but originate from different reconstruction paths (Qdrant
    payload vs. in-memory reference) — when a document appears in both
    rankings, the dense-side object is kept (see `retrieve()`'s docstring).
"""

from __future__ import annotations

from eiger.core.interfaces import BaseRetriever
from eiger.core.models import Document, RetrievalResult, RetrievedDocument
from eiger.retrieval.retriever import DenseRetriever
from eiger.retrieval.sparse_retriever import SparseRetriever
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# Default RRF constant, per the original RRF paper (Cormack, Clarke &
# Buettcher, 2009) and eiger/retrieval/README.md's documented formula.
_DEFAULT_RRF_K = 60


class HybridRetriever(BaseRetriever):
    """
    Retriever that fuses DenseRetriever and SparseRetriever rankings via
    Reciprocal Rank Fusion (RRF).

    Args:
        dense:  A DenseRetriever instance, already constructed with the same
                embedder/vector_store/collection used elsewhere in the
                pipeline.
        sparse: A SparseRetriever instance. Must be `fit()` (directly, or via
                this class's own `fit()` passthrough — see below) before
                `retrieve()` returns any sparse-derived hits.
        rrf_k:  The RRF constant `k` in `1 / (k + rank)`. Higher values
                flatten the fusion (later ranks matter almost as much as
                early ones); lower values sharpen it. Default 60, per the
                original RRF paper.

    Example::

        hybrid = HybridRetriever(
            dense=DenseRetriever(embedder=embedder, vector_store=store, collection="corpus"),
            sparse=SparseRetriever(),
        )
        hybrid.fit(corpus.all_documents)  # populates the sparse side only —
                                           # the dense side must already have
                                           # been ingested into vector_store
        result = hybrid.retrieve(query="...", claim_id="TEST_001", top_k=5)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = _DEFAULT_RRF_K,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    # ─── Corpus indexing (sparse side only) ────────────────────────────────────

    def fit(self, documents: list[Document]) -> None:
        """
        Fit the sparse (BM25) side of this hybrid retriever.

        This is a thin passthrough to `SparseRetriever.fit()`. The dense
        side has no equivalent step — it is ready as soon as the caller has
        ingested the corpus into the shared vector store (see
        `DenseRetriever`'s own module docstring). Provided so that callers
        (e.g. `ExperimentRunner`) can call `retriever.fit(...)` uniformly
        for both "sparse" and "hybrid" retriever types.

        Args:
            documents: Full corpus to index on the sparse side (typically
                       `CorpusBuilderResult.all_documents`).
        """
        self.sparse.fit(documents)

    # ─── BaseRetriever interface ────────────────────────────────────────────────

    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult:
        """
        Retrieve the top_k documents for a query, fused across dense and
        sparse rankings via RRF.

        Args:
            query:    Natural language query (typically Claim.context_query).
            claim_id: ID of the claim this query belongs to, propagated into
                      the RetrievalResult for downstream traceability.
            top_k:    Maximum number of documents to retrieve, from each
                      underlying retriever and in the final fused result.

        Returns:
            RetrievalResult with hits ranked 1..N by descending fused RRF
            score (re-normalized into [0, 1]).

        Raises:
            RetrievalError: If either underlying retriever's search fails
                             (propagated as-is from DenseRetriever/
                             SparseRetriever, which already wrap their own
                             backend errors).
        """
        dense_result = self.dense.retrieve(query, claim_id, top_k)
        sparse_result = self.sparse.retrieve(query, claim_id, top_k)

        fused_scores: dict[str, float] = {}
        # First-seen document object wins when a doc_id appears in both
        # rankings — dense_result is processed first, so the dense-side
        # reconstruction is preferred (see module docstring's last "does
        # NOT do" note). Content should be equivalent either way for a
        # document that has not been re-poisoned between the two indexes.
        documents_by_id: dict[str, Document] = {}

        for retrieval_result in (dense_result, sparse_result):
            for hit in retrieval_result.hits:
                doc_id = hit.document.doc_id
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (
                    self.rrf_k + hit.rank
                )
                documents_by_id.setdefault(doc_id, hit.document)

        ranked_ids = sorted(
            fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True
        )[:top_k]
        max_score = fused_scores[ranked_ids[0]] if ranked_ids else 0.0

        hits = [
            RetrievedDocument(
                document=documents_by_id[doc_id],
                score=self._normalize_score(fused_scores[doc_id], max_score),
                rank=rank,
            )
            for rank, doc_id in enumerate(ranked_ids, start=1)
        ]

        log.info(
            "hybrid_retriever.retrieved",
            claim_id=claim_id,
            n_hits=len(hits),
            top_k=top_k,
            n_dense_hits=len(dense_result.hits),
            n_sparse_hits=len(sparse_result.hits),
        )
        return RetrievalResult(query=query, claim_id=claim_id, hits=hits, top_k=top_k)

    # ─── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_score(raw_score: float, max_score: float) -> float:
        """
        Rescale a raw fused RRF score into [0, 1] via per-query
        max-normalization — mirroring SparseRetriever._normalize_score(),
        since RRF scores (like BM25 scores) have no fixed range.

        Args:
            raw_score: This document's fused RRF score.
            max_score: The maximum fused RRF score among this query's own
                       candidate results (already restricted to top_k).

        Returns:
            float in [0.0, 1.0]. 0.0 if max_score is not positive (which,
            since RRF terms are always positive when a document appears in
            at least one ranking, only happens when there are no candidates
            at all).
        """
        if max_score <= 0.0:
            return 0.0
        normalized = raw_score / max_score
        return max(0.0, min(1.0, normalized))
