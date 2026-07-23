"""
IngestionPipeline: embeds a built corpus and upserts it into a vector store.

This module closes the loop between CorpusBuilder (Phase 2: claims → mixed
ground-truth/poisoned Documents) and the retrieval layer (Phase 3: a
populated, searchable vector store). It is the only component in EIGER
that touches both BaseEmbedder and BaseVectorStore during corpus
preparation — DenseRetriever uses the same two interfaces later, but only
for a single query vector at retrieval time.

Pipeline position
------------------
    CorpusBuilderResult (ground_truth_docs + poisoned_docs)
        │
        ▼  corpus.all_documents (ground-truth first, then poisoned)
    list[Document]
        │
        ▼  vector_store.reset_collection(collection, dim)   [if reset=True]
        ▼  embedder.encode([doc.text for doc in documents])
    list[list[float]]
        │
        ▼  vector_store.upsert(collection, documents, vectors)
    IngestionResult(collection, embedding_dim, n_ground_truth, n_poisoned)

Design decisions
----------------
- **reset_collection over create_collection**: reset_collection() (Qdrant's
  recreate_collection) both creates the collection on first use and wipes
  it clean on subsequent runs, so IngestionPipeline never needs to check
  whether the collection already exists. Callers that want to append to an
  existing collection instead of starting fresh can pass ``reset=False``.
- **Same embedder as retrieval time**: IngestionPipeline does not select or
  configure the embedder itself; the caller (typically ExperimentRunner)
  must pass the same BaseEmbedder instance/model that DenseRetriever will
  later use, or similarity scores at retrieval time will be meaningless.
- **Single encode() call for the whole corpus**: BaseEmbedder.encode()
  already batches internally (see SentenceTransformerEmbedder), so the
  pipeline embeds all document texts in one call rather than looping
  document-by-document.
- **Fail loud**: any failure while resetting the collection, encoding
  documents, or upserting is wrapped in IngestionError so callers only
  need to catch one exception type at the ingestion boundary.
- **Empty corpus is a no-op, not an error**: an empty CorpusBuilderResult
  (e.g. a dry run with zero claims) still resets the collection (if
  requested) but skips the upsert call entirely, since there is nothing
  to store.

What this module does NOT do:
  - It does not build the corpus itself; that is CorpusBuilder's job.
  - It does not compute embeddings; that is BaseEmbedder's job.
  - It does not perform retrieval; that is DenseRetriever's job.
"""

from __future__ import annotations

from dataclasses import dataclass

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseEmbedder, BaseVectorStore
from eiger.core.models import Document
from eiger.ingestion.corpus_builder import CorpusBuilderResult
from eiger.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class IngestionResult:
    """
    Summary of a completed ingestion run.

    Returned by IngestionPipeline.ingest() so callers (typically
    ExperimentRunner) can record ingestion provenance without needing to
    re-inspect the original CorpusBuilderResult.

    Attributes:
        collection:     Name of the vector store collection that was populated.
        embedding_dim:  Dimensionality of the vectors written to the collection.
        n_ground_truth: Number of ground-truth documents ingested.
        n_poisoned:     Number of poisoned documents ingested.
    """

    collection: str
    embedding_dim: int
    n_ground_truth: int
    n_poisoned: int

    @property
    def n_documents(self) -> int:
        """
        Total number of documents ingested (ground-truth + poisoned).

        Returns:
            int: n_ground_truth + n_poisoned.
        """
        return self.n_ground_truth + self.n_poisoned


class IngestionPipeline:
    """
    Embeds a CorpusBuilderResult and upserts it into a vector store.

    Args:
        embedder:     Any BaseEmbedder implementation. Must be the same
                      model DenseRetriever will use at retrieval time.
        vector_store: Any BaseVectorStore implementation.
        collection:   Name of the vector store collection to populate.

    Example::

        pipeline = IngestionPipeline(
            embedder=SentenceTransformerEmbedder(),
            vector_store=QdrantVectorStore(),
            collection="eiger_corpus",
        )
        result = pipeline.ingest(corpus_builder_result)
        # result.n_documents, result.embedding_dim, ...
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

    # ─── Public API ────────────────────────────────────────────────────────────

    def ingest(self, corpus: CorpusBuilderResult, reset: bool = True) -> IngestionResult:
        """
        Embed every document in the corpus and upsert it into the vector store.

        Args:
            corpus: Output of CorpusBuilder.build(), containing ground-truth
                     and poisoned documents.
            reset:   If True (default), drop and recreate the target collection
                     before upserting, guaranteeing a clean corpus. Set to False
                     to append to an already-populated collection.

        Returns:
            IngestionResult summarizing the collection name, embedding
            dimension, and document counts.

        Raises:
            IngestionError: If resetting the collection, encoding documents,
                             or upserting fails.
        """
        documents = corpus.all_documents
        n_ground_truth = len(corpus.ground_truth_docs)
        n_poisoned = len(corpus.poisoned_docs)

        log.info(
            "ingestion.start",
            collection=self.collection,
            n_ground_truth=n_ground_truth,
            n_poisoned=n_poisoned,
            reset=reset,
        )

        # embedding_dim may trigger model loading (see BaseEmbedder), but we
        # need it up front regardless — both to size the collection and to
        # report it in the IngestionResult.
        embedding_dim = self.embedder.embedding_dim

        if reset:
            self._reset_collection(embedding_dim)

        if not documents:
            log.warning("ingestion.empty_corpus", collection=self.collection)
        else:
            vectors = self._encode_documents(documents)
            self._upsert(documents, vectors)

        log.info(
            "ingestion.complete",
            collection=self.collection,
            n_documents=len(documents),
        )
        return IngestionResult(
            collection=self.collection,
            embedding_dim=embedding_dim,
            n_ground_truth=n_ground_truth,
            n_poisoned=n_poisoned,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _reset_collection(self, embedding_dim: int) -> None:
        """
        Drop and recreate the target collection with the correct dimension.

        Args:
            embedding_dim: Vector dimensionality for the recreated collection.

        Raises:
            IngestionError: If the vector store fails to reset the collection.
        """
        try:
            self.vector_store.reset_collection(self.collection, dim=embedding_dim)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise IngestionError(
                f"Failed to reset collection '{self.collection}' "
                f"(dim={embedding_dim}): {exc}"
            ) from exc

    def _encode_documents(self, documents: list[Document]) -> list[list[float]]:
        """
        Embed the text of every document in the corpus.

        Args:
            documents: Documents to embed (ground-truth + poisoned, combined).

        Returns:
            One embedding vector per document, in the same order as documents.

        Raises:
            IngestionError: If the embedder fails to encode the document texts.
        """
        texts = [doc.text for doc in documents]
        try:
            return self.embedder.encode(texts)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise IngestionError(
                f"Failed to encode {len(texts)} document(s) "
                f"for collection '{self.collection}': {exc}"
            ) from exc

    def _upsert(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """
        Upsert documents and their pre-computed vectors into the vector store.

        Args:
            documents: Documents to store.
            vectors:   Pre-computed embedding vectors, same length and order
                       as documents.

        Raises:
            IngestionError: If the vector store fails to upsert the batch.
        """
        try:
            self.vector_store.upsert(self.collection, documents, vectors)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise IngestionError(
                f"Failed to upsert {len(documents)} document(s) "
                f"into collection '{self.collection}': {exc}"
            ) from exc
