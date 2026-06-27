"""
Abstract base classes (interfaces) for all EIGER extension points.

Every pluggable component — attacks, datasets, embedders, vector stores,
retrievers, LLM backends, and metrics — implements one of these ABCs.
This ensures new implementations can be dropped in without modifying
any orchestration or experiment code.

Design rationale:
  - Using ABC + @abstractmethod enforces the contract at class definition
    time (raised immediately on instantiation if a method is missing),
    rather than only at call time.
  - Class-level attributes (name, description) are declared but not
    enforced by ABC; concrete implementations must set them so that the
    registry and logging systems can identify components by name.
  - The interfaces import from eiger.core.models but nothing else, keeping
    the dependency graph acyclic and the module importable without any
    infrastructure being initialized.

What this file does NOT do:
  - It does not implement any concrete functionality.
  - It does not register implementations; that is done by decorators in
    eiger.attacks.registry, eiger.metrics.registry, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from eiger.core.models import (
    Claim,
    Document,
    PoisonedDocument,
    RetrievalResult,
    GenerationResult,
    MetricScore,
    EvaluationRecord,
    ExperimentConfig,
    ExperimentResult,
)


# ─── Attack interface ─────────────────────────────────────────────────────────

class BaseAttack(ABC):
    """
    Interface for adversarial poisoning strategies.

    Implementations must be stateless with respect to the corpus —
    all randomness is driven by an explicit seed passed at call time.
    This statelessness is what makes the attack pipeline reproducible:
    given the same document and the same seed, apply() must always
    return the same PoisonedDocument.

    What this class does NOT do:
      - It does not manage which documents to poison or at what rate;
        that logic lives in CorpusBuilder.
      - It does not store any results; it returns new PoisonedDocument
        objects and leaves persistence to the caller.

    Attributes:
        name:        Unique string identifier used in config YAML and registry.
        description: Human-readable description written to experiment logs.
    """

    name: str        # Unique identifier, used in config YAML and registry
    description: str # Human-readable description for experiment logs

    @abstractmethod
    def apply(self, document: Document, seed: int, **kwargs: Any) -> PoisonedDocument:
        """
        Apply the attack to a single document.

        Args:
            document: The original ground-truth document to poison.
            seed:     Deterministic seed for all random operations within
                      this call. The same seed must always produce the
                      same output for the same document.
            **kwargs: Attack-specific parameters that override the defaults
                      set in the attack's __init__.

        Returns:
            A PoisonedDocument with full provenance metadata (original_text,
            attack_name, attack_params populated).

        Raises:
            GenerationError: If the attack relies on an LLM and generation fails.
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """
        Return a JSON-serializable dict describing the attack's parameters.

        Used by the experiment runner to record attack configuration in the
        provenance log, even for attacks whose parameters are not exposed
        through AttackConfig.params.

        Returns:
            dict with at minimum {"name": ..., "description": ...}.
        """


# ─── Dataset interface ────────────────────────────────────────────────────────

class BaseDataset(ABC):
    """
    Interface for fact-checking dataset loaders.

    Implementations are responsible for downloading, parsing, and
    returning Claim objects from a specific dataset (e.g. AVeriTeC,
    PolitiFact). They do NOT perform any poisoning or augmentation.

    Attributes:
        name:        Dataset identifier string (must match the name field
                     in DatasetConfig YAML).
        description: Human-readable description for logs and documentation.
    """

    name: str
    description: str

    @abstractmethod
    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """
        Load and return claims from the dataset.

        Args:
            split:      Dataset split to load ("train", "test", "dev").
                        Not all datasets support all splits.
            max_claims: If set, return at most this many claims. The
                        selection should be deterministic (e.g. first N
                        after sorting by claim_id) for reproducibility.

        Returns:
            List of Claim objects. May be empty if the split has no data.

        Raises:
            IngestionError: If the dataset files are missing or corrupt.
        """

    @abstractmethod
    def download(self, target_dir: str) -> None:
        """
        Download the raw dataset to target_dir if not already present.

        Implementations should be idempotent — if the data already exists
        at target_dir, download() should be a no-op rather than re-downloading.

        Args:
            target_dir: Directory where raw dataset files should be stored.

        Raises:
            IngestionError: If the download fails or the files cannot be written.
        """

    @property
    @abstractmethod
    def content_hash(self) -> str:
        """
        SHA-256 of the loaded dataset content, for provenance tracking.

        Must be consistent across invocations for the same dataset version.
        The hash should cover all fields that affect Claim content, so that
        any upstream dataset update is detectable.

        Returns:
            Hexadecimal hash string (length >= 16 characters).
        """


# ─── Embedder interface ───────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """
    Interface for text embedding models.

    Embedders are used at two points in the pipeline:
      1. During corpus ingestion, to embed documents for storage.
      2. During retrieval, to embed the query for similarity search.

    Using the same embedder instance at both points is critical — mixing
    embedders would make similarity scores meaningless.

    Attributes:
        model_name: Identifier string (e.g. HuggingFace model ID or API name).
                    Recorded in the experiment provenance log.
    """

    model_name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into dense vectors.

        Implementations should handle batching internally. The output
        list must be the same length as the input list, in the same order.

        Args:
            texts: List of strings to embed. May be empty.

        Returns:
            List of float lists, each of length self.embedding_dim.

        Raises:
            EigerError: If the underlying model call fails.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Dimensionality of the output embedding vectors.

        Used by the vector store to create collections with the correct
        vector size. Must be consistent across all calls to encode().

        Returns:
            Positive integer (e.g. 384 for all-MiniLM-L6-v2).
        """


# ─── Vector store interface ───────────────────────────────────────────────────

class BaseVectorStore(ABC):
    """
    Interface for vector databases.

    Abstracts over different vector store backends (Qdrant, FAISS, Chroma, etc.)
    so that the retrieval and ingestion pipeline is backend-agnostic.

    What this class does NOT do:
      - It does not compute embeddings; it receives pre-computed vectors.
      - It does not manage connection lifecycle; that is handled by
        concrete implementations (e.g. via a context manager or __init__).
    """

    @abstractmethod
    def create_collection(self, name: str, dim: int) -> None:
        """
        Create a new collection (index) with the given vector dimension.

        Args:
            name: Collection name. Must be unique within the vector store.
            dim:  Dimensionality of vectors to be stored in this collection.

        Raises:
            IngestionError: If the collection already exists or creation fails.
        """

    @abstractmethod
    def reset_collection(self, name: str, dim: int) -> None:
        """
        Drop and recreate a collection, discarding all existing data.

        Used at the start of each experiment run to ensure a clean corpus.
        Callers must be aware that this is a destructive, irreversible operation.

        Args:
            name: Collection name to reset.
            dim:  Dimensionality for the recreated collection.

        Raises:
            IngestionError: If the reset fails.
        """

    @abstractmethod
    def upsert(self, collection: str, documents: list[Document], vectors: list[list[float]]) -> None:
        """
        Insert or update documents with their pre-computed vectors.

        The documents and vectors lists must have the same length and be
        in the same order (vectors[i] is the embedding of documents[i]).

        Args:
            collection: Target collection name.
            documents:  Document objects to store (metadata is stored alongside vectors).
            vectors:    Pre-computed embedding vectors, one per document.

        Raises:
            IngestionError: If the upsert fails (e.g. dimension mismatch).
        """

    @abstractmethod
    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """
        Return the top_k most similar documents as raw result dicts.

        Returns raw dicts rather than Document objects so that the
        BaseRetriever implementation can apply its own result-mapping logic
        (e.g. re-ranking, score normalization) without coupling to the
        vector store's internal representation.

        Args:
            collection:   Collection to search.
            query_vector: Embedding of the query, same dimension as stored vectors.
            top_k:        Maximum number of results to return.

        Returns:
            List of dicts, each containing at minimum {"doc_id", "score", "payload"}.
            Ordered by descending similarity score.

        Raises:
            RetrievalError: If the search fails.
        """


# ─── Retriever interface ──────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Interface for retrieval strategies (dense, sparse, hybrid).

    The retriever is the component that orchestrates embedding a query,
    querying the vector store, and assembling a RetrievalResult. It is
    deliberately separate from BaseVectorStore so that hybrid retrievers
    can compose multiple stores or query strategies internally.
    """

    @abstractmethod
    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult:
        """
        Retrieve top_k documents for a query.

        Args:
            query:    Natural language query string (typically the claim's
                      context_query field).
            claim_id: ID of the claim this query belongs to (propagated
                      into the RetrievalResult for traceability).
            top_k:    Number of documents to retrieve.

        Returns:
            RetrievalResult with hits ranked by similarity score.

        Raises:
            RetrievalError: If the retrieval fails.
        """


# ─── LLM interface ────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """
    Interface for LLM generation backends.

    Abstracts over local (Ollama) and API-based (OpenAI) backends.
    All implementations must be synchronous — async support is out of
    scope for the current EIGER version, which runs experiments sequentially.

    Attributes:
        model_name: Identifier string recorded in experiment provenance.
    """

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response given a prompt string.

        Args:
            prompt:   The fully-assembled prompt to send to the model.
            **kwargs: Backend-specific generation parameters (temperature,
                      max_tokens, etc.) that override the configured defaults.

        Returns:
            The model's response as a plain string.

        Raises:
            GenerationError: If the backend returns an error or times out.
        """

    @abstractmethod
    def build_rag_prompt(self, query: str, context_docs: list[str]) -> str:
        """
        Construct a RAG prompt from a query and retrieved document texts.

        Separating prompt construction from generation allows different
        backends to use backend-specific prompt formats (chat templates,
        instruction prefixes, etc.) without changing the calling code.

        Args:
            query:        The user's question or claim to be answered.
            context_docs: List of document text strings to include as context.
                          Ordered by retrieval rank (most similar first).

        Returns:
            A fully-assembled prompt string ready to pass to generate().
        """


# ─── Metric interface ─────────────────────────────────────────────────────────

class BaseMetric(ABC):
    """
    Interface for evaluation metrics.

    Metrics operate on EvaluationRecords and return a scalar score.
    They must be deterministic — same input always produces same output.

    Design notes:
      - compute() works on a single record; compute_batch() has a default
        implementation that maps over compute(). Override compute_batch()
        in implementations that can batch more efficiently (e.g. metrics
        that call an LLM judge and benefit from batching API requests).
      - aggregate() defaults to the mean, which is appropriate for most
        EIGER metrics. Override for metrics where the mean is not meaningful
        (e.g. a binary metric that should be reported as a proportion).

    Attributes:
        name:        Metric identifier (must be unique across all registered metrics).
        description: Human-readable description written to result files.
        range:       (min, max) tuple documenting the metric's output range.
                     Used only for documentation and plot scaling; not enforced.
    """

    name: str
    description: str
    range: tuple[float, float] = (0.0, 1.0)  # (min, max) for documentation

    @abstractmethod
    def compute(self, record: EvaluationRecord) -> MetricScore:
        """
        Compute the metric for a single evaluation record.

        Args:
            record: The evaluation record containing generation output,
                    retrieval result, and any previously computed metrics.

        Returns:
            MetricScore with the computed value and optional diagnostic metadata.

        Raises:
            EigerError: If computation fails (e.g. required fields are missing).
        """

    def compute_batch(self, records: list[EvaluationRecord]) -> list[MetricScore]:
        """
        Compute the metric for a list of records.

        Default implementation maps compute() over the list. Override this
        method in implementations that can batch more efficiently (e.g. by
        sending multiple records to an LLM judge in a single API call).

        Args:
            records: List of evaluation records to score.

        Returns:
            List of MetricScore objects in the same order as records.
        """
        return [self.compute(r) for r in records]

    def aggregate(self, scores: list[MetricScore]) -> float:
        """
        Aggregate a list of per-record scores into a single experiment-level value.

        Default implementation returns the arithmetic mean. Returns 0.0
        for an empty list to avoid ZeroDivisionError in experiment runners
        that don't pre-filter for non-empty score lists.

        Args:
            scores: List of MetricScore objects (typically all from one experiment).

        Returns:
            float: Single aggregate value (e.g. mean FFR across all claims).
        """
        if not scores:
            return 0.0
        return sum(s.value for s in scores) / len(scores)
