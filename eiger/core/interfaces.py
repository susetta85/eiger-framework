"""
Abstract base classes (interfaces) for all EIGER extension points.

Every pluggable component — attacks, datasets, embedders, vector stores,
retrievers, LLM backends, and metrics — implements one of these ABCs.
This ensures new implementations can be dropped in without modifying
any orchestration or experiment code.
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
    """

    name: str        # Unique identifier, used in config YAML and registry
    description: str # Human-readable description for experiment logs

    @abstractmethod
    def apply(self, document: Document, seed: int, **kwargs: Any) -> PoisonedDocument:
        """
        Apply the attack to a single document.

        Args:
            document: The original ground-truth document to poison.
            seed:     Deterministic seed for all random operations.
            **kwargs: Attack-specific parameters (override defaults).

        Returns:
            A PoisonedDocument with full provenance metadata.
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable dict describing the attack's parameters."""


# ─── Dataset interface ────────────────────────────────────────────────────────

class BaseDataset(ABC):
    """Interface for fact-checking dataset loaders."""

    name: str
    description: str

    @abstractmethod
    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """Load and return claims from the dataset."""

    @abstractmethod
    def download(self, target_dir: str) -> None:
        """Download the raw dataset to target_dir if not already present."""

    @property
    @abstractmethod
    def content_hash(self) -> str:
        """SHA-256 of the loaded dataset content (for provenance)."""


# ─── Embedder interface ───────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Interface for text embedding models."""

    model_name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into dense vectors."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""


# ─── Vector store interface ───────────────────────────────────────────────────

class BaseVectorStore(ABC):
    """Interface for vector databases."""

    @abstractmethod
    def create_collection(self, name: str, dim: int) -> None:
        """Create a new collection. Raises if it already exists."""

    @abstractmethod
    def reset_collection(self, name: str, dim: int) -> None:
        """Drop and recreate a collection (use carefully)."""

    @abstractmethod
    def upsert(self, collection: str, documents: list[Document], vectors: list[list[float]]) -> None:
        """Insert or update documents with their pre-computed vectors."""

    @abstractmethod
    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Return the top_k most similar documents as raw result dicts."""


# ─── Retriever interface ──────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """Interface for retrieval strategies (dense, sparse, hybrid)."""

    @abstractmethod
    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult:
        """Retrieve top_k documents for a query."""


# ─── LLM interface ────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Interface for LLM generation backends."""

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response given a prompt string."""

    @abstractmethod
    def build_rag_prompt(self, query: str, context_docs: list[str]) -> str:
        """Construct a RAG prompt from query and retrieved document texts."""


# ─── Metric interface ─────────────────────────────────────────────────────────

class BaseMetric(ABC):
    """
    Interface for evaluation metrics.

    Metrics operate on EvaluationRecords and return a scalar score.
    They must be deterministic — same input always produces same output.
    """

    name: str
    description: str
    range: tuple[float, float] = (0.0, 1.0)  # (min, max) for documentation

    @abstractmethod
    def compute(self, record: EvaluationRecord) -> MetricScore:
        """Compute the metric for a single evaluation record."""

    def compute_batch(self, records: list[EvaluationRecord]) -> list[MetricScore]:
        """Compute the metric for a list of records. Default: map over compute()."""
        return [self.compute(r) for r in records]

    def aggregate(self, scores: list[MetricScore]) -> float:
        """Aggregate a list of per-record scores into a single experiment-level value."""
        if not scores:
            return 0.0
        return sum(s.value for s in scores) / len(scores)
