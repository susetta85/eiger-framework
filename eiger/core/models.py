"""
Domain models for the EIGER framework.

All models are Pydantic v2 dataclasses — they validate on construction,
serialize cleanly to JSON for experiment provenance, and carry no
infrastructure dependencies.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ─── Dataset layer ────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """
    A single fact-checked claim, as loaded from a source dataset.

    This is the fundamental unit of data in EIGER. Every downstream
    document (ground-truth or poisoned) originates from a Claim.
    """

    claim_id: str = Field(description="Unique identifier within the dataset")
    original_fact: str = Field(description="Verified factual statement (ground truth)")
    context_query: str = Field(description="Query used to retrieve relevant documents")
    source_dataset: str = Field(default="unknown", description="Origin dataset (e.g. averitec, politifact)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the original fact text. Used for dataset versioning."""
        return hashlib.sha256(self.original_fact.encode()).hexdigest()[:16]


class Document(BaseModel):
    """
    A document stored in the vector corpus.

    Can represent either a ground-truth document or an adversarial variant.
    """

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    claim_id: str = Field(description="Parent claim this document belongs to")
    text: str = Field(description="Document text content")
    doc_type: str = Field(
        default="ground_truth",
        description="One of: ground_truth, poisoned",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PoisonedDocument(Document):
    """
    A document that has been adversarially modified by an attack.

    Extends Document with full poisoning provenance.
    """

    doc_type: str = "poisoned"
    attack_name: str = Field(description="Name of the attack that produced this document")
    attack_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters used by the attack",
    )
    original_text: str = Field(description="Unmodified text before poisoning")
    annotation: PoisonAnnotation | None = Field(default=None)


class PoisonAnnotation(BaseModel):
    """
    Human or model annotation describing the epistemic risk of a poisoned document.

    Scores are in [1, 5] — higher means riskier.
    """

    plausibility: float = Field(ge=1.0, le=5.0, description="How believable the falsehood is")
    verification_difficulty: float = Field(ge=1.0, le=5.0, description="How hard it is to fact-check")
    editorial_risk: float = Field(ge=1.0, le=5.0, description="Risk of passing editorial review")


# ─── Retrieval layer ──────────────────────────────────────────────────────────

class RetrievedDocument(BaseModel):
    """A single document returned by a retriever, with its similarity score."""

    document: Document
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (higher = more similar)")
    rank: int = Field(ge=1, description="Position in the ranked result list")


class RetrievalResult(BaseModel):
    """Full output of a single retrieval operation."""

    query: str
    claim_id: str
    hits: list[RetrievedDocument] = Field(default_factory=list)
    top_k: int

    @property
    def contains_poisoned(self) -> bool:
        return any(h.document.doc_type == "poisoned" for h in self.hits)

    @property
    def poison_ratio(self) -> float:
        if not self.hits:
            return 0.0
        return sum(1 for h in self.hits if h.document.doc_type == "poisoned") / len(self.hits)


# ─── Generation layer ─────────────────────────────────────────────────────────

class GenerationResult(BaseModel):
    """Output from an LLM given a query and retrieved context."""

    claim_id: str
    query: str
    context_docs: list[str] = Field(description="Texts of retrieved documents fed to the LLM")
    answer: str = Field(description="LLM-generated answer")
    model_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Evaluation layer ─────────────────────────────────────────────────────────

class MetricScore(BaseModel):
    """Result of a single metric computation on a single example."""

    metric_name: str
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    """
    Full evaluation record for a single claim.

    Combines the generation result with all computed metric scores.
    """

    claim_id: str
    generation: GenerationResult
    retrieval: RetrievalResult
    metrics: dict[str, float] = Field(default_factory=dict)

    # Derived convenience properties
    @property
    def faithfulness_score(self) -> float:
        return self.metrics.get("ragas_faithfulness", 0.0)

    @property
    def factual_correctness_score(self) -> float:
        return self.metrics.get("ragas_answer_correctness", 0.0)


# ─── Experiment layer ─────────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """
    Full, validated specification for a single experiment run.

    Loaded from a YAML file and resolved against environment settings.
    Serialized to JSON alongside every result file for provenance.
    """

    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid4().hex[:8]}")
    seed: int = Field(default=42)
    dataset: DatasetConfig
    attacks: list[AttackConfig] = Field(default_factory=list)
    retriever: RetrieverConfig
    llm: LLMConfig
    metrics: list[str] = Field(default_factory=lambda: ["ffr", "source_integrity", "ers"])
    output_dir: str = Field(default="results/")
    description: str = Field(default="")

    @property
    def config_hash(self) -> str:
        """Fingerprint of the full configuration for reproducibility tracking."""
        serialized = self.model_dump_json(exclude={"experiment_id"})
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class DatasetConfig(BaseModel):
    name: str = Field(description="Dataset identifier: averitec | politifact | json_fixture")
    split: str = Field(default="test")
    max_claims: int | None = Field(default=None, description="Cap number of claims (None = all)")
    path: str | None = Field(default=None, description="Local path override")


class AttackConfig(BaseModel):
    name: str = Field(description="Attack identifier (must be registered in attack registry)")
    poison_rate: float = Field(ge=0.0, le=1.0, description="Fraction of corpus to poison")
    params: dict[str, Any] = Field(default_factory=dict, description="Attack-specific hyperparameters")


class RetrieverConfig(BaseModel):
    type: str = Field(default="dense", description="dense | sparse | hybrid")
    embedder: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    vector_store: str = Field(default="qdrant")
    top_k: int = Field(default=5, ge=1)
    collection_name: str = Field(default="eiger_corpus")


class LLMConfig(BaseModel):
    backend: str = Field(default="ollama", description="ollama | openai")
    model: str = Field(default="llama3.1:8b")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)


class ExperimentResult(BaseModel):
    """
    Full output of a completed experiment run.

    Serialized to JSON in output_dir/results.json.
    Includes full provenance for reproducibility.
    """

    experiment_id: str
    config_hash: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    git_commit: str = Field(default="unknown")
    config: ExperimentConfig
    records: list[EvaluationRecord] = Field(default_factory=list)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
