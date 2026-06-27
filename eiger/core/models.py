"""
Domain models for the EIGER framework.

All models are Pydantic v2 BaseModel subclasses — they validate on
construction, serialize cleanly to JSON for experiment provenance, and
carry no infrastructure dependencies (no DB handles, no HTTP clients).

Why Pydantic v2:
  - model_dump_json() produces deterministic, schema-validated JSON that
    is used as the provenance record for every experiment run.
  - Field-level validation (ge/le constraints) catches configuration
    errors early, before any expensive computation begins.
  - The `from __future__ import annotations` import combined with Pydantic
    v2's model_rebuild() mechanism allows forward references between model
    classes defined in the same file (e.g. PoisonedDocument -> PoisonAnnotation).

What this file does NOT do:
  - It does not define any database schemas or ORM mappings.
  - It does not implement any business logic (that belongs in pipeline
    orchestrators and attack/metric implementations).
  - It does not import any infrastructure modules (vector stores, LLMs, etc.).
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
    Claims are immutable after construction — all downstream processing
    creates new Document objects rather than mutating the claim.

    What this class does NOT do:
      - It does not load itself from disk; dataset loaders (BaseDataset
        implementations) are responsible for constructing Claim objects.
      - It does not validate whether original_fact is factually correct;
        that is assumed to be guaranteed by the source dataset.
    """

    claim_id: str = Field(description="Unique identifier within the dataset")
    original_fact: str = Field(description="Verified factual statement (ground truth)")
    context_query: str = Field(description="Query used to retrieve relevant documents")
    # Default "unknown" lets fixtures omit the field without breaking validation.
    source_dataset: str = Field(default="unknown", description="Origin dataset (e.g. averitec, politifact)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """
        SHA-256 fingerprint of the original fact text, truncated to 16 hex chars.

        Used for dataset versioning: if the upstream dataset is updated and
        the fact text changes, the hash will differ, flagging a provenance
        mismatch when comparing old and new experiment results.

        Why SHA-256: cryptographically strong (collision-resistant), widely
        available in the Python standard library, and produces consistent
        output across platforms and Python versions.

        Returns:
            A 16-character hexadecimal string (64-bit prefix of SHA-256).
        """
        return hashlib.sha256(self.original_fact.encode()).hexdigest()[:16]


class Document(BaseModel):
    """
    A document stored in the vector corpus.

    Can represent either a ground-truth document or an adversarial variant.
    The doc_type field distinguishes between the two; downstream code that
    needs to treat them differently should branch on doc_type rather than
    checking isinstance(doc, PoisonedDocument), so that future doc types
    can be added without breaking existing branches.

    What this class does NOT do:
      - It does not store embedding vectors; those are managed by the
        vector store and keyed by doc_id.
      - It does not validate that text is non-empty; empty documents are
        allowed so that attack implementations can produce them as edge cases.
    """

    # uuid4 default ensures every Document has a globally unique ID even
    # when created outside a database transaction.
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

    Extends Document with full poisoning provenance: which attack was
    applied, with what parameters, and what the original text looked like
    before modification. This provenance is critical for post-hoc analysis
    of which attack strategies are most effective at deceiving the RAG system.

    What this class does NOT do:
      - It does not apply the attack itself; attacks are applied by
        BaseAttack.apply(), which returns a PoisonedDocument.
      - It does not score or annotate itself; PoisonAnnotation is
        optional and must be added by a separate annotation pipeline.
    """

    # Override the parent's default to lock the type for all instances.
    doc_type: str = "poisoned"
    attack_name: str = Field(description="Name of the attack that produced this document")
    attack_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters used by the attack",
    )
    # Storing the original text alongside the poisoned version enables
    # automatic diff generation for human review.
    original_text: str = Field(description="Unmodified text before poisoning")
    # Optional: annotation may be added by a human or a model judge in a
    # separate pass after the corpus is built.
    annotation: PoisonAnnotation | None = Field(default=None)


class PoisonAnnotation(BaseModel):
    """
    Human or model annotation describing the epistemic risk of a poisoned document.

    Scores are in [1, 5] — higher means riskier. The three dimensions
    capture distinct aspects of how dangerous a poisoned document is:

      - plausibility:            Would a reader believe it without checking?
      - verification_difficulty: How hard is it to debunk with a web search?
      - editorial_risk:          Would it pass a fact-checker's review?

    These scores are used as covariates in downstream analysis to understand
    which attack characteristics correlate with high false-fact rates (FFR).

    What this class does NOT do:
      - It does not compute scores automatically; it is a container for
        externally provided annotations.
    """

    plausibility: float = Field(ge=1.0, le=5.0, description="How believable the falsehood is")
    verification_difficulty: float = Field(ge=1.0, le=5.0, description="How hard it is to fact-check")
    editorial_risk: float = Field(ge=1.0, le=5.0, description="Risk of passing editorial review")


# ─── Retrieval layer ──────────────────────────────────────────────────────────

class RetrievedDocument(BaseModel):
    """
    A single document returned by a retriever, with its similarity score.

    Wraps a Document with retrieval-specific metadata (score, rank) so
    that downstream evaluation code can reason about retrieval quality
    independently from the document content.
    """

    document: Document
    # Score is normalized to [0, 1]; raw cosine similarities from the
    # vector store are rescaled by the retriever implementation.
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (higher = more similar)")
    rank: int = Field(ge=1, description="Position in the ranked result list")


class RetrievalResult(BaseModel):
    """
    Full output of a single retrieval operation.

    Captures the query, the claim it belongs to, and the ranked list of
    hits. The convenience properties (contains_poisoned, poison_ratio)
    are used heavily by metrics like FFR and Source Integrity Score.
    """

    query: str
    claim_id: str
    hits: list[RetrievedDocument] = Field(default_factory=list)
    top_k: int

    @property
    def contains_poisoned(self) -> bool:
        """
        Return True if at least one hit is a poisoned document.

        Used as a fast boolean check before computing more expensive
        metrics — if no poisoned document was retrieved, FFR is 0 for
        this result by definition.

        Returns:
            bool: True if any hit has doc_type == "poisoned".
        """
        return any(h.document.doc_type == "poisoned" for h in self.hits)

    @property
    def poison_ratio(self) -> float:
        """
        Fraction of retrieved hits that are poisoned documents.

        Returns 0.0 for empty hit lists to avoid division-by-zero.

        Returns:
            float in [0.0, 1.0].
        """
        if not self.hits:
            return 0.0
        return sum(1 for h in self.hits if h.document.doc_type == "poisoned") / len(self.hits)


# ─── Generation layer ─────────────────────────────────────────────────────────

class GenerationResult(BaseModel):
    """
    Output from an LLM given a query and retrieved context.

    Records everything needed to reproduce the LLM call: the exact
    context documents fed into the prompt, the model name, and any
    additional metadata (e.g. token counts, latency). This enables
    offline re-evaluation of answers without re-running the LLM.
    """

    claim_id: str
    query: str
    # Storing the raw texts (not Document objects) avoids coupling the
    # generation record to the vector store schema.
    context_docs: list[str] = Field(description="Texts of retrieved documents fed to the LLM")
    answer: str = Field(description="LLM-generated answer")
    model_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Evaluation layer ─────────────────────────────────────────────────────────

class MetricScore(BaseModel):
    """
    Result of a single metric computation on a single example.

    Keeping metric results as structured objects (rather than plain floats)
    allows each metric to attach additional diagnostic metadata — for example,
    the RAGAS faithfulness metric can include per-statement scores.
    """

    metric_name: str
    value: float
    # Optional diagnostic data from the metric implementation.
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    """
    Full evaluation record for a single claim.

    Combines the generation result with all computed metric scores.
    This is the atomic unit of experiment output — one record per claim
    per experiment run. All aggregate statistics are derived from a list
    of EvaluationRecords.

    What this class does NOT do:
      - It does not compute metrics itself; that is done by BaseMetric
        implementations which receive EvaluationRecords as input.
    """

    claim_id: str
    generation: GenerationResult
    retrieval: RetrievalResult
    # Metrics stored as a flat dict of name -> scalar value for easy
    # serialization and aggregation. Full MetricScore objects (with
    # metadata) are produced by metrics but collapsed here for compactness.
    metrics: dict[str, float] = Field(default_factory=dict)

    # ─── Convenience accessors ────────────────────────────────────────────

    @property
    def faithfulness_score(self) -> float:
        """
        Convenience accessor for RAGAS faithfulness score.

        Returns 0.0 if the metric was not computed for this record,
        making it safe to call unconditionally in aggregation loops.

        Returns:
            float: The ragas_faithfulness score, or 0.0 if absent.
        """
        return self.metrics.get("ragas_faithfulness", 0.0)

    @property
    def factual_correctness_score(self) -> float:
        """
        Convenience accessor for RAGAS answer correctness score.

        Returns:
            float: The ragas_answer_correctness score, or 0.0 if absent.
        """
        return self.metrics.get("ragas_answer_correctness", 0.0)


# ─── Experiment layer ─────────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """
    Full, validated specification for a single experiment run.

    Loaded from a YAML file and resolved against environment settings.
    Serialized to JSON alongside every result file for provenance.

    The config_hash property uniquely identifies the configuration so
    that two runs with identical configs can be detected and compared,
    even if they were run independently.

    What this class does NOT do:
      - It does not load itself from YAML; that is done by the experiment
        runner which calls ExperimentConfig.model_validate(yaml_data).
      - It does not interact with any infrastructure (files, DBs, etc.).
    """

    # Auto-generated ID keeps each run identifiable even without a user
    # supplied name. hex[:8] gives 8 hex chars = ~4 billion possibilities.
    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid4().hex[:8]}")
    seed: int = Field(default=42)
    dataset: DatasetConfig
    attacks: list[AttackConfig] = Field(default_factory=list)
    retriever: RetrieverConfig
    llm: LLMConfig
    # Default metrics cover the three primary EIGER KPIs.
    metrics: list[str] = Field(default_factory=lambda: ["ffr", "source_integrity", "ers"])
    output_dir: str = Field(default="results/")
    description: str = Field(default="")

    @property
    def config_hash(self) -> str:
        """
        SHA-256 fingerprint of the full configuration (excluding experiment_id).

        experiment_id is excluded because it is auto-generated and varies
        between runs even when all other config fields are identical. The
        hash is used to detect configuration drift between a stored result
        and a re-run attempt.

        Returns:
            A 16-character hexadecimal string (64-bit prefix of SHA-256).
        """
        # model_dump_json produces canonical, deterministic JSON (keys sorted
        # by field definition order in Pydantic v2), so the hash is stable.
        serialized = self.model_dump_json(exclude={"experiment_id"})
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class DatasetConfig(BaseModel):
    """Configuration for the source dataset to load claims from."""

    name: str = Field(description="Dataset identifier: averitec | politifact | json_fixture")
    split: str = Field(default="test")
    # None means load all available claims — useful for full benchmark runs.
    max_claims: int | None = Field(default=None, description="Cap number of claims (None = all)")
    # Local path override bypasses the default dataset download directory.
    path: str | None = Field(default=None, description="Local path override")


class AttackConfig(BaseModel):
    """Configuration for a single adversarial attack to apply during corpus building."""

    name: str = Field(description="Attack identifier (must be registered in attack registry)")
    # poison_rate is validated to [0, 1] by Pydantic's ge/le constraints.
    poison_rate: float = Field(ge=0.0, le=1.0, description="Fraction of corpus to poison")
    params: dict[str, Any] = Field(default_factory=dict, description="Attack-specific hyperparameters")


class RetrieverConfig(BaseModel):
    """Configuration for the retrieval component."""

    type: str = Field(default="dense", description="dense | sparse | hybrid")
    # Default embedder balances quality and speed for benchmark-scale corpora.
    embedder: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    vector_store: str = Field(default="qdrant")
    top_k: int = Field(default=5, ge=1)
    collection_name: str = Field(default="eiger_corpus")


class LLMConfig(BaseModel):
    """Configuration for the LLM generation backend."""

    backend: str = Field(default="ollama", description="ollama | openai")
    model: str = Field(default="llama3.1:8b")
    # temperature=0.0 ensures deterministic generation, which is critical
    # for reproducibility — same prompt must produce the same answer.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)


class ExperimentResult(BaseModel):
    """
    Full output of a completed experiment run.

    Serialized to JSON in output_dir/results.json.
    Includes full provenance for reproducibility: config hash, git commit,
    timestamp, and the complete configuration that produced the results.

    What this class does NOT do:
      - It does not write itself to disk; the experiment runner calls
        result.to_json() and handles the file I/O.
      - It does not compute aggregate metrics; those are computed by the
        evaluation pipeline and passed in as the aggregate_metrics dict.
    """

    experiment_id: str
    # config_hash stored redundantly here (also in config.config_hash) so
    # that provenance can be checked without deserializing the full config.
    config_hash: str
    # UTC ISO-8601 timestamp with trailing "Z" suffix for unambiguous
    # timezone identification in log files and result filenames.
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    # Git commit SHA (filled by the experiment runner via subprocess/git).
    # "unknown" is the safe default when not running inside a git repo.
    git_commit: str = Field(default="unknown")
    config: ExperimentConfig
    records: list[EvaluationRecord] = Field(default_factory=list)
    # Aggregate metrics (mean FFR, mean SIS, mean ERS, etc.) computed
    # across all records after the experiment completes.
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    # Captured environment variables (Python version, platform, package
    # versions) for debugging results that differ across machines.
    environment: dict[str, str] = Field(default_factory=dict)

    def to_json(self) -> str:
        """
        Serialize the full experiment result to a pretty-printed JSON string.

        Uses Pydantic's model_dump_json to ensure all nested models are
        serialized according to their field definitions (including aliases
        and exclude rules), rather than relying on Python's built-in json
        module which would miss Pydantic-specific serialization logic.

        Returns:
            str: Indented JSON string ready to write to a result file.
        """
        return self.model_dump_json(indent=2)
