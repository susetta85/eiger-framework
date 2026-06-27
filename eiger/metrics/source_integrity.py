"""
Source Integrity (SI) metric.

SI measures the factual integrity of retrieved documents relative to ground truth.
  1.0 = all retrieved context is factually consistent with ground truth
  0.0 = all retrieved context contradicts ground truth

Implementation uses Natural Language Inference (NLI) via a cross-encoder model
to assess entailment/contradiction between each retrieved document and the
ground-truth fact.

NLI model used: cross-encoder/nli-MiniLM2-L6-H768 (MIT license)
This is a lightweight model that runs on CPU without GPU requirements.

Status: SPRINT 4 — NLI backend is implemented but the model must be
        downloaded on first use. The interface is complete.
"""

from __future__ import annotations

import warnings
from typing import Any

from eiger.core.interfaces import BaseMetric
from eiger.core.models import EvaluationRecord, MetricScore
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# NLI label indices (cross-encoder output order)
_CONTRADICTION_IDX = 0
_ENTAILMENT_IDX = 1
_NEUTRAL_IDX = 2

_DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


class SourceIntegrityMetric(BaseMetric):
    """
    Source Integrity (SI).

    For each retrieved document, uses a cross-encoder NLI model to classify
    its relationship to the ground-truth claim:
      - ENTAILMENT → contributes positively to SI
      - CONTRADICTION → contributes negatively
      - NEUTRAL → treated as non-informative (weight 0.5)

    SI = mean(entailment_score) across all retrieved documents in the batch.

    NOTE: Requires `transformers` and `torch` to be installed.
          Falls back to 0.0 with a warning if unavailable.
    """

    name: str = "source_integrity"
    description: str = (
        "Source Integrity: NLI-based measure of factual consistency "
        "between retrieved documents and ground-truth claims."
    )
    range: tuple[float, float] = (0.0, 1.0)

    def __init__(self, model_name: str = _DEFAULT_NLI_MODEL) -> None:
        self.model_name = model_name
        self._pipeline: Any = None  # Lazy-loaded on first call

    def _load_pipeline(self) -> None:
        """Lazy-load the NLI cross-encoder pipeline."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import]
            log.info("source_integrity.loading_nli_model", model=self.model_name)
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU; set to 0 for CUDA
            )
            log.info("source_integrity.model_ready", model=self.model_name)
        except ImportError:
            warnings.warn(
                "transformers/torch not installed. SourceIntegrityMetric will return 0.0. "
                "Run: pip install transformers torch",
                stacklevel=2,
            )

    def compute(self, record: EvaluationRecord) -> MetricScore:
        """
        Compute SI for a single evaluation record.

        Compares each retrieved document against the ground-truth claim
        using NLI, then averages the entailment probabilities.
        """
        self._load_pipeline()

        # Retrieve the ground-truth text for the claim
        ground_truth = self._get_ground_truth(record)
        if not ground_truth:
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                metadata={"warning": "Ground truth not available for this record"},
            )

        if self._pipeline is None:
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                metadata={"warning": "NLI pipeline not available (transformers not installed)"},
            )

        retrieved_texts = [h.document.text for h in record.retrieval.hits]
        if not retrieved_texts:
            return MetricScore(metric_name=self.name, value=0.0)

        entailment_scores: list[float] = []
        for doc_text in retrieved_texts:
            result = self._pipeline(
                doc_text,
                candidate_labels=["consistent", "contradictory"],
                hypothesis_template="This text is {} with the following claim: " + ground_truth,
            )
            # Extract "consistent" label score
            label_scores = dict(zip(result["labels"], result["scores"]))
            entailment_scores.append(label_scores.get("consistent", 0.5))

        si = sum(entailment_scores) / len(entailment_scores)

        return MetricScore(
            metric_name=self.name,
            value=si,
            metadata={
                "n_documents": len(retrieved_texts),
                "entailment_scores": entailment_scores,
                "nli_model": self.model_name,
            },
        )

    @staticmethod
    def _get_ground_truth(record: EvaluationRecord) -> str | None:
        """
        Extract ground-truth text from the evaluation record.

        Ground truth is expected to be stored in the retrieval result metadata
        or derivable from the claim. This method is a placeholder for Sprint 4
        when the full dataset integration is complete.
        """
        # Will be populated when the full dataset loader is integrated.
        # For now: return query as a proxy (claim question approximates the fact).
        return record.retrieval.query or None

    def aggregate(self, scores: list[MetricScore]) -> float:
        if not scores:
            return 0.0
        return sum(s.value for s in scores) / len(scores)
