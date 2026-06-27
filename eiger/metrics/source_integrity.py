"""
Source Integrity (SI) metric.

SI measures the factual integrity of retrieved documents relative to ground truth.
  1.0 = all retrieved context is factually consistent with ground truth
  0.0 = all retrieved context contradicts ground truth

Implementation uses Natural Language Inference (NLI) via a cross-encoder model
to assess entailment/contradiction between each retrieved document and the
ground-truth fact. The NLI model produces a probability that the document text
is *consistent* with the claim; these probabilities are averaged across all
retrieved documents for a given query.

NLI model used: cross-encoder/nli-MiniLM2-L6-H768 (MIT license)
This is a lightweight model that runs on CPU without GPU requirements.
Average inference time: ~20 ms per document pair on modern hardware.

Status: SPRINT 4 — NLI backend is implemented but the model must be
        downloaded on first use (~90 MB). The interface is complete.

What this metric does NOT do:
  - Fine-tune or train the NLI model (inference only).
  - Handle multi-hop reasoning across documents.
  - Replace FFR or ERS; it is a complementary signal measuring source quality
    rather than final-answer quality.
"""

from __future__ import annotations

import warnings
from typing import Any

from eiger.core.interfaces import BaseMetric
from eiger.core.models import EvaluationRecord, MetricScore
from eiger.utils.logging import get_logger

log = get_logger(__name__)

# ─── NLI model constants ──────────────────────────────────────────────────────
# Label indices correspond to the output order of the cross-encoder model.
# These are defined as module-level constants rather than magic numbers so that
# a future model switch only requires updating them in one place.
_CONTRADICTION_IDX = 0
_ENTAILMENT_IDX = 1
_NEUTRAL_IDX = 2

# Default model name; can be overridden at instantiation time for experiments
# that require a larger/different NLI backbone.
_DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


class SourceIntegrityMetric(BaseMetric):
    """
    Source Integrity (SI).

    For each retrieved document, uses a cross-encoder NLI model to classify
    its relationship to the ground-truth claim:
      - ENTAILMENT   → document supports the ground truth (scores high)
      - CONTRADICTION → document conflicts with the ground truth (scores low)
      - NEUTRAL      → document is uninformative (treated as 0.5 by the
                        zero-shot-classification pipeline)

    SI = mean(consistency_score) across all retrieved documents in the record.

    The pipeline is lazy-loaded on first use to avoid penalising startup time
    in experiments that do not use this metric.

    NOTE: Requires ``transformers`` and ``torch`` to be installed.
          Falls back to 0.0 with a ``UserWarning`` if unavailable.

    What this class does NOT do:
      - Fine-tune the NLI model on EIBench data.
      - Distinguish between different types of factual errors in the documents.
      - Use GPU automatically (set ``device=0`` in ``_load_pipeline`` for CUDA).
    """

    # Registry key and human-readable metadata.
    name: str = "source_integrity"
    description: str = (
        "Source Integrity: NLI-based measure of factual consistency "
        "between retrieved documents and ground-truth claims."
    )
    range: tuple[float, float] = (0.0, 1.0)

    def __init__(self, model_name: str = _DEFAULT_NLI_MODEL) -> None:
        """
        Initialise SourceIntegrityMetric.

        Args:
            model_name: HuggingFace model identifier for the NLI cross-encoder.
                Defaults to ``cross-encoder/nli-MiniLM2-L6-H768``. Override
                to use a larger model (e.g. ``cross-encoder/nli-deberta-v3-base``)
                for higher accuracy at the cost of inference speed.
        """
        self.model_name = model_name
        # Pipeline is intentionally None at construction time.
        # Lazy loading prevents import-time network calls and allows the
        # metric to be instantiated even without transformers installed.
        self._pipeline: Any = None

    # ─── Pipeline lifecycle ───────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        """
        Lazy-load the NLI cross-encoder pipeline from HuggingFace.

        Called automatically on the first ``compute`` invocation. Subsequent
        calls return immediately because ``self._pipeline`` is already set.

        If ``transformers`` or ``torch`` are not installed, emits a
        ``UserWarning`` and leaves ``self._pipeline`` as ``None``.
        Callers must check for ``None`` before using the pipeline.
        """
        # Idempotent: do nothing if the pipeline was already loaded.
        if self._pipeline is not None:
            return
        try:
            # Import inside the method to make the top-level import of this
            # module succeed even without transformers installed.
            from transformers import pipeline as hf_pipeline  # type: ignore[import]

            log.info("source_integrity.loading_nli_model", model=self.model_name)
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # -1 = CPU; change to 0 for the first CUDA device
            )
            log.info("source_integrity.model_ready", model=self.model_name)
        except ImportError:
            # Warn rather than raise so that experiments mixing SI with other
            # metrics can still run partially if transformers is absent.
            warnings.warn(
                "transformers/torch not installed. SourceIntegrityMetric will return 0.0. "
                "Run: pip install transformers torch",
                stacklevel=2,
            )

    # ─── Core metric interface ────────────────────────────────────────────────

    def compute(self, record: EvaluationRecord) -> MetricScore:
        """
        Compute SI for a single evaluation record.

        Compares each retrieved document against the ground-truth claim
        using NLI, then averages the entailment (consistency) probabilities.

        Args:
            record: A completed evaluation record with retrieval hits populated.
                The record must have a retrievable ground-truth string (see
                ``_get_ground_truth``).

        Returns:
            MetricScore with:
              - value: Mean consistency score in [0.0, 1.0], or 0.0 with a
                       warning when the pipeline or ground truth is unavailable.
              - metadata: per-document scores, document count, and model name.
        """
        # Ensure the NLI pipeline is available before proceeding.
        self._load_pipeline()

        # ─── Ground truth extraction ──────────────────────────────────────
        ground_truth = self._get_ground_truth(record)
        if not ground_truth:
            # Without a ground-truth string we cannot run NLI comparison.
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                metadata={"warning": "Ground truth not available for this record"},
            )

        # ─── Pipeline availability check ──────────────────────────────────
        if self._pipeline is None:
            # transformers not installed; already warned in _load_pipeline.
            return MetricScore(
                metric_name=self.name,
                value=0.0,
                metadata={"warning": "NLI pipeline not available (transformers not installed)"},
            )

        # ─── Collect retrieved document texts ─────────────────────────────
        retrieved_texts = [h.document.text for h in record.retrieval.hits]
        if not retrieved_texts:
            # Empty retrieval result: no documents to score.
            return MetricScore(metric_name=self.name, value=0.0)

        # ─── Run NLI inference per document ──────────────────────────────
        entailment_scores: list[float] = []
        for doc_text in retrieved_texts:
            result = self._pipeline(
                doc_text,
                # Two-class zero-shot classification: consistent vs contradictory.
                # The neutral case is implicitly absorbed by the softmax.
                candidate_labels=["consistent", "contradictory"],
                hypothesis_template=(
                    "This text is {} with the following claim: " + ground_truth
                ),
            )
            # Convert the label-score lists into a lookup dict for safe access.
            label_scores = dict(zip(result["labels"], result["scores"]))
            # Use 0.5 as the fallback (neutral / uncertain) when the key is missing.
            entailment_scores.append(label_scores.get("consistent", 0.5))

        # ─── Aggregate and return ─────────────────────────────────────────
        # Simple mean: treats all retrieved documents equally regardless of rank.
        si = sum(entailment_scores) / len(entailment_scores)

        return MetricScore(
            metric_name=self.name,
            value=si,
            metadata={
                "n_documents": len(retrieved_texts),
                # Store per-document scores for post-hoc analysis.
                "entailment_scores": entailment_scores,
                "nli_model": self.model_name,
            },
        )

    # ─── Static helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_ground_truth(record: EvaluationRecord) -> str | None:
        """
        Extract ground-truth text from the evaluation record.

        Ground truth is expected to be stored in the retrieval result metadata
        or derivable from the claim. This method is a placeholder for Sprint 4
        when the full dataset integration is complete.

        Current approximation: returns the retrieval query string, which
        approximates the factual claim being evaluated. This is intentionally
        imprecise and will be replaced by the actual claim text once the
        dataset loader stores it in ``RetrievalResult``.

        Args:
            record: The evaluation record to inspect.

        Returns:
            The ground-truth text string, or ``None`` if unavailable.
        """
        # Will be populated when the full dataset loader is integrated.
        # For now: return query as a proxy (claim question approximates the fact).
        return record.retrieval.query or None

    # ─── Aggregation ──────────────────────────────────────────────────────────

    def aggregate(self, scores: list[MetricScore]) -> float:
        """
        Aggregate per-record SI scores into an experiment-level SI.

        Args:
            scores: List of MetricScore objects produced by ``compute``.

        Returns:
            Mean SI across all records in [0.0, 1.0].
            Returns 0.0 for an empty list.
        """
        if not scores:
            return 0.0
        # Simple mean across all records, including those with SI=0.0 caused by
        # pipeline unavailability. This conservatively reflects missing coverage.
        return sum(s.value for s in scores) / len(scores)
