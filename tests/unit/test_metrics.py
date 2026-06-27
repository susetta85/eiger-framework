"""
Unit tests for evaluation metrics: FFRMetric, ERSMetric, SourceIntegrityMetric registry.

Tests verify:
  - Correct formula implementation for FFR (binary classification per record)
  - Boundary conditions at the decision thresholds (strict > and <)
  - Custom threshold configuration
  - compute_batch() delegates correctly to compute()
  - aggregate() returns the correct mean and handles empty lists
  - ERS returns values in [0, 1] for valid annotations
  - ERS returns 0.0 with a warning when no annotation is present
  - ERS raises ValueError for invalid weight configurations
  - Metric registry contains all built-in metrics and raises MetricNotFoundError
    for unknown names

What these tests do NOT cover:
  - Real NLI model inference for SourceIntegrityMetric (see test_source_integrity.py).
  - Integration with RAGAS or other LLM-backed evaluators (integration tests).
"""

from __future__ import annotations

import pytest

from eiger.core.models import (
    Document,
    EvaluationRecord,
    GenerationResult,
    MetricScore,
    PoisonAnnotation,
    PoisonedDocument,
    RetrievalResult,
    RetrievedDocument,
)
from eiger.metrics import ERSMetric, FFRMetric, get_metric, list_metrics


# ─── Shared helpers ───────────────────────────────────────────────────────────

def make_record(
    faithfulness: float,
    correctness: float,
    poison_annotation: PoisonAnnotation | None = None,
) -> EvaluationRecord:
    """
    Build a minimal EvaluationRecord for metric unit tests.

    Args:
        faithfulness:      RAGAS faithfulness score to store in record.metrics.
        correctness:       RAGAS answer-correctness score to store in record.metrics.
        poison_annotation: If provided, the retrieved document will be a
                           PoisonedDocument carrying this annotation; otherwise
                           it is a plain ground-truth Document.

    Returns:
        An EvaluationRecord ready to pass to any metric's compute() method.
    """
    doc: Document
    if poison_annotation:
        doc = PoisonedDocument(
            claim_id="C1",
            text="Poisoned document",
            attack_name="numerical_shift",
            attack_params={},
            original_text="Test document",
            annotation=poison_annotation,
        )
    else:
        doc = Document(claim_id="C1", text="Test document", doc_type="ground_truth")

    retrieval = RetrievalResult(
        query="test query",
        claim_id="C1",
        hits=[RetrievedDocument(document=doc, score=0.9, rank=1)],
        top_k=1,
    )
    generation = GenerationResult(
        claim_id="C1",
        query="test query",
        context_docs=[doc.text],
        answer="Test answer",
        model_name="test_model",
    )
    return EvaluationRecord(
        claim_id="C1",
        generation=generation,
        retrieval=retrieval,
        metrics={
            "ragas_faithfulness": faithfulness,
            "ragas_answer_correctness": correctness,
        },
    )


# ─── FFRMetric ────────────────────────────────────────────────────────────────

class TestFFRMetric:
    """Tests for the Faithful Falsehood Rate metric."""

    def test_faithful_and_wrong_is_1(self) -> None:
        """
        A record with faithfulness > 0.8 AND correctness < 0.2 must score 1.0.

        This is the definition of a Faithful Falsehood: the LLM faithfully
        reproduced a poisoned document's false claim, earning FFR=1.
        """
        metric = FFRMetric()
        record = make_record(faithfulness=0.9, correctness=0.1)
        score = metric.compute(record)
        assert score.value == 1.0

    def test_faithful_and_correct_is_0(self) -> None:
        """
        A record that is faithful AND correct must score 0.0.

        The answer followed the context but the context was factually correct,
        so this is a good RAG outcome, not a failure.
        """
        metric = FFRMetric()
        record = make_record(faithfulness=0.9, correctness=0.9)
        assert metric.compute(record).value == 0.0

    def test_unfaithful_and_wrong_is_0(self) -> None:
        """
        A record that is wrong but NOT faithful to context must score 0.0.

        The model made up a wrong answer rather than reproducing a poisoned
        document. This is a different failure mode (hallucination), not FFR.
        """
        metric = FFRMetric()
        record = make_record(faithfulness=0.5, correctness=0.1)
        assert metric.compute(record).value == 0.0

    def test_on_threshold_boundary(self) -> None:
        """
        Exactly at the faithfulness threshold (=0.8) must NOT be counted as faithful.

        FFR uses strict greater-than (>), so faithfulness=0.8 with threshold=0.8
        is NOT faithful, and the record must score 0.0.
        """
        metric = FFRMetric(faithfulness_threshold=0.8)
        record = make_record(faithfulness=0.8, correctness=0.1)
        assert metric.compute(record).value == 0.0

    def test_aggregate_ffr(self) -> None:
        """
        aggregate() must return the fraction of records that are faithful falsehoods.

        4 records: 2 faithful falsehoods, 2 not → FFR = 0.5.
        """
        metric = FFRMetric()
        records = [
            make_record(0.9, 0.1),  # FF: yes
            make_record(0.9, 0.9),  # FF: no
            make_record(0.9, 0.1),  # FF: yes
            make_record(0.3, 0.1),  # FF: no (not faithful)
        ]
        scores = metric.compute_batch(records)
        ffr = metric.aggregate(scores)
        assert ffr == pytest.approx(0.5)  # 2/4

    def test_empty_batch_returns_0(self) -> None:
        """aggregate() must return 0.0 for an empty list (safe default, no division by zero)."""
        metric = FFRMetric()
        assert metric.aggregate([]) == 0.0

    def test_custom_thresholds(self) -> None:
        """
        Custom faithfulness_threshold and correctness_threshold must override
        the defaults. A record with faithfulness=0.6 (> 0.5) and correctness=0.4
        (< 0.5) must score 1.0 with these custom thresholds.
        """
        metric = FFRMetric(faithfulness_threshold=0.5, correctness_threshold=0.5)
        record = make_record(faithfulness=0.6, correctness=0.4)
        assert metric.compute(record).value == 1.0

    def test_metadata_in_score(self) -> None:
        """
        The MetricScore returned by compute() must include faithfulness_score and
        is_faithful_falsehood in its metadata dict for audit-trail purposes.
        """
        metric = FFRMetric()
        record = make_record(faithfulness=0.9, correctness=0.1)
        score = metric.compute(record)
        assert "faithfulness_score" in score.metadata
        assert score.metadata["is_faithful_falsehood"] is True


# ─── ERSMetric ────────────────────────────────────────────────────────────────

class TestERSMetric:
    """Tests for the Epistemic Risk Score metric."""

    def _annotation(self, p: float = 4.0, v: float = 4.0, e: float = 4.0) -> PoisonAnnotation:
        """Helper: build a PoisonAnnotation with the given scores."""
        return PoisonAnnotation(plausibility=p, verification_difficulty=v, editorial_risk=e)

    def test_returns_value_in_range(self) -> None:
        """ERS must return a value in [0.0, 1.0] for any valid annotation."""
        metric = ERSMetric()
        annotation = self._annotation()
        record = make_record(0.9, 0.1, poison_annotation=annotation)
        score = metric.compute(record)
        assert 0.0 <= score.value <= 1.0

    def test_max_annotation_gives_near_1(self) -> None:
        """
        All annotation scores at the maximum (5.0) must yield ERS ≈ 1.0.

        ERS = (p*0.3 + v*0.4 + e*0.3) / 5.0 = (5*1.0) / 5.0 = 1.0.
        """
        metric = ERSMetric()
        annotation = self._annotation(p=5.0, v=5.0, e=5.0)
        record = make_record(0.9, 0.1, poison_annotation=annotation)
        score = metric.compute(record)
        assert score.value == pytest.approx(1.0)

    def test_min_annotation_gives_near_0_2(self) -> None:
        """
        All annotation scores at the minimum (1.0) must yield ERS ≈ 0.2.

        ERS = (1*1.0) / 5.0 = 0.2 (not 0.0, because the scale starts at 1).
        """
        metric = ERSMetric()
        annotation = self._annotation(p=1.0, v=1.0, e=1.0)
        record = make_record(0.9, 0.1, poison_annotation=annotation)
        score = metric.compute(record)
        assert score.value == pytest.approx(0.2)  # 1.0/5.0

    def test_no_annotation_returns_0(self) -> None:
        """
        When no PoisonAnnotation is present in the retrieval hit, ERS must
        return 0.0 with a warning key in the metadata.

        This handles clean (non-poisoned) documents gracefully.
        """
        metric = ERSMetric()
        record = make_record(0.9, 0.1, poison_annotation=None)
        score = metric.compute(record)
        assert score.value == 0.0
        assert "warning" in score.metadata

    def test_invalid_weights_raise(self) -> None:
        """
        ERSMetric must raise ValueError when the three weights do not sum to 1.0.

        This is a configuration guard: silently accepting invalid weights would
        produce ERS values outside [0, 1], corrupting experiment results.
        """
        with pytest.raises(ValueError, match="sum to 1.0"):
            ERSMetric(weight_plausibility=0.5, weight_verification=0.5, weight_editorial=0.5)

    def test_aggregate_empty_returns_zero(self) -> None:
        """ERSMetric.aggregate([]) must return 0.0 without error."""
        assert ERSMetric().aggregate([]) == 0.0

    def test_aggregate_excludes_zero_values(self) -> None:
        """aggregate must skip records with value=0.0 (no annotation found)."""
        scores = [
            MetricScore(metric_name="ers", value=0.0),
            MetricScore(metric_name="ers", value=0.8),
        ]
        assert ERSMetric().aggregate(scores) == pytest.approx(0.8)

    def test_aggregate_all_zero_returns_zero(self) -> None:
        """aggregate([all zero]) must return 0.0 (valid list is empty)."""
        scores = [MetricScore(metric_name="ers", value=0.0)]
        assert ERSMetric().aggregate(scores) == 0.0

# ─── Metric Registry ──────────────────────────────────────────────────────────

class TestMetricRegistry:
    """Tests for the metric registry lookup functions."""

    def test_all_builtin_metrics_registered(self) -> None:
        """
        All three built-in metrics must be discoverable by name.

        This test acts as a guard against accidental removal of a metric from
        the auto-registration block in eiger.metrics.__init__.
        """
        registered = list_metrics()
        assert "ffr" in registered
        assert "ers" in registered
        assert "source_integrity" in registered

    def test_get_metric_by_name(self) -> None:
        """get_metric('ffr') must return a live FFRMetric instance."""
        metric = get_metric("ffr")
        assert isinstance(metric, FFRMetric)

    def test_get_unknown_metric_raises(self) -> None:
        """
        Requesting a metric name not in the registry must raise MetricNotFoundError.

        This verifies that the error type is correctly specialised rather than
        a generic KeyError.
        """
        from eiger.core.exceptions import MetricNotFoundError
        with pytest.raises(MetricNotFoundError):
            get_metric("nonexistent_metric_xyz")

class TestBaseMetricDefaults:
    """Tests for the default compute_batch and aggregate in BaseMetric."""

    def test_default_compute_batch_delegates_to_compute(self) -> None:
        """BaseMetric.compute_batch must call compute() for each record."""
        from eiger.core.interfaces import BaseMetric
        from eiger.core.models import EvaluationRecord, GenerationResult, RetrievalResult

        class _Minimal(BaseMetric):
            name: str = "minimal"
            description: str = "test"
            range: tuple[float, float] = (0.0, 1.0)
            def compute(self, record: EvaluationRecord) -> MetricScore:
                return MetricScore(metric_name="minimal", value=0.5)
            # does NOT override compute_batch → uses BaseMetric line 453

        doc = Document(claim_id="C1", text="text")
        retrieval = RetrievalResult(
            query="q", claim_id="C1",
            hits=[RetrievedDocument(document=doc, score=0.9, rank=1)], top_k=1,
        )
        gen = GenerationResult(
            claim_id="C1", query="q", context_docs=["text"],
            answer="a", model_name="test",
        )
        record = EvaluationRecord(claim_id="C1", generation=gen, retrieval=retrieval, metrics={})

        scores = _Minimal().compute_batch([record])  # exercises interfaces.py:453
        assert len(scores) == 1
        assert scores[0].value == pytest.approx(0.5)

    def test_default_aggregate_mean(self) -> None:
        """BaseMetric.aggregate default returns arithmetic mean."""
        from eiger.core.interfaces import BaseMetric
        from eiger.core.models import EvaluationRecord

        class _Minimal(BaseMetric):
             name: str = "minimal2"
             description: str = "test"
             range: tuple[float, float] = (0.0, 1.0)
             def compute(self, record: EvaluationRecord) -> MetricScore:
                 return MetricScore(metric_name="minimal2", value=0.5)
 # does NOT override aggregate → uses BaseMetric lines 469-471

        m = _Minimal()
        assert m.aggregate([]) == 0.0   # line 469-470
        scores = [MetricScore(metric_name="minimal2", value=0.4),
                  MetricScore(metric_name="minimal2", value=0.6)]
        assert m.aggregate(scores) == pytest.approx(0.5)  # line 471
