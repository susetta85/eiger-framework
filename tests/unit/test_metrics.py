"""
Unit tests for evaluation metrics (FFR, ERS).

Tests verify:
  - Correct formula implementation
  - Edge cases (empty input, all faithful, none faithful)
  - Configurability (custom thresholds)
  - Registry integration
"""

from __future__ import annotations

import pytest

from eiger.metrics import FFRMetric, ERSMetric, get_metric, list_metrics
from eiger.core.models import (
    EvaluationRecord, GenerationResult, RetrievalResult,
    RetrievedDocument, Document, PoisonedDocument, PoisonAnnotation, MetricScore,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_record(
    faithfulness: float,
    correctness: float,
    poison_annotation: PoisonAnnotation | None = None,
) -> EvaluationRecord:
    """Build a minimal EvaluationRecord for metric testing."""
    doc = Document(claim_id="C1", text="Test document", doc_type="ground_truth")
    if poison_annotation:
        doc = PoisonedDocument(
            claim_id="C1",
            text="Poisoned document",
            attack_name="numerical_shift",
            attack_params={},
            original_text="Test document",
            annotation=poison_annotation,
        )

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
    record = EvaluationRecord(
        claim_id="C1",
        generation=generation,
        retrieval=retrieval,
        metrics={
            "ragas_faithfulness": faithfulness,
            "ragas_answer_correctness": correctness,
        },
    )
    return record


# ─── FFRMetric ─────────────────────────────────────────────────────────────────

class TestFFRMetric:
    def test_faithful_and_wrong_is_1(self) -> None:
        metric = FFRMetric()
        record = make_record(faithfulness=0.9, correctness=0.1)
        score = metric.compute(record)
        assert score.value == 1.0

    def test_faithful_and_correct_is_0(self) -> None:
        metric = FFRMetric()
        record = make_record(faithfulness=0.9, correctness=0.9)
        assert metric.compute(record).value == 0.0

    def test_unfaithful_and_wrong_is_0(self) -> None:
        metric = FFRMetric()
        record = make_record(faithfulness=0.5, correctness=0.1)
        assert metric.compute(record).value == 0.0

    def test_on_threshold_boundary(self) -> None:
        # Exactly at threshold: faithfulness=0.8 is NOT > 0.8
        metric = FFRMetric(faithfulness_threshold=0.8)
        record = make_record(faithfulness=0.8, correctness=0.1)
        assert metric.compute(record).value == 0.0

    def test_aggregate_ffr(self) -> None:
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
        metric = FFRMetric()
        assert metric.aggregate([]) == 0.0

    def test_custom_thresholds(self) -> None:
        metric = FFRMetric(faithfulness_threshold=0.5, correctness_threshold=0.5)
        record = make_record(faithfulness=0.6, correctness=0.4)
        assert metric.compute(record).value == 1.0

    def test_metadata_in_score(self) -> None:
        metric = FFRMetric()
        record = make_record(faithfulness=0.9, correctness=0.1)
        score = metric.compute(record)
        assert "faithfulness_score" in score.metadata
        assert score.metadata["is_faithful_falsehood"] is True


# ─── ERSMetric ────────────────────────────────────────────────────────────────

class TestERSMetric:
    def _annotation(self, p: float = 4.0, v: float = 4.0, e: float = 4.0) -> PoisonAnnotation:
        return PoisonAnnotation(plausibility=p, verification_difficulty=v, editorial_risk=e)

    def test_returns_value_in_range(self) -> None:
        metric = ERSMetric()
        annotation = self._annotation()
        record = make_record(0.9, 0.1, poison_annotation=annotation)
        score = metric.compute(record)
        assert 0.0 <= score.value <= 1.0

    def test_max_annotation_gives_near_1(self) -> None:
        metric = ERSMetric()
        annotation = self._annotation(p=5.0, v=5.0, e=5.0)
        record = make_record(0.9, 0.1, poison_annotation=annotation)
        score = metric.compute(record)
        assert score.value == pytest.approx(1.0)

    def test_min_annotation_gives_near_0(self) -> None:
        metric = ERSMetric()
        annotation = self._annotation(p=1.0, v=1.0, e=1.0)
        record = make_record(0.9, 0.1, poison_annotation=annotation)
        score = metric.compute(record)
        assert score.value == pytest.approx(0.2)  # 1.0/5.0

    def test_no_annotation_returns_0(self) -> None:
        metric = ERSMetric()
        record = make_record(0.9, 0.1, poison_annotation=None)
        score = metric.compute(record)
        assert score.value == 0.0
        assert "warning" in score.metadata

    def test_invalid_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            ERSMetric(weight_plausibility=0.5, weight_verification=0.5, weight_editorial=0.5)


# ─── Metric Registry ──────────────────────────────────────────────────────────

class TestMetricRegistry:
    def test_all_builtin_metrics_registered(self) -> None:
        registered = list_metrics()
        assert "ffr" in registered
        assert "ers" in registered
        assert "source_integrity" in registered

    def test_get_metric_by_name(self) -> None:
        metric = get_metric("ffr")
        assert isinstance(metric, FFRMetric)

    def test_get_unknown_metric_raises(self) -> None:
        from eiger.core.exceptions import MetricNotFoundError
        with pytest.raises(MetricNotFoundError):
            get_metric("nonexistent_metric_xyz")
