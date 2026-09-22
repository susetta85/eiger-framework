"""
Unit tests for PCSMetric (eiger.metrics.pcs).

Tests verify:
  - compute() returns 0.0 with a "reason" when no poisoned context was
    retrieved at all (nothing to be sensitive to)
  - compute() returns 0.0 with a "reason" when contains_poisoned is True but
    generation.metadata has no "counterfactual_answer" (ExperimentRunner
    would only have populated it with "pcs" configured)
  - compute() returns 0.0 with a "reason" when the real or counterfactual
    answer is blank
  - compute() returns PCS = 1 - cosine_similarity(answer, counterfactual)
    for the normal case, and includes both answers + the counterfactual
    context docs in metadata for audit trails
  - compute_batch() maps compute() element-wise
  - aggregate() is a plain mean, 0.0 for an empty list
"""

from __future__ import annotations

from eiger.core.interfaces import BaseEmbedder
from eiger.core.models import (
    Document,
    EvaluationRecord,
    GenerationResult,
    PoisonedDocument,
    RetrievalResult,
    RetrievedDocument,
)
from eiger.metrics.pcs import PCSMetric


class _StubEmbedder(BaseEmbedder):
    """Returns fixed vectors so cosine similarity is fully predictable."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]

    @property
    def embedding_dim(self) -> int:
        return 2


def _clean_hit() -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(claim_id="C1", text="clean doc", doc_type="ground_truth"),
        score=0.9,
        rank=1,
    )


def _poisoned_hit() -> RetrievedDocument:
    return RetrievedDocument(
        document=PoisonedDocument(
            claim_id="C1",
            text="poisoned doc",
            attack_name="numerical_shift",
            original_text="clean doc",
        ),
        score=0.8,
        rank=2,
    )


def _make_record(
    hits: list[RetrievedDocument],
    answer: str = "the real answer",
    generation_metadata: dict | None = None,
) -> EvaluationRecord:
    generation = GenerationResult(
        claim_id="C1",
        query="what happened?",
        context_docs=[hit.document.text for hit in hits],
        answer=answer,
        model_name="mock-llm",
        metadata=generation_metadata or {},
    )
    retrieval = RetrievalResult(query="what happened?", claim_id="C1", hits=hits, top_k=len(hits))
    return EvaluationRecord(claim_id="C1", generation=generation, retrieval=retrieval, metrics={})


class TestNoPoisonedContext:
    def test_returns_zero_with_reason(self) -> None:
        metric = PCSMetric(embedder=_StubEmbedder({}))
        record = _make_record([_clean_hit()])
        score = metric.compute(record)
        assert score.value == 0.0
        assert score.metadata["reason"] == "no_poisoned_context_retrieved"


class TestMissingCounterfactual:
    def test_returns_zero_with_reason_when_metadata_key_absent(self) -> None:
        metric = PCSMetric(embedder=_StubEmbedder({}))
        record = _make_record([_poisoned_hit()], generation_metadata={})
        score = metric.compute(record)
        assert score.value == 0.0
        assert "counterfactual_answer missing" in score.metadata["reason"]


class TestBlankAnswers:
    def test_blank_real_answer_returns_zero(self) -> None:
        metric = PCSMetric(embedder=_StubEmbedder({}))
        record = _make_record(
            [_poisoned_hit()],
            answer="   ",
            generation_metadata={"counterfactual_answer": "something"},
        )
        score = metric.compute(record)
        assert score.value == 0.0
        assert score.metadata["reason"] == "blank real or counterfactual answer"

    def test_blank_counterfactual_answer_returns_zero(self) -> None:
        metric = PCSMetric(embedder=_StubEmbedder({}))
        record = _make_record(
            [_poisoned_hit()],
            answer="real answer",
            generation_metadata={"counterfactual_answer": "   "},
        )
        score = metric.compute(record)
        assert score.value == 0.0
        assert score.metadata["reason"] == "blank real or counterfactual answer"


class TestNormalComputation:
    def test_identical_answers_give_pcs_zero(self) -> None:
        embedder = _StubEmbedder({"same answer": [1.0, 0.0]})
        metric = PCSMetric(embedder=embedder)
        record = _make_record(
            [_poisoned_hit(), _clean_hit()],
            answer="same answer",
            generation_metadata={
                "counterfactual_answer": "same answer",
                "counterfactual_context_docs": ["clean doc"],
            },
        )
        score = metric.compute(record)
        assert score.value == 0.0
        assert score.metadata["answer"] == "same answer"
        assert score.metadata["counterfactual_answer"] == "same answer"
        assert score.metadata["counterfactual_context_docs"] == ["clean doc"]
        assert score.metadata["answer_similarity"] == 1.0

    def test_opposite_answers_give_pcs_one(self) -> None:
        embedder = _StubEmbedder({"real": [1.0, 0.0], "counterfactual": [-1.0, 0.0]})
        metric = PCSMetric(embedder=embedder)
        record = _make_record(
            [_poisoned_hit()],
            answer="real",
            generation_metadata={"counterfactual_answer": "counterfactual"},
        )
        score = metric.compute(record)
        assert score.value == 1.0
        assert score.metadata["answer_similarity"] == 0.0

    def test_orthogonal_answers_give_pcs_half(self) -> None:
        embedder = _StubEmbedder({"real": [1.0, 0.0], "counterfactual": [0.0, 1.0]})
        metric = PCSMetric(embedder=embedder)
        record = _make_record(
            [_poisoned_hit()],
            answer="real",
            generation_metadata={"counterfactual_answer": "counterfactual"},
        )
        score = metric.compute(record)
        assert score.value == 0.5

    def test_missing_counterfactual_context_docs_defaults_to_empty_list(self) -> None:
        embedder = _StubEmbedder({"real": [1.0, 0.0], "counterfactual": [1.0, 0.0]})
        metric = PCSMetric(embedder=embedder)
        record = _make_record(
            [_poisoned_hit()],
            answer="real",
            generation_metadata={"counterfactual_answer": "counterfactual"},
        )
        score = metric.compute(record)
        assert score.metadata["counterfactual_context_docs"] == []


class TestComputeBatchAndAggregate:
    def test_compute_batch_maps_element_wise(self) -> None:
        embedder = _StubEmbedder({"real": [1.0, 0.0], "counterfactual": [-1.0, 0.0]})
        metric = PCSMetric(embedder=embedder)
        records = [
            _make_record([_clean_hit()]),  # no poisoned -> 0.0
            _make_record(
                [_poisoned_hit()],
                answer="real",
                generation_metadata={"counterfactual_answer": "counterfactual"},
            ),  # opposite -> 1.0
        ]
        scores = metric.compute_batch(records)
        assert [s.value for s in scores] == [0.0, 1.0]

    def test_aggregate_is_mean(self) -> None:
        metric = PCSMetric(embedder=_StubEmbedder({}))
        from eiger.core.models import MetricScore

        scores = [
            MetricScore(metric_name="pcs", value=0.0),
            MetricScore(metric_name="pcs", value=1.0),
        ]
        assert metric.aggregate(scores) == 0.5

    def test_aggregate_empty_list_returns_zero(self) -> None:
        metric = PCSMetric(embedder=_StubEmbedder({}))
        assert metric.aggregate([]) == 0.0
