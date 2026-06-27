"""
Unit tests for core domain models.

Validates Pydantic schema enforcement, content hashing,
and derived properties.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eiger.core.models import (
    Claim, Document, PoisonedDocument, PoisonAnnotation,
    RetrievalResult, RetrievedDocument, ExperimentConfig,
    DatasetConfig, RetrieverConfig, LLMConfig,
)


class TestClaim:
    def test_valid_claim(self) -> None:
        claim = Claim(
            claim_id="C1",
            original_fact="The inflation rate was 2.1% in 2024.",
            context_query="What was the 2024 inflation rate?",
        )
        assert claim.claim_id == "C1"

    def test_content_hash_is_16_chars(self) -> None:
        claim = Claim(claim_id="C1", original_fact="fact", context_query="query")
        assert len(claim.content_hash) == 16

    def test_same_fact_same_hash(self) -> None:
        c1 = Claim(claim_id="A", original_fact="same fact", context_query="q")
        c2 = Claim(claim_id="B", original_fact="same fact", context_query="q")
        assert c1.content_hash == c2.content_hash

    def test_different_facts_different_hash(self) -> None:
        c1 = Claim(claim_id="A", original_fact="fact one", context_query="q")
        c2 = Claim(claim_id="A", original_fact="fact two", context_query="q")
        assert c1.content_hash != c2.content_hash


class TestDocument:
    def test_doc_id_auto_generated(self) -> None:
        doc = Document(claim_id="C1", text="text")
        assert doc.doc_id  # Not empty
        assert isinstance(doc.doc_id, str)

    def test_default_type_is_ground_truth(self) -> None:
        doc = Document(claim_id="C1", text="text")
        assert doc.doc_type == "ground_truth"


class TestPoisonAnnotation:
    def test_scores_in_valid_range(self) -> None:
        ann = PoisonAnnotation(plausibility=3.0, verification_difficulty=4.0, editorial_risk=5.0)
        assert 1.0 <= ann.plausibility <= 5.0

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            PoisonAnnotation(plausibility=6.0, verification_difficulty=4.0, editorial_risk=5.0)

        with pytest.raises(ValidationError):
            PoisonAnnotation(plausibility=0.5, verification_difficulty=4.0, editorial_risk=5.0)


class TestRetrievalResult:
    def _make_result(self, doc_types: list[str]) -> RetrievalResult:
        hits = []
        for i, dtype in enumerate(doc_types, 1):
            doc = Document(claim_id="C1", text="text", doc_type=dtype)
            hits.append(RetrievedDocument(document=doc, score=0.9, rank=i))
        return RetrievalResult(query="q", claim_id="C1", hits=hits, top_k=len(hits))

    def test_contains_poisoned_true(self) -> None:
        result = self._make_result(["ground_truth", "poisoned"])
        assert result.contains_poisoned is True

    def test_contains_poisoned_false(self) -> None:
        result = self._make_result(["ground_truth", "ground_truth"])
        assert result.contains_poisoned is False

    def test_poison_ratio(self) -> None:
        result = self._make_result(["ground_truth", "poisoned", "poisoned"])
        assert result.poison_ratio == pytest.approx(2 / 3)

    def test_empty_result_poison_ratio_zero(self) -> None:
        result = RetrievalResult(query="q", claim_id="C1", hits=[], top_k=5)
        assert result.poison_ratio == 0.0


class TestExperimentConfig:
    def test_valid_config(self) -> None:
        cfg = ExperimentConfig(
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg.seed == 42

    def test_config_hash_deterministic(self) -> None:
        cfg1 = ExperimentConfig(
            experiment_id="fixed",
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        cfg2 = ExperimentConfig(
            experiment_id="fixed",
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg1.config_hash == cfg2.config_hash

    def test_different_configs_different_hash(self) -> None:
        cfg1 = ExperimentConfig(
            experiment_id="x",
            seed=42,
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        cfg2 = ExperimentConfig(
            experiment_id="x",
            seed=99,
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg1.config_hash != cfg2.config_hash
