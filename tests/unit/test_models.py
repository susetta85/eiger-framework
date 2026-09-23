"""
Unit tests for core domain models (eiger.core.models).

Tests verify:
  - Pydantic schema enforcement (invalid values raise ValidationError)
  - Content hashing (SHA-256 fingerprints are stable and collision-resistant)
  - Auto-generated fields (doc_id, experiment_id) are non-empty strings
  - Derived properties (contains_poisoned, poison_ratio, config_hash) are correct
  - ExperimentResult.to_json() serializes to valid JSON

What these tests do NOT cover:
  - Network or disk I/O (models are pure Python / Pydantic, no side effects).
  - ORM mappings or database schemas.
  - Attack or metric business logic (covered in test_attacks.py / test_metrics.py).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from eiger.core.models import (
    AttackConfig,
    Claim,
    DatasetConfig,
    Document,
    EvaluationRecord,
    ExperimentConfig,
    ExperimentResult,
    GenerationResult,
    LLMConfig,
    MetricScore,
    PoisonAnnotation,
    PoisonedDocument,
    RetrieverConfig,
    RetrievalResult,
    RetrievedDocument,
)


class TestClaim:
    """Tests for the Claim domain model."""

    def test_valid_claim(self) -> None:
        """A Claim constructed with required fields must be accessible without error."""
        claim = Claim(
            claim_id="C1",
            original_fact="The inflation rate was 2.1% in 2024.",
            context_query="What was the 2024 inflation rate?",
        )
        assert claim.claim_id == "C1"

    def test_content_hash_is_16_chars(self) -> None:
        """content_hash must be a 16-character hex string (64-bit SHA-256 prefix)."""
        claim = Claim(claim_id="C1", original_fact="fact", context_query="query")
        assert len(claim.content_hash) == 16

    def test_same_fact_same_hash(self) -> None:
        """
        Two claims with the same original_fact must produce the same content_hash,
        even if their claim_id or other fields differ.
        """
        c1 = Claim(claim_id="A", original_fact="same fact", context_query="q")
        c2 = Claim(claim_id="B", original_fact="same fact", context_query="q")
        assert c1.content_hash == c2.content_hash

    def test_different_facts_different_hash(self) -> None:
        """Different original_fact strings must produce different content hashes."""
        c1 = Claim(claim_id="A", original_fact="fact one", context_query="q")
        c2 = Claim(claim_id="A", original_fact="fact two", context_query="q")
        assert c1.content_hash != c2.content_hash

    def test_sensitivity_class_and_risk_level_default_to_none(self) -> None:
        """
        Regression test (docs/ETHICS_AND_THREAT_MODEL.md §7): an unclassified
        claim must default to None on both fields, never a guessed "safe"
        value like "S0" or risk_level=1.
        """
        claim = Claim(claim_id="C1", original_fact="fact", context_query="q")
        assert claim.sensitivity_class is None
        assert claim.risk_level is None

    def test_valid_sensitivity_class_accepted(self) -> None:
        """Each of the four defined sensitivity classes must be accepted."""
        for cls in ("S0", "S1", "S2", "S3"):
            claim = Claim(
                claim_id="C1", original_fact="fact", context_query="q", sensitivity_class=cls,
            )
            assert claim.sensitivity_class == cls

    def test_invalid_sensitivity_class_raises(self) -> None:
        """A sensitivity_class outside {S0,S1,S2,S3} must raise ValidationError."""
        with pytest.raises(ValidationError):
            Claim(claim_id="C1", original_fact="fact", context_query="q", sensitivity_class="S4")

    def test_risk_level_out_of_range_raises(self) -> None:
        """risk_level outside [1, 5] must raise ValidationError."""
        with pytest.raises(ValidationError):
            Claim(claim_id="C1", original_fact="fact", context_query="q", risk_level=0)
        with pytest.raises(ValidationError):
            Claim(claim_id="C1", original_fact="fact", context_query="q", risk_level=6)

    def test_risk_level_in_range_accepted(self) -> None:
        """Every integer risk_level in [1, 5] must be accepted."""
        for level in (1, 2, 3, 4, 5):
            claim = Claim(
                claim_id="C1", original_fact="fact", context_query="q", risk_level=level,
            )
            assert claim.risk_level == level

    def test_ground_truth_label_defaults_to_none(self) -> None:
        """
        None means the source dataset does not carry this rating — never
        implicitly "verified_true" (see GroundTruthLabel's own comment,
        eiger/core/models.py).
        """
        claim = Claim(claim_id="C1", original_fact="fact", context_query="q")
        assert claim.ground_truth_label is None

    def test_valid_ground_truth_label_values_accepted(self) -> None:
        for label in ("verified_true", "verified_false"):
            claim = Claim(
                claim_id="C1", original_fact="fact", context_query="q",
                ground_truth_label=label,
            )
            assert claim.ground_truth_label == label

    def test_invalid_ground_truth_label_raises(self) -> None:
        with pytest.raises(ValidationError):
            Claim(
                claim_id="C1", original_fact="fact", context_query="q",
                ground_truth_label="true",
            )


class TestDocument:
    """Tests for the Document domain model."""

    def test_doc_id_auto_generated(self) -> None:
        """
        doc_id must be auto-generated (UUID4) when not supplied, producing
        a non-empty string. Every call produces a different ID.
        """
        doc = Document(claim_id="C1", text="text")
        assert doc.doc_id  # Not empty
        assert isinstance(doc.doc_id, str)

    def test_default_type_is_ground_truth(self) -> None:
        """doc_type must default to 'ground_truth' when not specified."""
        doc = Document(claim_id="C1", text="text")
        assert doc.doc_type == "ground_truth"

    def test_sensitivity_class_and_risk_level_default_to_none(self) -> None:
        """
        Regression test (docs/ETHICS_AND_THREAT_MODEL.md §7): an unclassified
        document must default to None on both fields, never a guessed "safe"
        value.
        """
        doc = Document(claim_id="C1", text="text")
        assert doc.sensitivity_class is None
        assert doc.risk_level is None

    def test_invalid_sensitivity_class_raises(self) -> None:
        """A sensitivity_class outside {S0,S1,S2,S3} must raise ValidationError."""
        with pytest.raises(ValidationError):
            Document(claim_id="C1", text="text", sensitivity_class="S9")

    def test_risk_level_out_of_range_raises(self) -> None:
        """risk_level outside [1, 5] must raise ValidationError."""
        with pytest.raises(ValidationError):
            Document(claim_id="C1", text="text", risk_level=0)

    def test_ground_truth_label_defaults_to_none(self) -> None:
        doc = Document(claim_id="C1", text="text")
        assert doc.ground_truth_label is None

    def test_valid_ground_truth_label_values_accepted(self) -> None:
        for label in ("verified_true", "verified_false"):
            doc = Document(claim_id="C1", text="text", ground_truth_label=label)
            assert doc.ground_truth_label == label

    def test_invalid_ground_truth_label_raises(self) -> None:
        with pytest.raises(ValidationError):
            Document(claim_id="C1", text="text", ground_truth_label="false")

    def test_ground_truth_label_independent_of_doc_type(self) -> None:
        """
        A verified_false claim can still be doc_type='ground_truth'
        ("clean but naturally false") — the two axes are not coupled.
        """
        doc = Document(
            claim_id="C1", text="text", doc_type="ground_truth",
            ground_truth_label="verified_false",
        )
        assert doc.doc_type == "ground_truth"
        assert doc.ground_truth_label == "verified_false"


class TestPoisonAnnotation:
    """Tests for the PoisonAnnotation model with [1.0, 5.0] range validation."""

    def test_scores_in_valid_range(self) -> None:
        """Scores within [1.0, 5.0] must be accepted without error."""
        ann = PoisonAnnotation(plausibility=3.0, verification_difficulty=4.0, editorial_risk=5.0)
        assert 1.0 <= ann.plausibility <= 5.0

    def test_out_of_range_raises(self) -> None:
        """
        Scores outside [1.0, 5.0] must raise Pydantic ValidationError.

        This validates that the ge/le constraints are active: plausibility=6.0
        violates le=5.0, and plausibility=0.5 violates ge=1.0.
        """
        with pytest.raises(ValidationError):
            PoisonAnnotation(plausibility=6.0, verification_difficulty=4.0, editorial_risk=5.0)

        with pytest.raises(ValidationError):
            PoisonAnnotation(plausibility=0.5, verification_difficulty=4.0, editorial_risk=5.0)


class TestRetrievalResult:
    """Tests for RetrievalResult computed properties."""

    def _make_result(self, doc_types: list[str]) -> RetrievalResult:
        """Build a RetrievalResult with hits of the specified doc_type values."""
        hits = []
        for i, dtype in enumerate(doc_types, 1):
            doc = Document(claim_id="C1", text="text", doc_type=dtype)
            hits.append(RetrievedDocument(document=doc, score=0.9, rank=i))
        return RetrievalResult(query="q", claim_id="C1", hits=hits, top_k=len(hits))

    def test_contains_poisoned_true(self) -> None:
        """contains_poisoned must be True when at least one hit has doc_type='poisoned'."""
        result = self._make_result(["ground_truth", "poisoned"])
        assert result.contains_poisoned is True

    def test_contains_poisoned_false(self) -> None:
        """contains_poisoned must be False when all hits have doc_type='ground_truth'."""
        result = self._make_result(["ground_truth", "ground_truth"])
        assert result.contains_poisoned is False

    def test_poison_ratio(self) -> None:
        """poison_ratio must return the fraction of poisoned hits."""
        result = self._make_result(["ground_truth", "poisoned", "poisoned"])
        assert result.poison_ratio == pytest.approx(2 / 3)

    def test_empty_result_poison_ratio_zero(self) -> None:
        """poison_ratio must return 0.0 for an empty hits list (no ZeroDivisionError)."""
        result = RetrievalResult(query="q", claim_id="C1", hits=[], top_k=5)
        assert result.poison_ratio == 0.0

    def test_serialized_hit_preserves_poisoned_document_provenance(self) -> None:
        """
        Regression test: found while inspecting a real experiment's
        results.json after the Sprint 4 ERS/Qdrant fix. RetrievedDocument.document
        is declared as the base `Document` type (correct for validation — a
        PoisonedDocument IS a Document), but Pydantic v2 by default serializes
        a field according to its DECLARED type, not the runtime instance's
        actual type. Before adding SerializeAsAny, a real PoisonedDocument
        stored in this field was silently truncated to just Document's own
        fields on every model_dump_json()/to_json() call — attack_name,
        attack_params (including the no_op flag), original_text, and
        annotation were all dropped from the persisted result, even though
        ERSMetric computed correctly from the live (pre-serialization) object.
        This defeated the entire purpose of results.json as a provenance
        record: which retrieved document was poisoned, by which attack, was
        unrecoverable from the file alone.
        """
        poisoned = PoisonedDocument(
            doc_id="d1",
            claim_id="C1",
            text="poisoned text",
            attack_name="numerical_shift",
            attack_params={"attack": "numerical_shift", "no_op": False},
            original_text="original text",
        )
        result = RetrievalResult(
            query="q",
            claim_id="C1",
            hits=[RetrievedDocument(document=poisoned, score=0.9, rank=1)],
            top_k=1,
        )
        dumped = json.loads(result.model_dump_json())
        serialized_doc = dumped["hits"][0]["document"]
        assert serialized_doc["attack_name"] == "numerical_shift"
        assert serialized_doc["attack_params"] == {"attack": "numerical_shift", "no_op": False}
        assert serialized_doc["original_text"] == "original text"


class TestExperimentConfig:
    """Tests for ExperimentConfig and its config_hash property."""

    def test_valid_config(self) -> None:
        """A minimally-valid ExperimentConfig must be constructed without error."""
        cfg = ExperimentConfig(
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg.seed == 42

    def test_faithfulness_scorer_defaults_to_embedding(self) -> None:
        """
        Backward compatibility: configs written before Sprint 5 (when this
        field did not exist) must still resolve to the CLI's pre-existing
        default scorer.
        """
        cfg = ExperimentConfig(
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg.faithfulness_scorer == "embedding"

    def test_faithfulness_scorer_accepts_ragas_and_none(self) -> None:
        for value in ("ragas", "none"):
            cfg = ExperimentConfig(
                dataset=DatasetConfig(name="json_fixture"),
                retriever=RetrieverConfig(),
                llm=LLMConfig(),
                faithfulness_scorer=value,
            )
            assert cfg.faithfulness_scorer == value

    def test_allow_non_benchmark_attacks_defaults_to_false(self) -> None:
        """
        Safe-by-default: a config that doesn't mention this field must not
        silently permit M03/M05 attacks (see eiger/ingestion/corpus_builder.py).
        """
        cfg = ExperimentConfig(
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg.allow_non_benchmark_attacks is False

    def test_allow_non_benchmark_attacks_accepts_true(self) -> None:
        cfg = ExperimentConfig(
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
            allow_non_benchmark_attacks=True,
        )
        assert cfg.allow_non_benchmark_attacks is True

    def test_config_hash_deterministic(self) -> None:
        """
        Two ExperimentConfig objects with the same field values must produce
        the same config_hash (excluding experiment_id, which is excluded from hashing).
        """
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
        """Two configs that differ in seed must produce different config hashes."""
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

    def test_config_hash_independent_of_free_form_dict_key_order(self) -> None:
        """
        Regression test (Sprint 4 audit): two configs whose only difference
        is the insertion order of a free-form dict[str, Any] field (here,
        AttackConfig.params) must hash identically, since they represent
        the exact same configuration.

        Before the fix, config_hash used model_dump_json() directly, which
        is only canonical for a model's own declared fields (fixed
        definition order) — NOT for the *contents* of a dict[str, Any]
        field, where Python preserves insertion order. Two semantically
        identical AttackConfig.params dicts built in different key order
        therefore produced different config_hash values, defeating the
        property's purpose of detecting identical configurations.
        """
        cfg1 = ExperimentConfig(
            experiment_id="x",
            dataset=DatasetConfig(name="json_fixture"),
            attacks=[
                AttackConfig(
                    name="date_manipulation",
                    poison_rate=0.1,
                    params={"min_shift": 1, "max_shift": 5},
                )
            ],
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        cfg2 = ExperimentConfig(
            experiment_id="x",
            dataset=DatasetConfig(name="json_fixture"),
            attacks=[
                AttackConfig(
                    name="date_manipulation",
                    poison_rate=0.1,
                    params={"max_shift": 5, "min_shift": 1},
                )
            ],
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg1.config_hash == cfg2.config_hash

    def test_config_hash_still_differs_on_real_param_change(self) -> None:
        """
        Sanity check alongside the canonicalization fix: config_hash must
        still change when a param VALUE (not just key order) differs, so
        the fix doesn't accidentally make the hash insensitive to real
        configuration differences.
        """
        cfg1 = ExperimentConfig(
            experiment_id="x",
            dataset=DatasetConfig(name="json_fixture"),
            attacks=[AttackConfig(name="date_manipulation", poison_rate=0.1, params={"max_shift": 5})],
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        cfg2 = ExperimentConfig(
            experiment_id="x",
            dataset=DatasetConfig(name="json_fixture"),
            attacks=[AttackConfig(name="date_manipulation", poison_rate=0.1, params={"max_shift": 10})],
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        assert cfg1.config_hash != cfg2.config_hash


class TestExperimentResult:
    """Tests for ExperimentResult and its to_json() serialization method."""

    def _make_result(self) -> ExperimentResult:
        """Build a minimal ExperimentResult for serialization tests."""
        cfg = ExperimentConfig(
            experiment_id="test-exp-001",
            dataset=DatasetConfig(name="json_fixture"),
            retriever=RetrieverConfig(),
            llm=LLMConfig(),
        )
        return ExperimentResult(
            experiment_id="test-exp-001",
            config_hash=cfg.config_hash,
            config=cfg,
        )

    def test_to_json_returns_string(self) -> None:
        """to_json() must return a str (not bytes or dict)."""
        result = self._make_result()
        assert isinstance(result.to_json(), str)

    def test_to_json_is_valid_json(self) -> None:
        """to_json() output must be parseable by the stdlib json module."""
        result = self._make_result()
        parsed = json.loads(result.to_json())
        assert isinstance(parsed, dict)

    def test_to_json_contains_experiment_id(self) -> None:
        """The serialized JSON must include the experiment_id field."""
        result = self._make_result()
        assert "experiment_id" in result.to_json()

    def test_to_json_contains_config_hash(self) -> None:
        """The serialized JSON must include the config_hash for provenance."""
        result = self._make_result()
        assert "config_hash" in result.to_json()


class TestEvaluationRecordProperties:
    """Tests for EvaluationRecord's convenience accessor properties."""

    def _make_eval_record(self, metrics: dict) -> EvaluationRecord:
        """Build a minimal EvaluationRecord with the given metrics dict."""
        doc = Document(claim_id="C1", text="text")
        retrieval = RetrievalResult(
            query="q", claim_id="C1",
            hits=[RetrievedDocument(document=doc, score=0.9, rank=1)],
            top_k=1,
        )
        generation = GenerationResult(
            claim_id="C1", query="q", context_docs=["text"],
            answer="ans", model_name="test",
        )
        return EvaluationRecord(
            claim_id="C1", generation=generation, retrieval=retrieval, metrics=metrics,
        )

    def test_faithfulness_score_present(self) -> None:
        """faithfulness_score must return the stored ragas_faithfulness value."""
        record = self._make_eval_record({"ragas_faithfulness": 0.9})
        assert record.faithfulness_score == pytest.approx(0.9)

    def test_faithfulness_score_default_zero(self) -> None:
        """faithfulness_score must return 0.0 when ragas_faithfulness is absent."""
        record = self._make_eval_record({})
        assert record.faithfulness_score == 0.0

    def test_factual_correctness_score_present(self) -> None:
        """factual_correctness_score must return the stored ragas_answer_correctness value."""
        record = self._make_eval_record({"ragas_answer_correctness": 0.3})
        assert record.factual_correctness_score == pytest.approx(0.3)

    def test_factual_correctness_score_default_zero(self) -> None:
        """factual_correctness_score must return 0.0 when ragas_answer_correctness is absent."""
        record = self._make_eval_record({})
        assert record.factual_correctness_score == 0.0
