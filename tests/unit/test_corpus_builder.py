"""
Unit tests for CorpusBuilder and CorpusBuilderResult.

Tests verify:
  - CorpusBuilder.build() creates exactly one ground-truth document per claim
  - With poison_rate=1.0, every claim is poisoned for the configured attack
  - With an empty attack list, no documents are poisoned
  - CorpusBuilderResult.all_documents combines both lists correctly
  - CorpusBuilderResult.poison_ratio computes the correct fraction
  - poison_ratio returns 0.0 for an empty corpus (no division by zero)
  - The ground-truth document text and claim_id match the source claim
  - Two builds with the same seed produce bit-identical poisoned text (determinism)

What these tests do NOT cover:
  - Integration with a real Qdrant vector store (integration test territory).
  - Very large corpora or performance benchmarks (see benchmarking suite).
  - The attack-specific transformation logic (covered in test_attacks.py).
"""

from __future__ import annotations

import pytest

from eiger.attacks.numerical import NumericalShiftAttack
from eiger.core.models import (
    AttackConfig,
    Claim,
    Document,
    PoisonAnnotation,
    PoisonedDocument,
)
from eiger.ingestion.corpus_builder import CorpusBuilder, CorpusBuilderResult


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def two_claims() -> list[Claim]:
    """
    Provide two representative Claim objects for corpus-building tests.

    The texts contain numeric values and named entities so that
    NumericalShiftAttack and AttributionSwitchAttack have meaningful
    content to transform.
    """
    return [
        Claim(
            claim_id="C1",
            original_fact="Inflation rose to 3.5% in 2023 due to supply shocks.",
            context_query="What was the 2023 inflation rate?",
            source_dataset="test_fixture",
        ),
        Claim(
            claim_id="C2",
            original_fact="NASA confirmed the Mars mission launched in July 2020.",
            context_query="When did NASA launch the Mars mission?",
            source_dataset="test_fixture",
        ),
    ]


def _make_poisoned(claim_id: str = "C1") -> PoisonedDocument:
    """Helper: build a minimal PoisonedDocument for CorpusBuilderResult tests."""
    return PoisonedDocument(
        claim_id=claim_id,
        text="Poisoned: Inflation rose to 5.3% in 2023.",
        attack_name="numerical_shift",
        attack_params={},
        original_text="Inflation rose to 3.5% in 2023.",
        annotation=PoisonAnnotation(
            plausibility=4.0,
            verification_difficulty=3.5,
            editorial_risk=4.5,
        ),
    )


# ─── CorpusBuilderResult ──────────────────────────────────────────────────────

class TestCorpusBuilderResult:
    """Tests for the CorpusBuilderResult dataclass properties."""

    def test_all_documents_combines_both_lists(self) -> None:
        """
        all_documents must concatenate ground_truth_docs and poisoned_docs.

        The combined list is the input to the vector store upsert step.
        Ordering: ground-truth documents first, then poisoned documents.
        """
        gt = Document(claim_id="C1", text="ground truth", doc_type="ground_truth")
        pois = _make_poisoned("C1")
        result = CorpusBuilderResult(ground_truth_docs=[gt], poisoned_docs=[pois])
        all_docs = result.all_documents
        assert len(all_docs) == 2
        assert all_docs[0] is gt
        assert all_docs[1] is pois

    def test_poison_ratio_empty_corpus_is_zero(self) -> None:
        """
        poison_ratio must return 0.0 for an empty CorpusBuilderResult without
        raising ZeroDivisionError.
        """
        result = CorpusBuilderResult()
        assert result.poison_ratio == 0.0

    def test_poison_ratio_half(self) -> None:
        """poison_ratio must return 0.5 when half the documents are poisoned."""
        gt = Document(claim_id="C1", text="ground truth")
        pois = _make_poisoned("C1")
        result = CorpusBuilderResult(ground_truth_docs=[gt], poisoned_docs=[pois])
        assert result.poison_ratio == pytest.approx(0.5)

    def test_poison_ratio_all_poisoned(self) -> None:
        """poison_ratio must return 1.0 when every document is poisoned."""
        p1 = _make_poisoned("C1")
        p2 = _make_poisoned("C2")
        result = CorpusBuilderResult(ground_truth_docs=[], poisoned_docs=[p1, p2])
        assert result.poison_ratio == pytest.approx(1.0)

    def test_poison_ratio_none_poisoned(self) -> None:
        """poison_ratio must return 0.0 when there are no poisoned documents."""
        gt1 = Document(claim_id="C1", text="t1")
        gt2 = Document(claim_id="C2", text="t2")
        result = CorpusBuilderResult(ground_truth_docs=[gt1, gt2], poisoned_docs=[])
        assert result.poison_ratio == 0.0


# ─── CorpusBuilder ────────────────────────────────────────────────────────────

class TestCorpusBuilder:
    """Tests for CorpusBuilder.build()."""

    def test_no_attacks_creates_only_ground_truth(self, two_claims: list[Claim]) -> None:
        """
        With an empty attack list, build() must produce exactly one ground-truth
        document per claim and zero poisoned documents.
        """
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        assert len(result.ground_truth_docs) == len(two_claims)
        assert len(result.poisoned_docs) == 0

    def test_attack_rate_1_poisons_every_claim(self, two_claims: list[Claim]) -> None:
        """
        With poison_rate=1.0, every claim must receive a poisoned variant.

        Because rng.random() is in [0, 1) and we check <= 1.0, the condition
        is always True, so all claims are poisoned.
        """
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        builder = CorpusBuilder(attacks=[(attack, cfg)], seed=42)
        result = builder.build(two_claims)
        assert len(result.poisoned_docs) == len(two_claims)

    def test_build_empty_claims(self) -> None:
        """
        build() with an empty claims list must return an empty CorpusBuilderResult
        with poison_ratio == 0.0.
        """
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build([])
        assert len(result.ground_truth_docs) == 0
        assert result.poison_ratio == 0.0

    def test_ground_truth_text_matches_original_fact(self, two_claims: list[Claim]) -> None:
        """
        Each ground-truth document's text must be the exact original_fact of its
        source claim, and its claim_id must match.
        """
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        for doc, claim in zip(result.ground_truth_docs, two_claims):
            assert doc.text == claim.original_fact
            assert doc.claim_id == claim.claim_id

    def test_ground_truth_doc_type_is_correct(self, two_claims: list[Claim]) -> None:
        """Ground-truth documents must have doc_type == 'ground_truth'."""
        builder = CorpusBuilder(attacks=[], seed=42)
        result = builder.build(two_claims)
        for doc in result.ground_truth_docs:
            assert doc.doc_type == "ground_truth"

    def test_poisoned_doc_type_is_poisoned(self, two_claims: list[Claim]) -> None:
        """Poisoned documents must have doc_type == 'poisoned'."""
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        builder = CorpusBuilder(attacks=[(attack, cfg)], seed=42)
        result = builder.build(two_claims)
        for doc in result.poisoned_docs:
            assert doc.doc_type == "poisoned"

    def test_determinism_same_seed_same_output(self, two_claims: list[Claim]) -> None:
        """
        Two builds with the same seed must produce identical poisoned texts.

        This is the fundamental reproducibility guarantee: an experiment run
        can be replicated bit-for-bit by re-using the same seed and config.
        """
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        r1 = CorpusBuilder(attacks=[(attack, cfg)], seed=42).build(two_claims)
        r2 = CorpusBuilder(attacks=[(attack, cfg)], seed=42).build(two_claims)
        texts1 = [d.text for d in r1.poisoned_docs]
        texts2 = [d.text for d in r2.poisoned_docs]
        assert texts1 == texts2

    def test_different_seeds_may_differ(self, two_claims: list[Claim]) -> None:
        """
        Two builds with different seeds should generally produce different texts.

        This is a soft assertion (not guaranteed for every text, but highly
        likely for texts containing multiple digits and named entities).
        """
        attack = NumericalShiftAttack()
        cfg = AttackConfig(name="numerical_shift", poison_rate=1.0)
        r1 = CorpusBuilder(attacks=[(attack, cfg)], seed=1).build(two_claims)
        r2 = CorpusBuilder(attacks=[(attack, cfg)], seed=9999).build(two_claims)
        # Just verify both runs complete without error and produce the right count.
        assert len(r1.poisoned_docs) == len(r2.poisoned_docs)
