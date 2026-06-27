"""
Unit tests for all adversarial attack implementations.

Tests verify:
  - Correct transformation of text content
  - Determinism: same seed → same output
  - Isolation: no global random state mutation
  - Provenance: PoisonedDocument has correct metadata
  - Registry: attacks can be retrieved by name
"""

from __future__ import annotations

import random

import pytest

from eiger.attacks import (
    NumericalShiftAttack,
    AttributionSwitchAttack,
    DateManipulationAttack,
    CausalManipulationAttack,
    get_attack,
    list_attacks,
)
from eiger.core.models import Document, PoisonedDocument

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def base_doc() -> Document:
    return Document(
        doc_id="test-doc-001",
        claim_id="TEST_CLAIM_001",
        text="The WHO reported that inflation rose to 3.5% in 2023 due to supply shocks.",
        doc_type="ground_truth",
    )


SEED = 42


# ─── NumericalShiftAttack ─────────────────────────────────────────────────────

class TestNumericalShiftAttack:
    def test_returns_poisoned_document(self, base_doc: Document) -> None:
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert isinstance(result, PoisonedDocument)

    def test_text_is_modified(self, base_doc: Document) -> None:
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        # At least one digit must have been swapped
        assert result.text != base_doc.text

    def test_original_text_preserved(self, base_doc: Document) -> None:
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.original_text == base_doc.text

    def test_attack_name_set(self, base_doc: Document) -> None:
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.attack_name == "numerical_shift"

    def test_determinism(self, base_doc: Document) -> None:
        attack = NumericalShiftAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_different_seeds_may_differ(self, base_doc: Document) -> None:
        attack = NumericalShiftAttack()
        r1 = attack.apply(base_doc, seed=1)
        r2 = attack.apply(base_doc, seed=9999)
        # Not guaranteed to differ for every text, but very likely for a text with multiple numbers
        # This is a probabilistic assertion; we just check neither crashes.
        assert isinstance(r1.text, str)
        assert isinstance(r2.text, str)

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        """Applying the attack must not change Python's global random state."""
        random.seed(0)
        before = random.random()
        random.seed(0)  # Reset to same state
        NumericalShiftAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after, "Attack must not mutate global random state"

    def test_annotation_present(self, base_doc: Document) -> None:
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.annotation is not None
        assert 1.0 <= result.annotation.plausibility <= 5.0


# ─── AttributionSwitchAttack ──────────────────────────────────────────────────

class TestAttributionSwitchAttack:
    def test_replaces_who(self, base_doc: Document) -> None:
        attack = AttributionSwitchAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert "WHO" not in result.text

    def test_original_preserved(self, base_doc: Document) -> None:
        attack = AttributionSwitchAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.original_text == base_doc.text

    def test_custom_entity_map(self, base_doc: Document) -> None:
        attack = AttributionSwitchAttack()
        custom_map = {"WHO": "A random blog"}
        result = attack.apply(base_doc, seed=SEED, entity_map=custom_map)
        assert "A random blog" in result.text

    def test_determinism(self, base_doc: Document) -> None:
        attack = AttributionSwitchAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        random.seed(0)
        before = random.random()
        random.seed(0)
        AttributionSwitchAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after


# ─── DateManipulationAttack ───────────────────────────────────────────────────

class TestDateManipulationAttack:
    def test_shifts_year(self, base_doc: Document) -> None:
        attack = DateManipulationAttack()
        result = attack.apply(base_doc, seed=SEED, direction="past")
        # Original has "2023"; result should NOT have "2023" unchanged
        # (it may still appear if no valid shift is made, but shift > 0 guarantees change)
        assert "2023" not in result.text or result.text != base_doc.text

    def test_determinism(self, base_doc: Document) -> None:
        attack = DateManipulationAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_attack_name(self, base_doc: Document) -> None:
        result = DateManipulationAttack().apply(base_doc, seed=SEED)
        assert result.attack_name == "date_manipulation"

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        random.seed(0)
        before = random.random()
        random.seed(0)
        DateManipulationAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after


# ─── CausalManipulationAttack ─────────────────────────────────────────────────

class TestCausalManipulationAttack:
    def test_text_is_longer(self, base_doc: Document) -> None:
        attack = CausalManipulationAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert len(result.text) > len(base_doc.text)

    def test_determinism(self, base_doc: Document) -> None:
        attack = CausalManipulationAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_custom_injections(self, base_doc: Document) -> None:
        attack = CausalManipulationAttack()
        custom = ["due to a secret government policy"]
        result = attack.apply(base_doc, seed=SEED, causal_injections=custom)
        assert "secret government policy" in result.text

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        random.seed(0)
        before = random.random()
        random.seed(0)
        CausalManipulationAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after


# ─── Registry ─────────────────────────────────────────────────────────────────

class TestAttackRegistry:
    def test_all_builtin_attacks_registered(self) -> None:
        registered = list_attacks()
        assert "numerical_shift" in registered
        assert "attribution_switch" in registered
        assert "date_manipulation" in registered
        assert "causal_manipulation" in registered

    def test_get_attack_returns_instance(self) -> None:
        attack = get_attack("numerical_shift")
        assert isinstance(attack, NumericalShiftAttack)

    def test_get_unknown_attack_raises(self) -> None:
        from eiger.core.exceptions import AttackNotFoundError
        with pytest.raises(AttackNotFoundError):
            get_attack("nonexistent_attack_xyz")
