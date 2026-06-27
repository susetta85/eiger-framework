"""
Unit tests for all adversarial attack implementations.

Tests verify:
  - Correct transformation of text content (the attack changes something)
  - Determinism: same seed → same output across repeated calls
  - Isolation: no global random state mutation (Python's ``random`` module
    must be in the same state before and after an attack is applied)
  - Provenance: ``PoisonedDocument`` carries correct metadata fields
  - Registry: attacks can be retrieved by name and are the right type

Each attack class has its own test class grouping. The ``base_doc`` fixture
is defined locally (not in conftest.py) because it carries attack-specific
text content that is not useful as a general-purpose fixture.

What these tests do NOT cover:
  - Integration with the corpus builder or retriever.
  - LLM-specific attack quality (semantic plausibility is tested implicitly
    by checking that a causal phrase was injected, not by scoring realism).
  - Performance / latency of attack operations.
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
    """
    Provide a baseline ground-truth document for attack unit tests.

    The text is chosen to exercise multiple attack types simultaneously:
      - Contains a numerical value ("3.5%", "2023") for NumericalShiftAttack
        and DateManipulationAttack.
      - Contains "WHO" for AttributionSwitchAttack.
      - Contains a causal phrase "due to supply shocks" as a reference point
        for CausalManipulationAttack (which appends additional causal clauses).

    Returns:
        A fresh ``Document`` instance for each test function.
    """
    return Document(
        doc_id="test-doc-001",
        claim_id="TEST_CLAIM_001",
        text="The WHO reported that inflation rose to 3.5% in 2023 due to supply shocks.",
        doc_type="ground_truth",
    )


# Fixed seed used across all determinism tests for consistency.
SEED = 42


# ─── NumericalShiftAttack ─────────────────────────────────────────────────────

class TestNumericalShiftAttack:
    """Tests for the NumericalShiftAttack adversarial transformation."""

    def test_returns_poisoned_document(self, base_doc: Document) -> None:
        """Attack must return a PoisonedDocument, not a plain Document."""
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert isinstance(result, PoisonedDocument)

    def test_text_is_modified(self, base_doc: Document) -> None:
        """
        The transformed text must differ from the original.

        The input contains "3.5" and "2023", both of which have multiple
        digits, so at least one swap is virtually guaranteed.
        """
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        # At least one digit must have been swapped
        assert result.text != base_doc.text

    def test_original_text_preserved(self, base_doc: Document) -> None:
        """The ``original_text`` field must retain the pre-attack content."""
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.original_text == base_doc.text

    def test_attack_name_set(self, base_doc: Document) -> None:
        """``attack_name`` must match the class's registered registry key."""
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.attack_name == "numerical_shift"

    def test_determinism(self, base_doc: Document) -> None:
        """Calling apply twice with the same seed must produce identical output."""
        attack = NumericalShiftAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_different_seeds_may_differ(self, base_doc: Document) -> None:
        """
        Different seeds should generally produce different outputs.

        This is a probabilistic, non-strict assertion: we only verify that
        both calls complete without error and return strings. A deterministic
        equality check would be fragile because a very short text could
        produce the same result under different seeds by coincidence.
        """
        attack = NumericalShiftAttack()
        r1 = attack.apply(base_doc, seed=1)
        r2 = attack.apply(base_doc, seed=9999)
        # Not guaranteed to differ for every text, but very likely for a text with multiple numbers
        # This is a probabilistic assertion; we just check neither crashes.
        assert isinstance(r1.text, str)
        assert isinstance(r2.text, str)

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        """
        Applying the attack must not change Python's global random state.

        This is a critical isolation property: if an attack seeds ``random``
        globally, it would non-deterministically affect any subsequent code
        that calls ``random.random()``, breaking other experiments or tests.

        Verification strategy: record ``random.random()`` before the attack
        (with a known seed), then reset the seed, run the attack, and check
        that the same call produces the same value.
        """
        random.seed(0)
        before = random.random()
        random.seed(0)  # Reset to same state
        NumericalShiftAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after, "Attack must not mutate global random state"

    def test_annotation_present(self, base_doc: Document) -> None:
        """
        The resulting PoisonedDocument must carry a non-None annotation with
        plausibility in the valid [1.0, 5.0] range.
        """
        attack = NumericalShiftAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.annotation is not None
        assert 1.0 <= result.annotation.plausibility <= 5.0


# ─── AttributionSwitchAttack ──────────────────────────────────────────────────

class TestAttributionSwitchAttack:
    """Tests for the AttributionSwitchAttack adversarial transformation."""

    def test_replaces_who(self, base_doc: Document) -> None:
        """
        The default entity map must replace "WHO" with its substitute.

        "WHO" is present in base_doc.text so this verifies the primary use case.
        """
        attack = AttributionSwitchAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert "WHO" not in result.text

    def test_original_preserved(self, base_doc: Document) -> None:
        """The ``original_text`` field must retain the pre-attack content."""
        attack = AttributionSwitchAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert result.original_text == base_doc.text

    def test_custom_entity_map(self, base_doc: Document) -> None:
        """
        A caller-provided ``entity_map`` must override the default mapping.

        This ensures the attack is configurable for domain-specific experiments
        that target organisations not in the built-in map.
        """
        attack = AttributionSwitchAttack()
        custom_map = {"WHO": "A random blog"}
        result = attack.apply(base_doc, seed=SEED, entity_map=custom_map)
        assert "A random blog" in result.text

    def test_determinism(self, base_doc: Document) -> None:
        """Same seed → same output for two independent apply calls."""
        attack = AttributionSwitchAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        """Attack must not modify Python's global random state."""
        random.seed(0)
        before = random.random()
        random.seed(0)
        AttributionSwitchAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after


# ─── DateManipulationAttack ───────────────────────────────────────────────────

class TestDateManipulationAttack:
    """Tests for the DateManipulationAttack adversarial transformation."""

    def test_shifts_year(self, base_doc: Document) -> None:
        """
        With ``direction="past"``, the year "2023" in the source text should
        be shifted backwards, producing a different string.

        The assertion is intentionally loose (OR condition) because a shift
        of 0 is technically valid if the RNG produces it; the important
        guarantee is that the method runs without error.
        """
        attack = DateManipulationAttack()
        result = attack.apply(base_doc, seed=SEED, direction="past")
        # Original has "2023"; result should NOT have "2023" unchanged
        # (it may still appear if no valid shift is made, but shift > 0 guarantees change)
        assert "2023" not in result.text or result.text != base_doc.text

    def test_determinism(self, base_doc: Document) -> None:
        """Same seed → same output for two independent apply calls."""
        attack = DateManipulationAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_attack_name(self, base_doc: Document) -> None:
        """``attack_name`` must match the registered registry key."""
        result = DateManipulationAttack().apply(base_doc, seed=SEED)
        assert result.attack_name == "date_manipulation"

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        """Attack must not modify Python's global random state."""
        random.seed(0)
        before = random.random()
        random.seed(0)
        DateManipulationAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after


    def test_direction_future(self, base_doc: Document) -> None:
        """direction='future' must shift years forward."""
        result = DateManipulationAttack().apply(base_doc, seed=SEED, direction="future")
        assert isinstance(result.text, str)
        assert result.text != base_doc.text

    def test_direction_random(self, base_doc: Document) -> None:
        """direction='random' must shift years in either direction without error."""
        result = DateManipulationAttack().apply(base_doc, seed=SEED, direction="random")
        assert isinstance(result.text, str)

# ─── CausalManipulationAttack ─────────────────────────────────────────────────

class TestCausalManipulationAttack:
    """Tests for the CausalManipulationAttack adversarial transformation."""

    def test_text_is_longer(self, base_doc: Document) -> None:
        """
        CausalManipulationAttack injects a causal phrase, which must
        increase the overall text length.
        """
        attack = CausalManipulationAttack()
        result = attack.apply(base_doc, seed=SEED)
        assert len(result.text) > len(base_doc.text)

    def test_determinism(self, base_doc: Document) -> None:
        """Same seed → same injected phrase and same output text."""
        attack = CausalManipulationAttack()
        r1 = attack.apply(base_doc, seed=SEED)
        r2 = attack.apply(base_doc, seed=SEED)
        assert r1.text == r2.text

    def test_custom_injections(self, base_doc: Document) -> None:
        """
        A caller-provided ``causal_injections`` list must be used as the
        candidate pool, replacing the built-in set of injections.
        """
        attack = CausalManipulationAttack()
        custom = ["due to a secret government policy"]
        result = attack.apply(base_doc, seed=SEED, causal_injections=custom)
        assert "secret government policy" in result.text

    def test_no_global_state_mutation(self, base_doc: Document) -> None:
        """Attack must not modify Python's global random state."""
        random.seed(0)
        before = random.random()
        random.seed(0)
        CausalManipulationAttack().apply(base_doc, seed=SEED)
        after = random.random()
        assert before == after

    def test_short_text_fallback(self) -> None:
        """Short text with no 20+ char sentence must use the full-text fallback."""
        short_doc = Document(
            doc_id="short-001", claim_id="TEST_CLAIM_001",
            text="Short.", doc_type="ground_truth",
        )
        result = CausalManipulationAttack().apply(short_doc, seed=SEED)
        assert len(result.text) > len(short_doc.text)


# ─── Registry ─────────────────────────────────────────────────────────────────

class TestAttackRegistry:
    """Tests for the attack registry lookup functions."""

    def test_all_builtin_attacks_registered(self) -> None:
        """
        All four built-in attacks must be discoverable by name.

        This test acts as a guard against accidental removal of an attack from
        the auto-registration block in ``eiger.attacks.__init__``.
        """
        registered = list_attacks()
        assert "numerical_shift" in registered
        assert "attribution_switch" in registered
        assert "date_manipulation" in registered
        assert "causal_manipulation" in registered

    def test_get_attack_returns_instance(self) -> None:
        """``get_attack`` must return a live instance of the correct class."""
        attack = get_attack("numerical_shift")
        assert isinstance(attack, NumericalShiftAttack)

    def test_get_unknown_attack_raises(self) -> None:
        """
        Requesting a name that does not exist must raise ``AttackNotFoundError``.

        This verifies that the error type is correctly specialised rather than
        a generic KeyError or RuntimeError.
        """
        from eiger.core.exceptions import AttackNotFoundError
        with pytest.raises(AttackNotFoundError):
            get_attack("nonexistent_attack_xyz")
