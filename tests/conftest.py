"""
Shared pytest fixtures for the EIGER test suite.

Fixtures defined here are automatically available to every test module in the
``tests/`` directory tree without explicit imports, because pytest discovers
conftest.py files automatically at collection time.

Scope conventions used in this file:
  - ``scope="session"``: fixture is created once per test session and shared
    across all tests. Use for expensive-to-build objects that are read-only.
  - (default) ``scope="function"``: fixture is recreated for every test function
    that requests it, ensuring test isolation.

What this file does NOT do:
  - Start external services (Qdrant, LLM APIs). Integration tests that need
    those dependencies should define their own fixtures in a separate conftest.
  - Define mock/patch fixtures; those live in the test modules that use them.
"""

from __future__ import annotations

import pytest

from eiger.core.models import Claim, Document


# ─── Session-scoped fixtures ──────────────────────────────────────────────────
# These are expensive to create (even if trivially fast now) and are logically
# shared across all tests, so ``scope="session"`` prevents redundant construction.

@pytest.fixture(scope="session")
def sample_claims() -> list[Claim]:
    """
    Provide a minimal set of representative claims for use across the test suite.

    Returns two ``Claim`` objects sourced from the ``test_fixture`` pseudo-dataset:
      - ``TEST_001``: a claim about WHO inflation statistics (contains a number
        and an organisation name, making it suitable for NumericalShift and
        AttributionSwitch attack tests).
      - ``TEST_002``: a claim about a NASA mission (tests attribution switching
        to ESA and date manipulation on the launch year).

    The ``source_dataset`` field is set to ``"test_fixture"`` so that pipeline
    components that filter by dataset do not confuse these with real EIBench data.

    Returns:
        List of two ``Claim`` objects that exercise different attack surface areas.
    """
    return [
        Claim(
            claim_id="TEST_001",
            original_fact="The WHO reported that inflation rose to 3.5% in 2023.",
            context_query="What did the WHO report about 2023 inflation?",
            source_dataset="test_fixture",
        ),
        Claim(
            claim_id="TEST_002",
            original_fact="NASA confirmed the Mars mission launched in July 2020.",
            context_query="When did NASA launch the Mars mission?",
            source_dataset="test_fixture",
        ),
    ]


# ─── Function-scoped fixtures ─────────────────────────────────────────────────
# Recreated per test to guarantee no state leaks between tests that may
# modify the returned object.

@pytest.fixture
def sample_document() -> Document:
    """
    Provide a single ground-truth ``Document`` for use in unit tests.

    The document text is the elaborated version of claim ``TEST_001`` with an
    added causal phrase ("due to supply shocks") so that
    ``CausalManipulationAttack`` tests have a realistic baseline to compare
    against.

    Returns:
        A ``Document`` with ``doc_type="ground_truth"`` linked to claim ``TEST_001``.
    """
    return Document(
        doc_id="doc-fixture-001",
        claim_id="TEST_001",
        text="The WHO reported that inflation rose to 3.5% in 2023 due to supply shocks.",
        doc_type="ground_truth",
    )
