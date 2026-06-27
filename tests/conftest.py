"""Shared pytest fixtures for the EIGER test suite."""

from __future__ import annotations

import pytest

from eiger.core.models import Claim, Document


@pytest.fixture(scope="session")
def sample_claims() -> list[Claim]:
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


@pytest.fixture
def sample_document() -> Document:
    return Document(
        doc_id="doc-fixture-001",
        claim_id="TEST_001",
        text="The WHO reported that inflation rose to 3.5% in 2023 due to supply shocks.",
        doc_type="ground_truth",
    )
