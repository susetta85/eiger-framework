"""
End-to-end integration test against REAL infrastructure: Qdrant, Ollama, and
a real sentence-transformers model.

Unlike test_pipeline_fake_infra.py (always runs, no external dependencies),
this test exercises the actual production stack: SentenceTransformerEmbedder,
QdrantVectorStore, and OllamaLLM, wired together exactly as ExperimentRunner
expects them to be used in a real experiment run.

Requirements to actually execute this test (rather than skip it):
  - Qdrant reachable at EIGER_QDRANT_HOST:EIGER_QDRANT_PORT — `make up`
  - Ollama reachable at EIGER_OLLAMA_HOST:EIGER_OLLAMA_PORT, with _MODEL_NAME
    already pulled: `docker exec eiger-ollama ollama pull llama3.1:8b`
  - `sentence-transformers` installed (downloads ~22MB on first use)

If either service is unreachable, the test SKIPS (not fails), so `pytest
tests/` remains safe to run without Docker — matching the project's
convention that only tests/unit/ is required for CI / the 100% coverage gate
(see pyproject.toml: addopts targets `--cov=eiger`, and Makefile's
`test-unit` vs `test-integration` targets).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from eiger.config import get_settings
from eiger.core.models import (
    AttackConfig,
    Claim,
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    LLMConfig,
    RetrieverConfig,
)
from eiger.experiments import ExperimentRunner
from eiger.llm import OllamaLLM
from eiger.metrics import EmbeddingFaithfulnessScorer
from eiger.retrieval import SentenceTransformerEmbedder
from eiger.vector_stores import QdrantVectorStore

# Must already be pulled on the target Ollama server for this test to pass
# once infra is up (see module docstring).
_MODEL_NAME = "llama3.1:8b"


def _port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def _require_live_infra() -> None:
    """
    Skip the module's tests if Qdrant or Ollama are not reachable.

    Checked once per module (not per test) since the reachability check
    itself has a real (if short) network timeout cost.
    """
    settings = get_settings()
    qdrant_up = _port_open(settings.qdrant_host, settings.qdrant_port)
    ollama_up = _port_open(settings.ollama_host, settings.ollama_port)
    if not (qdrant_up and ollama_up):
        pytest.skip(
            "Qdrant and/or Ollama not reachable at the configured "
            f"EIGER_QDRANT_HOST/PORT and EIGER_OLLAMA_HOST/PORT. Run `make up` "
            f"and `docker exec eiger-ollama ollama pull {_MODEL_NAME}` to "
            "exercise this test against real infrastructure."
        )


def test_full_pipeline_end_to_end_against_live_services(
    _require_live_infra: None, tmp_path: Path
) -> None:
    """
    Corpus -> ingestion -> retrieval -> generation -> metrics -> ExperimentResult,
    against a real Qdrant server, a real Ollama server, and a real
    sentence-transformers embedding model.
    """
    settings = get_settings()

    claim = Claim(
        claim_id="INTEG_LIVE_001",
        original_fact="The WHO reported that inflation rose to 3.5% in 2023.",
        context_query="What did the WHO report about 2023 inflation?",
        source_dataset="integration_fixture",
    )

    embedder = SentenceTransformerEmbedder()
    config = ExperimentConfig(
        dataset=DatasetConfig(name="integration_fixture"),
        attacks=[AttackConfig(name="numerical_shift", poison_rate=1.0)],
        retriever=RetrieverConfig(collection_name="eiger_integration_live", top_k=5),
        llm=LLMConfig(model=_MODEL_NAME, temperature=0.0, max_tokens=128),
        metrics=["ffr", "ers"],
        output_dir=str(tmp_path),
    )

    runner = ExperimentRunner(
        config=config,
        embedder=embedder,
        vector_store=QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port),
        llm=OllamaLLM(
            model_name=_MODEL_NAME, host=settings.ollama_host, port=settings.ollama_port
        ),
        faithfulness_scorer=EmbeddingFaithfulnessScorer(embedder),
    )

    result = runner.run([claim])

    assert isinstance(result, ExperimentResult)
    assert len(result.records) == 1

    record = result.records[0]
    # Corpus had 2 documents (1 ground-truth + 1 poisoned, poison_rate=1.0);
    # top_k=5 must retrieve both.
    assert len(record.retrieval.hits) == 2
    assert record.retrieval.contains_poisoned is True

    # A real LLM must produce some non-empty text.
    assert record.generation.answer.strip() != ""

    assert set(result.aggregate_metrics.keys()) == {"ffr", "ers"}
    assert 0.0 <= result.aggregate_metrics["ffr"] <= 1.0
    assert 0.0 <= result.aggregate_metrics["ers"] <= 1.0

    assert (tmp_path / "results.json").exists()
