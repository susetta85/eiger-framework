"""
Unit tests for ExperimentRunner (eiger.experiments.runner).

Tests verify:
  - __init__ stores config/embedder/vector_store/llm/faithfulness_scorer
  - collection and top_k are derived from config.retriever
  - ingestion_pipeline and retriever are built with the shared embedder/
    vector_store/collection
  - run() resolves configured attacks via the real attack registry and
    builds ground-truth + poisoned documents accordingly
  - run() raises AttackNotFoundError for an unregistered attack name
  - run() ingests the corpus (reset_collection + upsert called)
  - run() retrieves context and calls build_rag_prompt()/generate() per claim
    with the configured temperature/max_tokens
  - run() assembles GenerationResult/EvaluationRecord correctly
  - run() computes configured metrics via the real metric registry and
    writes per-record scores back into record.metrics
  - run() raises MetricNotFoundError for an unregistered metric name
  - the faithfulness_scorer hook's output is merged into record.metrics
    before FFR is computed
  - a warning is logged when "ffr" is configured without a faithfulness_scorer
  - _get_git_commit() returns a SHA on success and "unknown" on failure
  - _capture_environment() returns python_version/platform keys
  - run(save=True) writes {output_dir}/results.json; run(save=False) does not
  - an empty claims list produces a valid, empty ExperimentResult

What these tests do NOT cover:
  - A real running Ollama/Qdrant/sentence-transformers stack (covered by
    each component's own unit tests and by integration tests).
  - SourceIntegrityMetric's NLI inference (covered in test_source_integrity.py;
    here it is exercised only via its documented transformers-missing fallback).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.exceptions import AttackNotFoundError, MetricNotFoundError
from eiger.core.interfaces import BaseEmbedder, BaseLLM, BaseVectorStore
from eiger.core.models import (
    AttackConfig,
    Claim,
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    GenerationResult,
    LLMConfig,
    RetrieverConfig,
)
from eiger.experiments.runner import ExperimentRunner

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """
    Patch the module-level structlog logger for every test in this file.

    Matches the pattern used in test_retriever.py / test_pipeline.py /
    test_ollama.py to avoid a structlog version quirk (PrintLogger has no
    .name) that surfaces whenever configure_logging() has not been called.
    """
    with patch("eiger.experiments.runner.log"):
        yield


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_claim(claim_id: str = "C1", text: str = "Inflation rose to 3.5% in 2023.") -> Claim:
    """Return a minimal Claim for runner tests."""
    return Claim(
        claim_id=claim_id,
        original_fact=text,
        context_query=f"What happened in {claim_id}?",
        source_dataset="test_fixture",
    )


def _make_config(
    tmp_path: Path,
    attacks: list[AttackConfig] | None = None,
    metrics: list[str] | None = None,
    top_k: int = 5,
    collection_name: str = "eiger_corpus",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> ExperimentConfig:
    """Build a minimal, valid ExperimentConfig for runner tests."""
    return ExperimentConfig(
        seed=42,
        dataset=DatasetConfig(name="test_fixture"),
        attacks=attacks if attacks is not None else [],
        retriever=RetrieverConfig(top_k=top_k, collection_name=collection_name),
        llm=LLMConfig(temperature=temperature, max_tokens=max_tokens),
        metrics=metrics if metrics is not None else [],
        output_dir=str(tmp_path),
    )


def _make_runner(
    tmp_path: Path,
    embedding_dim: int = 8,
    search_results: list[dict] | None = None,
    llm_answer: str = "the answer",
    **config_kwargs: object,
) -> tuple[ExperimentRunner, MagicMock, MagicMock, MagicMock]:
    """
    Build an ExperimentRunner wired to mock embedder/vector_store/llm.

    Returns:
        (runner, mock_embedder, mock_vector_store, mock_llm)
    """
    mock_embedder = MagicMock(spec=BaseEmbedder)
    mock_embedder.embedding_dim = embedding_dim
    mock_embedder.encode.side_effect = lambda texts: [[0.1] * embedding_dim for _ in texts]

    mock_vector_store = MagicMock(spec=BaseVectorStore)
    mock_vector_store.search.return_value = (
        search_results if search_results is not None else []
    )

    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.model_name = "mock-llm"
    mock_llm.build_rag_prompt.side_effect = (
        lambda query, context_docs: f"PROMPT[{query}|{len(context_docs)} docs]"
    )
    mock_llm.generate.return_value = llm_answer

    config = _make_config(tmp_path, **config_kwargs)  # type: ignore[arg-type]
    runner = ExperimentRunner(
        config=config, embedder=mock_embedder, vector_store=mock_vector_store, llm=mock_llm
    )
    return runner, mock_embedder, mock_vector_store, mock_llm


def _make_search_hit(doc_id: str = "gt_C1", claim_id: str = "C1", text: str = "ctx") -> dict:
    """Build a raw search hit as returned by BaseVectorStore.search()."""
    return {
        "doc_id": doc_id,
        "score": 0.9,
        "payload": {
            "doc_id": doc_id,
            "claim_id": claim_id,
            "text": text,
            "doc_type": "ground_truth",
        },
    }


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestExperimentRunnerInit:
    """Tests for __init__ attribute storage and internal wiring."""

    def test_stores_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        assert runner.config.dataset.name == "test_fixture"

    def test_stores_embedder_vector_store_llm(self, tmp_path: Path) -> None:
        runner, mock_embedder, mock_vector_store, mock_llm = _make_runner(tmp_path)
        assert runner.embedder is mock_embedder
        assert runner.vector_store is mock_vector_store
        assert runner.llm is mock_llm

    def test_faithfulness_scorer_defaults_to_none(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        assert runner.faithfulness_scorer is None

    def test_collection_from_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, collection_name="my_collection")
        assert runner.collection == "my_collection"

    def test_top_k_from_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, top_k=9)
        assert runner.top_k == 9

    def test_ingestion_pipeline_shares_embedder_and_store(self, tmp_path: Path) -> None:
        runner, mock_embedder, mock_vector_store, _ = _make_runner(tmp_path)
        assert runner.ingestion_pipeline.embedder is mock_embedder
        assert runner.ingestion_pipeline.vector_store is mock_vector_store
        assert runner.ingestion_pipeline.collection == runner.collection

    def test_retriever_shares_embedder_and_store(self, tmp_path: Path) -> None:
        runner, mock_embedder, mock_vector_store, _ = _make_runner(tmp_path)
        assert runner.retriever.embedder is mock_embedder
        assert runner.retriever.vector_store is mock_vector_store
        assert runner.retriever.collection == runner.collection


# ─── run() — corpus / attack resolution ───────────────────────────────────────

class TestRunCorpusBuilding:
    """Tests for attack resolution and corpus construction in run()."""

    def test_no_attacks_produces_only_ground_truth(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(tmp_path)
        claims = [_make_claim("C1"), _make_claim("C2")]
        runner.run(claims, save=False)
        # upsert receives all_documents; with no attacks, that's one
        # ground-truth doc per claim.
        args, _ = mock_vector_store.upsert.call_args
        assert len(args[1]) == 2

    def test_registered_attack_produces_poisoned_documents(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(
            tmp_path,
            attacks=[AttackConfig(name="numerical_shift", poison_rate=1.0)],
        )
        claims = [_make_claim("C1", text="Inflation rose to 3.5% in 2023.")]
        runner.run(claims, save=False)
        args, _ = mock_vector_store.upsert.call_args
        # 1 ground-truth + 1 poisoned (poison_rate=1.0 always poisons).
        assert len(args[1]) == 2

    def test_unregistered_attack_raises(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(
            tmp_path,
            attacks=[AttackConfig(name="does_not_exist", poison_rate=1.0)],
        )
        with pytest.raises(AttackNotFoundError):
            runner.run([_make_claim()], save=False)


# ─── run() — ingestion ─────────────────────────────────────────────────────────

class TestRunIngestion:
    """Tests for the ingestion step of run()."""

    def test_resets_collection(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(tmp_path, embedding_dim=16)
        runner.run([_make_claim()], save=False)
        args, kwargs = mock_vector_store.reset_collection.call_args
        assert args[0] == runner.collection
        assert kwargs["dim"] == 16

    def test_upserts_documents(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(tmp_path)
        runner.run([_make_claim()], save=False)
        mock_vector_store.upsert.assert_called_once()


# ─── run() — retrieval & generation ────────────────────────────────────────────

class TestRunRetrievalAndGeneration:
    """Tests for the per-claim retrieval/generation step of run()."""

    def test_llm_generate_receives_configured_temperature_and_max_tokens(
        self, tmp_path: Path
    ) -> None:
        runner, _, _, mock_llm = _make_runner(tmp_path, temperature=0.42, max_tokens=99)
        runner.run([_make_claim()], save=False)
        _, kwargs = mock_llm.generate.call_args
        assert kwargs["temperature"] == 0.42
        assert kwargs["max_tokens"] == 99

    def test_build_rag_prompt_called_with_query_and_context(self, tmp_path: Path) -> None:
        hit = _make_search_hit(text="retrieved context")
        runner, _, _, mock_llm = _make_runner(tmp_path, search_results=[hit])
        claim = _make_claim("C1")
        runner.run([claim], save=False)
        args, _ = mock_llm.build_rag_prompt.call_args
        assert args[0] == claim.context_query
        assert args[1] == ["retrieved context"]

    def test_generation_result_fields(self, tmp_path: Path) -> None:
        hit = _make_search_hit(text="ctx text")
        runner, _, _, mock_llm = _make_runner(
            tmp_path, search_results=[hit], llm_answer="42% inflation"
        )
        claim = _make_claim("C1")
        result = runner.run([claim], save=False)
        record = result.records[0]
        assert record.generation.claim_id == "C1"
        assert record.generation.query == claim.context_query
        assert record.generation.context_docs == ["ctx text"]
        assert record.generation.answer == "42% inflation"
        assert record.generation.model_name == "mock-llm"
        assert record.generation.metadata["temperature"] == runner.config.llm.temperature
        assert record.generation.metadata["max_tokens"] == runner.config.llm.max_tokens

    def test_one_record_per_claim(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        claims = [_make_claim("C1"), _make_claim("C2"), _make_claim("C3")]
        result = runner.run(claims, save=False)
        assert len(result.records) == 3
        assert [r.claim_id for r in result.records] == ["C1", "C2", "C3"]


# ─── run() — metrics ────────────────────────────────────────────────────────────

class TestRunMetrics:
    """Tests for metric resolution and computation in run()."""

    def test_unregistered_metric_raises(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["does_not_exist"])
        with pytest.raises(MetricNotFoundError):
            runner.run([_make_claim()], save=False)

    def test_ffr_written_back_into_record_metrics(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        result = runner.run([_make_claim()], save=False)
        assert "ffr" in result.records[0].metrics
        assert "ffr" in result.aggregate_metrics

    def test_ffr_defaults_to_zero_without_scorer(self, tmp_path: Path) -> None:
        """
        Without a faithfulness_scorer, faithfulness/correctness default to
        0.0, so FFR (which requires faithfulness > 0.8) must be 0.0.
        """
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        result = runner.run([_make_claim()], save=False)
        assert result.aggregate_metrics["ffr"] == 0.0

    def test_faithfulness_scorer_output_drives_ffr(self, tmp_path: Path) -> None:
        def scorer(claim: Claim, generation: GenerationResult) -> dict[str, float]:
            return {"ragas_faithfulness": 0.95, "ragas_answer_correctness": 0.05}

        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        runner.faithfulness_scorer = scorer
        result = runner.run([_make_claim()], save=False)
        # faithfulness 0.95 > 0.8 and correctness 0.05 < 0.2 => faithful falsehood.
        assert result.aggregate_metrics["ffr"] == 1.0
        assert result.records[0].metrics["ragas_faithfulness"] == 0.95

    def test_warns_when_ffr_configured_without_scorer(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        with patch("eiger.experiments.runner.log") as mock_log:
            runner.run([_make_claim()], save=False)
        mock_log.warning.assert_called_once()

    def test_no_warning_when_scorer_provided(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        runner.faithfulness_scorer = lambda claim, generation: {}
        with patch("eiger.experiments.runner.log") as mock_log:
            runner.run([_make_claim()], save=False)
        mock_log.warning.assert_not_called()

    def test_ers_runs_without_annotations_and_returns_zero(self, tmp_path: Path) -> None:
        """ERS on a corpus with no PoisonAnnotations must gracefully return 0.0."""
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ers"])
        result = runner.run([_make_claim()], save=False)
        assert result.aggregate_metrics["ers"] == 0.0


# ─── run() — result assembly & empty input ────────────────────────────────────

class TestRunResultAssembly:
    """Tests for the ExperimentResult object returned by run()."""

    def test_returns_experiment_result(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        result = runner.run([_make_claim()], save=False)
        assert isinstance(result, ExperimentResult)

    def test_experiment_id_and_config_hash_match_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        result = runner.run([_make_claim()], save=False)
        assert result.experiment_id == runner.config.experiment_id
        assert result.config_hash == runner.config.config_hash

    def test_environment_captured(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        result = runner.run([_make_claim()], save=False)
        assert "python_version" in result.environment
        assert "platform" in result.environment

    def test_empty_claims_produces_empty_but_valid_result(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, mock_llm = _make_runner(tmp_path)
        result = runner.run([], save=False)
        assert result.records == []
        mock_llm.generate.assert_not_called()
        # reset_collection still happens (clean corpus per run), upsert does not
        # (IngestionPipeline skips upsert entirely for an empty document list).
        mock_vector_store.reset_collection.assert_called_once()
        mock_vector_store.upsert.assert_not_called()


# ─── _get_git_commit ───────────────────────────────────────────────────────────

class TestGetGitCommit:
    """Tests for the git commit SHA capture helper."""

    def test_returns_stripped_stdout_on_success(self) -> None:
        fake_completed = MagicMock()
        fake_completed.stdout = "abc123\n"
        with patch("eiger.experiments.runner.subprocess.run", return_value=fake_completed):
            assert ExperimentRunner._get_git_commit() == "abc123"

    def test_returns_unknown_on_failure(self) -> None:
        with patch(
            "eiger.experiments.runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            assert ExperimentRunner._get_git_commit() == "unknown"

    def test_returns_unknown_when_git_missing(self) -> None:
        with patch(
            "eiger.experiments.runner.subprocess.run", side_effect=FileNotFoundError()
        ):
            assert ExperimentRunner._get_git_commit() == "unknown"


# ─── _capture_environment ──────────────────────────────────────────────────────

class TestCaptureEnvironment:
    """Tests for the environment metadata capture helper."""

    def test_contains_expected_keys(self) -> None:
        env = ExperimentRunner._capture_environment()
        assert "python_version" in env
        assert "platform" in env
        assert isinstance(env["python_version"], str)
        assert isinstance(env["platform"], str)


# ─── save_result / run(save=...) ───────────────────────────────────────────────

class TestSaveResult:
    """Tests for result persistence."""

    def test_run_save_true_writes_results_json(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        runner.run([_make_claim()], save=True)
        result_path = tmp_path / "results.json"
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["experiment_id"] == runner.config.experiment_id

    def test_run_save_false_does_not_write(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        runner.run([_make_claim()], save=False)
        assert not (tmp_path / "results.json").exists()

    def test_save_result_creates_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "dir"
        runner, _, _, _ = _make_runner(tmp_path)
        runner.config.output_dir = str(nested)
        result = runner.run([_make_claim()], save=False)
        path = runner.save_result(result)
        assert path == nested / "results.json"
        assert path.exists()
