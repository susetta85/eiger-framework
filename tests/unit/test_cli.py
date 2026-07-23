"""
Unit tests for the eiger CLI (eiger.__main__).

Tests verify:
  - _load_config: valid YAML -> ExperimentConfig; missing file, invalid
    YAML, non-mapping top-level, and schema-invalid content all raise
    ConfigurationError.
  - _build_dataset: json_fixture with/without a path override; a path
    override on any other dataset name raises ConfigurationError; an
    unregistered name propagates DatasetNotFoundError.
  - _build_runner: unsupported retriever.type/vector_store/llm.backend
    values raise ConfigurationError; a supported config constructs
    SentenceTransformerEmbedder/QdrantVectorStore/OllamaLLM with the
    right arguments and injects them (plus EmbeddingFaithfulnessScorer)
    into ExperimentRunner.
  - _cmd_run: happy path prints a summary and returns 0; an empty claims
    list prints a message and returns 1, without calling ExperimentRunner.
  - _cmd_list_datasets / _cmd_list_attacks / _cmd_list_metrics print one
    name per line and return 0.
  - main(): dispatches argv to the right subcommand; a raised EigerError
    is caught, printed to stderr, and converted to exit code 1 instead of
    propagating as a traceback.

What these tests do NOT cover:
  - Real SentenceTransformerEmbedder/QdrantVectorStore/OllamaLLM behavior
    (covered by their own unit tests) or a real ExperimentRunner.run() call
    (covered by test_runner.py and the integration tests).
  - The `if __name__ == "__main__":` guard itself (marked `# pragma: no
    cover` in eiger/__main__.py; process-entry glue only, main() itself
    is fully covered by calling it directly below).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import eiger.__main__ as cli
from eiger.core.exceptions import ConfigurationError, DatasetNotFoundError
from eiger.core.models import (
    Claim,
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    LLMConfig,
    RetrieverConfig,
)
from eiger.datasets import JSONFixtureDataset

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """
    Patch every module-level `log` object touched transitively by these
    tests (the CLI's own, plus ExperimentRunner's when _build_runner or
    _cmd_run exercises a real ExperimentRunner instance), matching the
    project-wide convention (see test_pipeline.py).
    """
    with patch("eiger.__main__.log"), patch("eiger.experiments.runner.log"):
        yield


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _minimal_config_yaml() -> str:
    return """
dataset:
  name: json_fixture
retriever:
  collection_name: eiger_cli_test
llm:
  model: llama3.1:8b
metrics: []
"""


def _minimal_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(name="json_fixture"),
        retriever=RetrieverConfig(collection_name="eiger_cli_test"),
        llm=LLMConfig(),
        metrics=[],
    )


# ─── _load_config ──────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_valid_yaml_parses_to_experiment_config(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        _write_yaml(path, _minimal_config_yaml())
        config = cli._load_config(path)
        assert isinstance(config, ExperimentConfig)
        assert config.dataset.name == "json_fixture"
        assert config.retriever.collection_name == "eiger_cli_test"

    def test_missing_file_raises_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="Could not read"):
            cli._load_config(tmp_path / "does_not_exist.yaml")

    def test_invalid_yaml_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        _write_yaml(path, "dataset: [unclosed")
        with pytest.raises(ConfigurationError, match="not valid YAML"):
            cli._load_config(path)

    def test_non_mapping_top_level_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        _write_yaml(path, "- one\n- two\n")
        with pytest.raises(ConfigurationError, match="top-level YAML mapping"):
            cli._load_config(path)

    def test_schema_invalid_content_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.yaml"
        # Missing required 'dataset', 'retriever', 'llm' fields.
        _write_yaml(path, "description: incomplete\n")
        with pytest.raises(ConfigurationError, match="Invalid experiment config"):
            cli._load_config(path)


# ─── _build_dataset ─────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_json_fixture_without_path_uses_registry(self) -> None:
        dataset = cli._build_dataset(DatasetConfig(name="json_fixture"))
        assert isinstance(dataset, JSONFixtureDataset)

    def test_json_fixture_with_path_override(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom.json"
        dataset = cli._build_dataset(DatasetConfig(name="json_fixture", path=str(custom)))
        assert isinstance(dataset, JSONFixtureDataset)
        assert dataset.path == Path(custom)

    def test_path_override_on_other_dataset_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="json_fixture"):
            cli._build_dataset(DatasetConfig(name="averitec", path="/tmp/whatever.json"))

    def test_unregistered_dataset_name_raises_dataset_not_found_error(self) -> None:
        with pytest.raises(DatasetNotFoundError):
            cli._build_dataset(DatasetConfig(name="nonexistent_dataset_xyz"))


# ─── _build_runner ──────────────────────────────────────────────────────────────

class TestBuildRunner:
    def test_unsupported_retriever_type_raises(self) -> None:
        config = _minimal_config()
        config.retriever.type = "sparse"
        with pytest.raises(ConfigurationError, match="retriever.type"):
            cli._build_runner(config)

    def test_unsupported_vector_store_raises(self) -> None:
        config = _minimal_config()
        config.retriever.vector_store = "faiss"
        with pytest.raises(ConfigurationError, match="vector_store"):
            cli._build_runner(config)

    def test_unsupported_llm_backend_raises(self) -> None:
        config = _minimal_config()
        config.llm.backend = "openai"
        with pytest.raises(ConfigurationError, match="llm.backend"):
            cli._build_runner(config)

    def test_supported_config_builds_runner_with_expected_components(self) -> None:
        config = _minimal_config()

        with (
            patch("eiger.__main__.SentenceTransformerEmbedder") as mock_embedder_cls,
            patch("eiger.__main__.QdrantVectorStore") as mock_store_cls,
            patch("eiger.__main__.OllamaLLM") as mock_llm_cls,
            patch("eiger.__main__.EmbeddingFaithfulnessScorer") as mock_scorer_cls,
            patch("eiger.__main__.ExperimentRunner") as mock_runner_cls,
        ):
            mock_embedder = mock_embedder_cls.return_value
            runner = cli._build_runner(config)

            mock_embedder_cls.assert_called_once_with(
                model_name=config.retriever.embedder
            )
            mock_store_cls.assert_called_once()
            mock_llm_cls.assert_called_once_with(
                model_name=config.llm.model,
                host="localhost",
                port=11434,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )
            mock_scorer_cls.assert_called_once_with(mock_embedder)
            mock_runner_cls.assert_called_once_with(
                config=config,
                embedder=mock_embedder,
                vector_store=mock_store_cls.return_value,
                llm=mock_llm_cls.return_value,
                faithfulness_scorer=mock_scorer_cls.return_value,
            )
            assert runner is mock_runner_cls.return_value


# ─── _cmd_run ───────────────────────────────────────────────────────────────────

class TestCmdRun:
    def test_happy_path_prints_summary_and_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, _minimal_config_yaml())

        claim = Claim(
            claim_id="C1", original_fact="fact", context_query="query?",
            source_dataset="json_fixture",
        )
        fake_result = ExperimentResult(
            experiment_id="exp_test01",
            config_hash="abc123",
            config=_minimal_config(),
            records=[],
            aggregate_metrics={"ffr": 0.25},
        )

        with (
            patch.object(cli, "_build_dataset") as mock_build_dataset,
            patch.object(cli, "_build_runner") as mock_build_runner,
        ):
            mock_build_dataset.return_value.load.return_value = [claim]
            mock_build_runner.return_value.run.return_value = fake_result

            args = cli._build_parser().parse_args(["run", str(config_path)])
            exit_code = cli._cmd_run(args)

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "exp_test01" in out
        assert "ffr: 0.2500" in out
        assert "results.json" in out

    def test_empty_claims_prints_message_and_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "config.yaml"
        _write_yaml(config_path, _minimal_config_yaml())

        with (
            patch.object(cli, "_build_dataset") as mock_build_dataset,
            patch.object(cli, "_build_runner") as mock_build_runner,
        ):
            mock_build_dataset.return_value.load.return_value = []

            args = cli._build_parser().parse_args(["run", str(config_path)])
            exit_code = cli._cmd_run(args)

        assert exit_code == 1
        assert "nothing to run" in capsys.readouterr().out
        mock_build_runner.assert_not_called()


# ─── list-* commands ────────────────────────────────────────────────────────────

class TestListCommands:
    def test_list_datasets_prints_registered_names(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = cli._cmd_list_datasets(MagicMock())
        assert exit_code == 0
        assert "json_fixture" in capsys.readouterr().out.splitlines()

    def test_list_attacks_prints_registered_names(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = cli._cmd_list_attacks(MagicMock())
        assert exit_code == 0
        assert "numerical_shift" in capsys.readouterr().out.splitlines()

    def test_list_metrics_prints_registered_names(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = cli._cmd_list_metrics(MagicMock())
        assert exit_code == 0
        assert "ffr" in capsys.readouterr().out.splitlines()


# ─── main() ─────────────────────────────────────────────────────────────────────

class TestMain:
    def test_main_dispatches_list_datasets(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = cli.main(["list-datasets"])
        assert exit_code == 0
        assert "json_fixture" in capsys.readouterr().out.splitlines()

    def test_main_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            cli.main([])

    def test_main_converts_eiger_error_to_exit_code_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "nope.yaml"
        exit_code = cli.main(["run", str(missing)])
        assert exit_code == 1
        assert "Error:" in capsys.readouterr().err
