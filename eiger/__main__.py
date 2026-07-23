"""
EIGER command-line interface.

Provides the ``eiger`` console script (registered via ``[project.scripts]``
in pyproject.toml as ``eiger = "eiger.__main__:main"``) and ``python -m eiger``
execution. Before this module, there was no way to actually run an
experiment except by writing a Python script that manually constructed
every component (embedder, vector store, LLM, dataset) — this closes that
gap, fulfilling a console-script entry that pyproject.toml already declared.

Commands
--------
  eiger run <config.yaml>     Run a full experiment from a YAML config file.
  eiger list-datasets         List registered dataset names.
  eiger list-attacks          List registered attack names.
  eiger list-metrics          List registered metric names.

Design decisions
-----------------
- **Component factory lives here, not in ExperimentRunner**: ExperimentRunner
  is deliberately constructed with already-instantiated embedder/vector_store/
  llm objects (dependency injection — see its own module docstring). This
  CLI is the "caller" that docstring refers to: it is responsible for
  resolving ``config.retriever`` / ``config.llm`` string fields into concrete
  instances and injecting them.
- **Only currently-implemented backends are supported**: ``retriever.type``
  must be "dense", ``retriever.vector_store`` must be "qdrant", and
  ``llm.backend`` must be "ollama" — matching what actually exists in
  eiger.retrieval / eiger.vector_stores / eiger.llm today. Requesting any
  other value (e.g. the "sparse"/"hybrid"/"openai" values the config models
  already accept for forward-compatibility) raises ConfigurationError with
  an explicit, actionable message rather than failing confusingly deeper in
  the stack.
- **EmbeddingFaithfulnessScorer is always wired in**: it is dependency-free
  and safe by construction (see its own module docstring), so there is no
  reason to make the CLI user opt in explicitly. It is still just a proxy,
  not real RAGAS — this is unchanged from ExperimentRunner's own documented
  limitation and is not hidden from the printed summary.
- **dataset.path override is currently JSONFixtureDataset-only**: the
  dataset registry's ``get_dataset(name)`` always uses the class's default
  constructor (no arguments), so a path override can only be honored for
  the one dataset class whose constructor actually accepts a path. This is
  a real, temporary limitation of the registry's convenience path — it does
  not exist for AVeriTeC/PolitiFact/FactCheck.org because those loaders are
  not implemented yet.

What this module does NOT do
-----------------------------
- It does not implement retrieval, generation, or metric logic itself; it
  only resolves config into component instances and delegates entirely to
  ExperimentRunner.
- It does not support multiple experiments per invocation, resuming a
  partially-completed run, or any output format beyond the
  ``results.json`` that ExperimentRunner itself writes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from eiger.attacks import list_attacks
from eiger.config import get_settings
from eiger.core.exceptions import ConfigurationError, EigerError
from eiger.core.interfaces import BaseDataset
from eiger.core.models import DatasetConfig, ExperimentConfig
from eiger.datasets import JSONFixtureDataset, get_dataset, list_datasets
from eiger.experiments import ExperimentRunner
from eiger.llm import OllamaLLM
from eiger.metrics import EmbeddingFaithfulnessScorer, list_metrics
from eiger.retrieval import SentenceTransformerEmbedder
from eiger.utils.logging import get_logger
from eiger.vector_stores import QdrantVectorStore

log = get_logger(__name__)

# Only these values are backed by a real implementation today. Kept as
# explicit sets (rather than inferring from a factory dict) so the error
# message can list every supported option without extra bookkeeping.
_SUPPORTED_RETRIEVER_TYPES = {"dense"}
_SUPPORTED_VECTOR_STORES = {"qdrant"}
_SUPPORTED_LLM_BACKENDS = {"ollama"}


# ─── Config loading ────────────────────────────────────────────────────────────

def _load_config(path: Path) -> ExperimentConfig:
    """
    Read, parse, and validate a YAML experiment config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A validated ExperimentConfig.

    Raises:
        ConfigurationError: If the file cannot be read, is not valid YAML,
                            is not a top-level mapping, or fails
                            ExperimentConfig's Pydantic validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Could not read config file '{path}': {exc}") from exc

    try:
        raw_data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Config file '{path}' is not valid YAML: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigurationError(
            f"Config file '{path}' must contain a top-level YAML mapping, "
            f"got {type(raw_data).__name__}."
        )

    try:
        # ExperimentConfig.model_validate(yaml_data) is the exact call
        # documented in ExperimentConfig's own docstring as the intended
        # loading path (eiger/core/models.py).
        return ExperimentConfig.model_validate(raw_data)
    except Exception as exc:
        # Pydantic's ValidationError (and any other construction failure)
        # is re-raised as ConfigurationError so callers only need to catch
        # EigerError, matching the rest of the framework's error hierarchy.
        raise ConfigurationError(f"Invalid experiment config in '{path}': {exc}") from exc


# ─── Component factory ─────────────────────────────────────────────────────────

def _build_dataset(dataset_config: DatasetConfig) -> BaseDataset:
    """
    Resolve a DatasetConfig into a concrete, ready-to-load BaseDataset.

    Args:
        dataset_config: The ``dataset`` section of the experiment config.

    Returns:
        A BaseDataset instance ready to call .load() on.

    Raises:
        ConfigurationError: If ``path`` is set for a dataset other than
                            "json_fixture" (see module docstring), or if
                            ``name`` is not registered.
        DatasetNotFoundError: If ``name`` is not registered (propagated
                            from get_dataset()).
    """
    if dataset_config.path is not None:
        if dataset_config.name != "json_fixture":
            raise ConfigurationError(
                f"dataset.path override is currently only supported for the "
                f"'json_fixture' dataset (see eiger/datasets/README.md); "
                f"got name='{dataset_config.name}' with path set."
            )
        return JSONFixtureDataset(path=dataset_config.path)
    return get_dataset(dataset_config.name)


def _build_runner(config: ExperimentConfig) -> ExperimentRunner:
    """
    Resolve an ExperimentConfig into a fully-wired ExperimentRunner.

    Constructs the embedder, vector store, and LLM described by
    ``config.retriever`` / ``config.llm``, using ``EigerSettings`` for
    infrastructure coordinates (host/port), and injects them into a new
    ExperimentRunner alongside an EmbeddingFaithfulnessScorer (see module
    docstring for why it is always attached).

    Args:
        config: The validated experiment configuration.

    Returns:
        A ready-to-run ExperimentRunner.

    Raises:
        ConfigurationError: If ``retriever.type``, ``retriever.vector_store``,
                            or ``llm.backend`` name a value with no
                            implementation yet.
    """
    if config.retriever.type not in _SUPPORTED_RETRIEVER_TYPES:
        raise ConfigurationError(
            f"retriever.type='{config.retriever.type}' is not implemented yet "
            f"(supported: {sorted(_SUPPORTED_RETRIEVER_TYPES)})."
        )
    if config.retriever.vector_store not in _SUPPORTED_VECTOR_STORES:
        raise ConfigurationError(
            f"retriever.vector_store='{config.retriever.vector_store}' is not "
            f"implemented yet (supported: {sorted(_SUPPORTED_VECTOR_STORES)})."
        )
    if config.llm.backend not in _SUPPORTED_LLM_BACKENDS:
        raise ConfigurationError(
            f"llm.backend='{config.llm.backend}' is not implemented yet "
            f"(supported: {sorted(_SUPPORTED_LLM_BACKENDS)})."
        )

    settings = get_settings()

    embedder = SentenceTransformerEmbedder(model_name=config.retriever.embedder)
    vector_store = QdrantVectorStore(host=settings.qdrant_host, port=settings.qdrant_port)
    llm = OllamaLLM(
        model_name=config.llm.model,
        host=settings.ollama_host,
        port=settings.ollama_port,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    return ExperimentRunner(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        faithfulness_scorer=EmbeddingFaithfulnessScorer(embedder),
    )


# ─── Commands ──────────────────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> int:
    """
    Load a config, load claims from its dataset, run the experiment, and
    print a summary to stdout.

    Returns:
        0 on success, 1 if the dataset produced zero claims (nothing to run).
    """
    config = _load_config(Path(args.config))

    dataset = _build_dataset(config.dataset)
    claims = dataset.load(split=config.dataset.split, max_claims=config.dataset.max_claims)
    if not claims:
        print(
            f"No claims loaded from dataset '{config.dataset.name}' "
            f"(split={config.dataset.split!r}) — nothing to run."
        )
        return 1

    runner = _build_runner(config)
    result = runner.run(claims)

    print(f"Experiment '{result.experiment_id}' complete ({len(result.records)} records).")
    for name, value in sorted(result.aggregate_metrics.items()):
        print(f"  {name}: {value:.4f}")
    print(f"Results written to {Path(config.output_dir) / 'results.json'}")
    return 0


def _cmd_list_datasets(_args: argparse.Namespace) -> int:
    """Print every registered dataset name, one per line."""
    for name in list_datasets():
        print(name)
    return 0


def _cmd_list_attacks(_args: argparse.Namespace) -> int:
    """Print every registered attack name, one per line."""
    for name in list_attacks():
        print(name)
    return 0


def _cmd_list_metrics(_args: argparse.Namespace) -> int:
    """Print every registered metric name, one per line."""
    for name in list_metrics():
        print(name)
    return 0


# ─── Argument parsing ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="eiger", description="EIBench: run RAG poisoning experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config file.")
    run_parser.add_argument("config", help="Path to the experiment YAML config file.")
    run_parser.set_defaults(func=_cmd_run)

    datasets_parser = subparsers.add_parser(
        "list-datasets", help="List registered dataset names."
    )
    datasets_parser.set_defaults(func=_cmd_list_datasets)

    attacks_parser = subparsers.add_parser("list-attacks", help="List registered attack names.")
    attacks_parser.set_defaults(func=_cmd_list_attacks)

    metrics_parser = subparsers.add_parser("list-metrics", help="List registered metric names.")
    metrics_parser.set_defaults(func=_cmd_list_metrics)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point: parse arguments, dispatch to the selected subcommand,
    and translate any EigerError into a clean stderr message + exit code 1
    rather than a raw traceback.

    Args:
        argv: Argument list to parse (defaults to sys.argv[1:] via argparse
              when None — this parameter exists so tests can pass an
              explicit argv without touching sys.argv).

    Returns:
        Process exit code: 0 on success, 1 on any EigerError or an empty
        dataset.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # args.func is set via set_defaults(func=...) in _build_parser(), which
    # argparse.Namespace exposes as Any. The explicit annotation here (rather
    # than `return args.func(args)` directly) gives mypy a concrete Callable
    # return type instead of silently propagating Any past main()'s own
    # declared `-> int`.
    dispatch: Callable[[argparse.Namespace], int] = args.func

    try:
        return dispatch(args)
    except EigerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - process-entry glue, see module docstring
    raise SystemExit(main())
