# eiger.experiments

**Status: `ExperimentRunner` implemented (Sprint 2, Step 6). CLI entry point (`eiger` / `python -m eiger`) implemented in Sprint 3 — see `eiger/__main__.py`.**

This module provides the orchestration layer that wires the other Sprint 2
components together and executes a complete evaluation run.

---

## `ExperimentRunner`

```python
from eiger.experiments import ExperimentRunner
from eiger.retrieval import SentenceTransformerEmbedder
from eiger.vector_stores import QdrantVectorStore
from eiger.llm import OllamaLLM
from eiger.metrics import EmbeddingFaithfulnessScorer

embedder = SentenceTransformerEmbedder()

runner = ExperimentRunner(
    config=experiment_config,          # ExperimentConfig
    embedder=embedder,
    vector_store=QdrantVectorStore(),
    llm=OllamaLLM(model_name=experiment_config.llm.model),
    faithfulness_scorer=EmbeddingFaithfulnessScorer(embedder),  # optional — see below
)
result = runner.run(claims)            # list[Claim] -> ExperimentResult
```

**Dependency injection, not a factory.** `ExperimentRunner` is constructed with
already-instantiated `embedder`, `vector_store`, and `llm` objects rather than
building them from `config.retriever.embedder` / `config.retriever.vector_store`
/ `config.llm.backend` internally. There is no `PipelineOrchestrator` and no
component factory yet — those config string fields exist purely for
provenance (serialized into every result file); the caller is responsible for
constructing matching objects and injecting them. This mirrors the DI pattern
already used by `DenseRetriever`/`SparseRetriever` and `IngestionPipeline`, and keeps
`ExperimentRunner` trivially testable with mocks.

**Dataset loading stays outside `ExperimentRunner`.** `run()` accepts an
already-loaded `list[Claim]` directly rather than a dataset name.
`eiger/datasets/` now provides five real loaders (`SnopesDataset`,
`AVeriTecDataset`, `PolitiFactDataset`, `FactCheckDataset`, plus the local
JSON fixture — see `docs/DATASETS.md`), but resolving `ExperimentConfig.dataset`
into a `list[Claim]` is the CLI's job (`eiger/__main__.py`), not
`ExperimentRunner`'s — this keeps the runner trivially testable with an
in-memory claim list regardless of which dataset backs a given run.

---

## Pipeline steps (`run()`)

```
list[Claim]
    │
    ▼  seed_everything(config.seed)                — reproducibility
    ▼  CorpusBuilder.build(claims)                  — attacks resolved via get_attack()
    ▼    (raises ConfigurationError if any attack has excluded_from_benchmark=True
    ▼     and config.allow_non_benchmark_attacks is not True — see eiger/attacks/README.md)
CorpusBuilderResult
    │
    ▼  retriever.type == "dense": IngestionPipeline.ingest(corpus) — embed + upsert
    ▼  retriever.type == "sparse" (Sprint 4): SparseRetriever.fit(corpus.all_documents)
(vector store populated, or BM25 index built — see ExperimentRunner's own docstring)
    │
    ▼  for each claim:
    │     retriever.retrieve()  → RetrievalResult   (Dense- or SparseRetriever)
    │     BaseLLM.build_rag_prompt() + generate()   → GenerationResult
    │     ["pcs" configured AND retrieval.contains_poisoned]:
    │         second build_rag_prompt()+generate(), poisoned hits filtered
    │         out of context_docs → generation.metadata["counterfactual_answer"]
    │     [faithfulness_scorer(claim, generation)]  → optional pre-metric scores
    │   → EvaluationRecord
    │
    ▼  for each name in config.metrics:
    │     BaseMetric.compute_batch(records) → per-record scores (written back into
    │                                          each record.metrics)
    │     BaseMetric.aggregate(scores)       → experiment-level scalar
    │
    ▼  ExperimentResult(records, aggregate_metrics, git_commit, environment, ...)
    ▼  {config.output_dir}/results.json   [if save=True, the default]
```

- **Attacks and metrics resolved via the existing registries**: `config.attacks`
  entries are resolved through `eiger.attacks.get_attack`, `config.metrics`
  names through `eiger.metrics.get_metric`. Unregistered names raise
  `AttackNotFoundError` / `MetricNotFoundError`.
- **Reproducibility**: `seed_everything(config.seed)` is called once at the
  start of `run()`, seeding Python/numpy/torch global RNGs before any
  stochastic operation (attack application). `git_commit` (via
  `git rev-parse HEAD`, resolved relative to this file regardless of the
  caller's cwd) and `environment` (Python version, platform) are populated by
  `ExperimentRunner` itself.
- **Fail loud**: a single claim's retrieval or generation failure aborts the
  whole run (`RetrievalError`/`GenerationError` propagate unchanged) — no
  per-claim error swallowing. Silently skipping a failed claim would silently
  bias aggregate metrics like FFR.
- **Result file**: written to exactly `{config.output_dir}/results.json`
  (matching `ExperimentResult`'s own docstring contract). Give each
  experiment run a distinct `output_dir` (e.g. incorporating
  `experiment_id`) to avoid overwriting a previous run's results.
  `run(save=False)` skips writing entirely; `save_result(result)` can persist
  a result independently at any later point.

---

## FFR and the `faithfulness_scorer` hook

`FFRMetric` needs `EvaluationRecord.metrics["ragas_faithfulness"]` and
`["ragas_answer_correctness"]` to be populated *before* it runs.

`ExperimentRunner` exposes an optional `faithfulness_scorer` constructor
argument: any callable `(Claim, GenerationResult) -> dict[str, float]`.
Its return value is merged into each `EvaluationRecord.metrics` before metrics
are computed. The CLI selects which one to wire in via
`ExperimentConfig.faithfulness_scorer` (see `eiger/__main__.py`):

- **`eiger.metrics.EmbeddingFaithfulnessScorer`** (Sprint 2 addition, CLI
  default — `faithfulness_scorer: "embedding"`) is a ready-to-use,
  LLM-judge-free proxy: cosine similarity between the answer and the
  retrieved context (faithfulness proxy) / the ground truth (correctness
  proxy), using the same `BaseEmbedder` abstraction already in the project.
  See `eiger/metrics/README.md` for exactly what it does and does not
  capture — **it must be reported as a proxy** ("FFR
  (embedding-similarity proxy)"), not as RAGAS, in any published result. It
  logs a warning once on construction as a reminder of this.
- **`eiger.metrics.RAGASFaithfulnessScorer`** (Sprint 5 addition, opt-in —
  `faithfulness_scorer: "ragas"`) is a real RAGAS integration: RAGAS's
  `Faithfulness`/`AnswerCorrectness` metrics with an Ollama-served LLM as
  judge via `ragas.llms.LangchainLLMWrapper`. Requires the pinned `ragas`
  optional-dependency group (`pip install -e ".[ragas]"` — see
  `eiger/metrics/ragas_scorer.py`'s own docstring for the exact versions
  and why they are pinned so tightly). Ollama-as-judge configurations have
  documented upstream reliability issues, so this is a real judge, not a
  proxy, but still not validated against human judgments for EIBench's
  claims — report results as "FFR (RAGAS, `<model>` judge)".
- **`faithfulness_scorer: "none"`** wires in no scorer at all.
- If `"ffr"` is configured with no scorer at all, `ExperimentRunner` logs a
  warning once per run: faithfulness/correctness would default to 0.0 for
  every record, making the resulting FFR trivially 0.0 — not a valid
  measurement.

---

## Configuration

`ExperimentRunner` is driven entirely by `ExperimentConfig` (`eiger.core.models`):

```python
class ExperimentConfig(BaseModel):
    experiment_id: str          # auto-generated (exp_<8 hex chars>) if not set
    seed: int = 42
    dataset: DatasetConfig
    attacks: list[AttackConfig] = []
    retriever: RetrieverConfig
    llm: LLMConfig
    metrics: list[str] = ["ffr", "source_integrity", "ers"]
    faithfulness_scorer: str = "embedding"  # "embedding" | "ragas" | "none" (Sprint 5)
    output_dir: str = "results/"
    description: str = ""
```

`config_hash` (a property on `ExperimentConfig`) is a SHA-256 fingerprint of
the full config excluding `experiment_id`, letting two result files be
compared for configuration equivalence without diffing the whole config.

---

## Output

```
{output_dir}/
    results.json    # ExperimentResult: full records + aggregate_metrics + provenance
```

`results.json` includes `experiment_id`, `config_hash`, `timestamp`,
`git_commit`, the full resolved `config`, every `EvaluationRecord`, and
`aggregate_metrics` — no separate `config.json` snapshot file (the config is
embedded directly in `results.json`).

---

## Test coverage

`tests/unit/test_runner.py` (34 tests) covers construction, attack/metric
resolution (against the real registries), retrieval/generation orchestration,
the `faithfulness_scorer` hook, git-commit/environment capture, and result
persistence, with 100% line coverage — using mocked `embedder`/
`vector_store`/`llm` (no real Qdrant/Ollama/sentence-transformers required).

## Remaining work

- [x] `BaseDataset` integration — the CLI's `_build_dataset()` resolves a
      `DatasetConfig` into a registered loader, calls `.load()`, and passes
      the resulting `list[Claim]` into `ExperimentRunner.run()` (see
      `eiger/__main__.py`). `ExperimentRunner.run()` itself still accepts
      `list[Claim]` directly rather than a `DatasetConfig`, by design —
      this keeps the runner decoupled from dataset resolution.
- [x] Real RAGAS-based `faithfulness_scorer` — `RAGASFaithfulnessScorer`
      (Sprint 5), complementing (not replacing) `EmbeddingFaithfulnessScorer`;
      opt-in via `faithfulness_scorer: "ragas"`.
- [x] `__main__.py` CLI entry point (`python -m eiger run <config.yaml>`) —
      implemented Sprint 3.
- [x] Integration tests: full end-to-end run against live Qdrant + Ollama —
      `tests/integration/test_pipeline_live_infra.py` (skips gracefully if
      either service is unreachable).
- [x] Real RAGAS scoring integration tests —
      `tests/integration/test_ragas_scorer_real.py` (skips gracefully if the
      `ragas` extra isn't installed, and further skips its live-scoring test
      if Ollama isn't reachable — see `eiger/metrics/ragas_scorer.py`).
- [x] PCS (Poisoned Context Sensitivity) metric — `eiger/metrics/pcs.py`,
      requiring the counterfactual-generation support described above
      (`ExperimentRunner._add_counterfactual_generation`); opt-in via
      `"pcs"` in `ExperimentConfig.metrics`.
