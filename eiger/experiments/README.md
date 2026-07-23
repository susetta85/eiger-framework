# eiger.experiments

**Status: `ExperimentRunner` implemented (Sprint 2, Step 6). No CLI entry point / `__main__.py` yet.**

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
already used by `DenseRetriever` and `IngestionPipeline`, and keeps
`ExperimentRunner` trivially testable with mocks.

**No dataset loader yet.** `run()` accepts an already-loaded `list[Claim]`
directly rather than a dataset name — `eiger/datasets/` is still empty (see
`eiger/ingestion/README.md`). Wiring in a `BaseDataset` loader later only
changes the caller, not `ExperimentRunner` itself.

---

## Pipeline steps (`run()`)

```
list[Claim]
    │
    ▼  seed_everything(config.seed)                — reproducibility
    ▼  CorpusBuilder.build(claims)                  — attacks resolved via get_attack()
CorpusBuilderResult
    │
    ▼  IngestionPipeline.ingest(corpus)             — embed + upsert
(vector store populated)
    │
    ▼  for each claim:
    │     DenseRetriever.retrieve()  → RetrievalResult
    │     BaseLLM.build_rag_prompt() + generate()   → GenerationResult
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
`["ragas_answer_correctness"]` to be populated *before* it runs. No real RAGAS
(LLM-judge) integration exists in EIGER yet — RAGAS's faithfulness /
answer_correctness metrics both require an LLM judge wrapped via
`LangchainLLMWrapper`, which would pull in a new, heavy `langchain` +
`langchain-ollama` dependency, and Ollama-as-judge configurations have
documented upstream reliability issues.

`ExperimentRunner` instead exposes an optional `faithfulness_scorer`
constructor argument: any callable `(Claim, GenerationResult) -> dict[str, float]`.
Its return value is merged into each `EvaluationRecord.metrics` before metrics
are computed.

- **`eiger.metrics.EmbeddingFaithfulnessScorer`** (Sprint 2 addition) is a
  ready-to-use, LLM-judge-free proxy: cosine similarity between the answer
  and the retrieved context (faithfulness proxy) / the ground truth
  (correctness proxy), using the same `BaseEmbedder` abstraction already in
  the project. See `eiger/metrics/README.md` for exactly what it does and
  does not capture — **it must be reported as a proxy** ("FFR
  (embedding-similarity proxy)"), not as RAGAS, in any published result. It
  logs a warning once on construction as a reminder of this.
- If `"ffr"` is configured with no scorer at all, `ExperimentRunner` logs a
  warning once per run: faithfulness/correctness would default to 0.0 for
  every record, making the resulting FFR trivially 0.0 — not a valid
  measurement.
- A real RAGAS-based scorer remains future work and can replace
  `EmbeddingFaithfulnessScorer` without any change to `ExperimentRunner`.

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

- [ ] `BaseDataset` integration (load claims from a named dataset instead of
      accepting `list[Claim]` directly)
- [ ] Real RAGAS-based `faithfulness_scorer` (replacing/complementing
      `EmbeddingFaithfulnessScorer`)
- [ ] `__main__.py` CLI entry point (`python -m eiger run <config.yaml>`)
- [ ] Integration tests: full end-to-end run against live Qdrant + Ollama
