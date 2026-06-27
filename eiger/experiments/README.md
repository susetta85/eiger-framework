# eiger.experiments

**Status: Not yet implemented — planned for Sprint 3.**

This module will provide the orchestration layer that wires all EIGER components
together and executes a complete evaluation run from a single config file.

---

## Planned Components

**`PipelineOrchestrator`** — Constructs all pipeline components (dataset loader,
attack instances, embedder, vector store, retriever, LLM backend, metrics) from
an `ExperimentConfig` using dependency injection. No component is instantiated
outside the orchestrator; this makes the pipeline fully testable and replaceable.

**`ExperimentRunner`** — Iterates over the loaded dataset, applies the configured
attacks to build the poisoned corpus, ingests documents into the vector store,
runs retrieval and generation for each claim, computes all configured metrics,
and collects the results into `EvaluationRecord` objects.

---

## Data Flow

```
ExperimentConfig (YAML)
        |
        v
PipelineOrchestrator
  |- DatasetLoader      --> list[Claim]
  |- AttackInstances    --> list[PoisonedDocument]
  |- Embedder + VectorStore --> populated corpus
  |- Retriever          --> RetrievalResult per claim
  |- LLM backend        --> GenerationResult per claim
  |- Metrics            --> MetricScore per claim
        |
        v
ExperimentRunner
  |- iterates over claims
  |- collects EvaluationRecord per claim
        |
        v
ExperimentResult
  |- aggregate_metrics
  |- full records list
  |- config_hash (provenance fingerprint)
  |- git_commit, timestamp, environment
        |
        v
output_dir/results.json
output_dir/config.json
```

---

## Entry Point

The experiment runner is invoked via the `eiger` package entry point:

```bash
python -m eiger run experiments/baseline_v1.yaml

# Or via Make
make run CFG=experiments/baseline_v1.yaml
```

The `__main__.py` module parses the YAML config into an `ExperimentConfig`,
constructs a `PipelineOrchestrator`, and hands control to `ExperimentRunner`.

---

## Output Directory Structure

```
results/<experiment_id>/
    results.json    # ExperimentResult: all EvaluationRecords + aggregate metrics
    config.json     # Resolved ExperimentConfig snapshot (provenance)
```

`results.json` includes the `config_hash` fingerprint (SHA-256 of the resolved
config, excluding `experiment_id`) so that any two result files can be compared
for configuration equivalence without inspecting the full config.

---

## Sprint 3 Milestone

- [ ] `PipelineOrchestrator` — dependency injection from `ExperimentConfig`
- [ ] `ExperimentRunner` — claim iteration, corpus construction, metric collection
- [ ] `__main__.py` entry point (`python -m eiger run <config.yaml>`)
- [ ] JSON serialization of `ExperimentResult` to `output_dir/`
- [ ] Unit tests: orchestrator wiring, runner iteration, output schema
- [ ] Integration tests: full end-to-end run against live services
