# experiments/

This directory contains YAML experiment definitions. Each file is a
fully-specified, reproducible experiment configuration that can be passed
directly to the EIGER runner.

---

## How to Run

```bash
# Via Make (recommended)
make run CFG=experiments/baseline_v1.yaml
make run CFG=experiments/ablation_attacks.yaml

# Directly via the package entry point
python -m eiger run experiments/baseline_v1.yaml
```

Results are written to the `output_dir` specified in the config file
(default: `results/<experiment_id>/`).

---

## YAML Schema Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `experiment_id` | string | Yes | — | Unique run identifier. Must be distinct across all results. Used as the results directory name. |
| `description` | string | No | `""` | Human-readable description of the experiment. |
| `seed` | integer | Yes | `42` | Global random seed. Controls attack sampling, dataset shuffling, and all stochastic components. |
| `dataset.name` | string | Yes | — | Dataset identifier: `averitec`, `politifact`, or `json_fixture`. |
| `dataset.split` | string | No | `"test"` | Dataset split to load. |
| `dataset.max_claims` | integer or null | No | `null` | Cap on number of claims to process. `null` means all claims. |
| `dataset.path` | string or null | No | `null` | Local file path override (used by `json_fixture`). |
| `attacks` | list | No | `[]` | List of attack configurations. Empty list means no poisoning (baseline). |
| `attacks[].name` | string | Yes | — | Attack identifier. Must be registered in the attack registry. |
| `attacks[].poison_rate` | float | Yes | — | Fraction of corpus documents to poison, in [0.0, 1.0]. |
| `attacks[].params` | dict | No | `{}` | Attack-specific hyperparameters. See each attack's documentation. |
| `retriever.type` | string | No | `"dense"` | Retrieval strategy: `dense`, `sparse`, or `hybrid`. |
| `retriever.embedder` | string | No | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace model ID for embedding (dense and hybrid only). |
| `retriever.vector_store` | string | No | `"qdrant"` | Vector store backend: `qdrant`, `chroma`, or `faiss`. |
| `retriever.top_k` | integer | No | `5` | Number of documents to retrieve per query. |
| `retriever.collection_name` | string | No | `"eiger_corpus"` | Named collection in the vector store. Must be unique per experiment. |
| `llm.backend` | string | No | `"ollama"` | LLM backend: `ollama` or `openai`. |
| `llm.model` | string | No | `"llama3.1:8b"` | Model name as recognized by the backend. |
| `llm.temperature` | float | No | `0.0` | Sampling temperature. Set to `0.0` for reproducible results. |
| `llm.max_tokens` | integer | No | `512` | Maximum tokens in the generated response. |
| `metrics` | list of strings | No | `["ffr", "source_integrity", "ers"]` | Metrics to compute. Must be registered in the metric registry. |
| `output_dir` | string | No | `"results/"` | Directory where results are written. |

---

## Existing Experiments

### `baseline_v1.yaml`

Establishes the baseline performance of the RAG pipeline with no adversarial
poisoning. All claims from `eibench_raw_claims.json` are processed using a
dense Qdrant retriever and Llama 3.1 8B at temperature 0.0. The resulting FFR,
Source Integrity, and ERS scores serve as the reference point against which all
attack experiments are compared.

Key settings: `attacks: []`, `poison_rate: 0.0` (implicit), `seed: 42`.

### `ablation_attacks.yaml`

Ablation study over all four attack types at a uniform `poison_rate` of 0.3.
All attacks are applied simultaneously to the same corpus, allowing the
aggregate degradation across attack categories to be measured in a single run.
Uses the same retriever and LLM configuration as the baseline for a direct
comparison.

Attacks included: `numerical_shift`, `attribution_switch`, `date_manipulation`
(shift 1-5 years into the past), `causal_manipulation` (1 injected causal
reversal per document).

---

## Creating a New Experiment

1. Copy an existing YAML file as a starting point.
2. Set a new, unique `experiment_id`. Results are keyed by this identifier;
   reusing an ID will overwrite prior results.
3. Adjust the fields you want to vary (attacks, retriever type, LLM model, etc.).
4. Run with `make run CFG=experiments/<your_file>.yaml`.

---

## Output Structure

Each run produces two files in `results/<experiment_id>/`:

```
results/<experiment_id>/
    results.json    # Full ExperimentResult: all EvaluationRecords,
                    # aggregate metrics, timestamp, git_commit, environment
    config.json     # Resolved ExperimentConfig snapshot
```

**Provenance and reproducibility.** Every `results.json` embeds a `config_hash`
field — a SHA-256 fingerprint of the resolved configuration (excluding
`experiment_id`). Two runs with identical configs will produce the same hash,
making it straightforward to detect accidental configuration drift between runs.
The `git_commit` field records the exact framework version used.
