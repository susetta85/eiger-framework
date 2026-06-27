# eiger.core

The core package has no dependencies on infrastructure. It defines the domain
language of EIBench. All orchestration code, attack implementations, metric
implementations, and experiment runners import from here; nothing in this
package imports from them.

**Rule:** No infrastructure import is allowed in `eiger/core/`. If you need
`qdrant_client`, `transformers`, `torch`, or any network/filesystem library
here, you are in the wrong module.

---

## Contents

| Module | Purpose |
|---|---|
| `models.py` | Pydantic v2 domain models — the data language of the framework |
| `interfaces.py` | Abstract base classes for every extension point |
| `exceptions.py` | Domain-specific exception hierarchy |

---

## Models (`models.py`)

All models are Pydantic v2 `BaseModel` subclasses. They validate on
construction, serialize cleanly to JSON for experiment provenance, and carry
no infrastructure dependencies.

### Dataset layer

| Model | Key fields | Purpose |
|---|---|---|
| `Claim` | `claim_id`, `original_fact`, `context_query`, `source_dataset`, `metadata` | Single fact-checked claim; the fundamental unit of data. Every downstream document originates from a `Claim`. Exposes `content_hash` (SHA-256 of the fact text) for dataset versioning. |
| `Document` | `doc_id`, `claim_id`, `text`, `doc_type`, `metadata` | A document in the vector corpus. `doc_type` is `"ground_truth"` or `"poisoned"`. |
| `PoisonedDocument` | inherits `Document`; adds `attack_name`, `attack_params`, `original_text`, `annotation` | A document adversarially modified by an attack. Carries full poisoning provenance including the unmodified original text. |
| `PoisonAnnotation` | `plausibility`, `verification_difficulty`, `editorial_risk` | Epistemic risk annotation for a poisoned document. All three fields are floats in [1, 5]. Higher values mean higher risk. |

### Retrieval layer

| Model | Key fields | Purpose |
|---|---|---|
| `RetrievedDocument` | `document`, `score`, `rank` | A single retrieval hit with its similarity score and rank position. |
| `RetrievalResult` | `query`, `claim_id`, `hits`, `top_k` | Full output of one retrieval operation. Exposes `contains_poisoned` and `poison_ratio` as computed properties. |

### Generation layer

| Model | Key fields | Purpose |
|---|---|---|
| `GenerationResult` | `claim_id`, `query`, `context_docs`, `answer`, `model_name`, `metadata` | Output from an LLM given a query and retrieved context documents. |

### Evaluation layer

| Model | Key fields | Purpose |
|---|---|---|
| `MetricScore` | `metric_name`, `value`, `metadata` | Result of a single metric computation on a single record. |
| `EvaluationRecord` | `claim_id`, `generation`, `retrieval`, `metrics` | Full evaluation record for one claim. `metrics` is a `dict[str, float]` populated by upstream RAGAS scoring. Exposes `faithfulness_score` and `factual_correctness_score` as convenience properties. |

### Experiment layer

| Model | Key fields | Purpose |
|---|---|---|
| `ExperimentConfig` | `experiment_id`, `seed`, `dataset`, `attacks`, `retriever`, `llm`, `metrics`, `output_dir` | Complete, validated specification for one experiment run. Loaded from YAML; serialized to JSON alongside results for provenance. Exposes `config_hash` (SHA-256 fingerprint) for reproducibility tracking. |
| `DatasetConfig` | `name`, `split`, `max_claims`, `path` | Dataset selection: `averitec`, `politifact`, or `json_fixture`. |
| `AttackConfig` | `name`, `poison_rate`, `params` | Attack selection and per-attack hyperparameters. `name` must be registered in the attack registry. |
| `RetrieverConfig` | `type`, `embedder`, `vector_store`, `top_k`, `collection_name` | Retriever selection: `dense`, `sparse`, or `hybrid`. Defaults to `sentence-transformers/all-MiniLM-L6-v2` over Qdrant. |
| `LLMConfig` | `backend`, `model`, `temperature`, `max_tokens` | LLM backend selection: `ollama` or `openai`. |
| `ExperimentResult` | `experiment_id`, `config_hash`, `timestamp`, `git_commit`, `config`, `records`, `aggregate_metrics`, `environment` | Full output of a completed experiment. Serialized to `output_dir/results.json`. |

---

## Interfaces (`interfaces.py`)

Every pluggable component in EIGER implements one of these abstract base
classes. New implementations can be dropped in without modifying any
orchestration code.

| ABC | Key abstract methods | Primary implementors |
|---|---|---|
| `BaseAttack` | `apply(document, seed, **kwargs) -> PoisonedDocument`; `describe() -> dict` | `NumericalShiftAttack`, `DateManipulationAttack`, `AttributionSwitchAttack`, `CausalManipulationAttack` in `eiger.attacks` |
| `BaseDataset` | `load(split, max_claims) -> list[Claim]`; `download(target_dir)`; `content_hash` (property) | Dataset loaders in `eiger.datasets` |
| `BaseEmbedder` | `encode(texts) -> list[list[float]]`; `embedding_dim` (property) | Embedder adapters in `eiger.retrieval` |
| `BaseVectorStore` | `create_collection(name, dim)`; `reset_collection(name, dim)`; `upsert(collection, documents, vectors)`; `search(collection, query_vector, top_k)` | Qdrant adapter in `eiger.vector_stores` |
| `BaseRetriever` | `retrieve(query, claim_id, top_k) -> RetrievalResult` | Dense/sparse/hybrid retrievers in `eiger.retrieval` |
| `BaseLLM` | `generate(prompt, **kwargs) -> str`; `build_rag_prompt(query, context_docs) -> str` | Ollama and OpenAI adapters in `eiger.llm` |
| `BaseMetric` | `compute(record) -> MetricScore` | `FFRMetric`, `ERSMetric`, `SourceIntegrityMetric` in `eiger.metrics`. Default implementations of `compute_batch` and `aggregate` are provided by the base class. |

---

## Exceptions (`exceptions.py`)

| Exception | Raised when |
|---|---|
| `EigerError` | Base class for all framework errors. Catch this to handle any EIGER-specific failure. |
| `AttackNotFoundError` | `get_attack(name)` is called with a name not present in the attack registry. Message includes the list of available names. |
| `MetricNotFoundError` | `get_metric(name)` is called with a name not present in the metric registry. Message includes the list of available names. |
| `ConfigurationError` | An `ExperimentConfig` (or sub-config) fails validation — missing required fields, incompatible option combinations, or unresolvable references. |
| `IngestionError` | Corpus ingestion fails — document embedding, vector store upsert, or dataset loading error. |
| `RetrievalError` | A retrieval operation against the vector store fails. |
| `GenerationError` | An LLM generation call fails — backend unavailable, timeout, or malformed response. |
| `ReproducibilityError` | Reproducibility checks fail — seed mismatch between runs, or metric values drift beyond tolerance on re-run. |

---

## Design note

`eiger.core` is intentionally kept dependency-free beyond `pydantic`. The
separation is enforced by convention and by CI import checks: if a PR adds
any non-stdlib, non-pydantic import to a file under `eiger/core/`, it must be
rejected. Keeping the domain models and interfaces clean means the entire
framework can be understood by reading only this package.
