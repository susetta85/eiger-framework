# EIGER Framework — Architecture Reference

> Version: 0.1.0 | Sprint 1 baseline  
> Package: `eiger` | Python >= 3.10

---

## Table of Contents

1. [Overview](#1-overview)
2. [Six-Layer Pipeline](#2-six-layer-pipeline)
3. [Module Dependency Graph](#3-module-dependency-graph)
4. [Domain Model](#4-domain-model)
5. [Abstract Interfaces](#5-abstract-interfaces)
6. [Plugin Architecture and Registry](#6-plugin-architecture-and-registry)
7. [Configuration and Settings](#7-configuration-and-settings)
8. [Seeding and Reproducibility](#8-seeding-and-reproducibility)
9. [Infrastructure](#9-infrastructure)
10. [Design Decisions](#10-design-decisions)

---

## 1. Overview

EIGER (Epistemic Integrity Gauge for Epistemic Robustness) is a research framework for measuring how susceptible Retrieval-Augmented Generation (RAG) systems are to corpus poisoning attacks. The companion benchmark suite is called EIBench.

The framework is organized around three guiding principles:

**Clean Architecture.** Domain logic (models, interfaces, metrics) has no dependency on infrastructure (Qdrant, Ollama, Docker). The core layer defines contracts; outer layers implement them. A unit test can run the entire evaluation pipeline with mock implementations and no network access.

**Dependency injection over singletons.** Every pluggable component — embedders, vector stores, LLM backends, attack strategies, metrics — is resolved through explicit construction rather than global state. Configuration is the single source of truth; nothing is hard-coded.

**Plugin extensibility via registry and entry points.** Third parties (or future sprint work) can introduce new attack strategies, dataset loaders, and metrics without modifying framework code. The registry pattern and Python entry points make drop-in extension safe and explicit.

---

## 2. Six-Layer Pipeline

The experiment pipeline is a directed acyclic graph of six processing layers. Each layer is independently testable and replaceable.

### Layer 1 — Corpus Builder (`eiger/ingestion/`)

Transforms raw `Claim` objects into a vector corpus. For each claim, a ground-truth `Document` is produced. The layer drives embedding via a `BaseEmbedder` implementation and upserts the result into a `BaseVectorStore`.

| Responsibility | Detail |
|---|---|
| Claim ingestion | Accepts a `list[Claim]` from any `BaseDataset` |
| Document creation | Wraps each claim's text in a `Document` (doc_type=`ground_truth`) |
| Embedding | Calls `BaseEmbedder.encode()` on batch document texts |
| Indexing | Calls `BaseVectorStore.upsert()` to write vectors |
| Status | Sprint 1 — interface complete; Qdrant implementation in progress |

### Layer 2 — Poisoning Engine (`eiger/attacks/`)

Applies one or more adversarial attacks to a fraction of the corpus (the `poison_rate`). Each attack receives a deterministic per-document seed derived from the experiment seed and the document's `claim_id`.

| Responsibility | Detail |
|---|---|
| Attack dispatch | Looks up attack class in registry by name string |
| Poisoning | Calls `BaseAttack.apply(document, seed)` per document |
| Output | Returns `list[PoisonedDocument]`, each with full provenance |
| Corpus update | Upserts poisoned documents into the vector store alongside originals |
| Status | Sprint 1 — all four built-in attacks implemented and tested |

### Layer 3 — Retrieval (`eiger/retrieval/`, `eiger/vector_stores/`)

Given a `context_query` from a `Claim`, retrieves the top-k most similar documents from the vector corpus. The retriever is strategy-agnostic (dense, sparse, or hybrid).

| Responsibility | Detail |
|---|---|
| Query embedding | Encodes the query string via `BaseEmbedder.encode()` |
| Vector search | Calls `BaseVectorStore.search()` returning ranked document dicts |
| Result wrapping | Produces a `RetrievalResult` with ranked `RetrievedDocument` hits |
| Poison detection | `RetrievalResult.contains_poisoned` and `poison_ratio` are derived properties |
| Status | Sprint 2 — dense retrieval via Qdrant implemented; sparse/hybrid planned |

### Layer 4 — Generation (`eiger/llm/`)

Feeds the retrieved document texts and the original query to an LLM backend to produce a natural-language answer. The `BaseLLM` interface isolates this layer from specific backend choices (Ollama, OpenAI, etc.).

| Responsibility | Detail |
|---|---|
| Prompt construction | `BaseLLM.build_rag_prompt(query, context_docs)` |
| Generation | `BaseLLM.generate(prompt)` returns the answer string |
| Output | Produces a `GenerationResult` with query, context, answer, and model name |
| Status | Sprint 2 — Ollama backend implemented; OpenAI adapter planned |

### Layer 5 — Evaluation (`eiger/metrics/`)

Computes one or more `BaseMetric` values for each `(query, retrieved context, generated answer)` triple. Metrics are registered by name and can be composed freely per experiment.

| Responsibility | Detail |
|---|---|
| Record assembly | Combines `RetrievalResult` and `GenerationResult` into `EvaluationRecord` |
| Metric dispatch | Iterates over metric names from `ExperimentConfig.metrics` |
| Score collection | Calls `BaseMetric.compute(record)` for each metric |
| Aggregation | Calls `BaseMetric.aggregate(scores)` for experiment-level summary |
| Status | Sprint 1 — FFR and ERS implemented; SourceIntegrity NLI backend in Sprint 4 |

### Layer 6 — Analytics (`eiger/experiments/`)

Persists the full `ExperimentResult` to disk, computes aggregate statistics, and supports downstream analysis notebooks. Provenance is guaranteed by embedding the full `ExperimentConfig` (including `config_hash`) in every result file.

| Responsibility | Detail |
|---|---|
| Result serialization | `ExperimentResult.to_json()` writes results to `output_dir/results.json` |
| Provenance | Git commit hash, timestamp, config hash, environment dict |
| Reproducibility | `config_hash` is a SHA-256 fingerprint of the full config |
| Status | Sprint 3 — basic JSON output complete; analytics dashboard planned |

---

## 3. Module Dependency Graph

Dependencies flow strictly downward. Higher layers import lower layers; the reverse is prohibited. The `core` layer has no imports from any other `eiger` subpackage.

```
eiger.core          (models, interfaces, exceptions)
    ^
    |   imported by
    +-- eiger.config         (settings, env resolution)
    +-- eiger.utils          (seeding, logging)
    +-- eiger.attacks        (attack implementations + registry)
    +-- eiger.metrics        (metric implementations + registry)
    +-- eiger.datasets       (dataset loaders)
    +-- eiger.vector_stores  (Qdrant adapter)
    +-- eiger.llm            (Ollama adapter)
    +-- eiger.retrieval      (dense/sparse/hybrid retrievers)
    +-- eiger.ingestion      (corpus builder, uses vector_stores + attacks)
    +-- eiger.experiments    (orchestration, uses all layers)
```

As a table showing which modules each layer may import:

| Module | May import from |
|---|---|
| `eiger.core` | stdlib only |
| `eiger.config` | `eiger.core` |
| `eiger.utils` | `eiger.core`, `eiger.config` |
| `eiger.attacks` | `eiger.core`, `eiger.utils` |
| `eiger.metrics` | `eiger.core`, `eiger.utils` |
| `eiger.datasets` | `eiger.core`, `eiger.utils`, `eiger.config` |
| `eiger.vector_stores` | `eiger.core`, `eiger.config` |
| `eiger.llm` | `eiger.core`, `eiger.config` |
| `eiger.retrieval` | `eiger.core`, `eiger.vector_stores` |
| `eiger.ingestion` | `eiger.core`, `eiger.vector_stores`, `eiger.attacks` |
| `eiger.experiments` | All modules above |

---

## 4. Domain Model

All domain models are Pydantic v2 `BaseModel` subclasses. They validate on construction, serialize cleanly to JSON, and carry no infrastructure dependencies. The full model graph is defined in `eiger/core/models.py`.

### 4.1 Dataset Layer

#### `Claim`

The fundamental unit of data. Every downstream document originates from a `Claim`.

| Field | Type | Description |
|---|---|---|
| `claim_id` | `str` | Unique identifier within the source dataset |
| `original_fact` | `str` | Verified factual statement (ground truth) |
| `context_query` | `str` | Query string used for retrieval |
| `source_dataset` | `str` | Origin dataset name (e.g. `averitec`) |
| `metadata` | `dict` | Arbitrary extensible metadata |
| `content_hash` (property) | `str` | SHA-256[:16] of `original_fact` for versioning |

#### `Document`

A corpus entry, either ground-truth or poisoned.

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | UUID4, auto-generated |
| `claim_id` | `str` | Foreign key to parent `Claim` |
| `text` | `str` | Document text content |
| `doc_type` | `str` | `ground_truth` or `poisoned` |
| `metadata` | `dict` | Extensible metadata |

#### `PoisonedDocument` (extends `Document`)

Adds full poisoning provenance. `doc_type` is always `poisoned`.

| Field | Type | Description |
|---|---|---|
| `attack_name` | `str` | Registry name of the attack that produced this document |
| `attack_params` | `dict` | Hyperparameters used at attack time |
| `original_text` | `str` | Unmodified text before poisoning |
| `annotation` | `PoisonAnnotation \| None` | Human or model epistemic risk annotation |

#### `PoisonAnnotation`

Captures human or LLM-judged epistemic risk across three dimensions, each on a 1-5 scale.

| Field | Range | Description |
|---|---|---|
| `plausibility` | [1, 5] | How believable the falsehood appears |
| `verification_difficulty` | [1, 5] | How hard it is to fact-check |
| `editorial_risk` | [1, 5] | Likelihood of passing editorial review |

### 4.2 Retrieval Layer

#### `RetrievedDocument`

A single search result, coupling a `Document` with its rank and similarity score.

| Field | Type | Description |
|---|---|---|
| `document` | `Document` | The retrieved document |
| `score` | `float` [0, 1] | Similarity score (higher = more similar) |
| `rank` | `int` | Position in the ranked list (1-indexed) |

#### `RetrievalResult`

Full output of a retrieval operation.

| Field / Property | Type | Description |
|---|---|---|
| `query` | `str` | The query string |
| `claim_id` | `str` | Parent claim |
| `hits` | `list[RetrievedDocument]` | Ranked results |
| `top_k` | `int` | Number of results requested |
| `contains_poisoned` (property) | `bool` | True if any hit is a `PoisonedDocument` |
| `poison_ratio` (property) | `float` | Fraction of hits that are poisoned |

### 4.3 Generation Layer

#### `GenerationResult`

Output from one LLM inference call.

| Field | Type | Description |
|---|---|---|
| `claim_id` | `str` | Parent claim |
| `query` | `str` | The original query |
| `context_docs` | `list[str]` | Texts of retrieved documents fed to the LLM |
| `answer` | `str` | LLM-generated answer |
| `model_name` | `str` | LLM identifier (e.g. `llama3.1:8b`) |
| `metadata` | `dict` | Backend-specific metadata (tokens, latency, etc.) |

### 4.4 Evaluation Layer

#### `MetricScore`

Result of one metric computation on one example.

| Field | Type | Description |
|---|---|---|
| `metric_name` | `str` | Registry name of the metric |
| `value` | `float` | Computed score |
| `metadata` | `dict` | Per-computation diagnostics |

#### `EvaluationRecord`

Complete evaluation record for a single claim, combining retrieval, generation, and all metric scores.

| Field / Property | Type | Description |
|---|---|---|
| `claim_id` | `str` | Parent claim |
| `generation` | `GenerationResult` | LLM output |
| `retrieval` | `RetrievalResult` | Retrieval output |
| `metrics` | `dict[str, float]` | Metric name → aggregated score |
| `faithfulness_score` (property) | `float` | Shortcut to `ragas_faithfulness` metric |
| `factual_correctness_score` (property) | `float` | Shortcut to `ragas_answer_correctness` metric |

### 4.5 Experiment Layer

#### `ExperimentConfig`

Full, validated specification for a single experiment run. Loaded from YAML and resolved against environment settings. Serialized alongside every result file.

| Field / Property | Type | Description |
|---|---|---|
| `experiment_id` | `str` | Auto-generated (`exp_<8 hex chars>`) |
| `seed` | `int` | Global experiment seed (default: 42) |
| `dataset` | `DatasetConfig` | Dataset specification |
| `attacks` | `list[AttackConfig]` | Attack specifications |
| `retriever` | `RetrieverConfig` | Retrieval strategy |
| `llm` | `LLMConfig` | LLM backend |
| `metrics` | `list[str]` | Metric names to compute |
| `output_dir` | `str` | Results output directory |
| `config_hash` (property) | `str` | SHA-256[:16] of full config JSON |

Sub-configs (`DatasetConfig`, `AttackConfig`, `RetrieverConfig`, `LLMConfig`) are validated inline and serialized as nested objects.

#### `ExperimentResult`

Full output of a completed run. Written to `output_dir/results.json`.

| Field | Type | Description |
|---|---|---|
| `experiment_id` | `str` | Matches the config |
| `config_hash` | `str` | Fingerprint for reproducibility lookup |
| `timestamp` | `str` | UTC ISO-8601 |
| `git_commit` | `str` | Git SHA at run time |
| `config` | `ExperimentConfig` | Full embedded config |
| `records` | `list[EvaluationRecord]` | Per-claim results |
| `aggregate_metrics` | `dict[str, float]` | Experiment-level aggregates |
| `environment` | `dict[str, str]` | Python version, platform, dependency versions |

---

## 5. Abstract Interfaces

All extension points are defined in `eiger/core/interfaces.py` as Python `ABC` subclasses. Implementations in outer layers depend only on these contracts, never on concrete types.

### `BaseAttack`

Contract for adversarial poisoning strategies.

```python
class BaseAttack(ABC):
    name: str         # unique registry key
    description: str  # logged in experiment output

    @abstractmethod
    def apply(self, document: Document, seed: int, **kwargs) -> PoisonedDocument: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]: ...  # JSON-serializable parameter dict
```

**Contract rules:** Implementations must be stateless with respect to the corpus. All randomness is driven by the explicit `seed` argument — implementations must not read from global `random` state. `describe()` must return a dict that round-trips through `json.dumps`.

### `BaseDataset`

Contract for fact-checking dataset loaders.

```python
class BaseDataset(ABC):
    name: str
    description: str

    @abstractmethod
    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]: ...

    @abstractmethod
    def download(self, target_dir: str) -> None: ...

    @property
    @abstractmethod
    def content_hash(self) -> str: ...  # SHA-256 of loaded content
```

**Contract rules:** `load()` must be idempotent. `content_hash` must reflect the actual data loaded, not just the source path. `download()` must be a no-op if data is already present.

### `BaseEmbedder`

Contract for text embedding models.

```python
class BaseEmbedder(ABC):
    model_name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...
```

**Contract rules:** `encode([])` must return `[]`. Output length must equal input length. `embedding_dim` must be constant for the lifetime of the instance.

### `BaseVectorStore`

Contract for vector databases.

```python
class BaseVectorStore(ABC):
    @abstractmethod
    def create_collection(self, name: str, dim: int) -> None: ...
    @abstractmethod
    def reset_collection(self, name: str, dim: int) -> None: ...
    @abstractmethod
    def upsert(self, collection: str, documents: list[Document], vectors: list[list[float]]) -> None: ...
    @abstractmethod
    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[dict]: ...
```

**Contract rules:** `upsert` is idempotent on `doc_id`. `search` returns results ordered by descending similarity. `create_collection` raises if the collection already exists; `reset_collection` silently drops and recreates.

### `BaseRetriever`

Contract for retrieval strategies.

```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult: ...
```

**Contract rules:** `retrieve` must always return a `RetrievalResult` (never raise on empty results). `RetrievalResult.hits` may be empty if no results are found.

### `BaseLLM`

Contract for LLM generation backends.

```python
class BaseLLM(ABC):
    model_name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def build_rag_prompt(self, query: str, context_docs: list[str]) -> str: ...
```

**Contract rules:** `generate` must return a non-empty string or raise a typed exception. `build_rag_prompt` must be pure (no side effects, no network calls).

### `BaseMetric`

Contract for evaluation metrics.

```python
class BaseMetric(ABC):
    name: str
    description: str
    range: tuple[float, float] = (0.0, 1.0)

    @abstractmethod
    def compute(self, record: EvaluationRecord) -> MetricScore: ...

    def compute_batch(self, records: list[EvaluationRecord]) -> list[MetricScore]: ...
    def aggregate(self, scores: list[MetricScore]) -> float: ...
```

**Contract rules:** `compute` must be deterministic — same input always produces same output. `range` documents the theoretical bounds; actual values may differ. `aggregate` has a sensible default (arithmetic mean) which implementations may override.

---

## 6. Plugin Architecture and Registry

### Registry Pattern

Both attacks and metrics use an identical registry pattern: a module-level `dict` mapping string names to classes. Built-in implementations are registered automatically on package import.

```python
# eiger/attacks/registry.py (simplified)
_REGISTRY: dict[str, type[BaseAttack]] = {}

def register_attack(cls: type[BaseAttack]) -> type[BaseAttack]:
    _REGISTRY[cls.name] = cls
    return cls  # usable as a decorator

def get_attack(name: str) -> BaseAttack:
    if name not in _REGISTRY:
        raise AttackNotFoundError(name, list(_REGISTRY.keys()))
    return _REGISTRY[name]()
```

The auto-registration happens in `eiger/attacks/__init__.py`:

```python
register_attack(NumericalShiftAttack)
register_attack(AttributionSwitchAttack)
register_attack(DateManipulationAttack)
register_attack(CausalManipulationAttack)
```

### Registering a Custom Attack via Entry Points

Third-party packages can register attacks without modifying EIGER code. Add the following to your package's `pyproject.toml`:

```toml
[project.entry-points."eiger.attacks"]
my_attack = "my_package.attacks:MyCustomAttack"
```

The framework discovers and registers entry-point attacks at import time. After `pip install my-package`, the attack is available by name in experiment YAML files.

### Implementing a Custom Attack

```python
# my_package/attacks.py
from eiger.core.interfaces import BaseAttack
from eiger.core.models import Document, PoisonedDocument, PoisonAnnotation
from eiger.utils.seeding import make_rng

class MyCustomAttack(BaseAttack):
    name: str = "my_custom_attack"
    description: str = "Replaces technical terms with lay equivalents."

    def apply(self, document: Document, seed: int, **kwargs) -> PoisonedDocument:
        rng = make_rng(seed)  # isolated RNG — never touch random.random()
        # ... transform document.text using rng ...
        return PoisonedDocument(
            doc_id=document.doc_id,
            claim_id=document.claim_id,
            text=poisoned_text,
            attack_name=self.name,
            attack_params=self.describe(),
            original_text=document.text,
            annotation=PoisonAnnotation(
                plausibility=2.0,
                verification_difficulty=3.0,
                editorial_risk=2.5,
            ),
        )

    def describe(self) -> dict:
        return {"attack": self.name, "method": "terminology_replacement"}
```

The same pattern applies to custom metrics (entry point group: `eiger.metrics`) and custom datasets (entry point group: `eiger.datasets`).

---

## 7. Configuration and Settings

### Resolution Order

Settings are resolved from highest to lowest priority:

1. **Environment variables** prefixed with `EIGER_` (e.g. `EIGER_QDRANT_HOST=192.168.1.10`)
2. **`.env` file** in the working directory (UTF-8 encoded)
3. **Hardcoded defaults** in `EigerSettings`

The `EigerSettings` class is defined in `eiger/config/settings.py` and uses `pydantic-settings`:

```python
class EigerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EIGER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    default_embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    results_dir: str = "results/"
    log_level: str = "INFO"
    default_seed: int = 42
```

### Singleton Access

```python
from eiger.config import get_settings

settings = get_settings()  # lru_cache(maxsize=1) — same instance every call
print(settings.qdrant_url)  # "http://localhost:6333"
```

The `lru_cache` guarantees a single parse of the environment per process. In tests, clear the cache with `get_settings.cache_clear()` before patching environment variables.

### Experiment YAML

Experiment-level configuration is expressed as YAML and loaded into an `ExperimentConfig` Pydantic model (which validates all fields on load). The runtime settings (`EigerSettings`) supply infrastructure coordinates (hostnames, ports), while the YAML supplies experimental parameters (seed, dataset, attacks, metrics).

```yaml
# experiments/example.yaml
seed: 42
dataset:
  name: json_fixture
  split: test
attacks:
  - name: numerical_shift
    poison_rate: 0.3
    params:
      shift_factor: 10.0
retriever:
  type: dense
  top_k: 5
llm:
  backend: ollama
  model: llama3.1:8b
  temperature: 0.0
metrics:
  - ffr
  - ers
  - source_integrity
output_dir: results/
```

---

## 8. Seeding and Reproducibility

### Isolation Principle

EIGER never mutates global RNG state during an experiment run (with the sole exception of the one `seed_everything` call at the very start). Every stochastic operation receives an isolated `random.Random` instance.

### `make_rng(seed)`

Creates an independent `random.Random` instance seeded with the given integer. Does not affect `random.random()`, `numpy.random`, or any other global RNG.

```python
from eiger.utils.seeding import make_rng

rng = make_rng(seed=1234)
value = rng.uniform(0.5, 1.5)  # deterministic, isolated
```

### `derive_seed(parent_seed, *context)`

Derives a deterministic child seed from a parent seed and string context tokens. Uses SHA-256 to avoid collisions across different contexts.

```python
from eiger.utils.seeding import derive_seed

# Per-document, per-attack seed derived from the experiment seed
doc_seed = derive_seed(experiment_seed, claim_id, attack_name)
```

This ensures that adding a new attack to an experiment does not change the seeds of existing attacks.

### `seed_everything(seed)`

Seeds all available global RNGs at the start of an experiment run. Covers Python `random`, NumPy (if installed), and PyTorch (if installed, including CUDA). Should be called exactly once, before any model loading or data processing.

```python
from eiger.utils.seeding import seed_everything
seed_everything(config.seed)
```

### Reproducibility Guarantee

An experiment run is fully reproducible given:
- The same `ExperimentConfig` (verified via `config_hash`)
- The same dataset content (verified via `Claim.content_hash`)
- The same software environment (pinned in `ExperimentResult.environment`)
- The same infrastructure (pinned Docker image tags in `docker-compose.yml`)

---

## 9. Infrastructure

### Development Stack (Docker Compose)

The development environment is defined in `docker-compose.yml`. All images are pinned to exact versions for reproducibility.

| Service | Image | Ports | Purpose |
|---|---|---|---|
| `qdrant` | `qdrant/qdrant:v1.9.4` | 6333 (HTTP), 6334 (gRPC) | Vector store |
| `ollama` | `ollama/ollama:0.2.8` | 11434 | LLM inference backend |

Start the stack:

```bash
docker compose up -d
# Pull the default model after containers are healthy:
docker exec eiger-ollama ollama pull llama3.1:8b
```

Qdrant data and Ollama model weights are persisted in named Docker volumes (`qdrant_storage`, `ollama_data`).

### Distributed Research Stack (ContainerLab)

For distributed topology experiments, the framework includes a ContainerLab configuration in `infra/containerlab/`. ContainerLab is used to simulate multi-node network topologies where retrieval nodes, LLM backends, and the orchestration layer run on separate virtual hosts. This is not required for standard single-machine experiments.

---

## 10. Design Decisions

### D1: Pydantic v2 for All Domain Models

**Decision:** Use Pydantic v2 `BaseModel` for every domain object rather than Python `dataclasses` or plain dicts.

**Rationale:** Pydantic v2 provides construction-time validation (field type, range constraints with `ge`/`le`), zero-effort JSON serialization via `model_dump_json()`, and a stable schema for provenance. The `config_hash` and `content_hash` properties rely on deterministic JSON serialization, which Pydantic guarantees. The performance overhead of Pydantic v2 relative to v1 is negligible for this workload; the correctness guarantees outweigh it.

**Trade-off:** Pydantic models are slightly heavier than plain dataclasses. This is acceptable because EIGER processes at most thousands of documents per experiment, not millions.

---

### D2: Registry over Subclass Discovery

**Decision:** Use an explicit registry (a `dict` populated by `register_attack()` calls) rather than automatic subclass discovery via `__subclasses__()`.

**Rationale:** Automatic subclass discovery is fragile — it depends on import order and fails silently when a module is not yet imported. The explicit registry pattern makes all registered extensions visible at runtime via `list_attacks()`, produces a clear error on unknown names (`AttackNotFoundError`), and is compatible with the Python entry-point ecosystem for third-party extensions. The registry is populated at package import time in `__init__.py`, so by the time any orchestration code runs, all built-in implementations are available.

**Trade-off:** Third parties must call `register_attack()` or declare an entry point; they cannot simply subclass `BaseAttack` and expect the framework to find their implementation. This is an intentional friction that prevents accidental name collisions.

---

### D3: Docker Compose for Development, ContainerLab for Distributed Research

**Decision:** Maintain two separate infrastructure definitions rather than forcing ContainerLab for all use cases.

**Rationale:** ContainerLab is purpose-built for network topology simulation and requires more setup (kernel capabilities, network namespaces). For a researcher running a single-node experiment on a laptop, `docker compose up -d` is a two-second path to a working stack. ContainerLab is reserved for experiments that specifically study how distributed retrieval topology affects epistemic robustness. Keeping the two stacks separate means the framework can be used without ContainerLab installed.

---

### D4: Isolated RNG Instances over Global Seed State

**Decision:** Every stochastic component receives an isolated `random.Random(seed)` instance rather than relying on a globally seeded `random.random()`.

**Rationale:** Global RNG state is fragile under composition. If two components both call `random.random()`, inserting a new call in one component changes all subsequent outputs of the other. This makes experiments non-reproducible across code changes, even with the same seed. Isolated instances are completely independent — adding or removing a component has no effect on another component's random sequence. `derive_seed` further guarantees that per-document seeds do not collide even when the same attack is applied to many documents with the same experiment seed.
