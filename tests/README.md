# Tests

EIGER uses pytest throughout. The suite is split into two tiers: fast unit tests
that run with no external services, and integration tests that require the Docker
Compose stack (Qdrant + Ollama) to be running.

---

## Running Tests

```bash
make test               # All tests with coverage report
make test-unit          # Unit tests only (no Docker required)
make test-integration   # Integration tests (requires: make up)
```

Direct pytest invocations are also supported:

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ --cov=eiger --cov-report=term-missing
```

---

## Test Suite Overview

| Suite       | Location            | Count | Speed  | External Services |
|-------------|---------------------|-------|--------|--------------------|
| Unit        | `tests/unit/`       | 358   | Fast   | None               |
| Integration | `tests/integration/`| 3     | Fast–Slow | None required (see below) |

Integration tests are split into two files with different infrastructure
requirements:

- **`test_pipeline_fake_infra.py`** (2 tests) — always runs, no external
  services or heavy ML dependencies. Exercises the *real* orchestration
  logic (`CorpusBuilder` → `IngestionPipeline` → `DenseRetriever` →
  `ExperimentRunner` → metrics → `ExperimentResult`) against small but
  functionally working fake implementations of `BaseEmbedder`/
  `BaseVectorStore`/`BaseLLM` (real cosine similarity, real in-memory
  storage, a deterministic echo-LLM) — not `MagicMock` call-assertions like
  `tests/unit/`.
- **`test_pipeline_live_infra.py`** (1 test) — exercises the real production
  stack: `SentenceTransformerEmbedder`, `QdrantVectorStore`, `OllamaLLM`.
  Automatically **skips** (via a module-scoped fixture that checks TCP
  reachability of the configured Qdrant/Ollama hosts, not a pytest marker)
  when either service is unreachable, so `pytest tests/` remains safe to run
  without Docker. Run `make up` and pull the model referenced by
  `_MODEL_NAME` in that file to actually execute it.

---

## Unit Test Files

### `test_attacks.py` — 24 tests

Covers all four adversarial attack implementations. Each attack is verified for:

- **Text transformation** — the attack produces a materially different document
- **Determinism** — identical seed and input always produce identical output
- **Isolation** — the attack does not mutate global `random` state
- **Provenance** — the returned `PoisonedDocument` carries correct `attack_name`,
  `attack_params`, and `original_text` fields
- **Registry** — the attack is retrievable by name via `get_attack()`

### `test_metrics.py` — 13 tests

Covers `FFRMetric` (Factual Faithfulness Rate) and `ERSMetric` (Epistemic Risk
Score). Verified properties include:

- Correct formula implementation against hand-computed expected values
- Edge cases: empty input, all-faithful corpus, fully-poisoned corpus
- Configurable thresholds (where applicable)
- Registry integration via `get_metric()` and `list_metrics()`

### `test_models.py` — 13 tests

Validates the Pydantic domain models in `eiger.core.models`. Covers:

- Schema enforcement and field validation (type errors, range violations)
- Content hashing (`Claim.content_hash`, `ExperimentConfig.config_hash`)
- Derived properties (`RetrievalResult.contains_poisoned`, `poison_ratio`)
- Serialization round-trips (model to JSON and back)

### `test_seeding.py` — 8 tests

The scientific validity gate for the framework. Verifies that:

- `make_rng(seed)` produces the same pseudo-random sequence for the same seed
- `derive_seed(base, label)` is deterministic and collision-resistant
- `seed_everything(seed)` sets Python `random`, `numpy`, and `torch` (if available)
  to deterministic state without leaking state between calls

These tests must pass before any experiment results can be considered
reproducible.

### `conftest.py` — Shared fixtures

Provides `sample_claims` (session-scoped, two `Claim` objects) and
`sample_document` (function-scoped, one ground-truth `Document`). All unit
tests that need baseline data use these fixtures rather than constructing
objects inline.

---

## Sprint 2 unit test files

Sprint 2 added the retrieval, ingestion, LLM, and orchestration layers, each
with a corresponding unit test file mocking out its infrastructure boundary
(`sentence-transformers`, `qdrant-client`, `httpx`) so no real model download
or running service is required:

| File | Tests | Covers |
|---|---|---|
| `test_embedder.py` | 13 | `SentenceTransformerEmbedder`: lazy loading, batching, `embedding_dim` |
| `test_qdrant_store.py` | 19 | `QdrantVectorStore`: lazy client, collection lifecycle, upsert/search payload shape |
| `test_retriever.py` | 32 | `DenseRetriever`: query encoding, search, score normalization, `RetrievalResult` assembly |
| `test_pipeline.py` | 24 | `IngestionPipeline`: reset/encode/upsert orchestration, empty-corpus handling |
| `test_ollama.py` | 38 | `OllamaLLM`: HTTP request/response handling, `build_rag_prompt()`, error paths |
| `test_runner.py` | 34 | `ExperimentRunner`: full orchestration, attack/metric registry resolution, `faithfulness_scorer` hook, git/environment capture, result persistence |
| `test_heuristic_scorer.py` | 23 | `EmbeddingFaithfulnessScorer`: cosine-similarity proxy semantics, blank-input edge cases |

All of the above follow a common convention: a module-level `log` object is
patched (`with patch("eiger.<module>.log")` or an autouse fixture doing the
same) whenever the code under test would otherwise call `structlog` for
real, since a pre-existing structlog version quirk raises `AttributeError`
on the default `PrintLogger` when `configure_logging()` has run earlier in
the same pytest process (fixed for `test_logging.py` itself via
`structlog.reset_defaults()` in an autouse teardown fixture, to prevent that
module's tests from corrupting global logging state for every
alphabetically-later test file).

---

## Sprint 3 unit test files

Sprint 3 added the dataset layer:

| File | Tests | Covers |
|---|---|---|
| `test_datasets.py` | 30 | Dataset registry (`register_dataset`/`get_dataset`/`list_datasets`, `DatasetNotFoundError`) and `JSONFixtureDataset`: field mapping, `max_claims`/`split` handling, `download()` no-op, `content_hash` before/after load, error paths (missing file, invalid JSON, non-list top-level, missing required field), and the optional `source`/`domain`/`notes`/`verified` provenance passthrough |
| `test_cli.py` | 22 | `eiger.__main__` (the `eiger` CLI): YAML config loading/validation, the component factory (`_build_dataset`/`_build_runner` — unsupported retriever/vector_store/llm backend values, correct component wiring), `run`/`list-datasets`/`list-attacks`/`list-metrics` subcommands, and `main()`'s dispatch + `EigerError`/`ImportError` → exit-code-1 handling |
| `test_snopes.py` | 9 | `SnopesDataset`: identity/default path/registration, `source_dataset` correctly reporting `"snopes"` (the one behavior its `JSONFixtureDataset` parent couldn't provide unmodified), provenance passthrough, `download()` no-op |

Follows the same module-level `log`-patching convention described above
(patches `eiger.datasets.json_fixture.log` and, for `test_cli.py`,
`eiger.__main__.log` plus `eiger.experiments.runner.log`; for
`test_snopes.py`, both `eiger.datasets.json_fixture.log` and
`eiger.datasets.snopes.log`).

---

## Adding Tests for a New Component

When implementing a new module (e.g., a new attack, metric, or retriever),
add a corresponding test file in `tests/unit/` following this pattern:

1. **Determinism test** — call the component twice with the same seed and
   assert that outputs are identical.
2. **No global state mutation test** — record `random.getstate()` before and
   after calling the component and assert they are equal.
3. **Registry test** — if the component registers itself, assert that it can
   be retrieved by name and that `list_*()` includes it.
4. **Log patching** — if the component logs via `eiger.utils.logging.get_logger`,
   patch its module-level `log` object (see "Sprint 2 unit test files" above)
   rather than relying on `configure_logging()` having been called.

For components that require external services, add integration tests in
`tests/integration/`. There is no `@pytest.mark.integration` marker; instead,
check reachability directly (see `_port_open()` /
`_require_live_infra` in `test_pipeline_live_infra.py`) and call
`pytest.skip(...)` when the service isn't available, so the test suite
degrades gracefully without Docker rather than failing outright.

---

## Coverage

Coverage is configured in `pyproject.toml`. The target threshold is **100%**
(`--cov-fail-under=100`), enforced on every `pytest` invocation via
`addopts` — not just `make test`:

```bash
pytest tests/unit/ --cov=eiger --cov-report=term-missing
```

An HTML report is written to `htmlcov/` and is excluded from version control.
