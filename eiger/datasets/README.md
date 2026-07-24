# eiger.datasets

Status: **Sprint 3** — the registry and four concrete loaders,
`JSONFixtureDataset`, `SnopesDataset`, `AVeriTecDataset`, and
`PolitiFactDataset`, are implemented and tested. Both
`AVeriTecDataset.download()` and `PolitiFactDataset.download()` are
guards, not real fetchers (see below). A FactCheck.org loader is still
planned (see `docs/DATASETS.md` for its full spec and the roadmap).

---

## Contents

| Module          | Class                | Source                         | Status      |
|------------------|----------------------|--------------------------------|-------------|
| `registry.py`    | —                    | `register_dataset`/`get_dataset`/`list_datasets` | Implemented |
| `json_fixture.py`| `JSONFixtureDataset` | `eibench_raw_claims.json`      | Implemented |
| `snopes.py`      | `SnopesDataset`      | LLM-enriched Snopes export (`scripts/enrich_snopes_claims.py`) | Implemented |
| `averitec.py`    | `AVeriTecDataset`    | AVeriTeC `*.jsonl` splits (manual download — see `docs/DATASETS.md` §3) | Implemented (loader only; `download()` is a guard) |
| `politifact.py`  | `PolitiFactDataset`  | LIAR `*.tsv` splits (manual download — see `docs/DATASETS.md` §4) | Implemented (loader only; `download()` is a guard) |
| `factcheck.py`   | `FactCheckDataset`   | CheckThat! corpus              | Planned     |

`BaseDataset` itself is **not** re-declared here — it already lives in
`eiger/core/interfaces.py`, alongside every other core abstract interface
(`BaseAttack`, `BaseMetric`, `BaseEmbedder`, etc.).

---

## BaseDataset interface contract

```python
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim

class BaseDataset(ABC):
    name: str
    description: str

    @abstractmethod
    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """Load and return claims from the dataset."""

    @abstractmethod
    def download(self, target_dir: str) -> None:
        """Download the raw dataset to target_dir if not already present."""

    @property
    @abstractmethod
    def content_hash(self) -> str:
        """SHA-256 fingerprint of the loaded content, for provenance tracking."""
```

## Dataset registry

Mirrors `eiger.attacks.registry` / `eiger.metrics.registry` exactly:

```python
from eiger.datasets import get_dataset, list_datasets, register_dataset

list_datasets()          # -> ["averitec", "json_fixture", "politifact", "snopes"]
dataset = get_dataset("json_fixture")  # fresh JSONFixtureDataset() instance
```

Built-in datasets are auto-registered on `import eiger.datasets`. Requesting
an unregistered name raises `DatasetNotFoundError` (from
`eiger.core.exceptions`), listing the currently available names.

---

## JSONFixtureDataset

Loads `Claim` objects from the bundled `eibench_raw_claims.json` fixture,
committed at the repository root. No network access, no external
dependencies, no download step — `download()` is a documented no-op.

Real (correct) JSON schema, matching the actual `Claim` model
(`eiger/core/models.py`):

```json
[
  {
    "claim_id": "EIB_CLAIM_001",
    "original_fact": "Il report ufficiale dichiara che l'inflazione core nel 2024 è stabile al 2.1%.",
    "context_query": "Quali sono i dati ufficiali sull'inflazione core nel 2024?",
    "adversarial_variants": {
      "numerical_shift": "... variant text ...",
      "causal_manipulation": "... variant text ..."
    }
  }
]
```

Field mapping to `Claim`:

| JSON field              | Claim field                          | Required |
|--------------------------|---------------------------------------|----------|
| `claim_id`               | `claim_id`                            | Yes      |
| `original_fact`          | `original_fact`                       | Yes      |
| `context_query`          | `context_query`                       | Yes      |
| (fixed value)            | `source_dataset` = `"json_fixture"`   | —        |
| `adversarial_variants`   | `metadata["adversarial_variants"]`    | No       |
| `source`, `domain`, `notes`, `verified` (each, if present) | same key under `metadata` | No |

`adversarial_variants` is informational/example provenance only — it is
**not** consumed by `CorpusBuilder`, which generates its own poisoned
documents at ingestion time via the attack registry (`get_attack(...)`).

`source`/`domain`/`notes`/`verified` mirror the fields produced by
`scripts/import_claims_xlsx.py` exactly, and are only added to `metadata`
when present in the raw entry — older fixture entries without them are
unaffected. This means a reviewed candidate claim can be pasted into
`eibench_raw_claims.json` almost as-is (just flip `"verified"` to `true`)
without losing its provenance; see `scripts/README.md` for the full
collect → convert → verify → promote workflow.

### Usage

```python
from eiger.datasets import JSONFixtureDataset

dataset = JSONFixtureDataset()          # defaults to the repo-root fixture
claims = dataset.load(max_claims=10)    # split is accepted but ignored
print(dataset.content_hash)             # populated only after load()
```

`path` can be overridden to point at a different fixture file (e.g. in
tests), matching `DatasetConfig.path`'s "local path override" semantics:

```python
dataset = JSONFixtureDataset(path="/tmp/custom_claims.json")
```

The resulting `claims` list is passed directly to `CorpusBuilder.build()` /
`ExperimentRunner.run()`.

---

## SnopesDataset

Subclasses `JSONFixtureDataset` (reuses all its parsing/error-handling
unchanged) and overrides only `name`/`description`/the default `path`, so
that `Claim.source_dataset` correctly reports `"snopes"`. Loads an
LLM-enriched export produced by `scripts/enrich_snopes_claims.py` — see
that script's docstring, `scripts/README.md`, and `docs/DATASETS.md`
Section 8 for the full collect → filter → enrich → (team) verify
pipeline, including why every claim is tagged `metadata["verified"] =
false` even though Snopes itself already rated it `True`.

```python
from eiger.datasets import get_dataset

dataset = get_dataset("snopes")  # defaults to data/snopes/snopes_enriched.json
claims = dataset.load(max_claims=100)
print(claims[0].source_dataset)  # "snopes"
```

---

## AVeriTecDataset

Implements `BaseDataset` directly (does not subclass `JSONFixtureDataset`
— the raw format is JSONL, not a JSON array). Loads only
`label == "Supported"` records from AVeriTeC's own `<split>.jsonl` files
(default: `data/averitec/<split>.jsonl`), using each record's first
evidence question as `context_query` directly — no LLM enrichment step
needed, unlike Snopes. See `eiger/datasets/averitec.py`'s module docstring
and `docs/DATASETS.md` Section 3 for the full field mapping and rationale.

`download()` is a **guard, not a fetcher**: it no-ops if `*.jsonl` files
already exist under the target directory, and otherwise raises
`IngestionError` pointing at the manual download steps in
`docs/DATASETS.md` §3 (fetching requires the optional HuggingFace
`datasets` library and network access, deliberately not added as a core
dependency yet).

```python
from eiger.datasets import get_dataset

dataset = get_dataset("averitec")  # defaults to data/averitec/
dataset.download(target_dir="data/averitec")  # raises if files are missing
claims = dataset.load(split="test", max_claims=100)
print(claims[0].source_dataset)  # "averitec"
print(claims[0].metadata.get("evidence_urls"))  # real source URLs, if any
```

---

## PolitiFactDataset

Implements `BaseDataset` directly, like `AVeriTecDataset` (does not
subclass `JSONFixtureDataset` — the raw format is LIAR's headerless TSV).
Loads only `label == "true"` records from `<split>.tsv` files (default:
`data/politifact/<split>.tsv`). LIAR has no evidence Q&A like AVeriTeC and
no natural-language question like Snopes, so `context_query` is a simple
templated fallback (`"Is it true that {statement}?"`) — no LLM required.
See `eiger/datasets/politifact.py`'s module docstring and
`docs/DATASETS.md` Section 4 for the full field mapping, including a
correction of that section's own earlier (incorrect) example, which had
suggested importing false/pants-fire claims — contradicting the
verified-true-only philosophy actually implemented here and everywhere
else.

`download()` is a **guard, not a fetcher**, for the same reason as
`AVeriTecDataset`: it no-ops if `*.tsv` files already exist, and otherwise
raises `IngestionError` pointing at `docs/DATASETS.md` §4's manual
download steps.

```python
from eiger.datasets import get_dataset

dataset = get_dataset("politifact")  # defaults to data/politifact/
dataset.download(target_dir="data/politifact")  # raises if files are missing
claims = dataset.load(split="test", max_claims=100)
print(claims[0].source_dataset)  # "politifact"
```

---

## Adding a new dataset loader

1. Implement a `BaseDataset` subclass in a new module under `eiger/datasets/`
   (see `json_fixture.py` for the reference implementation).
2. Register it in `eiger/datasets/__init__.py`:
   `register_dataset(YourDatasetClass)`, and add it to `__all__`.
3. Reference it by name in a `DatasetConfig.name` (YAML or code).
4. Verify with `list_datasets()` and `get_dataset(name)`.

See `docs/DATASETS.md` for the full spec (expected raw format, download
instructions) of the still-planned FactCheck.org loader.

---

## Growing the JSON fixture: claim intake from non-technical contributors

`scripts/import_claims_xlsx.py` converts a filled-in spreadsheet
(`eiger_claims_template.xlsx`) into an **unverified candidate** JSON file
— never directly into `eibench_raw_claims.json`. See `scripts/README.md`
for the full collect → convert → verify → promote workflow. A candidate's
`source`/`domain`/`notes`/`verified` fields survive promotion unchanged:
`JSONFixtureDataset._to_claim()` carries each into `Claim.metadata` when
present (see the field-mapping table above) — a previously-documented
limitation, fixed in Sprint 3.
