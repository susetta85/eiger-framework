# eiger.datasets

Status: **Sprint 3 (in progress)** — the registry and one concrete loader,
`JSONFixtureDataset`, are implemented and tested. AVeriTeC, PolitiFact, and
FactCheck.org loaders are still planned (see `docs/DATASETS.md` for their
full specs and the roadmap).

---

## Contents

| Module          | Class                | Source                         | Status      |
|------------------|----------------------|--------------------------------|-------------|
| `registry.py`    | —                    | `register_dataset`/`get_dataset`/`list_datasets` | Implemented |
| `json_fixture.py`| `JSONFixtureDataset` | `eibench_raw_claims.json`      | Implemented |
| `averitec.py`    | `AVeriTeCDataset`    | HuggingFace `datasets` library | Planned     |
| `politifact.py`  | `PolitiFactDataset`  | LIAR TSV                       | Planned     |
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

list_datasets()          # -> ["json_fixture"]
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

`adversarial_variants` is informational/example provenance only — it is
**not** consumed by `CorpusBuilder`, which generates its own poisoned
documents at ingestion time via the attack registry (`get_attack(...)`).

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

## Adding a new dataset loader

1. Implement a `BaseDataset` subclass in a new module under `eiger/datasets/`
   (see `json_fixture.py` for the reference implementation).
2. Register it in `eiger/datasets/__init__.py`:
   `register_dataset(YourDatasetClass)`, and add it to `__all__`.
3. Reference it by name in a `DatasetConfig.name` (YAML or code).
4. Verify with `list_datasets()` and `get_dataset(name)`.

See `docs/DATASETS.md` for the full specs (expected raw formats, download
instructions) of the still-planned AVeriTeC, PolitiFact, and FactCheck.org
loaders.

---

## Growing the JSON fixture: claim intake from non-technical contributors

`scripts/import_claims_xlsx.py` converts a filled-in spreadsheet
(`eiger_claims_template.xlsx`) into an **unverified candidate** JSON file
— never directly into `eibench_raw_claims.json`. See `scripts/README.md`
for the full collect → convert → verify → promote workflow, and its
documented limitation (promotion currently drops the candidate's
`source`/`domain`/`notes`/`verified` fields, since `JSONFixtureDataset`
only preserves `adversarial_variants` in `Claim.metadata` today).
