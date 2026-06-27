# eiger.datasets

Status: planned for Sprint 2. This package does not exist yet. The documentation below
describes the intended design so that contributors can work to the same interface
contract.

For current experiments, claims are loaded from the fixture file
`eibench_raw_claims.json` directly. See `docs/DATASETS.md` for instructions.

---

## Planned contents

| Module           | Class                | Source                         |
|------------------|----------------------|--------------------------------|
| `base.py`        | `BaseDataset`        | Abstract base class            |
| `json_fixture.py`| `JSONFixtureDataset` | `eibench_raw_claims.json`      |
| `averitec.py`    | `AVeriTeCDataset`    | HuggingFace `datasets` library |
| `politifact.py`  | `PolitiFactDataset`  | PolitiFact API / scraper       |
| `factcheck.py`   | `FactCheckDataset`   | FactCheck.org loader           |

---

## BaseDataset interface contract

All dataset classes will inherit from `BaseDataset` and implement the following
abstract interface:

```python
from abc import ABC, abstractmethod
from eiger.models import Claim

class BaseDataset(ABC):

    @abstractmethod
    def load(self, split: str = "train", max_claims: int | None = None) -> list[Claim]:
        """Load claims from the dataset.

        Args:
            split:      Dataset split — "train", "dev", or "test".
            max_claims: If set, truncates the returned list to this length.

        Returns:
            A list of Claim objects in the order defined by the dataset.
        """
        ...

    @abstractmethod
    def download(self, target_dir: str) -> None:
        """Download or cache the raw dataset files to target_dir."""
        ...

    @property
    @abstractmethod
    def content_hash(self) -> str:
        """SHA-256 hex digest of the raw source files.

        Used to detect dataset version changes between experiment runs.
        """
        ...
```

The `load` method must return `Claim` objects as defined in `eiger.models`. Callers
must not assume any ordering unless the specific dataset class documents one.

---

## Expected JSON schema

All datasets, regardless of source, map to the same `Claim` model fields. The canonical
JSON representation used by `JSONFixtureDataset` and expected by importers is:

```json
{
  "id": "claim_001",
  "text": "The Eiffel Tower is 330 metres tall.",
  "label": "false",
  "evidence": [
    {
      "source": "wikipedia",
      "text": "The Eiffel Tower stands 300 metres tall."
    }
  ],
  "metadata": {
    "source_dataset": "eibench",
    "date": "2024-01-15"
  }
}
```

Field mapping to `Claim`:

| JSON field          | Claim field     | Required |
|---------------------|-----------------|----------|
| `id`                | `id`            | Yes      |
| `text`              | `text`          | Yes      |
| `label`             | `label`         | Yes      |
| `evidence`          | `evidence`      | No       |
| `metadata`          | `metadata`      | No       |

---

## Placeholder usage example

The following example shows the intended usage once Sprint 2 is complete. It will not
work until the package is implemented.

```python
# Sprint 2 — not yet available
from eiger.datasets.averitec import AVeriTeCDataset

dataset = AVeriTeCDataset()
dataset.download(target_dir="data/averitec/")
claims = dataset.load(split="dev", max_claims=500)

print(f"Loaded {len(claims)} claims")
print(f"Dataset hash: {dataset.content_hash}")
```

The resulting `claims` list is passed directly to `CorpusBuilder.build()`.

---

## Current workaround

Until Sprint 2 is delivered, load claims from the bundled fixture:

```python
import json
from eiger.models import Claim

with open("eibench_raw_claims.json") as f:
    raw = json.load(f)

claims = [Claim(**item) for item in raw]
```

See `docs/DATASETS.md` for the fixture file location, format details, and instructions
for adding new fixture claims.
