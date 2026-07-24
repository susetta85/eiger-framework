# EIGER Datasets — Reference Guide

> Version: 0.1.0 | Sprint 1 baseline

---

## Table of Contents

1. [Overview](#1-overview)
2. [Supported Datasets](#2-supported-datasets)
3. [AVeriTeC](#3-averitec)
4. [PolitiFact](#4-politifact)
5. [FactCheck.org](#5-factcheckorg)
6. [JSON Fixture](#6-json-fixture)
7. [JSON Fixture Format Reference](#7-json-fixture-format-reference)
8. [Snopes](#8-snopes)
9. [Dataset Versioning](#9-dataset-versioning)
10. [Adding a New Dataset](#10-adding-a-new-dataset)
11. [Sprint Roadmap](#11-sprint-roadmap)

---

## 1. Overview

EIBench requires datasets composed of verifiable factual claims — statements whose truth value can be established by consulting authoritative primary sources. This property is essential for two reasons:

**Metric validity.** The primary EIBench metric, FFR (Faithful Falsehood Rate), measures whether a RAG system produces answers that are internally consistent with retrieved context but factually wrong relative to ground truth. Computing FFR requires knowing the ground truth, which only verified fact-checking corpora provide.

**Controlled poisoning.** The poisoning engine modifies ground-truth documents to introduce specific epistemic errors (numerical shifts, misattributions, causal insertions, date manipulations). The original verified fact is preserved as the reference for both poison generation and downstream evaluation.

Fact-checking corpora — originally created for automated claim verification research — satisfy both requirements. They provide structured claims with verified verdicts, supporting evidence, and domain coverage across politics, science, economics, and health.

---

## 2. Supported Datasets

| Name | Language | Approx. Size | Domain | License | Status |
|---|---|---|---|---|---|
| Snopes | English | 4,832 verified-true claims (of 19,631 raw) | Multi-domain | Research use | **Implemented** (Sprint 3) |
| AVeriTeC | English | 4,500 claims | Multi-domain | CC BY 4.0 | Planned |
| PolitiFact | English | 21,000+ claims | US Politics | Research use | Planned |
| FactCheck.org | English | ~3,000 claims | Multi-domain | Research use | Planned |
| JSON Fixture | Italian (demo) | 1 claim | Economics | Internal | Implemented (Sprint 1) |

---

## 3. AVeriTeC

### Description

AVeriTeC (Automated Verification of Textual Claims over Evidence) is a benchmark for automated fact-checking with evidence retrieval. Each claim is paired with a verdict, a list of question-answer evidence pairs, and metadata about the source and date. AVeriTeC is the primary target corpus for EIBench because it is multi-domain, English, and freely licensed under CC BY 4.0.

The dataset was introduced at NeurIPS 2023 and contains approximately 4,500 claims spanning politics, science, health, and economics. Evidence is linked to web sources, making the retrieval context realistic.

### Download

```bash
# Create the data directory
mkdir -p data/averitec

# Download via the Hugging Face datasets CLI
pip install datasets
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("chenxwh/AVeriTeC", split="test")
ds.to_json("data/averitec/test.jsonl")
EOF
```

Alternatively, download directly from the AVeriTeC GitHub repository:

```bash
git clone https://github.com/Raldir/AVeriTeC.git /tmp/averitec_repo
cp /tmp/averitec_repo/data/*.json data/averitec/
```

### Expected Format

AVeriTeC JSONL records contain the following fields relevant to EIGER:

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | The factual claim text |
| `label` | `str` | Verdict: `Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting` |
| `evidence` | `list[dict]` | List of `{"question": str, "answer": str, "url": str}` |
| `claim_date` | `str` | Date the claim was made (ISO format) |
| `speaker` | `str` | Entity who made the claim |

### Loading with EIGER

```python
from eiger.datasets import get_dataset

# The AVeriTeC loader maps 'claim' -> Claim.original_fact
# and constructs a context_query from the claim text.
dataset = get_dataset("averitec")
dataset.download(target_dir="data/averitec")
claims = dataset.load(split="test", max_claims=100)

print(f"Loaded {len(claims)} claims")
print(f"Dataset content hash: {dataset.content_hash}")
# Example: Loaded 100 claims
# Dataset content hash: 3f8a1c2d9e4b7f0a
```

---

## 4. PolitiFact

### Description

PolitiFact is one of the largest publicly available fact-checking datasets, covering US political statements rated on a six-point scale: Pants on Fire, False, Mostly False, Half True, Mostly True, True. The dataset was introduced with the LIAR benchmark (Wang 2017) and has been extended in multiple follow-up works.

EIBench uses PolitiFact to study epistemic robustness in political discourse, a domain where misattribution and numerical shift attacks are particularly impactful.

### Download

```bash
mkdir -p data/politifact

# LIAR dataset (base version, 12,800 statements)
wget https://www.cs.ucsb.edu/~william/data/liar_dataset.zip -O /tmp/liar.zip
unzip /tmp/liar.zip -d data/politifact/

# Columns: id, label, statement, subject, speaker, job_title,
#           state_info, party_affiliation, context, justification
```

### Expected Format

The LIAR TSV format (no header row):

| Column | Field | Type | Description |
|---|---|---|---|
| 0 | `id` | `str` | Statement identifier |
| 1 | `label` | `str` | One of six truth labels |
| 2 | `statement` | `str` | The factual claim |
| 3 | `subject` | `str` | Topic tags |
| 4 | `speaker` | `str` | Who made the claim |
| 5 | `job_title` | `str` | Speaker's job title |
| 8 | `context` | `str` | Venue/context of the statement |

### Loading with EIGER

```python
from eiger.datasets import get_dataset

dataset = get_dataset("politifact")
dataset.download(target_dir="data/politifact")

# Filter to only verified-false claims (label in {"false", "pants-fire"})
# for use as adversarial ground truth
claims = dataset.load(split="test", max_claims=200)
```

---

## 5. FactCheck.org

### Description

FactCheck.org is a non-partisan US fact-checking organization. Their public corpus covers political and scientific claims with detailed rebuttals, primary source citations, and structured verdicts. The corpus is smaller than PolitiFact but has higher editorial depth per claim, making it useful for studying complex multi-hop poisoning scenarios.

### Download

FactCheck.org does not offer a bulk download API. The EIGER loader scrapes the public search endpoint or uses a pre-processed mirror:

```bash
mkdir -p data/factcheck

# Use the pre-processed CLEF 2020/2021 CheckThat! corpus which includes
# FactCheck.org claims under research-use terms
wget https://gitlab.com/checkthat_lab/clef2021-checkthat-lab/-/archive/main/data.zip \
     -O /tmp/checkthat.zip
unzip /tmp/checkthat.zip "*/task1*" -d data/factcheck/
```

### Expected Format

| Field | Type | Description |
|---|---|---|
| `claim_id` | `str` | Unique identifier |
| `claim` | `str` | The claim text |
| `verdict` | `str` | `true`, `false`, `mixture`, `unverifiable` |
| `article_url` | `str` | Link to the full fact-check article |
| `date` | `str` | Publication date |

### Loading with EIGER

```python
from eiger.datasets import get_dataset

dataset = get_dataset("factcheck_org")
claims = dataset.load(split="test", max_claims=50)
```

---

## 6. JSON Fixture

### Description

The JSON fixture is a lightweight, self-contained dataset bundled with the EIGER repository. It is the only dataset with active status, and as of Sprint 3 it is loaded through a real `JSONFixtureDataset` (`BaseDataset`) implementation registered in `eiger.datasets`, not through ad-hoc `json.load()` calls. It is used for:

- Unit and integration tests (deterministic, no network access required)
- Development and debugging of the poisoning engine
- CI pipeline validation

The fixture currently contains one claim in Italian, demonstrating the multi-lingual capability of the framework and reflecting the research team's initial prototype. English claims are now available via `SnopesDataset` (Section 8, implemented) and will be further expanded once the `AVeriTeCDataset` loader (still planned — see Section 11) is implemented.

### Location

```
eiger-framework/eibench_raw_claims.json
```

### Loading with EIGER

```python
from eiger.datasets import get_dataset

dataset = get_dataset("json_fixture")
# No download required — fixture is bundled with the package.
claims = dataset.load()

assert len(claims) == 1
claim = claims[0]
print(claim.claim_id)         # EIB_CLAIM_001
print(claim.source_dataset)   # json_fixture
print(claim.content_hash)     # e.g. "a3f1c9d2..."
```

---

## 7. JSON Fixture Format Reference

### Schema

The fixture file is a JSON array of objects. Each object represents one claim and its pre-generated adversarial variants (for reference and testing).

```json
[
  {
    "claim_id": "EIB_CLAIM_001",
    "original_fact": "Il report ufficiale dichiara che l'inflazione core nel 2024 e stabile al 2.1%.",
    "adversarial_variants": {
      "numerical_shift": "...",
      "causal_manipulation": "...",
      "attribution_switch": "..."
    },
    "context_query": "Quali sono i dati ufficiali sull'inflazione core nel 2024?"
  }
]
```

### Field Annotations

| Field | Type | Required | Description |
|---|---|---|---|
| `claim_id` | `str` | Yes | Unique identifier. Convention: `EIB_CLAIM_NNN` |
| `original_fact` | `str` | Yes | Verified factual statement. This becomes `Claim.original_fact` |
| `adversarial_variants` | `dict` | No | Pre-generated attack outputs, keyed by attack name. Stored for reference; the live pipeline regenerates these deterministically |
| `adversarial_variants.numerical_shift` | `str` | No | Output of `NumericalShiftAttack` on `original_fact` |
| `adversarial_variants.causal_manipulation` | `str` | No | Output of `CausalManipulationAttack` on `original_fact` |
| `adversarial_variants.attribution_switch` | `str` | No | Output of `AttributionSwitchAttack` on `original_fact` |
| `context_query` | `str` | Yes | Natural-language query for retrieval. Becomes `Claim.context_query` |

### Mapping to `Claim`

The `JsonFixtureDataset` loader maps fixture fields to `Claim` fields as follows:

| Fixture field | `Claim` field |
|---|---|
| `claim_id` | `claim_id` |
| `original_fact` | `original_fact` |
| `context_query` | `context_query` |
| _(constant)_ `"json_fixture"` | `source_dataset` |
| `adversarial_variants` | `metadata["adversarial_variants"]` |

The `adversarial_variants` dict is preserved in `Claim.metadata` for inspection and comparison during testing. It is not used by the live poisoning engine, which regenerates attacks from the registered implementations.

---

## 8. Snopes

### Description

Unlike AVeriTeC/PolitiFact/FactCheck.org above (still planned, Section 11), Snopes is **implemented**: `eiger/datasets/snopes.py`'s `SnopesDataset` (registry name `"snopes"`) loads it via the same JSON schema as the JSON Fixture (Section 7), inherited by subclassing `JSONFixtureDataset`.

The raw source is a 19,631-row export of Snopes fact-checks (1995-2025) with a `normalised_rating` per claim (`True`: 4,832 · `False`: 11,872 · `partially true`: 2,335 · `misleading`: 451 · `unverifiable`: 141) and a real source `url` per row — verifiable, unlike the bare `id`/`claim`/`date` PolitiFact export the team also has on hand (see Section 4; that export has no per-row source link).

Two things the raw export does NOT have, which `scripts/enrich_snopes_claims.py` adds:
1. Only `normalised_rating == True` rows are kept (a `Claim.original_fact` must be a verified true statement — EIGER generates falsehoods itself via the attack registry, it does not import externally-sourced false claims as ground truth).
2. A `context_query` (natural-language question) is generated per claim via a local Ollama LLM, since Snopes indexes fact-checks by claim, not by the question a person would ask to surface one.

**Verification status.** Every claim produced by the enrichment script is tagged `metadata["verified"] = false`, even though Snopes itself already rated it `True` — Snopes' own rating is not treated as a substitute for this research team's own review before a claim is reported on in published results. See `scripts/README.md` for the full collect → filter → enrich → (team) verify workflow, which mirrors the manual claim-intake workflow (`scripts/import_claims_xlsx.py`) but for bulk external data instead of hand-authored claims.

### Location & running the enrichment

The raw `Snopes.xlsx` and the enriched output are **not** committed to Git (`data/` is gitignored — too large, and not ours to redistribute). To (re)generate the enriched file locally:

```bash
pip install 'eiger[data-import]'   # one-time: installs openpyxl

# Pilot run first — check output quality before committing to the full batch:
python scripts/enrich_snopes_claims.py path/to/Snopes.xlsx --limit 20 -o /tmp/pilot.json

# Full run (requires a running Ollama server with the model pulled):
python scripts/enrich_snopes_claims.py path/to/Snopes.xlsx \
    -o data/snopes/snopes_enriched.json
```

The script is idempotent/resumable (checkpoints every 25 claims by default) — see its own docstring for the full design rationale.

### Loading with EIGER

```python
from eiger.datasets import get_dataset

dataset = get_dataset("snopes")  # defaults to data/snopes/snopes_enriched.json
claims = dataset.load(max_claims=100)

print(claims[0].source_dataset)          # "snopes"
print(claims[0].metadata["source"])      # the original snopes.com URL
print(claims[0].metadata["verified"])    # False, pending team review
```

---

## 9. Dataset Versioning

### Content Hashing

Every `Claim` exposes a `content_hash` property that is the first 16 hex characters of the SHA-256 hash of `original_fact`:

```python
import hashlib

claim.content_hash == hashlib.sha256(claim.original_fact.encode()).hexdigest()[:16]
```

This hash is stable across dataset reloads as long as the original fact text is unchanged. It serves as a lightweight fingerprint for detecting dataset mutations without storing the full text in provenance records.

Every `BaseDataset` implementation must also expose a `content_hash` property at the dataset level (SHA-256 of all loaded claim texts, concatenated in load order). This dataset-level hash is logged in `ExperimentResult` for full provenance.

### DVC Integration (Planned — Sprint 3)

EIGER will integrate Data Version Control (DVC) to track dataset files stored outside the Git repository (the raw JSON/JSONL files are too large for Git). The planned integration:

```bash
# Track a dataset file with DVC
dvc add data/averitec/test.jsonl

# Push to the configured remote (S3, GCS, or SSH)
dvc push

# Reproduce the exact dataset used in a prior experiment
dvc pull data/averitec/test.jsonl
```

DVC `.dvc` files (small text files containing the dataset SHA-256 and remote location) will be committed to Git alongside the experiment YAML files. This means any experiment result can be reproduced by:

1. Checking out the experiment's Git commit
2. Running `dvc pull` to restore the dataset snapshot
3. Running `eiger run experiments/config.yaml`

Until Sprint 3, dataset versioning relies solely on `content_hash` logged in `ExperimentResult.environment`.

---

## 10. Adding a New Dataset

Follow these four steps to integrate a new fact-checking corpus into EIGER.

### Step 1: Implement `BaseDataset`

Create a new file under `eiger/datasets/`:

```python
# eiger/datasets/my_corpus.py
from __future__ import annotations
import hashlib
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim

class MyCorpusDataset(BaseDataset):
    name: str = "my_corpus"
    description: str = "My custom fact-checking corpus."

    def __init__(self) -> None:
        self._claims: list[Claim] = []

    def download(self, target_dir: str) -> None:
        # Download raw files to target_dir if not already present.
        # Must be idempotent.
        ...

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        # Parse raw files and return Claim objects.
        # Populate self._claims for content_hash computation.
        raw = self._parse_raw_files(split)
        self._claims = [
            Claim(
                claim_id=row["id"],
                original_fact=row["statement"],
                context_query=self._build_query(row),
                source_dataset=self.name,
            )
            for row in raw
        ]
        if max_claims is not None:
            self._claims = self._claims[:max_claims]
        return self._claims

    @property
    def content_hash(self) -> str:
        combined = "".join(c.original_fact for c in self._claims)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _parse_raw_files(self, split: str) -> list[dict]:
        ...  # implementation-specific

    def _build_query(self, row: dict) -> str:
        return f"What is the factual status of: {row['statement']}"
```

### Step 2: Register the Dataset

Add the registration call to `eiger/datasets/__init__.py`:

```python
from eiger.datasets.registry import register_dataset
from eiger.datasets.my_corpus import MyCorpusDataset

register_dataset(MyCorpusDataset)
```

Or, for a third-party package, declare an entry point in `pyproject.toml`:

```toml
[project.entry-points."eiger.datasets"]
my_corpus = "my_package.datasets:MyCorpusDataset"
```

### Step 3: Reference in Experiment YAML

Once registered, the dataset is available by name in experiment configuration files:

```yaml
dataset:
  name: my_corpus
  split: test
  max_claims: 500
  path: data/my_corpus/   # optional local path override
```

### Step 4: Verify

```python
from eiger.datasets import get_dataset, list_datasets

print(list_datasets())  # should include "my_corpus"

ds = get_dataset("my_corpus")
ds.download(target_dir="data/my_corpus")
claims = ds.load(split="test", max_claims=10)
assert len(claims) > 0
assert all(isinstance(c.claim_id, str) for c in claims)
assert all(c.source_dataset == "my_corpus" for c in claims)
print(f"Content hash: {ds.content_hash}")
```

---

## 11. Sprint Roadmap

| Dataset | Sprint | Milestone |
|---|---|---|
| JSON Fixture (1 claim, Italian) | Sprint 1 | **Implemented.** `JSONFixtureDataset` + the `eiger.datasets` registry. Used for all unit and integration tests. |
| Snopes (English, 4,832 verified-true claims) | Sprint 3 | **Implemented.** `SnopesDataset` + `scripts/enrich_snopes_claims.py` (filter/dedupe/LLM-generated context_query). Not yet independently reviewed by the research team — see Section 8. |
| AVeriTeC (English, ~4,500 claims) | Sprint 2 | Planned. Primary research corpus. Loader implementation + DVC tracking. |
| PolitiFact via LIAR (English, ~12,800 claims) | Sprint 3 | Planned. Political domain expansion. Note: the team also has a bulk PolitiFact export on hand (`politifact_true.csv`/`politifact_false.csv`, id/claim/date only, no source URL) — lower priority than Snopes due to the missing per-row source link and minor non-English contamination in the false subset. |
| FactCheck.org via CheckThat! (English, ~3,000 claims) | Sprint 4 | Planned. Multi-domain expansion. Note: the team also has a 50-row bulk-extracted `factcheck_false.csv`, every row explicitly flagged `needs_manual_check: True` — candidate FALSE claims requiring manual review, not a source of `Claim.original_fact` ground truth. |
| Multi-lingual extension (Italian, French, German) | Sprint 5 | Planned. Cross-lingual epistemic robustness. |

Note: the "Sprint" column above is this document's own dataset-specific roadmap numbering, established during Sprint 1 planning, and does not necessarily align 1:1 with the project's actual sprint cadence (e.g. the retrieval/generation/orchestration layer built in the project's own "Sprint 2" did not touch datasets at all). The `eiger.datasets` registry and `JSONFixtureDataset` described in Sections 6-7, and `SnopesDataset` described in Section 8, were all implemented during the project's Sprint 3.

The JSON fixture will remain in the repository indefinitely as the canonical fast-test dataset. All CI pipelines run against the fixture (and, once independently reviewed, Snopes) only; full-scale experiments against AVeriTeC and PolitiFact are run on the research compute cluster and results are archived under `experiments/`.
