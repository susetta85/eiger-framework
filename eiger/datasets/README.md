# eiger.datasets

Status: **Sprint 3+** — the registry and six concrete loaders,
`JSONFixtureDataset`, `SnopesDataset`, `AVeriTecDataset`,
`PolitiFactDataset`, `FactCheckDataset`, and `CorpusClaimDataset`, are
implemented and tested. All five datasets originally documented in
`docs/DATASETS.md`'s roadmap now have loaders, plus the project's own
human-curated Mistral-generated corpus (`CorpusClaimDataset`, added
post-Sprint-5). `AVeriTecDataset.download()` and `PolitiFactDataset.download()` are now
real fetchers (HuggingFace `datasets` and stdlib `urllib`/`zipfile`
respectively — see below), each falling back to a guard-and-raise if the
fetch fails; neither has been exercised against a live network call yet.
`FactCheckDataset.download()` deliberately remains a guard, not a
fetcher — its own module docstring flags the CheckThat! archive's internal
path/format as unverified. None of the three has been independently
reviewed by the research team yet either. `CorpusClaimDataset.download()`
is also a guard: this is a private corpus with no automated source at all.

---

## Contents

| Module          | Class                | Source                         | Status      |
|------------------|----------------------|--------------------------------|-------------|
| `registry.py`    | —                    | `register_dataset`/`get_dataset`/`list_datasets` | Implemented |
| `json_fixture.py`| `JSONFixtureDataset` | `eibench_raw_claims.json`      | Implemented |
| `snopes.py`      | `SnopesDataset`      | LLM-enriched Snopes export (`scripts/enrich_snopes_claims.py`) | Implemented |
| `averitec.py`    | `AVeriTecDataset`    | AVeriTeC `*.jsonl` splits (auto-fetched via HuggingFace `datasets` — see `docs/DATASETS.md` §3) | Implemented, including automated `download()` (unverified against a live network call) |
| `politifact.py`  | `PolitiFactDataset`  | LIAR `*.tsv` splits (auto-fetched via stdlib `urllib`/`zipfile` — see `docs/DATASETS.md` §4) | Implemented, including automated `download()` (unverified against a live network call) |
| `factcheck.py`   | `FactCheckDataset`   | CheckThat! mirror `*.jsonl` splits (manual download — see `docs/DATASETS.md` §5) | Implemented (loader only; `download()` is a guard) |
| `corpus_claim.py`| `CorpusClaimDataset` | `Corpus_claim_RAG_Mistral_output_v3.xlsx`, sheet `01_Corpus_claim` (private, copy/symlink into `data/corpus_claim/`) | Implemented (loader only; `download()` is a guard; `load()` returns `[]` by default — ethical gate, see below) |

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

list_datasets()          # -> ["averitec", "corpus_claim", "factcheck_org", "json_fixture", "politifact", "snopes"]
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

`download()` now fetches automatically via the optional HuggingFace
`datasets` library (`pip install 'eiger[data-import]'` — install it in an
isolated venv, see that extra's own note in `pyproject.toml` about a real
`numpy` version conflict found doing this): it no-ops if `*.jsonl` files
already exist under the target directory, otherwise it fetches `train`/
`dev`/`test` from `datasets.load_dataset("chenxwh/AVeriTeC", ...)` and
writes whichever splits succeed, raising `IngestionError` only if every
split fails (see `docs/DATASETS.md` §3 for the manual fallback and this
fetcher's own "not yet exercised against a live network call" caveat).

```python
from eiger.datasets import get_dataset

dataset = get_dataset("averitec")  # defaults to data/averitec/
dataset.download(target_dir="data/averitec")  # fetches automatically; raises only if every split fails
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

`download()` now fetches automatically: it no-ops if `*.tsv` files already
exist, otherwise it downloads and extracts `liar_dataset.zip` via stdlib
`urllib`/`zipfile` (no new dependency), raising `IngestionError` if the
download or extraction fails (see `docs/DATASETS.md` §4 for the manual
fallback and this fetcher's own "not yet exercised against a live network
call" caveat).

```python
from eiger.datasets import get_dataset

dataset = get_dataset("politifact")  # defaults to data/politifact/
dataset.download(target_dir="data/politifact")  # raises if files are missing
claims = dataset.load(split="test", max_claims=100)
print(claims[0].source_dataset)  # "politifact"
```

---

## FactCheckDataset

Implements `BaseDataset` directly, like `AVeriTecDataset`/`PolitiFactDataset`.
Loads only `verdict == "true"` records from `<split>.jsonl` files (default:
`data/factcheck/<split>.jsonl`). No evidence Q&A documented for this
source, so `context_query` is a templated fallback, same as
`PolitiFactDataset` — no LLM required. Registry name is `"factcheck_org"`
(not `"factcheck"`, matching `docs/DATASETS.md` section 5's own usage
example) even though the module/class are named `factcheck.py`/
`FactCheckDataset`.

See `eiger/datasets/factcheck.py`'s module docstring and
`docs/DATASETS.md` Section 5 for the full field mapping and an important
caveat: the raw file format is *assumed* to be JSONL, since section 5
never explicitly specified one (unlike PolitiFact's documented TSV or
AVeriTeC's documented JSONL) — this may need revisiting once the real
CheckThat! mirror is downloaded.

`download()` is a **guard, not a fetcher**, same as the other two
implemented-loader-only datasets.

```python
from eiger.datasets import get_dataset

dataset = get_dataset("factcheck_org")  # defaults to data/factcheck/
dataset.download(target_dir="data/factcheck")  # raises if files are missing
claims = dataset.load(split="test", max_claims=50)
print(claims[0].source_dataset)  # "factcheck_org"
```

---

## CorpusClaimDataset

Implements `BaseDataset` directly, reading an Excel workbook via
`openpyxl` (no separate filter/enrich script needed — the sheet is
already the collaborator's own final, fully-processed corpus). Unlike
every other loader here, this is not a public fact-checking export: it
is a private, project-internal corpus (`Corpus_claim_RAG_Mistral_output_v3.xlsx`,
sheet `01_Corpus_claim`, 5,672 rows) built by a research collaborator's
own Mistral/Ollama pipeline. See `eiger/datasets/corpus_claim.py`'s module
docstring and `docs/DATASETS.md` §3a for the full field mapping, and
`docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7 for the two-pipeline background.

The workbook's own `normalized_label` column already uses exactly
`"verified_true"`/`"verified_false"` — mapped straight onto
`Claim.ground_truth_label` with no reinterpretation. `risk_level`/
`sensitivity_class` map straight across too (already this project's own
1-5 / S0-S3 scales). `modified_claim` (the collaborator's own
Mistral-generated poisoned variant) and its full provenance are carried
into `Claim.metadata` for inspection only — `CorpusBuilder`/the attack
registry still generates every experiment's actual poisoning
mechanically, the same as for every other loader.

**Ethical gate, not a bug:** every row currently has
`requires_human_review = True` (no human validation has happened on this
corpus yet), so `load()` returns an **empty list by default**. Pass
`include_unreviewed=True` explicitly for non-paper engineering work only
— mirrors this project's `excluded_from_benchmark`/
`allow_non_benchmark_attacks` gating pattern (see `eiger/attacks/README.md`).

`download()` is a **guard, not a fetcher**: this is a private corpus with
no automated source at all — copy or symlink the workbook into
`data/corpus_claim/` yourself, or pass `path=...` explicitly.

```python
from eiger.datasets import get_dataset

dataset = get_dataset("corpus_claim")  # defaults to data/corpus_claim/
claims = dataset.load()                        # [] today — nothing reviewed yet
claims = dataset.load(include_unreviewed=True)  # engineering-only, never paper-facing
print(claims[0].ground_truth_label)             # "verified_true" or "verified_false"
print(claims[0].metadata["manipulation_type_applied"])  # e.g. "M02"
```

---

## Adding a new dataset loader

1. Implement a `BaseDataset` subclass in a new module under `eiger/datasets/`
   (see `json_fixture.py` for the reference implementation).
2. Register it in `eiger/datasets/__init__.py`:
   `register_dataset(YourDatasetClass)`, and add it to `__all__`.
3. Reference it by name in a `DatasetConfig.name` (YAML or code).
4. Verify with `list_datasets()` and `get_dataset(name)`.

All five datasets originally documented in `docs/DATASETS.md`, plus
`CorpusClaimDataset`, now have implemented loaders; see that document for
their full specs and the (mostly completed) roadmap.

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
