# EIGER Datasets — Reference Guide

> Version: 0.1.0 | Sprint 3 complete (dataset layer) — all five loaders below implemented and registered

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

This is also what distinguishes EIBench's approach from benchmarks that sample naturally-occurring misinformation as-is (e.g. RAGuard) or inject adversarially-optimized text (e.g. PoisonedRAG): every poisoned document here starts from a real, independently-verified true claim, and the poisoning engine (Section 3's Layer 2 of `docs/ARCHITECTURE.md`) applies one specific, well-defined type of factual edit to it. See `docs/ARCHITECTURE.md` §2 ("Related Work and Positioning") for the full literature comparison and an honest list of current limitations.

**`ground_truth_label` and `include_verified_false` (added post-Sprint 5).** Every `Claim`/`Document` now carries an explicit `ground_truth_label` field (`"verified_true"` / `"verified_false"` / `None`), separate from `Document.doc_type` ("ground_truth"/"poisoned", EIGER's own manipulation-status axis) — see `eiger.core.models.GroundTruthLabel`'s docstring and `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7 for the full rationale (collapsing these two axes into one "clean" label was a real semantic mismatch between this codebase and the team's human-curated corpus). `AVeriTecDataset`/`PolitiFactDataset`/`FactCheckDataset.load()` all accept `include_verified_false: bool = False` — default unchanged (verified-true-only, exactly as before), opt-in `True` also returns the unambiguous false-end of each source's own rating (`"Refuted"`/`"false"`+`"pants-fire"`/`"false"`), tagged accordingly. Labels with no clear true/false meaning (LIAR's three middle-scale labels, AVeriTeC's "Not Enough Evidence"/"Conflicting Evidence") are never included either way. `SnopesDataset`'s cached `data/snopes/snopes_enriched.json` (2,928 claims) has been backfilled with `ground_truth_label: "verified_true"` (all of them passed the historical True-only filter); `scripts/enrich_snopes_claims.py --include-verified-false` exists to enrich the False side too, but has not yet been run against the full raw export (a further LLM pass, deferred pending compute — see Section 11) and has not had the same original_verdict contamination audit as the True side.

**Note on a second, parallel claim source.** The four loaders above are the automated path from raw fact-checking exports to `Claim` objects, feeding `eiger.attacks`' own mechanical poisoning strategies. The project team separately maintains a much larger, human-curated claim corpus (5,672 rows in the final, non-blocked set, across Snopes/PolitiFact/FactCheck.org, produced by a dedicated Mistral/Ollama pipeline with topic/risk/sensitivity classification and — for every row — an already-generated LLM poisoned variant). This corpus is now ingestible via `CorpusClaimDataset` ("corpus_claim" — see Section 3a below), but **every row still has `requires_human_review = True`**, so `load()` returns zero claims unless the caller explicitly opts in with `include_unreviewed=True` (engineering use only, never for paper-facing results) — see [`docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7](CLAIM_AND_RESEARCH_QUESTIONS.md#7-data-assets-two-parallel-claim-pipelines) for the full picture and the human-validation plan.

---

## 2. Supported Datasets

| Name | Language | Approx. Size | Domain | License | Status |
|---|---|---|---|---|---|
| Snopes | English | 4,832 verified-true claims (of 19,631 raw) | Multi-domain | Research use | **Implemented** (Sprint 3) |
| AVeriTeC | English | 4,500 claims (Supported-label subset) | Multi-domain | CC BY 4.0 | **Implemented**, including automated `download()` — see Section 3 (unverified against a live network call) |
| PolitiFact | English | 21,000+ claims (12,800 in base LIAR; "true"-label subset used) | US Politics | Research use | **Implemented**, including automated `download()` — see Section 4 (unverified against a live network call) |
| FactCheck.org | English | ~3,000 claims ("true"-verdict subset used) | Multi-domain | Research use | **Implemented** (loader only — see Section 5; automated download() still pending) |
| JSON Fixture | Italian (demo) | 1 claim | Economics | Internal | Implemented (Sprint 1) |
| Corpus Claim (Mistral) | Multilingual | 5,672 claims, 0 human-reviewed today | Multi-domain | Private (project-internal) | **Implemented** (loader only — see Section 3a; `load()` returns 0 claims by default, ethical gate) |

---

## 3. AVeriTeC

### Description

AVeriTeC (Automated Verification of Textual Claims over Evidence) is a benchmark for automated fact-checking with evidence retrieval. Each claim is paired with a verdict, a list of question-answer evidence pairs, and metadata about the source and date. AVeriTeC is the primary target corpus for EIBench because it is multi-domain, English, and freely licensed under CC BY 4.0.

The dataset was introduced at NeurIPS 2023 and contains approximately 4,500 claims spanning politics, science, health, and economics. Evidence is linked to web sources, making the retrieval context realistic.

**Status: `AVeriTecDataset` (registry name `"averitec"`) is implemented** — `eiger/datasets/averitec.py`, 35 unit tests (`tests/unit/test_averitec.py`). `download()` now fetches automatically via the HuggingFace `datasets` library (see below), falling back to a guard-and-raise if the fetch fails; parsing, filtering, and `Claim` construction are fully implemented and tested. The fetcher itself is unit-tested against a mocked Hub client but has not yet been exercised against a live network call — verify it once against the real Hub before relying on it in a real experiment.

Only `label == "Supported"` records are loaded, mirroring Snopes' `normalised_rating == True` filter: `Claim.original_fact` must be a verified TRUE statement (EIGER generates its own falsehoods via the attack registry, it does not import externally-sourced false claims as ground truth). Unlike Snopes, no LLM enrichment step is needed: each record's `evidence` list already contains real question/answer/url triples from AVeriTeC's own human annotators, so the loader uses the first evidence question directly as `Claim.context_query`. As with Snopes, claims are tagged `metadata["verified"] = False` until the research team independently spot-checks a sample.

### Download

`AVeriTecDataset.download()` fetches automatically: it no-ops if `*.jsonl` files already exist under the target directory; otherwise it attempts `datasets.load_dataset("chenxwh/AVeriTeC", split=...)` for each of `train`/`dev`/`test` and writes each split that succeeds to `<target_dir>/<split>.jsonl`, raising a clear `IngestionError` only if literally every split fails (e.g. no network, the `datasets` library isn't installed, or the Hub dataset ID has changed).

**Install the optional dependency in an isolated virtual environment** — installing `datasets` was found, in this project's own verification pass, to silently upgrade `numpy` past this project's `numpy<2.0` pin via its `pandas` dependency, which is a real risk if done in a venv shared with another project (see `pyproject.toml`'s `data-import` extra for the full note):

```bash
pip install 'eiger[data-import]'   # or: pip install datasets
```

```python
from eiger.datasets import AVeriTecDataset
AVeriTecDataset().download("data/averitec")
```

**Caveat**: this fetcher is unit-tested against a mocked Hub client but has not yet been exercised against a live network call (this project's own sandbox cannot reach huggingface.co). Run it once for real before relying on it, and report back if the Hub dataset ID or its split names have changed.

If the automated fetch fails, download directly from the AVeriTeC GitHub repository instead:

```bash
git clone https://github.com/Raldir/AVeriTeC.git /tmp/averitec_repo
cp /tmp/averitec_repo/data/*.json data/averitec/
```

Split files are expected at `data/averitec/<split>.jsonl` (e.g. `test.jsonl`, `dev.jsonl`, `train.jsonl`) — `load(split=...)` selects the file by exact name.

### Expected Format

AVeriTeC JSONL records contain the following fields relevant to EIGER:

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | The factual claim text |
| `label` | `str` | Verdict: `Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting` |
| `evidence` | `list[dict]` | List of `{"question": str, "answer": str, "url": str}` |
| `claim_date` | `str` | Date the claim was made (ISO format) |
| `speaker` | `str` | Entity who made the claim |

Field mapping to `Claim` (see `eiger/datasets/averitec.py`'s docstring for the full rationale):

| Raw field | Maps to |
|---|---|
| `claim` | `Claim.original_fact` |
| `evidence[0]["question"]` (or a templated fallback if no evidence) | `Claim.context_query` |
| *(constructed, see below)* | `Claim.claim_id` — `f"AVERITEC_{index:05d}"`, where `index` is the record's zero-based position in the raw split file (assigned before the Supported-only filter, so IDs stay stable across filter changes) |
| `label`, `claim_date`, `speaker`, `evidence[*]["url"]` (each if present) | `Claim.metadata["label"/"claim_date"/"speaker"/"evidence_urls"]` |
| *(always)* | `Claim.metadata["verified"] = False` |

### Loading with EIGER

```python
from eiger.datasets import get_dataset

# The AVeriTeC loader maps 'claim' -> Claim.original_fact
# and uses the first evidence question as context_query.
dataset = get_dataset("averitec")
dataset.download(target_dir="data/averitec")  # fetches automatically; raises only if every split fails
claims = dataset.load(split="test", max_claims=100)

print(f"Loaded {len(claims)} claims")
print(f"Dataset content hash: {dataset.content_hash}")
# Example: Loaded 100 claims
# Dataset content hash: 3f8a1c2d9e4b7f0a
```

---

## 3a. Corpus Claim (Mistral-generated, human-curated)

### Description

Unlike the four loaders above, this is not a public fact-checking export: it is a project-internal, human-curated claim corpus (`Corpus_claim_RAG_Mistral_output_v3.xlsx`, sheet `01_Corpus_claim`) built by a research collaborator's own Mistral/Ollama pipeline. 5,672 rows in the final set (of 21,804 raw candidates), spanning Snopes/PolitiFact/FactCheck.org source claims, each already annotated with `normalized_label` (`verified_true`/`verified_false` — mapped straight across to `Claim.ground_truth_label`), `risk_level` (1-5) and `sensitivity_class` (S0-S3, already this project's own scales), plus a full manipulation-provenance trail (`manipulation_type_applied`, `modified_claim`, `claim_change_description`, etc.) for a variant the collaborator's own pipeline already generated.

`CorpusClaimDataset` ("corpus_claim") loads `claim_original` as `Claim.original_fact` — `modified_claim` and its provenance are carried into `Claim.metadata` for inspection only, not consumed as a poisoned Document: EIGER's own attack registry generates every experiment's poisoning mechanically and deterministically, the same way it does for every other loader in this file, so a single seed-derived mechanism produces every manipulated document regardless of which claim source is in use.

**Every row currently has `requires_human_review = True`** (see `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7, point 4 — no human validation has happened yet). `load()` defaults to returning an empty list rather than silently treating unreviewed rows as usable; pass `include_unreviewed=True` for non-paper engineering work only. This mirrors this project's `excluded_from_benchmark`/`allow_non_benchmark_attacks` gating pattern (`eiger/attacks/README.md`) — an explicit opt-in, not a silent default. `sensitivity_class == "S3"` ("excluded", per `docs/ETHICS_AND_THREAT_MODEL.md` §5) is excluded unconditionally, the same way blocked rows are — no S3 rows exist in the file today, but the schema allows the value and this is the first loader to populate `sensitivity_class` from real external data, so the exclusion is enforced in code rather than assumed. A post-implementation review found and fixed several robustness gaps before this shipped (a blank spreadsheet row would otherwise become a claim with literal text `"None"`, a corrupt/non-`.xlsx` file raised an unhandled exception instead of `IngestionError`, and a malformed `risk_level`/`sensitivity_class` cell could abort loading every other valid row) — see the module's own docstring and test file for details.

### Setup

This is a private corpus with no automated download. Copy or symlink the workbook to `data/corpus_claim/Corpus_claim_RAG_Mistral_output_v3.xlsx`, or pass an explicit `path=...` to `CorpusClaimDataset()`.

### Loading with EIGER

```python
from eiger.datasets import get_dataset

dataset = get_dataset("corpus_claim")
claims = dataset.load()  # [] today — nothing has passed human review yet
claims = dataset.load(include_unreviewed=True)  # engineering-only, not paper-facing
```

---

## 4. PolitiFact

### Description

PolitiFact is one of the largest publicly available fact-checking datasets, covering US political statements rated on a six-point scale: Pants on Fire, False, Mostly False, Half True, Mostly True, True. The dataset was introduced with the LIAR benchmark (Wang 2017) and has been extended in multiple follow-up works.

EIBench uses PolitiFact to study epistemic robustness in political discourse, a domain where misattribution and numerical shift attacks are particularly impactful.

**Status: `PolitiFactDataset` (registry name `"politifact"`) is implemented** — `eiger/datasets/politifact.py`, 34 unit tests (`tests/unit/test_politifact.py`). Like AVeriTeC, `download()` now fetches automatically (see below), falling back to a guard-and-raise if the fetch fails.

Only `label == "true"` records are loaded — the strictest of the six ratings, and the same verified-true-only philosophy as Snopes/AVeriTeC (see Section 1's Overview): `Claim.original_fact` must be a verified TRUE statement, since EIGER generates its own falsehoods via the attack registry rather than importing externally-sourced false claims as ground truth. **Correction:** an earlier draft of this section's "Loading with EIGER" example suggested filtering to `label in {"false", "pants-fire"}` "for use as adversarial ground truth" — that contradicted the philosophy actually implemented here and in every other loader, and has been corrected below. LIAR has no evidence Q&A pairs (unlike AVeriTeC) and no natural-language question at all (like raw Snopes), so `context_query` is a simple templated fallback (`"Is it true that {statement}?"`) — no LLM enrichment step required, though a future `scripts/enrich_politifact_claims.py` (mirroring `scripts/enrich_snopes_claims.py`) could improve its quality later.

### Download

`PolitiFactDataset.download()` fetches automatically: it no-ops if `*.tsv` files already exist under the target directory; otherwise it downloads `liar_dataset.zip` via stdlib `urllib` (no new dependency) and extracts every `*.tsv` member into the target directory (flattened to just its filename, in case the archive nests them under a subdirectory), raising a clear `IngestionError` if the download or extraction fails.

```python
from eiger.datasets import PolitiFactDataset
PolitiFactDataset().download("data/politifact")
```

**Caveat**: this fetcher is unit-tested against a mocked HTTP response but has not yet been exercised against a live network call. Run it once for real before relying on it — and if you hit `SSLCertVerificationError` on macOS with a python.org-installed Python, that's a local certificate-bundle issue, not a bug: run that Python version's own "Install Certificates.command", or set `SSL_CERT_FILE` to `certifi.where()`.

If the automated fetch fails, download manually instead:

```bash
mkdir -p data/politifact

# LIAR dataset (base version, 12,800 statements)
wget https://www.cs.ucsb.edu/~william/data/liar_dataset.zip -O /tmp/liar.zip
unzip /tmp/liar.zip -d data/politifact/

# Columns: id, label, statement, subject, speaker, job_title,
#           state_info, party_affiliation, context, justification
```

Split files are expected at `data/politifact/<split>.tsv` (LIAR's own standard split names: `train.tsv`, `test.tsv`, `valid.tsv`) — `load(split=...)` selects the file by exact name.

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

**Caveat:** this column table (in particular `context` at index 8) has not been independently re-verified against a real downloaded LIAR file — it is carried over from this project's original dataset-planning notes. `PolitiFactDataset` is deliberately defensive about this: only columns 0-2 (`id`/`label`/`statement`) are strictly required per row, and `subject`/`speaker`/`job_title`/`context` are read best-effort and simply omitted from `Claim.metadata` if a row is too short to contain them. If real data reveals a different column layout, only the loader's column-index constants need correcting.

Field mapping to `Claim`:

| Raw field | Maps to |
|---|---|
| `statement` (col 2) | `Claim.original_fact` |
| *(templated, see above)* | `Claim.context_query` — `f"Is it true that {statement}?"` |
| `id` (col 0, `.json` suffix stripped if present) | `Claim.claim_id` — `f"POLITIFACT_{id}"` |
| `label`, `subject`, `speaker`, `job_title`, `context` (each if present/non-blank) | `Claim.metadata["label"/"subject"/"speaker"/"job_title"/"context"]` |
| *(always)* | `Claim.metadata["verified"] = False` |

### Loading with EIGER

```python
from eiger.datasets import get_dataset

dataset = get_dataset("politifact")
dataset.download(target_dir="data/politifact")  # fetches and extracts liar_dataset.zip automatically

# Only the verified 'true'-label subset is returned — see the Description
# above for why this differs from an earlier (incorrect) draft of this
# example that suggested filtering to false/pants-fire claims.
claims = dataset.load(split="test", max_claims=200)
```

---

## 5. FactCheck.org

### Description

FactCheck.org is a non-partisan US fact-checking organization. Their public corpus covers political and scientific claims with detailed rebuttals, primary source citations, and structured verdicts. The corpus is smaller than PolitiFact but has higher editorial depth per claim, making it useful for studying complex multi-hop poisoning scenarios.

**Status: `FactCheckDataset` (registry name `"factcheck_org"`) is implemented** — `eiger/datasets/factcheck.py`, 25 unit tests (`tests/unit/test_factcheck.py`). Unlike AVeriTeC/PolitiFact (which gained real fetchers — see Sections 3-4), `download()` here deliberately remains a manual/guard step: this class's own module docstring already flags the CheckThat! archive's internal JSONL path/format as unverified, and automating a fetch on top of an unconfirmed layout would be guessing rather than engineering.

Only `verdict == "true"` records are loaded, matching every other loader's verified-true-only philosophy (Section 1's Overview): `Claim.original_fact` must be a verified TRUE statement. Like PolitiFact, there is no evidence Q&A documented for this source, so `context_query` is a templated fallback (`"Is it true that {claim}?"`) — no LLM enrichment required.

**Format caveat:** unlike PolitiFact (explicitly TSV) or AVeriTeC (explicitly JSONL), this section does not specify a concrete raw file format for the CheckThat! mirror below — only the field table. `FactCheckDataset` assumes **JSON Lines** (one JSON object per line, matching AVeriTeC's format), since that's the most natural fit for the flat record shape below. This is an unverified assumption; if the real downloaded mirror uses a different format, only the loader's parsing method needs to change.

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

Split files are expected at `data/factcheck/<split>.jsonl` — `load(split=...)` selects the file by exact name.

### Expected Format

| Field | Type | Description |
|---|---|---|
| `claim_id` | `str` | Unique identifier |
| `claim` | `str` | The claim text |
| `verdict` | `str` | `true`, `false`, `mixture`, `unverifiable` |
| `article_url` | `str` | Link to the full fact-check article |
| `date` | `str` | Publication date |

Field mapping to `Claim`:

| Raw field | Maps to |
|---|---|
| `claim` | `Claim.original_fact` |
| *(templated)* | `Claim.context_query` — `f"Is it true that {claim}?"` |
| `claim_id` | `Claim.claim_id` — `f"FACTCHECK_{claim_id}"` |
| `verdict`, `article_url`, `date` (each if present) | `Claim.metadata["verdict"/"article_url"/"date"]` |
| *(always)* | `Claim.metadata["verified"] = False` |

### Loading with EIGER

```python
from eiger.datasets import get_dataset

dataset = get_dataset("factcheck_org")
dataset.download(target_dir="data/factcheck")  # guard: raises if files are missing
claims = dataset.load(split="test", max_claims=50)
```

---

## 6. JSON Fixture

### Description

The JSON fixture is a lightweight, self-contained dataset bundled with the EIGER repository. It is the only dataset with active status, and as of Sprint 3 it is loaded through a real `JSONFixtureDataset` (`BaseDataset`) implementation registered in `eiger.datasets`, not through ad-hoc `json.load()` calls. It is used for:

- Unit and integration tests (deterministic, no network access required)
- Development and debugging of the poisoning engine
- CI pipeline validation

The fixture currently contains one claim in Italian, demonstrating the multi-lingual capability of the framework and reflecting the research team's initial prototype. English claims are now available via `SnopesDataset` (Section 8) and `AVeriTecDataset` (Section 3), both implemented.

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

Like AVeriTeC/PolitiFact/FactCheck.org above (also implemented, Sections 3-5), Snopes is **implemented**: `eiger/datasets/snopes.py`'s `SnopesDataset` (registry name `"snopes"`) loads it via the same JSON schema as the JSON Fixture (Section 7), inherited by subclassing `JSONFixtureDataset`. Unlike those three, Snopes' `context_query` requires a separate LLM-enrichment step (`scripts/enrich_snopes_claims.py`) rather than a templated fallback or documented evidence field — see below.

The raw source is a 19,631-row export of Snopes fact-checks (1995-2025) with a `normalised_rating` per claim (`True`: 4,832 · `False`: 11,872 · `partially true`: 2,335 · `misleading`: 451 · `unverifiable`: 141) and a real source `url` per row — verifiable, unlike the bare `id`/`claim`/`date` PolitiFact export the team also has on hand (see Section 4; that export has no per-row source link).

Two things the raw export does NOT have, which `scripts/enrich_snopes_claims.py` adds:
1. Only `normalised_rating == True` rows are kept (a `Claim.original_fact` must be a verified true statement — EIGER generates falsehoods itself via the attack registry, it does not import externally-sourced false claims as ground truth).
2. A `context_query` (natural-language question) is generated per claim via a local Ollama LLM, since Snopes indexes fact-checks by claim, not by the question a person would ask to surface one.

**Verification status.** Every claim produced by the enrichment script is tagged `metadata["verified"] = false`, even though Snopes itself already rated it `True` — Snopes' own rating is not treated as a substitute for this research team's own review before a claim is reported on in published results. See `scripts/README.md` for the full collect → filter → enrich → (team) verify workflow, which mirrors the manual claim-intake workflow (`scripts/import_claims_xlsx.py`) but for bulk external data instead of hand-authored claims.

**Known data-quality issue, found and fixed (Sprint 4 audit).** `filter_and_dedupe()` originally trusted the raw export's `normalised_rating` column alone, without cross-checking the same row's own `original_verdict` (already carried into every output entry's `notes` field, but never validated). A full pass over the live `data/snopes/snopes_enriched.json` (via `scripts/clean_snopes_contamination.py --dry-run`) found 472 of 3,400 claims generated so far (13.9%) whose `original_verdict` directly contradicted "verified true" (`Mixture`: 251, `Unproven`: 85, `Miscaptioned`: 47, `Research In Progress`: 30, `Recall`: 25, `no`: 12, `Fake`: 9, `Misattributed`: 7, `Scam`: 4, `Unfounded`: 2) despite `normalised_rating` being `True` for every one of them — a direct violation of this project's core verified-true-only ground-truth invariant. `filter_and_dedupe()` now cross-checks `original_verdict` too (see `_VERIFIED_TRUE_ORIGINAL_VERDICTS` in `scripts/enrich_snopes_claims.py`), fixing this for future/resumed enrichment runs.

Note on the allowlist: an initial dry-run flagged 481 rows (9 more than the 472 above) with `original_verdict=Legit`. Manual spot-check of 3 sample rows (genuine Crown Royal/class-action-settlement notices) confirmed Snopes uses `Legit` as its own "true" rating for this claim category, so `"legit"` was added to `_VERIFIED_TRUE_ORIGINAL_VERDICTS` — excluding it would have been a false positive (discarding verified-true claims, not contamination). A separate `original_verdict=no` value (12 rows) was also inspected and is **not** in the allowlist: it isn't a real Snopes rating and contradicts `normalised_rating=True` on those same rows, which looks like a raw data-quality issue in the upstream Snopes export rather than a bug in this filter — those rows stay excluded.

Run `python scripts/clean_snopes_contamination.py` (no `--dry-run`) to apply this to the already-generated file before trusting any FFR/ERS/Source Integrity numbers computed from `data/snopes/snopes_enriched.json`; see that script's own docstring for exactly what it does.

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
| Snopes (English, up to 4,832 in the raw export; 3,400 enriched so far, 2,928 confirmed verified-true after Sprint 4 contamination cleanup) | Sprint 3 | **Implemented.** `SnopesDataset` + `scripts/enrich_snopes_claims.py` (filter/dedupe/LLM-generated context_query). Not yet independently reviewed by the research team — see Section 8. |
| AVeriTeC (English, ~4,500 claims, Supported-label subset) | Sprint 2 | **Implemented, including automated `download()`.** `AVeriTecDataset` — parsing/filtering/`Claim` construction fully implemented and unit-tested (Section 3). `download()` now fetches all three splits via HuggingFace `datasets.load_dataset("chenxwh/AVeriTeC", ...)` (falls back to the existing guard-and-raise behavior if the fetch fails); DVC tracking is still outstanding. Written and unit-tested against a mocked Hub client. **Real-world verification attempted, not completed**: installing the required `datasets` package into a real dev venv was found to silently upgrade `numpy` to 2.x (violating this project's own `numpy<2.0` pin) via its `pandas` dependency — a real risk if that venv is shared with any other project (this happened during this project's own verification pass, in a venv also used by an unrelated Flower/federated-learning experiment, and was caught and reverted before the network call itself was reached). **Install this extra in an isolated virtual environment only** (see the warning now in `pyproject.toml`'s `data-import` group), and re-attempt the live-network verification from there. |
| PolitiFact via LIAR (English, ~12,800 claims, "true"-label subset) | Sprint 3 | **Implemented, including automated `download()`.** `PolitiFactDataset` — parsing/filtering/`Claim` construction fully implemented and unit-tested (Section 4). `download()` now fetches and extracts `liar_dataset.zip` via stdlib `urllib`/`zipfile` (no new dependency, so no risk of the numpy conflict above). Written and unit-tested against a mocked HTTP response. **Real-world verification attempted, not completed**: hit a local `SSLCertVerificationError` on macOS running Python 3.14 installed from python.org (that distribution doesn't register the system CA bundle by default — a known, actionable, non-code issue: run that Python version's own "Install Certificates.command", or set `SSL_CERT_FILE` to `certifi.where()`) before the network call itself could be confirmed working. Note: the team also has a bulk PolitiFact export on hand (`politifact_true.csv`/`politifact_false.csv`, id/claim/date only, no source URL) — not used by this loader, which targets the standard LIAR TSV format instead; the team's export remains a lower-priority alternative source due to the missing per-row source link and minor non-English contamination in the false subset. |
| FactCheck.org via CheckThat! (English, ~3,000 claims, "true"-verdict subset) | Sprint 4 | **Implemented (loader only); `download()` deliberately still a guard, not a fetcher.** `FactCheckDataset` — parsing/filtering/`Claim` construction fully implemented and unit-tested (Section 5). Unlike AVeriTeC/PolitiFact above, this loader's own module docstring already flags the CheckThat! archive's internal JSONL path/format as "assumed", never independently re-verified — writing an automated fetcher on top of an unconfirmed internal layout would be guessing, so this was scoped out rather than built on sand. Verifying the archive's real structure (needs network access) is the prerequisite before a fetcher can be written here. Not yet independently reviewed. Note: the team also has a 50-row bulk-extracted `factcheck_false.csv`, every row explicitly flagged `needs_manual_check: True` — candidate FALSE claims requiring manual review, not a source of `Claim.original_fact` ground truth, and not used by this loader. |
| Corpus Claim / Mistral (Multilingual, 5,672 rows final, 21,804 raw) | Post-Sprint-5 | **Implemented (loader only).** `CorpusClaimDataset` — parsing/mapping/`Claim` construction fully implemented and unit-tested (Section 3a). `load()` returns 0 claims by default: 100% of rows still have `requires_human_review=True` — see `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7 for the stratified human-validation plan that will actually lower that count. `download()` is a guard only (private corpus, no automated fetch). |
| Multi-lingual extension (Italian, French, German) | Sprint 5 | **Scoped.** Investigated against the real corpus_claim data: only 17/5,672 rows (0.3%) are genuinely non-English (Spanish, via a PolitiFact source); 19 more rows are mistagged (language-ID false positives on short English strings) — no genuine IT/FR/DE claim content exists in the corpus today. Found and fixed a real bug where `original_language_iso` was silently dropped at load time. Found and documented (not yet fixed) that `CausalManipulationAttack`/`MissingContextAttack` assume English text — see `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7 for the full analysis and the open decision for the research team. |
| Mistral/Ollama v3 corpus integration (6,591 claims, topic/risk/sensitivity-tagged, partially LLM-poisoned already) | Sprint 4/5 | **Done.** Loader-vs-attack framing resolved: ingested via `CorpusClaimDataset` (loader only); the collaborator's own `modified_claim`/poisoning provenance stays in `Claim.metadata` for inspection, never consumed as EIGER poisoning — EIGER's own attack registry still generates every experiment's poisoning mechanically. See `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7. |

Note: the "Sprint" column above is this document's own dataset-specific roadmap numbering, established during Sprint 1 planning, and does not necessarily align 1:1 with the project's actual sprint cadence (e.g. the retrieval/generation/orchestration layer built in the project's own "Sprint 2" did not touch datasets at all). The `eiger.datasets` registry and `JSONFixtureDataset` described in Sections 6-7, `SnopesDataset` described in Section 8, `AVeriTecDataset` described in Section 3, `PolitiFactDataset` described in Section 4, and `FactCheckDataset` described in Section 5 were all implemented during the project's Sprint 3. All five datasets originally planned in this roadmap now have implemented loaders; multi-lingual extension (Sprint 5) is the only remaining planned item.

The JSON fixture will remain in the repository indefinitely as the canonical fast-test dataset. All CI pipelines run against the fixture (and, once independently reviewed, Snopes, AVeriTeC, PolitiFact, and FactCheck.org) only; full-scale experiments against the real corpora are run on the research compute cluster and results are archived under `experiments/`.
