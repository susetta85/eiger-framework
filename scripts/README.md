# scripts/

Standalone utility scripts. Unlike `eiger/`, this directory is not covered
by the 100% test-coverage gate (`pyproject.toml`'s `[tool.coverage.run]`
explicitly omits `scripts/*`) — these are one-off tools, not framework
code, matching the same convention already used by the repo-root
quickstart scripts (`pipeline_eibench.py`, `epistemic.py`, `engine.py`).

---

## `sample_corpus_claim_for_review.py` — stratified human-review sample

Selects a stratified, reproducible sample from the real corpus
(`Corpus_claim_RAG_Mistral_output_v3.xlsx`) and writes it out as an
`.xlsx` workbook ready for a human fact-checker to review — the concrete
tool proposed in the "EIGER x Corpus Claim - Protocollo di allineamento"
alignment document, point 4 ("definire un campione di validazione").
Every row in the corpus currently has `requires_human_review = True` (see
`eiger/datasets/corpus_claim.py`); reviewing all 5,672 at once isn't
realistic, so this produces a representative batch instead.

```bash
python scripts/sample_corpus_claim_for_review.py \
    /path/to/Corpus_claim_RAG_Mistral_output_v3.xlsx \
    -o review_batch_001.xlsx --sample-size 400 --min-per-stratum 20
```

Stratifies on `manipulation_type_applied` (M01/M02/M04/M06) x
`sensitivity_class` (S1/S2), proportional allocation with a floor per
stratum so small categories (e.g. M04/S1) aren't drowned out by large
ones (e.g. M06/S2). Deterministic: the same `--seed` always produces the
same sample (`eiger.utils.seeding`). Output columns include everything a
reviewer needs (`claim_original`, `modified_claim`,
`claim_change_description`, topic/subtopic, source URL) plus empty
columns for her own judgment (`plausibility`, `editorial_risk`,
`verification_difficulty`, `review_decision`, `reviewer_notes`) — see the
script's own module docstring. It does **not** write back to the master
workbook or flip `requires_human_review` anywhere; folding a completed
review batch back into the corpus is a separate, deliberate step.

---

## `sample_for_faithfulness_calibration.py` / `compute_calibration_correlation.py` — scorer calibration

Two-step tool to calibrate `EmbeddingFaithfulnessScorer`/`RAGASFaithfulnessScorer`
against real human judgments — closing the biggest open scientific-validity
gap flagged in `docs/ARCHITECTURE.md` Section 2 ("neither scorer has been
calibrated against human judgments").

**Step 1 — sample a batch for blind annotation** from any completed
experiment's `results.json`:

```bash
python scripts/sample_for_faithfulness_calibration.py \
    results/poison_rate_sweep/5pct/results.json \
    -o calibration/batch_001.xlsx --sample-size 40 --seed 42
```

Stratifies on whether each record's retrieval contained a poisoned
document, so the batch covers both clean and poisoned-context generations.
Writes two files: the annotation workbook itself (query, ground-truth fact
when its document was retrieved, retrieved context, generated answer, plus
empty `faithfulness_human`/`correctness_human` 1-5 columns and
`annotator_notes`) and a sibling `<output>.automated_scores.json` holding
the scorer's own numbers — **the workbook never shows the automated score
next to a record**, so the annotator's rating isn't anchored by the very
number being evaluated. Don't open the `.automated_scores.json` file while
annotating.

**Step 2 — after a human has filled in `faithfulness_human`/
`correctness_human`**, compute correlation against the automated scorer:

```bash
python scripts/compute_calibration_correlation.py calibration/batch_001.xlsx
```

Reports Pearson's r, Spearman's rho, and MAE for both faithfulness and
answer-correctness (human 1-5 ratings rescaled to [0, 1] first). Rows left
blank are skipped, not treated as 0. No pass/fail threshold is hard-coded —
conventional weak/moderate/strong bands are printed for reference only;
deciding what correlation is "good enough" for this project's purposes is
the research team's call, not this script's.

---

## `import_claims_xlsx.py` — claim collection intake

Converts a filled-in copy of `eiger_claims_template.xlsx` (the spreadsheet
handed to non-technical contributors, with an "Istruzioni" tab and a
"Claim" tab) into a JSON file of **candidate** claims.

```bash
pip install 'eiger[data-import]'   # one-time: installs openpyxl

python scripts/import_claims_xlsx.py path/to/filled_claims.xlsx \
    -o claims_candidate_batch1.json
```

### The verify-then-promote workflow

Claims collected this way are entered by a human contributor but have not
been fact-checked against their cited source yet. This script performs
**zero verification** — it only structures what was typed into the sheet
and tags every row `"verified": false`. It never writes to
`eibench_raw_claims.json` (the fixture actually consumed by
`JSONFixtureDataset` and exercised by the test suite / CI), so unverified
content can never silently leak into the framework's canonical regression
fixture or into a real experiment run.

1. **Collect** — a contributor fills in `eiger_claims_template.xlsx`.
2. **Convert** — run this script; get a `claims_candidate_*.json` file,
   every entry `"verified": false`.
3. **Verify** — a researcher opens each candidate's `"source"` field and
   confirms the `"original_fact"` is actually supported by that source.
4. **Promote** — for each verified claim, manually copy it into
   `eibench_raw_claims.json`, flipping `"verified"` to `true`. Every
   field is preserved: `JSONFixtureDataset._to_claim()` carries
   `source`/`domain`/`notes`/`verified` into `Claim.metadata` verbatim
   when present (see `eiger/datasets/json_fixture.py` and
   `eiger/datasets/README.md`), so nothing needs re-mapping except
   renaming `claim_id` from the `EIB_CANDIDATE_NNN` convention to
   `EIB_CLAIM_NNN` (a cosmetic convention, not a schema requirement).

   There is no automated promotion step (deliberately — promotion is the
   point at which a human takes responsibility for the fact-check).

### Spreadsheet structure this script expects

Sheet named exactly `Claim`, columns A-E: `Fatto verificato` (required),
`Domanda naturale` (required), `Fonte`, `Dominio`, `Note (facoltativo)`.
Row 1 is the header, row 2 is the template's own worked example (always
skipped), data starts at row 3. Rows missing either required field are
reported (with their real spreadsheet row number) and skipped, not
silently dropped.

---

## `enrich_snopes_claims.py` — bulk external dataset intake (Snopes)

Converts a raw Snopes fact-check export (e.g. `Snopes.xlsx`) into the JSON
schema `SnopesDataset` (`eiger/datasets/snopes.py`) expects: filters to
`normalised_rating == True` (a verified-true statement — EIGER generates
falsehoods itself via the attack registry, it does not import
externally-sourced false claims as ground truth), deduplicates by
`claim_id`, and generates a `context_query` per claim via a local Ollama
LLM (Snopes fact-checks are indexed by claim, not by a natural question).

```bash
pip install 'eiger[data-import]'   # one-time: installs openpyxl

# Pilot run first — always check output quality before the full batch:
python scripts/enrich_snopes_claims.py path/to/Snopes.xlsx --limit 20 -o /tmp/pilot.json

# Full run (requires a running Ollama server with the model pulled — already
# set up by `make bootstrap`, or manually via `make ollama-pull`):
python scripts/enrich_snopes_claims.py path/to/Snopes.xlsx \
    -o data/snopes/snopes_enriched.json
```

The script is idempotent and resumable: re-running it with the same
`-o` skips already-enriched `claim_id`s and only processes new ones, and
progress is checkpointed to disk every 25 claims by default
(`--checkpoint-every`) so an interruption loses at most that many claims.

Every output claim is tagged `"verified": false`, even though Snopes
itself already rated it `True` — this is exactly the same
verify-then-promote posture as `import_claims_xlsx.py` above: Snopes'
own rating is not a substitute for this research team's own review
before a claim is reported on in published results. See
`docs/DATASETS.md`, Section 8, for the full rationale, the exact raw
column requirements, and what the team decided *not* to use yet (a bulk
PolitiFact export with no per-row source URL, and a 50-row FactCheck.org
extraction explicitly flagged `needs_manual_check: True`).

`data/` (both the raw export and the enriched output) is gitignored —
not ours to redistribute and too large for Git.
