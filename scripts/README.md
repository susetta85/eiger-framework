# scripts/

Standalone utility scripts. Unlike `eiger/`, this directory is not covered
by the 100% test-coverage gate (`pyproject.toml`'s `[tool.coverage.run]`
explicitly omits `scripts/*`) — these are one-off tools, not framework
code, matching the same convention already used by the repo-root
quickstart scripts (`pipeline_eibench.py`, `epistemic.py`, `engine.py`).

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

# Full run (requires a running Ollama server with the model pulled —
# `make up`, then `docker exec eiger-ollama ollama pull llama3.1:8b`):
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
