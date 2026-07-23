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
   `eibench_raw_claims.json`, re-mapping fields to the schema
   `JSONFixtureDataset` expects:

   | Candidate JSON field | Promoted field in `eibench_raw_claims.json` |
   |---|---|
   | `claim_id` | `claim_id` (rename to the `EIB_CLAIM_NNN` convention) |
   | `original_fact` | `original_fact` |
   | `context_query` | `context_query` |
   | `source`, `domain`, `notes`, `verified` | **dropped** — see limitation below |
   | — | `adversarial_variants` (optional; hand-authored example poisoned texts, not required — see `eiger/datasets/json_fixture.py`) |

   There is no automated promotion step (deliberately — promotion is the
   point at which a human takes responsibility for the fact-check).

### Known limitation: provenance fields are not preserved on promotion

`JSONFixtureDataset._to_claim()` only carries `adversarial_variants` into
`Claim.metadata`; `source`/`domain`/`notes`/`verified` from the candidate
file are currently **lost** when a claim is promoted, unless you paste
them into `eibench_raw_claims.json`'s (currently unused) freedom to add
extra keys per entry AND extend `JSONFixtureDataset._to_claim()` to read
them into `metadata`. This is real, un-fixed technical debt — worth doing
before claim volume grows large enough that losing source provenance
becomes a real reproducibility problem, but out of scope for the initial
intake tooling.

### Spreadsheet structure this script expects

Sheet named exactly `Claim`, columns A-E: `Fatto verificato` (required),
`Domanda naturale` (required), `Fonte`, `Dominio`, `Note (facoltativo)`.
Row 1 is the header, row 2 is the template's own worked example (always
skipped), data starts at row 3. Rows missing either required field are
reported (with their real spreadsheet row number) and skipped, not
silently dropped.
