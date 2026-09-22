# EIGER / EIBench — Project Claim, Research Questions & Gap Analysis

> Version: 0.1.0 | Written Sprint 3→4 transition
> Source material: `Proposta_ricerca.docx`, `Integrità Epistemica e Vulnerabilità RAG.docx`, `Spunti fasi di lavoro.docx` (shared multidisciplinary project folder — Digital Humanities + Engineering team). This document distills that material into a single reference the whole team (including GitHub collaborators without access to the shared folder) can check the codebase against. It does not replace the source documents; when the two disagree, the source documents are authoritative and this file should be corrected.

---

## Table of Contents

1. [The Claim](#1-the-claim)
2. [Research Questions and Hypotheses](#2-research-questions-and-hypotheses)
3. [Experimental Design: "The Poisoned Newsroom"](#3-experimental-design-the-poisoned-newsroom)
4. [Threat Model Summary](#4-threat-model-summary)
5. [Manipulation Taxonomy (M01–M06)](#5-manipulation-taxonomy-m01m06)
6. [Metrics: Proposed vs. Implemented](#6-metrics-proposed-vs-implemented)
7. [Data Assets: Two Parallel Claim Pipelines](#7-data-assets-two-parallel-claim-pipelines)
8. [Human-in-the-Loop Protocol](#8-human-in-the-loop-protocol)
9. [Gap Analysis and Sprint Mapping](#9-gap-analysis-and-sprint-mapping)

---

## 1. The Claim

Retrieval-Augmented Generation (RAG) was designed to reduce LLM hallucination by grounding answers in an external, retrievable corpus. This project's central claim is that this design **relocates risk rather than removing it**: the corpus itself becomes the attack surface, and a poisoned document that is stylistically credible can be retrieved, cited, and incorporated into an answer that is *faithful to its context* while being *false relative to the world*.

In a newsroom or fact-checking setting, this is not merely a technical failure. A retrieved and cited source functions as an **epistemic warrant** — it signals to a human reviewer that the claim has already been checked. The project's claim is therefore two-part:

1. **Technical claim:** a small fraction of poisoned documents (the proposal's working hypothesis: under 5% of the corpus) is sufficient to alter retrieval and generation for targeted or semi-targeted queries, and standard RAG metrics (RAGAS faithfulness/answer-relevance) can score such an answer highly *because* they measure agreement with the retrieved context, not the integrity of that context.
2. **Epistemic/editorial claim:** the presence of a citation — even a corrupted one — measurably reduces a human reviewer's propensity to verify externally, especially under time pressure. This is the "Epistemic Vigilance Drop" the project sets out to measure directly rather than assume.

EIGER (the codebase in this repository) is the **engineering half** of this project: a reproducible pipeline (ingestion → poisoning → retrieval → generation → evaluation) for producing evidence on the technical claim. The human-in-the-loop study needed to test the second, editorial half of the claim is designed in the proposal but **not yet implemented** in this codebase (see Section 8).

---

## 2. Research Questions and Hypotheses

| # | Research Question | Companion Hypothesis | Status in this codebase |
|---|---|---|---|
| **RQ1** | To what extent do manipulated-but-semantically-optimized documents enter the top-k retrieval of a local RAG system (dense retrieval, sentence embeddings)? | **H1** — Even a poison rate below 5% of the corpus can alter the ranking of retrieved contexts for targeted or semi-targeted queries. | **Partial.** `DenseRetriever` + `AttackConfig.poison_rate` + `RetrievalResult.poison_ratio`/`contains_poisoned` make this measurable per-experiment, but no systematic poison-rate sweep (0/1/3/5/10%) has been run, and vector-void mapping (Section 4) — needed to test *where* poisoning is most effective — is not implemented. |
| **RQ2** | Can RAGAS-style metrics report high faithfulness and answer-relevance even when the retrieved context is epistemically corrupted? | **H2** — Faithfulness can remain high under poisoning because it measures answer-context coherence, not context integrity. | **Partially testable, proxy caveat applies.** This is exactly what FFR (Faithful Falsehood Rate) is designed to isolate — see `docs/ARCHITECTURE.md` §3 Layer 5. However, the faithfulness/correctness signal FFR consumes today is `EmbeddingFaithfulnessScorer`, a cosine-similarity heuristic, **not** real RAGAS. Results are directionally informative but not yet publication-grade (see `docs/ARCHITECTURE.md` §2 "Current limitations"). |
| **RQ3** | Does the presence of a formally credible citation reduce the probability that a journalist detects falsification? | **H3** — Citation presence reduces external-verification time/rate when the snippet's style is journalistically plausible. | **Not started.** Requires the human-in-the-loop study (Section 8) — no participant-facing interface, no A–E condition design, exists in this repo yet. |
| **RQ4** | Is a false snippet written in institutional/wire-service style more effective — technically and cognitively — than a stylistically weak one? | **H4** — Poisoning effect increases in "vector voids" (thematic areas with low corpus coverage/redundancy). | **Split across two tracks, not reconciled yet.** EIGER's own attacks (`eiger/attacks/`) are mechanical, non-LLM edits (digit swaps, source substitution, etc.) — they do not vary *writing style/plausibility* as an independent variable. The team's separate Mistral/Ollama v3 pipeline (Section 7) *does* produce stylistically fluent, LLM-written manipulated claims with human-review gating — closer to what RQ4 needs — but it is not yet wired into `eiger.datasets`/`eiger.attacks`, so no experiment can yet compare "mechanical edit" vs. "LLM-fluent edit" detectability head-to-head. Vector-void mapping itself is not implemented. |
| **RQ5** | Do hybrid retrieval, provenance scoring, and editorial audit significantly reduce poisoning impact vs. vanilla dense retrieval? | **H5** — A multi-layer defense (hybrid retrieval + source-integrity scoring + editorial checklist) reduces both attack success and Epistemic Vigilance Drop. | **Partially started.** `SourceIntegrity` (NLI-based, `eiger/metrics/source_integrity.py`) and, as of Sprint 4, `SparseRetriever` (BM25 — `eiger/retrieval/sparse_retriever.py`) are both implemented as standalone building blocks, but no experiment has yet compared dense vs. sparse poisoning susceptibility, and `HybridRetriever` (RRF fusion of the two), reranking, and contradiction-audit remain on the roadmap (see `eiger/retrieval/README.md`). |

---

## 3. Experimental Design: "The Poisoned Newsroom"

The proposal's experimental design has 8 phases (0–7). Status below reflects what exists in this repository as of Sprint 3.

| Phase | Description | Status |
|---|---|---|
| **0 — Ethical protocol & threat model** | Define adversary, sensitivity classes, containment rules, go/no-go criteria before any corpus work. | **Formalized in this repo as of Sprint 4** — see [`docs/ETHICS_AND_THREAT_MODEL.md`](ETHICS_AND_THREAT_MODEL.md), which condenses the team's `Spunti fasi di lavoro.docx` (also summarized in Section 4 below) and proposes a go/no-go checklist plus a `risk_level`/`sensitivity_class` field mapping (pending team ratification and implementation, respectively). |
| **1 — Clean/sound corpus** | ~1,000 journalistic documents across 3–4 domains, with metadata (source, date, domain, claims, reliability, redundancy). | **Partially done, via a different mechanism than planned.** `eiger.datasets` (Snopes/AVeriTeC/PolitiFact/FactCheck.org loaders) supplies verified-true claims, not full journalistic articles, and does not yet track domain redundancy/reliability metadata at the corpus level. Separately, the team's Mistral pipeline has already processed 6,591 claims across exactly this kind of domain taxonomy (Section 7). |
| **2 — Vector void mapping** | Identify low-coverage/low-redundancy semantic areas of the corpus. | **Not implemented.** No tooling in `eiger/` computes corpus density or redundancy per topic/subtopic. |
| **3 — Controlled poisoning (0/1/3/5/10%)** | Insert a progressive share of manipulated documents; manipulation types include numeric shift, date shift, causal inversion, false attribution, decontextualized citation. | **Partially implemented.** `eiger/attacks/` implements 4 of the 6 manipulation types in the project's own taxonomy (Section 5) as deterministic, seeded, non-LLM edits; `AttackConfig.poison_rate` supports arbitrary rates but no experiment has yet swept 0/1/3/5/10% systematically. |
| **4 — Querying & retrieval audit** | Query with target/paraphrased/query-agnostic phrasings; log top-k, rank, similarity, contradictions. | **Partially implemented.** `DenseRetriever` + `RetrievalResult` gives rank/score/poison-ratio per query; no paraphrase/query-agnostic query-set methodology yet, no contradiction-audit logging. |
| **5 — Generation & RAG metrics** | RAGAS + custom metrics distinguish irrelevant answer / hallucination / faithful-to-sound-source / faithful-to-corrupted-source. | **Implemented with a caveat.** FFR + ERS + SourceIntegrity exist; the faithfulness/correctness input is a heuristic proxy, not RAGAS (see RQ2 above). |
| **6 — Human-in-the-loop** | Journalists/fact-checkers/students decide publish/verify/reject/audit on RAG answers with cited sources. | **Not implemented.** No participant interface, no condition randomization, no data-collection protocol exists in this repo (Section 8). |
| **7 — Defenses & comparison** | Compare vanilla dense retrieval vs. hybrid, reranking, provenance audit, source-integrity scoring, risk-signal UI. | **Partially implemented.** `SourceIntegrity` and, as of Sprint 4, `SparseRetriever` both exist as standalone components, but no comparison experiment (dense vs. sparse vs. hybrid poisoning susceptibility) has been run yet; reranking, provenance audit, and risk-signal UI are not implemented. |

---

## 4. Threat Model Summary

Condensed from `Spunti fasi di lavoro.docx` (Fase 0). The full text (system description, asset list, adversary capability levels, attack-vector taxonomy, domain risk cards) should be preserved verbatim in the shared project folder; this is a working summary for engineering purposes.

**As of Sprint 4, this summary has been formalized into its own versioned document: [`docs/ETHICS_AND_THREAT_MODEL.md`](ETHICS_AND_THREAT_MODEL.md).** That file is now the canonical engineering reference (it also adds a proposed go/no-go phase checklist and a `risk_level`/`sensitivity_class` field mapping); this section remains as a shorter in-context summary and should stay consistent with it.

**System in scope:** a simulated newsroom/fact-checking desk querying a local RAG system (corpus → embedding → vector store → retriever → LLM → answer with cited sources). Entirely local/offline by design (privacy, containment).

**Adversary:** functional, not attributed to a real actor. Capability is **corpus/ingestion-level only** — the adversary can get a plausible document indexed, but has no access to model weights, retriever code, infrastructure, or admin functions. Knowledge is classified **black-box** (topic only) or **grey-box** (topic + house style + likely queries); **white-box** (full pipeline knowledge) is explicitly out of scope for the pilot.

**Assets to protect:** (1) corpus integrity — verifiable separation of sound/corrective/experimental documents; (2) retrieval reliability — corrupted documents should not be systematically over-ranked; (3) epistemic quality of generated answers — faithful ≠ truthful; (4) provenance traceability — every answer traceable to its source documents; (5) editorial responsibility — the human reviewer's ability to recognize a weak/isolated/contradicted source.

**Explicitly out of scope:** attacks on real/production systems, infrastructure compromise, model weight tampering, generation of reusable disinformation payloads, targeting of private individuals/minors/vulnerable groups, any white-box or fully-automated disinformation-campaign scenario.

**Sensitivity classes (apply to every corpus item):**

| Class | Meaning | Treatment |
|---|---|---|
| S0 | Neutral | Usable anywhere, including publication |
| S1 | Sensitive but manageable | Usable in the lab; publish only sanitized |
| S2 | Critical (health, civil rights, migration, etc.) | Requires dedicated ethics review; abstract/sanitized form only |
| S3 | Excluded | Must not be produced or included in operational form (private individuals, minors, vulnerable subjects, unverified accusations) |

Rule: when in doubt between two classes, apply the more restrictive one.

**Risk levels (1–5)** run from "usable, ordinary review" (1) to "very high/critical — normally exclude or heavily abstract" (5).

**Publication rule:** poisoned/manipulated corpus items are never published in reusable form; only pipeline/logging/auditing/defense code is released, plus abstracted or heavily sanitized examples in any paper.

*Sprint 4 task, done:* this section (and the source document's condensed content) is now also available as a standalone, versioned [`docs/ETHICS_AND_THREAT_MODEL.md`](ETHICS_AND_THREAT_MODEL.md), including a proposed go/no-go checklist for moving between corpus phases — see Section 9.

---

## 5. Manipulation Taxonomy (M01–M06)

This is the project's canonical manipulation taxonomy (from the team's `03_Codebook` sheet, also embedded as safety constraints in the Mistral/Ollama v3 pipeline's system prompt). It maps directly — but not completely — onto `eiger/attacks/`'s existing registry:

| Code | Category | Constraint | Maps to `eiger/attacks/` |
|---|---|---|---|
| M01 | Numerical alteration | Use with caution; requires a verified correction | `numerical_shift` (`NumericalShiftAttack`) — ✅ implemented |
| M02 | Temporal alteration | Historicized cases only; never current deadlines | `date_manipulation` (`DateManipulationAttack`) — ✅ implemented |
| M03 | False attribution | Sanitized only; avoid imitating real official sources | `attribution_switch` (`AttributionSwitchAttack`) — ✅ implemented |
| M04 | Causal inversion | Requires explicit corrective evidence | `causal_manipulation` (`CausalManipulationAttack`) — ✅ implemented |
| M05 | Cherry-picking | Annotate the omitted period/baseline/context | `cherry_picking` (`CherryPickingAttack`) — ✅ implemented (Sprint 5) |
| M06 | Missing context | Admissible only if the resulting distortion is measurable | `missing_context` (`MissingContextAttack`) — ✅ implemented (Sprint 5) |

**Sprint 5: taxonomy complete 1:1.** `CherryPickingAttack` (M05) deletes a comparative baseline/reference-period clause (e.g. "compared to 6.8% in 2020") while leaving the surviving statistic byte-for-byte intact; `MissingContextAttack` (M06) deletes an entire qualifying/caveat sentence (e.g. "However, this figure excludes housing costs.") while leaving every other sentence unchanged. No `BaseAttack` contract extension was needed in the end — `apply()`'s existing signature (`document, seed, **kwargs -> PoisonedDocument`) already permits returning text that is a strict subset of the input; the "different shape" concern flagged below turned out to be a non-issue once implementation started. Both attacks follow the same conservative no-fallback convention as `attribution_switch`: if nothing eligible is found to omit, `attack_params["no_op"]` is `True` rather than fabricating content to then delete. Both also record `attack_params["omitted_count"]`. See `eiger/attacks/README.md` for full usage examples.

---

## 6. Metrics: Proposed vs. Implemented

| Metric | Defined in proposal as | Status |
|---|---|---|
| FFR (Faithful Falsehood Rate) | Not named identically in the proposal, but is EIGER's own operationalization of the proposal's central "faithful ≠ truthful" dissociation (RQ2/H2) | ✅ Implemented (`eiger/metrics/ffr.py`) — heuristic faithfulness proxy caveat applies |
| SI / SIS (Source Integrity (Score)) | "Source Integrity Score" — source-level integrity, distinct from answer-level faithfulness | ✅ Implemented (`eiger/metrics/source_integrity.py`, NLI-based, falls back to 0.0 without `transformers`/`torch`) — name matches the proposal's SIS almost exactly |
| ERS (Epistemic Risk Score) | Not explicitly named in the proposal under this name; conceptually related to the proposal's plausibility/verification-difficulty/editorial-risk annotation dimensions | ✅ Implemented (`eiger/metrics/`, weighted combination of `PoisonAnnotation` dimensions) |
| PRR@k (Poisoned Retrieval Rate) | `# queries with ≥1 poisoned doc in top-k / # total queries` | ✅ **Implemented (Sprint 4)** — `PRRMetric` (`eiger/metrics/prr.py`), registered as `"prr"` |
| PRD@1 (Poisoned Rank-1 Dominance) | `# queries where the poisoned doc is rank 1 / # total queries` | ✅ **Implemented (Sprint 4)** — `PRDMetric` (`eiger/metrics/prd.py`), registered as `"prd"` |
| PCS (Poisoned Context Sensitivity) | `Δ output_score` when the suspect context is removed | ❌ Not implemented — requires a counterfactual re-generation step (generate with vs. without the suspect document), not present in `eiger/experiments/` today |
| EVD (Epistemic Vigilance Drop) | `P(verify \| no citation) − P(verify \| citation)` | ❌ Not implemented — requires the human-in-the-loop study (Section 8); this is fundamentally a human-subjects measurement, not a pipeline metric |

**Sprint 4, done:** PRR@k and PRD@1 were the cheapest wins here — both were pure aggregations over data EIGER already produces (`RetrievalResult.contains_poisoned`/rank) and needed no new infrastructure, just a new `BaseMetric` implementation each (`eiger/metrics/prr.py`, `eiger/metrics/prd.py`).

---

## 7. Data Assets: Two Parallel Claim Pipelines

The project currently has **two distinct, not-yet-reconciled** paths from raw fact-checking data to usable claims:

**(A) `eiger.datasets` automated loaders** (this repo, Sprint 3): `SnopesDataset`, `AVeriTecDataset`, `PolitiFactDataset`, `FactCheckDataset`. Each loads only verified-**true** claims and constructs a `context_query` (real, templated, or LLM-enriched — see `docs/DATASETS.md`). These feed directly into `eiger.ingestion` → the poisoning engine applies EIGER's own mechanical attacks (Section 5) at experiment time. This is the path the CLI (`eiger run ...`) actually exercises today.

**(B) The team's Mistral/Ollama v3 pipeline** (`CLAIM/ollama_mistral_rag_pipeline_v3.py`, outside this Git repo, per the project's data-file convention): a separate, more elaborate workflow that has **already processed 6,591 claims** (3,203 flagged priority) sourced from Snopes/PolitiFact/FactCheck.org, each carrying: `topic`/`subtopic`, `risk_level` (1–5), `sensitivity_class` (S0–S3), `allowed_manipulation` (one of M01–M06), and — for a large subset already run through `--mode generate` — an LLM-generated `modified_claim`, `modified_verdict_normalized/text`, `modified_fact_checker_explanation`, and full generation provenance (`manipulation_type_applied`, `explanation_generation_mode`, `requires_human_review`, safety flags, etc.). This pipeline enforces its own strict rules (documented in `NOTA_METODOLOGICA_Mistral_RAG_v3.md`/`PROMPT_Mistral_RAG_v3.txt`): `claim_original` is never mutated, language is preserved, only one manipulation per item, no invented entities/sources, and every output requires human review before use.

**Why this matters:** pipeline (B) already produces exactly the kind of stylistically-fluent, taxonomy-tagged, sensitivity-classified poisoned claims that RQ4 (Section 2) needs to compare against EIGER's own mechanical attacks — but it is not yet ingested anywhere in `eiger/`. Concretely open questions for Sprint 4/5, to raise with the whole team (not just the engineering side) before building:

1. Should the consolidated Mistral output (`Corpus_claim_RAG_Mistral_output_v3_CONSOLIDATO.xlsx`) become a new `BaseDataset` loader (e.g. registry name `mistral_enriched`), or should it instead feed a *new attack strategy* (`eiger/attacks/llm_generated.py`) that wraps pre-generated `modified_claim` text as a `PoisonedDocument`, keeping the existing four mechanical attacks as a separate, comparable condition?
2. `risk_level`/`sensitivity_class` do not exist yet on `Claim`/`Document` in `eiger/core/models.py` — needed either way to respect the threat model's containment rules (Section 4) once this data enters the pipeline.
3. Every row already has `requires_human_review = True` — a loader must not silently treat unreviewed rows as usable ground truth; this needs the same `metadata["verified"] = False`-until-spot-checked convention the four existing loaders already use (see `docs/DATASETS.md` §1).

No code changes have been made yet for this integration — it is recorded here as the most concrete, highest-value Sprint 4/5 candidate (Section 9), pending a decision on (1) above.

---

## 8. Human-in-the-Loop Protocol

Defined in the proposal, not implemented in this repo:

- **Conditions (A–E):** no source; sound-cited source; poisoned-but-plausible source; poisoned source + integrity warning; multi-source comparison.
- **Sample:** journalists, fact-checkers, communication/digital-humanities students. Pilot: 12–20 participants; extended study: 40–60.
- **Task:** per item, decide publish / verify / reject / request more sources; declare a confidence score and which signals drove the decision.
- **Measures:** acceptance rate, external-verification request rate, detection rate, decision time, stated confidence, verdict accuracy (quantitative); post-task interview on trust/suspicion signals (qualitative).
- **Tooling suggested in the proposal:** Gradio, for a no-code interface the non-engineering team member can run studies with.

This is entirely future work — no Sprint currently scopes it. Flagged in Section 9 as a Sprint 5+ candidate, since it depends on Sections 5–7 (attacks, metrics, and a decided claim-ingestion path) being stable first.

---

## 9. Gap Analysis and Sprint Mapping

Concrete, actionable deltas between the full research proposal and the current codebase, roughly ordered by cost/value:

| Task | Depends on | Suggested Sprint |
|---|---|---|
| ~~Formalize `docs/ETHICS_AND_THREAT_MODEL.md` (Section 4, full text)~~ | — | ~~4~~ **Done (Sprint 4)** — condensed version + proposed go/no-go checklist and `risk_level`/`sensitivity_class` field mapping; full verbatim source text (asset list, adversary capability levels, attack-vector taxonomy, domain risk cards) remains in the shared project folder, not reproduced here — see the new doc's own §0 |
| ~~Add `PRR@k`/`PRD@1` as registered `BaseMetric` implementations~~ | Existing `RetrievalResult` data | ~~4~~ **Done (Sprint 4)** — `eiger/metrics/prr.py`, `eiger/metrics/prd.py` |
| ~~Add `risk_level`/`sensitivity_class` fields to `Claim`/`Document`~~ | Threat model doc (for defining the field's allowed values authoritatively) | ~~4~~ **Done (Sprint 4)** — typed/validated fields (not nested in free-form `metadata`), propagated from `Claim` to ground-truth `Document` and from source `Document` to `PoisonedDocument` in all 4 attacks, plus a follow-on fix to the Qdrant payload round-trip (`_document_to_payload`/`_document_from_payload`) so classification survives the default dense retriever; no dataset loader populates them yet, so every claim loaded today is still unclassified (`None`) — see `docs/ETHICS_AND_THREAT_MODEL.md` §7 |
| Decide + implement ingestion path for the Mistral-generated corpus (Section 7) | Team decision on loader-vs-attack framing | 4/5 |
| ~~Implement `CherryPickingAttack` (M05) and `MissingContextAttack` (M06)~~ | Possible `BaseAttack` contract extension | ~~4/5~~ **Done (Sprint 5)** — no contract extension was actually needed; see §5 above and `eiger/attacks/README.md` |
| ~~`SparseRetriever` (BM25)~~ | — | ~~4/5~~ **Done (Sprint 4)** — `eiger/retrieval/sparse_retriever.py`, selectable via `retriever.type: sparse` |
| `HybridRetriever` (RRF fusion of dense + sparse) | `SparseRetriever` (done) + `DenseRetriever` (done) | 5 |
| Real RAGAS-based faithfulness scorer (replace `EmbeddingFaithfulnessScorer`) | `ragas` dependency (currently in `future` extra) | 5 |
| `PCS` (Poisoned Context Sensitivity) metric | Counterfactual re-generation support in `eiger/experiments/` | 5 |
| Vector-void mapping (Phase 2) | Corpus-level density/redundancy tooling, not yet designed | 5 |
| Poison-rate sweep experiment (0/1/3/5/10%) reproducing Phase 3 | Stable attacks + metrics above | 5 |
| Human-in-the-loop study (Section 8) | Stable claim/attack/metric pipeline; a participant-facing UI | 5+ |

### Sprint 4 audit findings (code review, not part of the original proposal gap analysis)

An independent multi-area code review (four parallel reviews covering attacks/metrics, datasets/ingestion, retrieval/vector-stores/LLM, and experiments/CLI/config/utils) turned up the following, ordered by what's already fixed vs. still open:

| Finding | Status |
|---|---|
| `ERSMetric` silently returned 0.0 for every record under dense retrieval (poisoning provenance dropped on the Qdrant round-trip) | ✅ **Fixed** — see `docs/ARCHITECTURE.md` §2's limitations list |
| `CausalManipulationAttack` could split a numeric fact (e.g. "3.5%") at its decimal point and corrupt it with an injected clause | ✅ **Fixed** — `eiger/attacks/causal.py`'s `_split_sentences()` |
| 472/3,400 (13.9%) of the live Snopes claims file (`data/snopes/snopes_enriched.json`) contradicts the verified-true-only invariant (allowlist also corrected: `legit` added, `no` deliberately excluded — see `docs/DATASETS.md` §8) | ✅ **Filter fixed** for future runs; run `scripts/clean_snopes_contamination.py` (no `--dry-run`) to apply to the existing file |
| `numerical_shift`/`attribution_switch`/`date_manipulation` can silently no-op (return a "poisoned" document byte-identical to ground truth, still fully annotated) | ✅ **Fixed** — each now records `attack_params["no_op"]` (`True`/`False`); no fallback text-injection was added (would be a design change to `attribution_switch` in particular, not a bug fix) |
| `derive_seed()`'s collision-freedom claim doesn't hold if a `claim_id`/`attack.name` ever contains a colon (currently only exercised via `json_fixture` claim IDs, which are unvalidated user input) | ✅ **Fixed** — `eiger/utils/seeding.py` now length-prefixes each piece before concatenation instead of joining with a plain `:` separator |
| `ExperimentConfig.config_hash` is not truly order-independent for free-form `dict[str, Any]` fields (e.g. `AttackConfig.params`) — two semantically identical configs can hash differently | ✅ **Fixed** — `config_hash` now uses `model_dump(mode="json")` + `json.dumps(sort_keys=True)` instead of `model_dump_json()` directly |
| `PolitiFactDataset._CONTEXT_COLUMN = 8` is very likely wrong against the real LIAR TSV schema (the 5 credibility-count columns are not accounted for) | ❌ Not yet fixed — currently dormant, since no real PolitiFact data has been downloaded yet |
| `RetrievedDocument.document` (declared as base `Document`) silently dropped `PoisonedDocument`'s provenance fields on every `results.json` write, because Pydantic v2 serializes per the declared type, not the runtime instance — found by inspecting `ablation_attacks_v1`'s real output after the ERS/Qdrant fix above | ✅ **Fixed** — `SerializeAsAny[Document]` in `eiger/core/models.py`; metric *values* in older result files remain valid, but their `retrieval.hits[].document` entries lack poisoning provenance |

These are tracked here (not only in code comments) so they survive across sessions and don't get lost between a code review and the next person picking up the codebase.

This table is the living answer to "what's next" for this project — update it whenever a row is completed or a new gap is discovered, and keep `docs/DATASETS.md` §11 (dataset-specific roadmap) and this table from drifting apart.
