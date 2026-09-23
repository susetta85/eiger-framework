# EIGER / EIBench — Project Claim, Research Questions & Gap Analysis

> Version: 0.2.0 | Written Sprint 3→4 transition, updated Sprint 5+ against the actual paper draft
> Source material: `Proposta_ricerca.docx`, `Integrità Epistemica e Vulnerabilità RAG.docx`, `docs/Fasi di lavoro.docx`, `docs/Spunti.docx` (shared multidisciplinary project folder — Digital Humanities + Engineering team), and, as of Sprint 5+, the actual paper draft PDF ("Epistemic Integrity Benchmark (EIB): Evaluating Source Poisoning and Faithful Falsehoods in Retrieval-Augmented Journalism", Imperatrice & Putortì) plus direct inspection of `CLAIM/Corpus_claim_RAG_Mistral_output_v3.xlsx`. This document distills that material into a single reference the whole team (including GitHub collaborators without access to the shared folder) can check the codebase against. It does not replace the source documents; when the two disagree, the source documents are authoritative and this file should be corrected.

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
| **RQ2** | Can RAGAS-style metrics report high faithfulness and answer-relevance even when the retrieved context is epistemically corrupted? | **H2** — Faithfulness can remain high under poisoning because it measures answer-context coherence, not context integrity. | **Testable with two scorer options, neither yet calibrated.** This is exactly what FFR (Faithful Falsehood Rate) is designed to isolate — see `docs/ARCHITECTURE.md` §3 Layer 5. The faithfulness/correctness signal FFR consumes is pluggable: `EmbeddingFaithfulnessScorer` (cosine-similarity heuristic, CLI default) or, as of Sprint 5, `RAGASFaithfulnessScorer` (real RAGAS via an Ollama LLM judge, opt-in via `faithfulness_scorer: "ragas"`). Results from either are directionally informative but not yet publication-grade until calibrated against human judgments (see `docs/ARCHITECTURE.md` §2 "Current limitations"). |
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
| **3 — Controlled poisoning (0/1/3/5/10%)** | Insert a progressive share of manipulated documents; manipulation types include numeric shift, date shift, causal inversion, false attribution, decontextualized citation. | **Attacks complete, sweep not yet run.** `eiger/attacks/` implements all 6 of the manipulation types in the project's own taxonomy (Section 5) as deterministic, seeded, non-LLM edits (Sprint 5); `AttackConfig.poison_rate` supports arbitrary rates but no experiment has yet swept 0/1/3/5/10% systematically. |
| **4 — Querying & retrieval audit** | Query with target/paraphrased/query-agnostic phrasings; log top-k, rank, similarity, contradictions. | **Partially implemented.** `DenseRetriever`/`SparseRetriever`/`HybridRetriever` + `RetrievalResult` gives rank/score/poison-ratio per query; no paraphrase/query-agnostic query-set methodology yet, no contradiction-audit logging. |
| **5 — Generation & RAG metrics** | RAGAS + custom metrics distinguish irrelevant answer / hallucination / faithful-to-sound-source / faithful-to-corrupted-source. | **Implemented with a caveat.** FFR + ERS + SourceIntegrity exist; the faithfulness/correctness input is pluggable — a heuristic proxy by default, or real RAGAS via an Ollama judge (Sprint 5, opt-in, itself not yet calibrated — see RQ2 above). |
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
| PCS (Poisoned Context Sensitivity) | `Δ output_score` when the suspect context is removed | ✅ **Implemented (Sprint 5+)** — `PCSMetric` (`eiger/metrics/pcs.py`), registered as `"pcs"`; operationalized as `1 - cosine_similarity(answer, counterfactual_answer)` (embedding-similarity proxy, same caveat as FFR's heuristic scorer). `ExperimentRunner` runs the counterfactual re-generation (poisoned hits filtered out of context) only when `"pcs"` is configured and a poisoned hit was actually retrieved — see `eiger/experiments/README.md` |
| EVD (Epistemic Vigilance Drop) | `P(verify \| no citation) − P(verify \| citation)` | ❌ Not implemented — requires the human-in-the-loop study (Section 8); this is fundamentally a human-subjects measurement, not a pipeline metric |

**Sprint 4, done:** PRR@k and PRD@1 were the cheapest wins here — both were pure aggregations over data EIGER already produces (`RetrievalResult.contains_poisoned`/rank) and needed no new infrastructure, just a new `BaseMetric` implementation each (`eiger/metrics/prr.py`, `eiger/metrics/prd.py`).

---

## 7. Data Assets: Two Parallel Claim Pipelines

> Updated after reading the actual paper draft ("Epistemic Integrity
> Benchmark (EIB): Evaluating Source Poisoning and Faithful Falsehoods in
> Retrieval-Augmented Journalism", Imperatrice & Putortì) and directly
> inspecting `CLAIM/Corpus_claim_RAG_Mistral_output_v3.xlsx` — the numbers
> below are verified against that file, not transcribed from an earlier
> summary.

The project has **two distinct, not-yet-reconciled** paths from raw fact-checking data to usable claims:

**(A) `eiger.datasets` automated loaders** (this repo, Sprint 3): `SnopesDataset`, `AVeriTecDataset`, `PolitiFactDataset`, `FactCheckDataset`. Each loads only verified-**true** claims and constructs a `context_query` (real, templated, or LLM-enriched — see `docs/DATASETS.md`). These feed directly into `eiger.ingestion` → the poisoning engine applies EIGER's own mechanical attacks (Section 5) at experiment time. This is the path the CLI (`eiger run ...`) actually exercises today. `data/snopes/snopes_enriched.json` (2,928 claims, IDs like `SNOPES_80214`) is this pipeline's own Snopes output — confirmed, by cross-checking claim text, to trace back to the *same* raw Snopes claims as pipeline (B) below, but processed independently, with a different ID scheme and no shared record between the two files.

**(B) The team's Mistral/Ollama v3 pipeline** (`CLAIM/Corpus_claim_RAG_Mistral_output_v3.xlsx`, outside this Git repo, per the project's data-file convention) — **this is the actual corpus the paper reports numbers on.** Verified directly from the file (sheet `01_Corpus_claim`, 5,672 rows):

- Raw collection: 21,804 claims (16,704 Snopes, 5,050 PolitiFact [2,650 True / 2,400 False], 50 FactCheck.org False).
- After thematic filtering (5 domains) + verdict normalization + risk/sensitivity screening: 5,672 retained instances (3,343 `verified_false`, 2,329 `verified_true`), from a 6,591-instance consolidated pool (919 rows `blocked` and removed — an intentional ethical exclusion, not a technical failure).
- Manipulation taxonomy **M01–M06 defined, but only M01/M02/M04/M06 actually automated**: M06=2,904 (51.2%), M01=1,696 (29.9%), M02=931 (16.4%), M04=141 (2.5%). **M03 = 0 rows** ("excluded from automation because of its higher reputational and dual-use risk", paper §6.0.2) and **M05 = 0 rows** ("does not occur in the final set of instances", same section) — confirmed directly in the file's own `03_Codebook` sheet ("Presenza nel corpus finale": M03→0, M05→0).
- `risk_level`/`sensitivity_class`: only two configurations appear — 3,925 rows (69.2%) at risk 5 / S2, 1,747 rows (30.8%) at risk 4 / S1. No S0/S3 rows (S3 would mean exclusion; S0 apparently didn't survive the domain/risk filtering for this corpus).
- **Every one of the 5,672 rows has `requires_human_review = True`** (verified in the file's own `00_Riepilogo` summary sheet, which states this explicitly: "needs_manual_validation indica la necessità di revisione umana e non un errore tecnico"). Of these, 1,938 (34.2%) are `generation_status = generated`, 3,734 (65.8%) are `needs_manual_validation` — i.e. the **majority of the benchmark corpus has not yet completed its own pipeline's human-review step.**
- **QA flags investigated (task closed, see below for the two findings and their disposition)**: 93 rows (1.6%) have `added_new_entity = True` and 1,037 rows (18.3%) have `same_language_confirmed = False` — both nominally violate constraints the paper states the generation pipeline enforces ("does not introduce new entities", "preserves the language of the claim"). Direct inspection of the raw file resolves both to concrete, narrow findings rather than corpus-wide problems:
  - **`added_new_entity = True` (93 rows) — a real, narrow M01 violation, already self-flagged by the pipeline.** 92/93 rows are `manipulation_type_applied = M01` (numerical_shift), 1/93 is M02. Every one of the 93 also carries `pipeline_safety_flags` containing both `model_reports_new_entity` and `m01_did_not_change_exactly_one_number` — i.e. the pipeline's own generation step already detected the deviation at generation time. Sampled rows (e.g. SNP_000130, SNP_000476, SNP_000544, SNP_000593, SNP_000620) show the model *fabricating a new numeric fact absent from the original claim* (e.g. "Added a heart rate of 450 beats per minute, which was not present in the original claim") rather than shifting a digit of an existing number — a different, more severe operation than M01's own definition. **Recommendation:** exclude these 93 rows from any M01-specific "does not introduce new entities" claim in the paper, or flag them to the collaborator for regeneration; no code change needed on the EIGER side, since `CorpusClaimDataset` already carries `modified_claim` into `Claim.metadata` only and never consumes it as poisoning content — this finding reinforces that design choice rather than requiring a new one.
  - **`same_language_confirmed = False` (1,037 rows) — root cause is the explanation field, not the claim text; likely zero impact on the paper's own claim-text guarantee.** `manipulation_type_applied`: M01=565, M06=328, M02=142, M04=2. In 973/1,037 rows (94%), `modified_claim_language_iso` already equals `original_language_iso` (overwhelmingly `en`) — the claim text itself is not corrupted. The mismatch driving the flag is `modified_explanation_language_iso`, which is `it` in 1,009/1,037 rows regardless of the claim's source language, with `explanation_generation_mode = generated_from_transformation_only` — consistent with the fact-checker explanation field defaulting to Italian (the team's own working language) rather than 1,037 individually broken claims. **Recommendation:** confirm with the collaborator whether the Italian-language explanation default is intentional; either way this affects `modified_fact_checker_explanation` only, a metadata field `CorpusClaimDataset` does not consume as experiment content, so it has no functional impact on EIGER's own pipeline — a documentation/paper-accuracy note for pipeline (B), not an EIGER code fix.
- The `03_Codebook` sheet documents every column (`item_id`, `claim_original`, `modified_claim`, `manipulation_type_applied`, `risk_level`, `sensitivity_class`, `requires_human_review`, `pipeline_safety_flags`, etc.) but does **not** include `plausibility`/`editorial_risk`/`verification_difficulty` fields the paper's abstract describes as a "humanities-oriented annotation layer" — if those annotations exist, they live somewhere not yet found in this repo's connected folders; if they don't yet exist, the abstract should be reconciled with what the corpus actually contains before submission.
- Semantics that needed getting right before wiring this in, now resolved: per the team's own working notes (`docs/Fasi di lavoro.docx`/`Spunti.docx`), **`clean` means "not experimentally manipulated", independent of `verified_true`/`verified_false`** — a `verified_false` claim that was never selected for manipulation is still `clean`. This differed from pipeline (A)'s original convention, where only verified-true claims were loaded as ground truth at all. **Fixed**: `Claim`/`Document` now carry two separate fields — `ground_truth_label` (`verified_true`/`verified_false`, this axis) and `doc_type`/manipulation_status (`clean`/`poisoned`, EIGER's own axis) — propagated end-to-end through `CorpusBuilder`, all six attacks, and the Qdrant retrieval round trip. Pipeline (A)'s three inline-filtered loaders (`AVeriTecDataset`/`PolitiFactDataset`/`FactCheckDataset`) also gained an opt-in `include_verified_false` parameter so they too can surface the false side of their own source labeling, tagged the same way.

**Status update: multilingual scoping investigation (Sprint 5 roadmap item "multi-lingual extension").** The real corpus's own `original_language_iso` column tags 36/5,672 rows (0.6%) as non-English (es=17, fr=4, unknown=4, ca=2, it=2, af=2, de=2, nl=2, da=1). Direct inspection shows the tag is **not reliable for most of these**: 19 of the 36 (fr/ca/it/af/de/nl/da/unknown rows) have `claim_original` text that is plainly English (e.g. `SNP_003927` tagged `da` reads "President Obama forgave Al Sharpton's huge tax debt.") — almost certainly a language-detection false positive on short strings, not real content. Only the 17 `es`-tagged rows (all `PFF_`-prefixed, i.e. sourced from a Spanish-language PolitiFact export) are genuinely non-English claim text. Two real findings from this investigation:
- **Bug, fixed**: `original_language_iso` was read from the workbook but never in `CorpusClaimDataset`'s `_PROVENANCE_METADATA_COLUMNS`, so it was silently dropped at load time — no downstream code (including the new pilot/calibration scripts) could tell a non-English claim apart from an English one. Fixed (now carried into `Claim.metadata`); 2 new regression tests added.
- **Design limitation, documented not silently patched**: two of the six attacks assume English text. `CausalManipulationAttack`'s `DEFAULT_CAUSAL_INJECTIONS` (`eiger/attacks/causal.py`) are hardcoded English clauses ("due to an unprecedented collapse in consumer confidence", etc.) — applied to one of the 17 genuine Spanish claims, it would append an English clause to Spanish text, producing an obviously mixed-language, unrealistic poisoned document rather than a crash. `MissingContextAttack`'s caveat-marker regex (`_CONTEXT_MARKER_RE` in `eiger/attacks/missing_context.py`: "however", "note that", "meanwhile", etc.) is English-only, so it silently no-ops on every non-English claim (never finds a caveat sentence to delete) — a quiet, undocumented-until-now reduction of the effective M06 poison rate to 0% on non-English data. Given genuinely non-English content is only 17/5,672 rows (0.3%) today, and given the language tag itself is unreliable for the other 19, this is flagged as a scoped, low-impact limitation for the research team to decide on (exclude non-English claims from M04/M06 paper-facing runs via a language filter, or invest in native-reviewed non-English injection phrases/caveat markers) rather than something to auto-fix by inventing translations without linguistic review.

**Status update: the engineering pilot ran successfully end-to-end on real data.** `pipeline_corpus_claim_pilot.py` (repo root) loads real claims via `CorpusClaimDataset(include_unreviewed=True)`, poisons them with EIGER's own M01/M02/M04/M06 attacks (`CorpusBuilder`), retrieves with the new in-memory `SparseRetriever` (no Qdrant/Ollama needed — deliberately lightweight), and computes ERS/Source Integrity. A 30-claim run confirmed the full path works against the real corpus: 30 ground-truth + 52 poisoned documents built, all 30 queries retrieved at least one poisoned document in their top-5, ERS aggregated to 0.806 (using each attack's own heuristic annotation, not a human/LLM-judge rating — a sanity signal, not a paper-facing score). This is explicitly non-paper-facing (see the script's own module docstring) since it uses unreviewed claims. Running it surfaced and fixed one real, previously-undiscovered bug: `eiger/utils/logging.py`'s `configure_logging()` paired `structlog.stdlib.add_logger_name` with `structlog.PrintLoggerFactory()`, whose `PrintLogger` objects have no `.name` attribute — every real log call after `configure_logging()` (i.e. every actual run of `pipeline_eibench.py`, `epistemic.py`, or `python -m eiger run ...`) raised `AttributeError` unconditionally. `tests/unit/test_logging.py` never caught this because it tested `configure_logging()` and `get_logger()` separately but never emitted a log line through the fully configured pipeline. Fixed by switching to `structlog.stdlib.LoggerFactory()` (matching the module's own stated "stdlib logging bridge" purpose and `get_logger()`'s own `structlog.stdlib.BoundLogger` return-type annotation); 4 new regression tests added that actually call `log.info/warning/error/debug()` post-configuration.

**Status update: pipeline (B) is now ingestible.** `CorpusClaimDataset` ("corpus_claim", `eiger/datasets/corpus_claim.py`) loads `Corpus_claim_RAG_Mistral_output_v3.xlsx` directly — `claim_original` → `Claim.original_fact`, `normalized_label` → `Claim.ground_truth_label` (a direct 1:1 mapping — the workbook already uses the exact same two values this project settled on), `risk_level`/`sensitivity_class` map straight across (already this project's own scales). `modified_claim` and its full manipulation provenance are carried into `Claim.metadata` for inspection, not consumed as a poisoned Document — EIGER's own attack registry still generates every experiment's poisoning mechanically, the same as for every other loader.

The one item from the list below that could **not** be closed by writing a loader is the human-review gap itself (point 3): **every one of the 5,672 rows still has `requires_human_review = True`**, so `load()` returns an empty list by default — `include_unreviewed=True` is required for any engineering use, and is documented as never appropriate for paper-facing results. Closing this for real requires the stratified human-validation sample described in the alignment document shared with the corpus-claim collaborator, not a code change.

1. ~~Build a `mistral_enriched` (or similarly named) `BaseDataset` loader~~ **Done** — `CorpusClaimDataset`.
2. ~~`risk_level`/`sensitivity_class` already exist on `Claim`/`Document`~~ **Done** — populated directly from the matching Excel columns.
3. **Tooling done, the review itself is still open** (not a code problem): the loader enforces the `requires_human_review` gate (returns `[]` by default). `scripts/sample_corpus_claim_for_review.py` now exists and has produced a first stratified batch (`review_batches/review_batch_001.xlsx`, 434 claims across the 8 real M0x/S* strata, seed 42 — see `scripts/README.md`) — but nothing in this repo can perform the human review itself; that batch is waiting on the collaborator.
4. `AttributionSwitchAttack`/`CherryPickingAttack` (M03/M05) are gated behind `ExperimentConfig.allow_non_benchmark_attacks` (see Section 5/9) specifically so that this loader's output cannot silently reintroduce a manipulation category the published corpus itself excludes (moot today anyway — M03/M05 have 0 rows in this corpus).

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
| ~~Add `risk_level`/`sensitivity_class` fields to `Claim`/`Document`~~ | Threat model doc (for defining the field's allowed values authoritatively) | ~~4~~ **Done (Sprint 4)** — typed/validated fields (not nested in free-form `metadata`), propagated from `Claim` to ground-truth `Document` and from source `Document` to `PoisonedDocument` in all 6 attacks (including `cherry_picking`/`missing_context`, added in Sprint 5), plus a follow-on fix to the Qdrant payload round-trip (`_document_to_payload`/`_document_from_payload`) so classification survives the default dense retriever. Now populated for real by `CorpusClaimDataset` (see the row below) — see `docs/ETHICS_AND_THREAT_MODEL.md` §7 |
| ~~Add `ground_truth_label` (verified_true/verified_false) separate from manipulation_status~~ | Resolves the `clean`-semantics mismatch documented in §7 | **Done (post-Sprint-5)** — new typed field on `Claim`/`Document`, propagated through `CorpusBuilder`, all 6 attacks, and the Qdrant round trip; `AVeriTecDataset`/`PolitiFactDataset`/`FactCheckDataset` gained an opt-in `include_verified_false` parameter |
| ~~Implement ingestion path for the Mistral-generated corpus (Section 7)~~ | Schema fully mapped (§7); decision made: `BaseDataset` loader, not a new attack strategy | **Done (post-Sprint-5)** — `CorpusClaimDataset` ("corpus_claim", `eiger/datasets/corpus_claim.py`); `normalized_label` maps 1:1 onto `ground_truth_label`. `load()` returns `[]` by default (100% of rows still `requires_human_review=True`) — the human-validation gap itself (§7 point 3) remains open, it is not a code problem |
| ~~Gate `AttributionSwitchAttack` (M03) / `CherryPickingAttack` (M05) behind explicit opt-in~~ | Confirmed against the paper draft: both are absent from the published EIB corpus for stated ethical/reputational reasons | ~~5+~~ **Done** — `BaseAttack.excluded_from_benchmark` class attribute + `CorpusBuilder`/`ExperimentConfig.allow_non_benchmark_attacks` raise `ConfigurationError` on silent inclusion; see `eiger/attacks/README.md` |
| ~~Implement `CherryPickingAttack` (M05) and `MissingContextAttack` (M06)~~ | Possible `BaseAttack` contract extension | ~~4/5~~ **Done (Sprint 5)** — no contract extension was actually needed; see §5 above and `eiger/attacks/README.md` |
| ~~`SparseRetriever` (BM25)~~ | — | ~~4/5~~ **Done (Sprint 4)** — `eiger/retrieval/sparse_retriever.py`, selectable via `retriever.type: sparse` |
| ~~`HybridRetriever` (RRF fusion of dense + sparse)~~ | `SparseRetriever` (done) + `DenseRetriever` (done) | ~~5~~ **Done (Sprint 5)** — `eiger/retrieval/hybrid_retriever.py`, composes both without modifying either, selectable via `retriever.type: hybrid`; wired into `ExperimentRunner` (ingests + fits both indexes) and the CLI. See `eiger/retrieval/README.md`'s "RRF Fusion" section |
| ~~Real RAGAS-based faithfulness scorer~~ | `ragas` dependency (pinned `ragas` extra) | ~~5~~ **Done (Sprint 5)** — `RAGASFaithfulnessScorer` (`eiger/metrics/ragas_scorer.py`), an Ollama-judge-based real RAGAS integration, opt-in via `ExperimentConfig.faithfulness_scorer: "ragas"`; does NOT replace `EmbeddingFaithfulnessScorer` (kept as the dependency-free default) — judge quality is itself not yet calibrated against human judgments, see that module's own docstring |
| ~~`PCS` (Poisoned Context Sensitivity) metric~~ | Counterfactual re-generation support in `eiger/experiments/` | ~~5~~ **Done (Sprint 5+)** — `PCSMetric` (`eiger/metrics/pcs.py`) + `ExperimentRunner._add_counterfactual_generation`; embedding-similarity proxy, same caveat as FFR's heuristic scorer applies |
| Vector-void mapping (Phase 2) | Corpus-level density/redundancy tooling, not yet designed | 5 |
| ~~Poison-rate sweep experiment (0/1/3/5/10%) reproducing Phase 3~~ | Stable attacks + metrics above | ~~5~~ **Executed (Sprint 5+)** — `experiments/poison_rate_sweep/sweep_{0,1,3,5,10}pct.yaml`, run against `data/snopes/snopes_enriched.json` (100-claim sample per point). **This is an engineering sweep, not an EIB benchmark reproduction**: it uses EIGER's own synthetic Snopes dataset (not `Corpus_claim_RAG_Mistral_output_v3.xlsx`) and includes M03/M05 via `allow_non_benchmark_attacks: true` — see each config's own header comment. Re-run against the real corpus once its loader exists before reporting these as EIB numbers. |
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
