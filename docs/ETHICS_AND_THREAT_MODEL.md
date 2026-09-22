# EIGER / EIBench — Ethics and Threat Model

> Version: 0.1.0 | Written Sprint 4
> Source material: `Spunti fasi di lavoro.docx` (Fase 0 — Ethical protocol & threat model), part of the shared multidisciplinary project folder (Digital Humanities + Engineering team). This document formalizes and condenses that source into a versioned, engineering-facing reference for this codebase, per the Sprint 4 task tracked in `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §4/§9.

---

## 0. Scope and honesty note about this document

This file operationalizes the ethical protocol and threat model that the research team defined in `Spunti fasi di lavoro.docx` before any corpus work began. Sections 1-5 below condense that source document (already summarized once in `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §4; this file expands on it and gives it a permanent, versioned home).

**What this document is not:** it is not a verbatim reproduction of the full source `.docx` (system description, complete asset list, adversary capability levels, attack-vector taxonomy, domain risk cards) — that full text was not available to reconstruct from inside this codebase, and remains authoritative in the shared project folder. If anything here conflicts with the source document, **the source document wins** and this file should be corrected.

Sections 6 (go/no-go checklist) and 7 (proposed `risk_level`/`sensitivity_class` field mapping) are **new engineering proposals written to operationalize the source protocol for this codebase** — they are not a transcription of pre-existing team decisions. They need review and ratification by the whole team (not just the engineering side) before being treated as binding, and are marked as such below.

---

## 1. System in Scope

A simulated newsroom / fact-checking desk querying a local RAG system: corpus → embedding → vector store → retriever → LLM → answer with cited sources. The system is entirely local/offline by design, for privacy and containment — no production system, no real user data, no external disinformation payload ever leaves this pipeline.

## 2. Adversary Model

The adversary is **functional**, not attributed to a real actor or organization. Capability is deliberately limited to **corpus/ingestion level only**: the adversary can get a plausible document indexed into the corpus, but has no access to model weights, retriever code, infrastructure, or admin functions.

Adversary knowledge is classified along a black-box/grey-box/white-box spectrum:

| Level | Knowledge | In scope for this project? |
|---|---|---|
| Black-box | Topic only | Yes |
| Grey-box | Topic + house style + likely queries | Yes |
| White-box | Full pipeline knowledge (retriever internals, embedding model, ranking logic) | **Explicitly out of scope** for the pilot |

## 3. Assets to Protect

1. **Corpus integrity** — a verifiable separation between sound, corrective, and experimental (poisoned) documents must always be maintainable and auditable.
2. **Retrieval reliability** — corrupted documents should not be systematically over-ranked relative to sound ones by an artifact of the pipeline itself (as opposed to being over-ranked as the deliberate subject of an experiment).
3. **Epistemic quality of generated answers** — faithful-to-context is not the same as truthful; the pipeline exists specifically to make this gap measurable, not to widen it carelessly.
4. **Provenance traceability** — every generated answer must be traceable back to the specific source document(s) that produced it, including whether that document was poisoned and by which attack (see `eiger/core/models.py`'s `PoisonedDocument` and the Sprint 4 fix ensuring this survives serialization to `results.json`).
5. **Editorial responsibility** — the human reviewer's own ability to recognize a weak, isolated, or contradicted source must not be undermined by how the pipeline or its outputs are presented.

## 4. Explicitly Out of Scope

This project does **not**, under any circumstance:

- Attack real or production RAG systems.
- Attempt infrastructure compromise or model-weight tampering.
- Generate disinformation payloads intended to be reusable outside this research pipeline.
- Target private individuals, minors, or vulnerable groups in any generated or poisoned content.
- Pursue a white-box or fully-automated disinformation-campaign scenario.

Any experiment design, prompt, or dataset addition that would cross one of these lines is out of scope for this repository regardless of its potential research value, and should be raised with the whole team rather than implemented unilaterally.

## 5. Sensitivity Classification and Risk Levels

Every corpus item (claim, document, or poisoned variant) is assigned a sensitivity class:

| Class | Meaning | Treatment |
|---|---|---|
| **S0** | Neutral | Usable anywhere, including publication |
| **S1** | Sensitive but manageable | Usable in the lab; publish only in sanitized form |
| **S2** | Critical (health, civil rights, migration, and similarly high-stakes domains) | Requires dedicated ethics review; abstracted/sanitized form only, even internally |
| **S3** | Excluded | Must not be produced or included in operational form at all (private individuals, minors, vulnerable subjects, unverified accusations against named people) |

**Rule:** when in doubt between two adjacent classes, apply the more restrictive one.

Separately, every corpus item also carries a **risk level from 1 to 5**, from "usable, ordinary review" (1) to "very high/critical — normally exclude or heavily abstract" (5). Risk level and sensitivity class are related but distinct: sensitivity class is about the *subject matter*, risk level is about the *practical consequence* of that item being mishandled (e.g. a S1 item about a public company's stock price might still warrant a risk level of 3 if the numeric attack could plausibly be mistaken for real financial guidance).

**Publication rule:** poisoned/manipulated corpus items are never published in directly reusable form. Only pipeline, logging, auditing, and defense code is released publicly; any example shown in a paper or presentation is abstracted or heavily sanitized first, regardless of the item's own sensitivity class.

## 6. Go/No-Go Checklist Between Corpus Phases (proposed, needs team ratification)

The experimental design ("Poisoned Newsroom", see `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §3) has 8 phases (0-7). The source protocol calls for "go/no-go criteria before any corpus work" but does not — in the material available to this codebase — spell out a phase-by-phase checklist. The following is a **first proposal**, written to give Phase 0's intent a concrete, checkable form; it should be reviewed and adjusted by the full team before being treated as a hard gate.

Before advancing from one phase to the next, confirm:

- [ ] **Phase 0 → 1 (start building the clean corpus):** this document and its sensitivity/risk framework are agreed by the whole team; every dataset loader in `eiger.datasets` tags unreviewed claims with `metadata["verified"] = False` (already true for all four implemented loaders — see `docs/DATASETS.md` §1).
- [ ] **Phase 1 → 2 (vector void mapping):** the clean corpus's provenance is fully traceable (source dataset + content hash per claim — already true via `Claim.content_hash`); no S3-classified item has entered the corpus.
- [ ] **Phase 2 → 3 (controlled poisoning):** the specific attack(s) to be applied, their `poison_rate`, and any per-attack params are recorded in an `experiments/*.yaml` config *before* the run (already the project's convention — see `eiger/experiments/README.md`); no attack introduces S2/S3-classified content as a side effect (e.g. a fabricated causal clause must not itself constitute a real accusation against a named private individual).
- [ ] **Phase 3 → 4 (querying & retrieval audit):** poisoned documents are tagged with full provenance (`attack_name`/`attack_params`/`original_text`/`annotation`) and that provenance is confirmed to survive to the persisted `results.json` (this was a real, found-and-fixed gap this sprint — see `docs/ARCHITECTURE.md` §2).
- [ ] **Phase 4 → 5 (generation & RAG metrics):** the LLM backend used for generation is local/offline (Ollama), not a third-party hosted API that would send poisoned content off-machine.
- [ ] **Phase 5 → 6 (human-in-the-loop):** a separate, dedicated ethics review has been completed for the human-subjects study design itself (participant consent, no S2/S3 material shown to participants without sanitization, IRB/equivalent approval per the team's institution) — this is a harder, additional gate beyond the rest of this checklist, since it involves real people rather than only synthetic pipeline data.
- [ ] **Any phase → publication:** every example intended for a paper or public repository has been re-checked against the sensitivity classification and publication rule in Section 5, independent of whatever classification it was assigned during the experiment itself.

## 7. Proposed `risk_level` / `sensitivity_class` fields (not yet implemented)

`docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §9 flags adding `risk_level`/`sensitivity_class` fields to `Claim`/`Document` metadata as a Sprint 4 task, dependent on this document existing to authoritatively define the fields' allowed values. With this document now in place, a concrete proposal:

- `Claim.metadata["sensitivity_class"]` / `Document.metadata["sensitivity_class"]`: one of `"S0"`, `"S1"`, `"S2"`, `"S3"` (Section 5 above).
- `Claim.metadata["risk_level"]` / `Document.metadata["risk_level"]`: integer `1`-`5` (Section 5 above).
- Both should default to `None`/absent rather than a guessed value when not explicitly classified — an unclassified item should never be silently treated as S0/low-risk. Code that reads these fields (dataset loaders, the results matrix, any future publication-export tooling) should treat "absent" as "requires classification before use," not as "safe by default."
- The Mistral/Ollama v3 corpus (`docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §7) already carries per-claim topic/risk/sensitivity classification from its own pipeline — whichever ingestion path is chosen for that corpus (loader vs. attack framing, still an open team decision) should map its existing classification into these same fields rather than inventing a second scheme.

This section is a proposal for the next implementation step, not yet implemented in `eiger/core/models.py` — tracked as an open Sprint 4 item in `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §9 until it is.

---

## Change log

- **Sprint 4:** initial version, condensing `Spunti fasi di lavoro.docx` §Fase 0 (already summarized once in `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §4) into this standalone file, plus a first proposed go/no-go checklist and `risk_level`/`sensitivity_class` field mapping (Sections 6-7) for team review.
