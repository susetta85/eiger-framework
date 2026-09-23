"""
EIBench Corpus-Claim Pilot — Layer 1-3(+5) Engineering Quickstart

A small, lightweight end-to-end run of the EIBench pipeline (corpus loading
-> poisoning -> retrieval -> retrieval-only metrics) against the REAL
Mistral/Ollama v3 corpus (`Corpus_claim_RAG_Mistral_output_v3.xlsx`), the
one the paper reports numbers on (see docs/CLAIM_AND_RESEARCH_QUESTIONS.md
Section 7).

Why this script exists
-----------------------
This is explicitly the third of three ordered steps the team agreed on
after closing the corpus-ingestion sprint: (1) build a stratified human-
review sample, (2) report on the corpus's own QA flags, (3) run a small
engineering-only pilot on the real corpus to confirm the pipeline
(poisoning + retrieval + metrics) actually works end-to-end against real
data, not just the synthetic JSON fixture.

What this pilot deliberately is NOT
-------------------------------------
- NOT a paper-facing experiment. It uses `include_unreviewed=True`, i.e.
  it consumes claims that have not passed the corpus's own human-review
  gate (`requires_human_review = True` for 100% of the real corpus today
  — see eiger/datasets/corpus_claim.py). Never cite these numbers in the
  paper; they exist purely to validate that the code path works.
- NOT a heavy compute job. It intentionally avoids both Qdrant (needs a
  running server) and any LLM backend (Ollama/generation):
    * Retrieval uses `SparseRetriever` (in-memory BM25 via rank-bm25) —
      no server, no embeddings model download.
    * Only retrieval-dependent metrics are computed (ERS, Source
      Integrity); FFR is skipped because it requires a real generation
      step (`GenerationResult.faithfulness_score`), which this pilot does
      not run. A placeholder `GenerationResult` (empty answer) is used
      only to satisfy `EvaluationRecord`'s schema.
  This matches the "light work only" compute constraint in place while
  the team's main machine is busy with an unrelated experiment.
- NOT using the corpus's own `modified_claim` (the collaborator's
  Mistral-generated poisoned variant) as poisoning content. Consistent
  with every other loader in this project, EIGER's own attack registry
  generates all poisoning mechanically; `modified_claim` stays in
  `Claim.metadata` for inspection only. See docs/DATASETS.md Section 3a.

Run with (from the eiger-framework repo root):
    python pipeline_corpus_claim_pilot.py \\
        --corpus-path ../CLAIM/Corpus_claim_RAG_Mistral_output_v3.xlsx \\
        --max-claims 30 --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eiger.attacks import (
    CausalManipulationAttack,
    DateManipulationAttack,
    MissingContextAttack,
    NumericalShiftAttack,
)
from eiger.core.models import AttackConfig, EvaluationRecord, GenerationResult
from eiger.datasets.corpus_claim import CorpusClaimDataset
from eiger.ingestion.corpus_builder import CorpusBuilder
from eiger.metrics.ers import ERSMetric
from eiger.metrics.source_integrity import SourceIntegrityMetric
from eiger.retrieval.sparse_retriever import SparseRetriever
from eiger.utils.logging import configure_logging, get_logger
from eiger.utils.seeding import seed_everything

log = get_logger(__name__)


def run_pilot(
    corpus_path: str | Path | None = None,
    max_claims: int = 30,
    top_k: int = 5,
    poison_rate: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Execute the corpus-claim engineering pilot end-to-end.

    Args:
        corpus_path: Path to Corpus_claim_RAG_Mistral_output_v3.xlsx. The
                     real file lives outside this Git repo (in the sibling
                     CLAIM/ folder, per the project's data-file convention
                     — see docs/CLAIM_AND_RESEARCH_QUESTIONS.md Section 7),
                     so this must normally be passed explicitly; None falls
                     back to CorpusClaimDataset's own default location
                     (data/corpus_claim/... inside the repo).
        max_claims:  Cap on claims loaded from the real corpus (kept small
                     deliberately — this is a pilot, not a full run).
        top_k:       Number of documents retrieved per query.
        poison_rate: Fraction of the corpus each attack independently
                      attempts to poison (see CorpusBuilder.build()).
        seed:        Seed for reproducible poisoning + retrieval ordering.
    """
    seed_everything(seed)

    # ── Layer 1: Load real corpus claims ──────────────────────────────────
    # include_unreviewed=True is required (100% of the real corpus has
    # requires_human_review=True today) and is the explicit, documented
    # opt-in for engineering-only use — see CorpusClaimDataset.load().
    dataset = CorpusClaimDataset(path=Path(corpus_path) if corpus_path else None)
    claims = dataset.load(split="test", max_claims=max_claims, include_unreviewed=True)
    if not claims:
        log.error("pilot.no_claims_loaded")
        print("No claims loaded — check that CLAIM/Corpus_claim_RAG_Mistral_output_v3.xlsx is reachable.")
        sys.exit(1)
    log.info("pilot.claims_loaded", n_claims=len(claims))

    # ── Layer 2: Poison via EIGER's own attack registry ───────────────────
    # M01/M02/M04/M06 are the four categories that actually occur in the
    # real corpus (see docs/CLAIM_AND_RESEARCH_QUESTIONS.md Section 7).
    # M03/attribution_switch and M05/cherry_picking are never used here —
    # they're excluded_from_benchmark and have 0 rows in the real corpus,
    # so allow_non_benchmark_attacks stays at its safe default (False).
    attacks = [
        (NumericalShiftAttack(), AttackConfig(name="numerical_shift", poison_rate=poison_rate)),
        (DateManipulationAttack(), AttackConfig(name="date_manipulation", poison_rate=poison_rate)),
        (CausalManipulationAttack(), AttackConfig(name="causal_manipulation", poison_rate=poison_rate)),
        (MissingContextAttack(), AttackConfig(name="missing_context", poison_rate=poison_rate)),
    ]
    builder = CorpusBuilder(attacks=attacks, seed=seed)
    corpus = builder.build(claims)
    log.info(
        "pilot.corpus_built",
        ground_truth=len(corpus.ground_truth_docs),
        poisoned=len(corpus.poisoned_docs),
        total=len(corpus.all_documents),
    )

    # ── Layer 3: Retrieval (in-memory BM25, no server required) ───────────
    retriever = SparseRetriever()
    retriever.fit(corpus.all_documents)

    # ── Layer 5: Retrieval-only metrics ────────────────────────────────────
    # FFR is intentionally NOT computed here: it needs a real generation
    # step this pilot skips by design (see module docstring).
    ers_metric = ERSMetric()
    si_metric = SourceIntegrityMetric()

    print("\n" + "=" * 70)
    print("CORPUS-CLAIM PILOT — real data, engineering-only, non-paper-facing")
    print("=" * 70)

    ers_scores = []
    si_scores = []
    n_contains_poisoned = 0

    for claim in claims:
        result = retriever.retrieve(query=claim.context_query, claim_id=claim.claim_id, top_k=top_k)
        if result.contains_poisoned:
            n_contains_poisoned += 1

        # Placeholder GenerationResult: EvaluationRecord requires one, but
        # ERS/Source-Integrity only read `record.retrieval` — no LLM call
        # is made to produce this.
        generation = GenerationResult(
            claim_id=claim.claim_id,
            query=claim.context_query,
            context_docs=[hit.document.text for hit in result.hits],
            answer="",
            model_name="none (retrieval-only engineering pilot, no generation performed)",
        )
        record = EvaluationRecord(claim_id=claim.claim_id, generation=generation, retrieval=result)

        ers_scores.append(ers_metric.compute(record))
        si_scores.append(si_metric.compute(record))

    print(f"\nClaims loaded:            {len(claims)}")
    print(f"Ground-truth documents:   {len(corpus.ground_truth_docs)}")
    print(f"Poisoned documents:       {len(corpus.poisoned_docs)} ({corpus.poison_ratio:.1%} of corpus)")
    print(f"Queries with >=1 poisoned hit in top-{top_k}: {n_contains_poisoned}/{len(claims)}")
    print(f"ERS (aggregate):          {ers_metric.aggregate(ers_scores):.4f}")
    print(f"Source Integrity (mean):  {sum(s.value for s in si_scores) / len(si_scores):.4f}")
    print(
        "\nNote: ERS uses each attack's own heuristic PoisonAnnotation (plausibility/"
        "verification_difficulty/editorial_risk), NOT a human or LLM-judge rating — "
        "treat it as a pipeline sanity signal here, not a paper-facing risk score. "
        "Source Integrity requires transformers/torch; if unavailable (as in this "
        "run) it returns 0.0 with a warning rather than failing."
    )
    print("=" * 70)
    log.info("pilot.complete")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Lightweight engineering pilot: EIBench pipeline on the real corpus_claim data."
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help=(
            "Path to Corpus_claim_RAG_Mistral_output_v3.xlsx (the real file lives "
            "outside this repo, e.g. ../CLAIM/Corpus_claim_RAG_Mistral_output_v3.xlsx). "
            "Defaults to CorpusClaimDataset's own in-repo default path."
        ),
    )
    parser.add_argument("--max-claims", type=int, default=30, help="Number of real claims to load (default: 30)")
    parser.add_argument("--top-k", type=int, default=5, help="Documents retrieved per query (default: 5)")
    parser.add_argument("--poison-rate", type=float, default=0.5, help="Per-attack poison rate (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed (default: 42)")
    args = parser.parse_args()
    run_pilot(
        corpus_path=args.corpus_path,
        max_claims=args.max_claims,
        top_k=args.top_k,
        poison_rate=args.poison_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
