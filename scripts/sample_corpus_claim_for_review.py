"""
scripts/sample_corpus_claim_for_review.py

Select a stratified, reproducible sample from the collaborator's real
corpus (`Corpus_claim_RAG_Mistral_output_v3.xlsx`) and write it out as an
Excel workbook for human review — this is the concrete tool proposed in
the "EIGER x Corpus Claim - Protocollo di allineamento" alignment document
(point 4: "definire un campione di validazione", not the entire 5,672-row
corpus at once).

Why this exists
-----------------
Every row in the corpus currently has `requires_human_review = True` (see
`eiger/datasets/corpus_claim.py` and `docs/CLAIM_AND_RESEARCH_QUESTIONS.md`
§7, point 4) — nothing has been human-validated yet, and reviewing all
5,672 rows by hand is not realistic in one pass. This script picks a
representative, stratified subset instead, so review effort can start now
and expand incrementally.

What this script does NOT do
-------------------------------
- It does NOT perform the review itself — plausibility/editorial_risk/
  verification_difficulty judgments require a human fact-checker (the
  collaborator), not code. This script only prepares the sample and the
  columns for her to fill in.
- It does NOT write back to the original workbook or update
  `requires_human_review` anywhere — the output is a separate file. Once
  she has reviewed a batch, updating the master corpus (however that
  round-trip happens) is a separate, deliberate step, not automated here.
- It is not part of the eiger package and is not covered by the 100%
  coverage gate (pyproject.toml's [tool.coverage.run] omits scripts/*),
  matching every other script in this directory.

Stratification
----------------
Two axes, matching the alignment document's own proposal: manipulation
category (`manipulation_type_applied`: M01/M02/M04/M06 — M03/M05 never
occur in this corpus, see docs/CLAIM_AND_RESEARCH_QUESTIONS.md §7) and
`sensitivity_class` (S1/S2 — S0/S3 do not occur either; S3 is excluded
from the corpus entirely by CorpusClaimDataset itself). Allocation within
each non-empty stratum is proportional to that stratum's share of the
total population, with a minimum floor (--min-per-stratum) so a small
stratum (e.g. M04/S1, 53 rows in the real file) still gets meaningful
coverage rather than being drowned out by the largest ones (e.g. M06/S2,
over 2,000 rows). Floors can make the actual output larger than
--sample-size; this script reports the real total rather than silently
trimming below the floor guarantee.

Sampling within each stratum uses this project's own deterministic RNG
utilities (`eiger.utils.seeding`), sorted by claim_id before drawing, so
the same --seed always produces the exact same sample — this matters for
a review batch that might be re-generated or extended later.

Usage
------
    python scripts/sample_corpus_claim_for_review.py \\
        /path/to/Corpus_claim_RAG_Mistral_output_v3.xlsx \\
        -o review_batch_001.xlsx --sample-size 400 --min-per-stratum 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from eiger.datasets.corpus_claim import CorpusClaimDataset
from eiger.utils.seeding import derive_seed, make_rng

# Columns carried into the output sheet from each sampled Claim, in order.
# See CorpusClaimDataset's own field mapping for where each comes from.
_OUTPUT_COLUMNS = (
    "item_id",
    "claim_original",
    "normalized_label",
    "manipulation_type_applied",
    "sensitivity_class",
    "risk_level",
    "topic",
    "subtopic",
    "modified_claim",
    "claim_change_description",
    "factcheck_url",
    "generation_status",
    "requires_human_review",
)

# Empty columns appended for the human reviewer to fill in — matching the
# codebook vocabulary referenced in docs/Fasi di lavoro.docx/Spunti.docx
# (plausibility / editorial_risk / verification_difficulty) plus a final
# go/no-go decision and free-text notes.
_REVIEW_COLUMNS = (
    "plausibility",
    "editorial_risk",
    "verification_difficulty",
    "review_decision",  # e.g. Approvata / Da rivedere / Scartata
    "reviewer_notes",
)


def _stratum_key(claim: Any) -> tuple[str | None, str | None]:
    return (claim.metadata.get("manipulation_type_applied"), claim.sensitivity_class)


def stratified_sample(
    claims: list[Any],
    sample_size: int,
    seed: int,
    min_per_stratum: int,
) -> list[Any]:
    """
    Select a stratified, reproducible sample of claims.

    Args:
        claims:          Full candidate population (already filtered by
                          the caller, e.g. via CorpusClaimDataset).
        sample_size:      Target total sample size (a soft target — see
                          the module docstring's note on floors).
        seed:            RNG seed; the same seed + population always
                          produces the same sample.
        min_per_stratum: Minimum claims drawn from each non-empty stratum,
                          capped at that stratum's own size.

    Returns:
        The sampled claims, grouped by stratum (M01/S1, M01/S2, M02/S1,
        ...) and sorted by claim_id within each stratum.
    """
    strata: dict[tuple[str | None, str | None], list[Any]] = {}
    for claim in claims:
        strata.setdefault(_stratum_key(claim), []).append(claim)

    total_population = len(claims)
    sampled: list[Any] = []
    for stratum_key in sorted(strata, key=lambda k: (k[0] or "", k[1] or "")):
        stratum_claims = sorted(strata[stratum_key], key=lambda c: c.claim_id)
        n_stratum = len(stratum_claims)
        proportional_share = round(sample_size * n_stratum / total_population) if total_population else 0
        n_draw = min(n_stratum, max(min_per_stratum, proportional_share))

        rng = make_rng(derive_seed(seed, "sample_corpus_claim", *stratum_key))
        sampled.extend(rng.sample(stratum_claims, n_draw))

    return sampled


def write_review_workbook(claims: list[Any], output_path: Path) -> None:
    """Write the sampled claims to an .xlsx workbook, ready for review."""
    try:
        import openpyxl
    except ImportError as exc:
        raise SystemExit(
            "openpyxl is required for this script. Install it with: "
            "pip install 'eiger[data-import]' (or: pip install openpyxl)"
        ) from exc

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Campione da revisionare"
    sheet.append(list(_OUTPUT_COLUMNS) + list(_REVIEW_COLUMNS))

    for claim in claims:
        row = [
            claim.claim_id,
            claim.original_fact,
            claim.ground_truth_label,
            claim.metadata.get("manipulation_type_applied"),
            claim.sensitivity_class,
            claim.risk_level,
            claim.metadata.get("topic"),
            claim.metadata.get("subtopic"),
            claim.metadata.get("modified_claim"),
            claim.metadata.get("claim_change_description"),
            claim.metadata.get("factcheck_url"),
            claim.metadata.get("generation_status"),
            claim.metadata.get("requires_human_review"),
        ]
        row.extend([None] * len(_REVIEW_COLUMNS))
        sheet.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 if the input file is missing."""
    parser = argparse.ArgumentParser(
        description=(
            "Select a stratified, reproducible sample from the real corpus_claim "
            "workbook and write it out for human review (see the module docstring)."
        )
    )
    parser.add_argument("input", help="Path to Corpus_claim_RAG_Mistral_output_v3.xlsx")
    parser.add_argument(
        "-o", "--output", default="review_batch.xlsx",
        help="Output .xlsx path (default: review_batch.xlsx)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=400,
        help="Target total sample size across all strata (default: 400, ~7%% of 5,667).",
    )
    parser.add_argument(
        "--min-per-stratum", type=int, default=20,
        help="Minimum claims drawn from each non-empty stratum (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    # include_unreviewed=True: sampling FROM the unreviewed pool is the
    # entire point of this script (see module docstring) — this is
    # engineering tooling to prepare a review batch, not a paper-facing
    # experiment consuming unreviewed claims as ground truth.
    dataset = CorpusClaimDataset(path=input_path)
    claims = dataset.load(include_unreviewed=True)
    if not claims:
        print("No claims loaded (workbook empty or all rows blocked/S3-excluded).")
        return 0

    sampled = stratified_sample(
        claims, sample_size=args.sample_size, seed=args.seed,
        min_per_stratum=args.min_per_stratum,
    )

    strata_counts: dict[tuple[str | None, str | None], int] = {}
    for claim in sampled:
        key = _stratum_key(claim)
        strata_counts[key] = strata_counts.get(key, 0) + 1

    print(f"{len(claims)} candidate claim(s) loaded (requires_human_review ignored).")
    print(f"Sampled {len(sampled)} claim(s) across {len(strata_counts)} strata:")
    for (manip, sens), count in sorted(strata_counts.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        print(f"  {manip or '(none)':>6} / {sens or '(none)':<4} -> {count}")

    output_path = Path(args.output)
    write_review_workbook(sampled, output_path)
    print(f"Written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
