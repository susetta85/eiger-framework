"""
scripts/sample_for_faithfulness_calibration.py

Select a sample of records from a completed experiment's `results.json` and
write out a BLIND annotation workbook for a human to rate faithfulness and
answer-correctness — the concrete first step towards closing the biggest
open scientific-validity gap in this project (see docs/ARCHITECTURE.md
Section 2, "Current limitations": neither EmbeddingFaithfulnessScorer nor
RAGASFaithfulnessScorer has been calibrated against human judgments).

Why "blind"
------------
The output workbook does NOT include either scorer's automated score next
to a record — showing it would anchor the human annotator's rating towards
the very number being evaluated, defeating the point of calibration. The
automated scores are instead written to a separate sibling JSON file
(`<output>.automated_scores.json`), keyed by claim_id, for
`compute_calibration_correlation.py` to join back in once annotation is
done. Never open that file while annotating.

Where "ground truth" comes from
----------------------------------
No claim text needs to be re-loaded from the original dataset: every
retrieved ground-truth Document's `text` field IS `Claim.original_fact`
verbatim (see CorpusBuilder.build()), so if a claim's own ground-truth
document was retrieved among its hits, its text is used directly. If it
was NOT retrieved (every hit for that claim happened to be poisoned), the
ground_truth_fact column is left blank and flagged — the human annotator
can still judge faithfulness-to-context from the context column alone, but
answer-correctness for that row is not annotatable and should be skipped.

What this script does NOT do
-------------------------------
- It does not compute or suggest a "right" correlation threshold — that is
  a judgment call for the research team once real numbers exist, not
  something to hard-code here.
- It does not run any scorer or generation step itself; it only samples
  and reformats an experiment's already-computed results.json.
- It is not covered by the 100% coverage gate (pyproject.toml's
  [tool.coverage.run] omits scripts/*), matching every other script here.

Usage
------
    python scripts/sample_for_faithfulness_calibration.py \\
        results/poison_rate_sweep/5pct/results.json \\
        -o calibration/batch_001.xlsx --sample-size 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eiger.core.models import ExperimentResult
from eiger.utils.seeding import derive_seed, make_rng

# Columns shown to the human annotator, in order. Deliberately excludes
# any automated scorer output (see module docstring's "Why blind" note).
_OUTPUT_COLUMNS = (
    "claim_id",
    "query",
    "ground_truth_fact",
    "retrieved_context",
    "generated_answer",
)

_ANNOTATION_COLUMNS = (
    "faithfulness_human",  # 1-5: is the answer supported by retrieved_context?
    "correctness_human",  # 1-5: does the answer match ground_truth_fact?
    "annotator_notes",
)


def _ground_truth_text(record: Any) -> str | None:
    """
    Return the ground-truth document's text for this record's claim, if its
    own ground-truth document was among the retrieved hits; None otherwise.
    """
    for hit in record.retrieval.hits:
        if hit.document.doc_type == "ground_truth" and hit.document.claim_id == record.claim_id:
            return hit.document.text
    return None


def select_sample(records: list[Any], sample_size: int, seed: int) -> list[Any]:
    """
    Select a reproducible sample of records, stratified on whether the
    record's retrieval contained a poisoned document.

    Stratifying on contains_poisoned (rather than plain random sampling)
    ensures the calibration batch covers both clean and poisoned-context
    generations — calibrating only on one would leave the other's scorer
    behaviour unverified.

    Args:
        records:     Full list of EvaluationRecords from results.json.
        sample_size: Target total sample size (soft target — see
                      stratified_sample()'s own floor behaviour below).
        seed:        RNG seed; same seed + records always produces the
                      same sample.

    Returns:
        Sampled records, sorted by claim_id within each stratum.
    """
    strata: dict[bool, list[Any]] = {True: [], False: []}
    for record in records:
        strata[record.retrieval.contains_poisoned].append(record)

    total = len(records)
    sampled: list[Any] = []
    for is_poisoned in (True, False):
        stratum = sorted(strata[is_poisoned], key=lambda r: r.claim_id)
        if not stratum:
            continue
        proportional_share = round(sample_size * len(stratum) / total) if total else 0
        # Floor of 1 so a non-empty stratum is never silently skipped.
        n_draw = min(len(stratum), max(1, proportional_share))
        rng = make_rng(derive_seed(seed, "faithfulness_calibration", str(is_poisoned)))
        sampled.extend(rng.sample(stratum, n_draw))

    return sampled


def write_annotation_workbook(records: list[Any], output_path: Path) -> dict[str, dict[str, float]]:
    """
    Write the blind annotation workbook and return the automated scores
    (for the caller to persist to the sibling reference file).

    Returns:
        Mapping of claim_id -> {"ragas_faithfulness": ..., "ragas_answer_correctness": ...}.
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise SystemExit(
            "openpyxl is required for this script. Install it with: "
            "pip install 'eiger[data-import]' (or: pip install openpyxl)"
        ) from exc

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = "Calibrazione da annotare"
    sheet.append(list(_OUTPUT_COLUMNS) + list(_ANNOTATION_COLUMNS))

    automated_scores: dict[str, dict[str, float]] = {}
    for record in records:
        ground_truth = _ground_truth_text(record)
        row = [
            record.claim_id,
            record.generation.query,
            ground_truth if ground_truth is not None else "(ground-truth document not retrieved for this claim)",
            "\n---\n".join(record.generation.context_docs),
            record.generation.answer,
        ]
        row.extend([None] * len(_ANNOTATION_COLUMNS))
        sheet.append(row)

        automated_scores[record.claim_id] = {
            "ragas_faithfulness": record.faithfulness_score,
            "ragas_answer_correctness": record.factual_correctness_score,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return automated_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a sample from results.json and write a blind annotation workbook."
    )
    parser.add_argument("results_json", help="Path to an experiment's results.json")
    parser.add_argument(
        "-o", "--output", default="calibration_batch.xlsx",
        help="Output .xlsx path (default: calibration_batch.xlsx)",
    )
    parser.add_argument("--sample-size", type=int, default=40, help="Target sample size (default: 40)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args(argv)

    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"Error: file not found: {results_path}", file=sys.stderr)
        return 1

    result = ExperimentResult.model_validate_json(results_path.read_text(encoding="utf-8"))
    if not result.records:
        print("No records found in results.json — nothing to sample.")
        return 0

    sampled = select_sample(result.records, sample_size=args.sample_size, seed=args.seed)

    output_path = Path(args.output)
    automated_scores = write_annotation_workbook(sampled, output_path)

    reference_path = output_path.with_suffix(output_path.suffix + ".automated_scores.json")
    reference_path.write_text(
        json.dumps(
            {
                "source_results_json": str(results_path),
                "faithfulness_scorer_config": result.config.faithfulness_scorer,
                "scores": automated_scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_poisoned = sum(1 for r in sampled if r.retrieval.contains_poisoned)
    print(f"{len(result.records)} candidate record(s) loaded from {results_path}.")
    print(f"Sampled {len(sampled)} record(s): {n_poisoned} with poisoned context, {len(sampled) - n_poisoned} clean.")
    print(f"Annotation workbook (blind, no automated scores): {output_path}")
    print(f"Automated-scores reference (do NOT open while annotating): {reference_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
