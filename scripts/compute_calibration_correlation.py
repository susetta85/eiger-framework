"""
scripts/compute_calibration_correlation.py

Compare a completed human annotation batch (from
sample_for_faithfulness_calibration.py) against the automated scorer's own
output, and report Pearson correlation, Spearman rank correlation, and mean
absolute error (MAE) for both faithfulness and answer-correctness.

Human ratings are expected on a 1-5 Likert scale (matching the annotation
workbook's own column comments) and are rescaled to [0, 1] before
comparison, since both EmbeddingFaithfulnessScorer and RAGASFaithfulnessScorer
report scores in [0, 1].

What this script does NOT do
-------------------------------
- It does not decide whether a given correlation is "good enough" — no
  threshold is hard-coded. Conventional rough bands for Pearson's r
  (weak <0.3, moderate 0.3-0.7, strong >0.7) are printed for reference
  only; the research team makes the actual call.
- It does not implement its own statistics library. Pearson/Spearman/MAE
  are computed with plain Python (no numpy/scipy dependency) since this
  project is deliberately conservative about adding new dependencies for
  one-off scripts (see enrich_snopes_claims.py's own Ollama-only approach
  and ragas_scorer.py's pinned-dependency caution).
- It is not covered by the 100% coverage gate (scripts/* is excluded).

Usage
------
    python scripts/compute_calibration_correlation.py calibration_batch.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _rescale_1_to_5(value: float) -> float:
    """Rescale a 1-5 Likert rating to [0, 1] (1 -> 0.0, 5 -> 1.0)."""
    return (value - 1.0) / 4.0


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """
    Pearson correlation coefficient between two equal-length sequences.

    Returns 0.0 if fewer than 2 points are given or either sequence has
    zero variance (undefined correlation, degrades gracefully rather than
    raising ZeroDivisionError).
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    return cov / denom if denom > 0 else 0.0


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation: Pearson's r computed on ranks instead of raw values."""
    return pearson_r(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    """
    Convert values to their ranks (1-indexed), averaging ranks for ties.

    E.g. [10, 20, 20, 30] -> [1.0, 2.5, 2.5, 4.0].
    """
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_indices):
        j = i
        # Extend the tie group while values are equal.
        while j + 1 < len(sorted_indices) and values[sorted_indices[j + 1]] == values[sorted_indices[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = average_rank
        i = j + 1
    return ranks


def mean_absolute_error(xs: list[float], ys: list[float]) -> float:
    """Mean absolute difference between paired values. 0.0 for an empty input."""
    if not xs:
        return 0.0
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def _interpret(r: float) -> str:
    """Non-authoritative, conventional Pearson's r band, for reference only."""
    magnitude = abs(r)
    if magnitude < 0.3:
        return "weak"
    if magnitude < 0.7:
        return "moderate"
    return "strong"


def load_annotations(workbook_path: Path) -> list[dict]:
    """
    Read the completed annotation workbook, keeping only rows where BOTH
    faithfulness_human and correctness_human were actually filled in.

    Rows left blank by the annotator (not yet reviewed) are skipped, not
    treated as 0 — a blank is "not annotated yet", not "worst possible
    score", and silently including it would corrupt the correlation.
    """
    import openpyxl

    wb = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    sheet = wb[wb.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    wb.close()

    header = list(rows[0])
    idx = {name: i for i, name in enumerate(header)}
    for required in ("claim_id", "faithfulness_human", "correctness_human"):
        if required not in idx:
            raise ValueError(f"Annotation workbook is missing required column: {required}")

    annotations = []
    for row in rows[1:]:
        if row is None:
            continue
        faithfulness = row[idx["faithfulness_human"]]
        correctness = row[idx["correctness_human"]]
        if faithfulness is None or correctness is None:
            continue  # not yet annotated
        annotations.append(
            {
                "claim_id": row[idx["claim_id"]],
                "faithfulness_human": float(faithfulness),
                "correctness_human": float(correctness),
            }
        )
    return annotations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare human faithfulness/correctness annotations against the automated scorer."
    )
    parser.add_argument("annotated_workbook", help="Path to the completed annotation .xlsx")
    parser.add_argument(
        "--automated-scores",
        default=None,
        help=(
            "Path to the automated-scores reference JSON. Defaults to "
            "<annotated_workbook>.automated_scores.json (the sibling file "
            "sample_for_faithfulness_calibration.py writes automatically)."
        ),
    )
    args = parser.parse_args(argv)

    workbook_path = Path(args.annotated_workbook)
    if not workbook_path.exists():
        print(f"Error: file not found: {workbook_path}", file=sys.stderr)
        return 1

    reference_path = (
        Path(args.automated_scores)
        if args.automated_scores
        else workbook_path.with_suffix(workbook_path.suffix + ".automated_scores.json")
    )
    if not reference_path.exists():
        print(f"Error: automated-scores reference file not found: {reference_path}", file=sys.stderr)
        return 1

    annotations = load_annotations(workbook_path)
    if not annotations:
        print("No fully-annotated rows found (faithfulness_human/correctness_human both blank everywhere).")
        return 0

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    automated = reference["scores"]

    human_faithfulness, auto_faithfulness = [], []
    human_correctness, auto_correctness = [], []
    skipped_no_match = 0
    for ann in annotations:
        scores = automated.get(ann["claim_id"])
        if scores is None:
            skipped_no_match += 1
            continue
        human_faithfulness.append(_rescale_1_to_5(ann["faithfulness_human"]))
        auto_faithfulness.append(scores["ragas_faithfulness"])
        human_correctness.append(_rescale_1_to_5(ann["correctness_human"]))
        auto_correctness.append(scores["ragas_answer_correctness"])

    scorer_label = reference.get("faithfulness_scorer_config") or "unknown"
    print(f"Scorer under calibration: {scorer_label!r} (from {reference['source_results_json']})")
    print(f"Annotated rows: {len(annotations)} ({skipped_no_match} had no matching automated score, skipped)")
    print(f"Rows used for correlation: {len(human_faithfulness)}\n")

    for label, human, auto in (
        ("Faithfulness", human_faithfulness, auto_faithfulness),
        ("Answer correctness", human_correctness, auto_correctness),
    ):
        r = pearson_r(human, auto)
        rho = spearman_rho(human, auto)
        mae = mean_absolute_error(human, auto)
        print(f"{label}:")
        print(f"  Pearson r  = {r:+.3f} ({_interpret(r)})")
        print(f"  Spearman rho = {rho:+.3f} ({_interpret(rho)})")
        print(f"  MAE        = {mae:.3f}")
        print()

    print(
        "Note: 'weak'/'moderate'/'strong' bands are conventional reference points only "
        "(|r| < 0.3 / 0.3-0.7 / > 0.7) — the research team decides what correlation is "
        "acceptable for this specific use case, not this script."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
