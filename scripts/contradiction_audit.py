"""
Contradiction audit: would an NLI check on the retrieved context notice the poison?

Defence question
----------------
A simple defence for a RAG system is to check its own top-k for documents that
contradict each other before answering. This script measures how far that
defence could go on saved results, without any new generation:

1. For every record, every pair of retrieved documents is scored with a
   3-way NLI cross-encoder (the Source Integrity model). NLI is not symmetric,
   so a pair's score is max(P(contradiction | a, b), P(contradiction | b, a)).
   The record's score is the maximum over its pairs.
2. The flag threshold is calibrated on the 0% runs, where no poison exists:
   tau = the (1 - fpr) quantile of record scores, so about ``fpr`` of clean
   records are flagged (false alarms). A fixed threshold (default 0.5) is
   reported as well.
3. Poisoned runs are then split by exposure (eiger.utils.records.classify):
   - own_and_truth: the variant AND its true document are both in the top-k.
     Only here can a pairwise check see the contradiction.
   - own_without_truth: the variant is retrieved but the truth is not. A
     pairwise check cannot see anything: there is nothing to contradict.
   - other_only / clean: flags here are false alarms.

Main outputs: share of own-exposed records that are detectable at all
(own_and_truth / own), detection rate within own_and_truth, false-alarm rate
on clean records, and the NLI contradiction score of the (truth, own variant)
pair itself (does the model even see a one-number change as a contradiction?).

Proportions carry 95% Wilson intervals computed over records. Records are not
independent (same claims across seeds), so these intervals are optimistic;
read them as descriptive.

Usage
-----
    python scripts/contradiction_audit.py \\
        --results-dir results/corpus_sweep_official_genq_preliminary
    # quick check on a few records:
    python scripts/contradiction_audit.py --results-dir ... --max-records 20

Writes ``contradiction_audit.md`` and ``contradiction_audit_records.csv`` into
the results directory.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from eiger.utils.nli import NLIScorer
from eiger.utils.records import (
    classify,
    find_runs,
    hit_doc,
    is_poisoned,
    load_records,
    ordered_hits,
    wilson_interval,
)

GROUPS = ("clean", "other_only", "own_without_truth", "own_and_truth")


def record_group(info: dict[str, Any]) -> str:
    if info["category"] == "own":
        return "own_and_truth" if info["own_and_truth"] else "own_without_truth"
    return str(info["category"])


def record_pairs(record: dict[str, Any]) -> list[tuple[int, int, str, str, bool]]:
    """
    (rank_a, rank_b, text_a, text_b, same_claim_variant) for every unordered pair
    of retrieved documents. ``same_claim_variant`` marks a (truth, variant) pair
    of the same claim, whichever claim it is.
    """
    out = []
    for a, b in combinations(ordered_hits(record), 2):
        da, db = hit_doc(a), hit_doc(b)
        same = da.get("claim_id") == db.get("claim_id") and is_poisoned(a) != is_poisoned(b)
        out.append(
            (
                a.get("rank", 0),
                b.get("rank", 0),
                str(da.get("text", "")),
                str(db.get("text", "")),
                same,
            )
        )
    return out


def prefetch(records: list[dict[str, Any]], nli: NLIScorer) -> None:
    """Score every pair of every record in batches (both directions), filling the cache."""
    pairs: list[tuple[str, str]] = []
    for record in records:
        for _, _, ta, tb, _ in record_pairs(record):
            pairs += [(ta, tb), (tb, ta)]
    nli.scores(pairs)


def score_record(record: dict[str, Any], nli: NLIScorer) -> dict[str, Any]:
    """Classify the record and attach its NLI contradiction scores."""
    info = classify(record)
    pairs = record_pairs(record)
    scores = [nli.symmetric_contradiction(ta, tb) for _, _, ta, tb, _ in pairs]
    same = [s for s, p in zip(scores, pairs) if p[4]]
    own_pair = None
    if info["own_and_truth"]:
        own_pair = nli.symmetric_contradiction(info["truth_text"], info["own_variant_text"])
    return {
        "claim_id": record.get("claim_id"),
        "group": record_group(info),
        "n_pairs": len(pairs),
        "max_contradiction": max(scores) if scores else 0.0,
        "max_same_claim_pair": max(same) if same else None,
        "own_pair_contradiction": own_pair,
    }


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (q in [0, 1])."""
    if not values:
        raise ValueError("quantile of an empty list")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def calibrate_threshold(rows: list[dict[str, Any]], fpr: float) -> float | None:
    """(1 - fpr) quantile of record scores in the 0% runs; None if there are none."""
    clean = [r["max_contradiction"] for r in rows if r["rate"] == 0]
    return quantile(clean, 1 - fpr) if clean else None


def _fmt_prop(k: int, n: int) -> str:
    if n == 0:
        return "n/a (0)"
    lo, hi = wilson_interval(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}] (n={n})"


def summarize(rows: list[dict[str, Any]], threshold: float) -> dict[int, dict[str, Any]]:
    """Per rate: group sizes, flag counts, detectable share, own-pair scores."""
    by_rate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rate[row["rate"]].append(row)
    out: dict[int, dict[str, Any]] = {}
    for rate in sorted(by_rate):
        rate_rows = by_rate[rate]
        groups = {g: [r for r in rate_rows if r["group"] == g] for g in GROUPS}
        flagged = {g: sum(r["max_contradiction"] > threshold for r in groups[g]) for g in GROUPS}
        own_pairs = [r["own_pair_contradiction"] for r in groups["own_and_truth"]]
        n_own = len(groups["own_and_truth"]) + len(groups["own_without_truth"])
        out[rate] = {
            "n": len(rate_rows),
            "groups": {g: len(groups[g]) for g in GROUPS},
            "flagged": flagged,
            "n_own": n_own,
            "own_pair_mean": statistics.fmean(own_pairs) if own_pairs else None,
            "own_pair_above": sum(p > threshold for p in own_pairs),
        }
    return out


def render(summary: dict[int, dict[str, Any]], threshold: float, label: str) -> list[str]:
    lines = [
        f"### Threshold {threshold:.3f} ({label})",
        "",
        "| Rate | Records | False alarms (clean) | Flagged (other only) | Detectable (truth + variant in top-k, share of own) "
        "| Detected (within detectable) | Detected (share of all own) | Truth-variant pair P(contr.) mean | Pair above threshold |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rate, s in summary.items():
        g, f = s["groups"], s["flagged"]
        detected_all = f["own_and_truth"] + f["own_without_truth"]
        pair_mean = "n/a" if s["own_pair_mean"] is None else f"{s['own_pair_mean']:.3f}"
        lines.append(
            f"| {rate}% | {s['n']} | {_fmt_prop(f['clean'], g['clean'])} | {_fmt_prop(f['other_only'], g['other_only'])} "
            f"| {_fmt_prop(g['own_and_truth'], s['n_own'])} | {_fmt_prop(f['own_and_truth'], g['own_and_truth'])} "
            f"| {_fmt_prop(detected_all, s['n_own'])} | {pair_mean} | {_fmt_prop(s['own_pair_above'], g['own_and_truth'])} |"
        )
    return lines + [""]


def main(argv: list[str] | None = None, nli: NLIScorer | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--fpr", type=float, default=0.05, help="false-alarm rate used to calibrate on 0%% runs"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="fixed threshold reported alongside"
    )
    parser.add_argument(
        "--rates", type=int, nargs="*", help="only these rates (0 is needed for calibration)"
    )
    parser.add_argument("--max-records", type=int, help="per run, for a quick check")
    parser.add_argument(
        "--model", default=None, help="NLI model (default: the Source Integrity model)"
    )
    args = parser.parse_args(argv)

    runs = [r for r in find_runs(args.results_dir) if not args.rates or r[0] in args.rates]
    if not runs:
        print(f"No runs found in {args.results_dir}", file=sys.stderr)
        return 1
    nli = nli or (NLIScorer(args.model) if args.model else NLIScorer())

    rows: list[dict[str, Any]] = []
    for rate, seed, path in runs:
        records = load_records(path)[: args.max_records]
        prefetch(records, nli)
        for record in records:
            rows.append({"rate": rate, "seed": seed} | score_record(record, nli))
        print(
            f"{path.parent.name}: {len(records)} records, NLI cache {nli.cache_size} pairs",
            file=sys.stderr,
        )

    calibrated = calibrate_threshold(rows, args.fpr)
    lines = [
        f"# Contradiction audit: {args.results_dir}",
        "",
        f"NLI model: {nli.model_name}. Runs: {len(runs)}. Records: {len(rows)}. "
        "Score per record = max over retrieved-document pairs of max(P(contr.|a,b), P(contr.|b,a)).",
        "",
    ]
    if calibrated is None:
        lines += [
            "No 0% runs: calibrated threshold not available; only the fixed threshold is reported.",
            "",
        ]
    else:
        lines += render(
            summarize(rows, calibrated),
            calibrated,
            f"calibrated on 0% runs, target false-alarm rate {args.fpr}",
        )
    lines += render(summarize(rows, args.threshold), args.threshold, "fixed")
    lines += [
        "Reading: 'Detectable' is the ceiling of any pairwise contradiction check: when the true document is not",
        "retrieved, the variant contradicts nothing in the context. Intervals are Wilson intervals over records",
        "(records are not independent across seeds: descriptive only).",
    ]
    report = "\n".join(lines)
    print(report)

    out_md = args.results_dir / "contradiction_audit.md"
    out_md.write_text(report + "\n", encoding="utf-8")
    out_csv = args.results_dir / "contradiction_audit_records.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten: {out_md}, {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
