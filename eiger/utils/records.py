"""
Helpers for analysis scripts that read saved ``results.json`` files.

Every analysis of a sweep needs the same three things: find the runs
(``<rate>pct_seed<seed>/results.json``), order a record's retrieved hits, and
classify the record by what kind of poison reached its top-k. Keeping this in
one tested place avoids each script re-deriving the categories slightly
differently (the PRR decomposition in ``aggregate_benchmark_sweep.py`` uses the
same definitions).

Categories (per record, i.e. per query)
---------------------------------------
- ``clean``:      no poisoned document in the top-k.
- ``own``:        the queried claim's OWN poisoned variant is in the top-k
                  (targeted exposure).
- ``other_only``: only poisoned variants of OTHER claims are in the top-k
                  (collateral exposure).

The two booleans that matter for answer-level and defence analyses:
- ``truth_retrieved``: the queried claim's ground-truth document is in the top-k.
- ``own_and_truth``:   both the true document and its own variant are in the
                       top-k, i.e. the context contains the contradiction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RUN_RE = re.compile(r"^(?P<rate>\d+)pct_seed(?P<seed>\d+)$")


def find_runs(results_dir: Path) -> list[tuple[int, int, Path]]:
    """Return ``(rate_pct, seed, results.json path)`` for every completed run, sorted."""
    runs = []
    for path in sorted(Path(results_dir).glob("*/results.json")):
        match = RUN_RE.match(path.parent.name)
        if match:
            runs.append((int(match["rate"]), int(match["seed"]), path))
    return sorted(runs)


def load_records(path: Path) -> list[dict[str, Any]]:
    """Records of one results.json (empty list if the key is missing)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = data.get("records", []) or []
    return records


def ordered_hits(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Retrieved hits sorted by rank (missing rank sorts last)."""
    hits = (record.get("retrieval", {}) or {}).get("hits", []) or []
    return sorted(hits, key=lambda h: h.get("rank", 10**9))


def hit_doc(hit: dict[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = hit.get("document", {}) or {}
    return doc


def is_poisoned(hit: dict[str, Any]) -> bool:
    doc = hit_doc(hit)
    return doc.get("doc_type") == "poisoned" or bool(doc.get("is_poisoned"))


def classify(record: dict[str, Any]) -> dict[str, Any]:
    """
    Exposure category and positions for one record.

    Returns a dict with: category, own_retrieved, truth_retrieved, own_and_truth,
    variant_outranks_truth (own variant ranked above the truth, or truth absent),
    poison_rank1, own_rank, truth_rank, own_variant_text, truth_text (None when
    not retrieved).
    """
    claim_id = record.get("claim_id")
    hits = ordered_hits(record)
    own_rank = truth_rank = None
    own_text = truth_text = None
    any_poison = False
    for hit in hits:
        doc = hit_doc(hit)
        if is_poisoned(hit):
            any_poison = True
            if doc.get("claim_id") == claim_id and own_rank is None:
                own_rank, own_text = hit.get("rank"), doc.get("text")
        elif doc.get("claim_id") == claim_id and truth_rank is None:
            truth_rank, truth_text = hit.get("rank"), doc.get("text")
    own = own_rank is not None
    category = "own" if own else ("other_only" if any_poison else "clean")
    return {
        "category": category,
        "own_retrieved": own,
        "truth_retrieved": truth_rank is not None,
        "own_and_truth": own and truth_rank is not None,
        "variant_outranks_truth": own and (truth_rank is None or own_rank < truth_rank),
        "poison_rank1": bool(hits) and hits[0].get("rank") == 1 and is_poisoned(hits[0]),
        "own_rank": own_rank,
        "truth_rank": truth_rank,
        "own_variant_text": own_text,
        "truth_text": truth_text,
    }


def ground_truth_text(record: dict[str, Any]) -> str | None:
    """
    The claim's true statement: from the generation metadata (stored by the
    runner since the Source Integrity fix), else from the retrieved true document.
    """
    meta = (record.get("generation", {}) or {}).get("metadata", {}) or {}
    text = meta.get("ground_truth")
    if text:
        return str(text)
    truth: str | None = classify(record)["truth_text"]
    return truth


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; (0, 0) when n == 0."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
