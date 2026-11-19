"""
Compare the counterfactual generations of two sweeps run on the same claims,
seeds and poisoned set (for example multi2 k=1 and k=2).

For every record with a counterfactual answer in both sweeps, it counts whether
the counterfactual context (top-k minus every poisoned hit, saved as
generation.metadata["counterfactual_context_docs"]) and the counterfactual
answer are the same. "Same context, different answer" > 0 would mean that
generation is not deterministic; 0 means every difference in the answers
comes from a different context.

Usage:
    python scripts/check_counterfactual_context.py SWEEP_DIR_A SWEEP_DIR_B
"""

import sys
from pathlib import Path
from typing import Any

from eiger.utils.records import find_runs, load_records


def load(results_dir: str) -> dict[tuple[Any, ...], tuple[Any, Any]]:
    """(rate, seed, claim_id) -> (counterfactual answer, counterfactual context)."""
    out: dict[tuple[Any, ...], tuple[Any, Any]] = {}
    for rate, seed, path in find_runs(Path(results_dir)):
        for record in load_records(path):
            meta = (record.get("generation") or {}).get("metadata") or {}
            if "counterfactual_answer" in meta:
                key = (rate, seed, record.get("claim_id"))
                out[key] = (meta["counterfactual_answer"], meta.get("counterfactual_context_docs"))
    return out


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: check_counterfactual_context.py SWEEP_DIR_A SWEEP_DIR_B")
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(a.keys() & b.keys(), key=str)
    counts = {
        "stesso contesto, stessa risposta": 0,
        "stesso contesto, risposta diversa": 0,
        "contesto diverso, stessa risposta": 0,
        "contesto diverso, risposta diversa": 0,
    }
    for key in keys:
        context = "stesso contesto" if a[key][1] == b[key][1] else "contesto diverso"
        answer = "stessa risposta" if a[key][0] == b[key][0] else "risposta diversa"
        counts[f"{context}, {answer}"] += 1
    print("coppie confrontate:", len(keys))
    for name, value in counts.items():
        print(f"{value:6d}  {name}")


if __name__ == "__main__":
    main()
