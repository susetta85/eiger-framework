"""
Parity check between two runs of the same configuration (for example the same
0% config run on two machines).

For the claims present in both runs it reports how many have:
- the same retrieved documents, in the same order (doc_id by rank);
- the same answer text;
- the same counterfactual answer (when both runs have one).

Two machines with the same code, models and data should give 100% on all
three. Retrieval below 100% points at the index or the retriever (for BM25:
tokenizer, rank-bm25 version, tie-breaking); answers below 100% with identical
retrieval point at generation (Ollama or model version, hardware).

Retrieval is compared by doc_id only, so it is meant for two runs of the same
configuration: k=1 and k=2 runs, for instance, share the poisoned doc ids but
not their text.

Usage:
    python scripts/compare_runs.py RUN_A RUN_B [--show N]
    RUN_x: a run folder (containing results.json) or a results.json path.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    return json.loads(p.read_text(encoding="utf-8"))


def by_claim(data: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    return {r.get("claim_id"): r for r in data.get("records", []) or []}


def doc_ids(record: dict[str, Any]) -> list[Any]:
    hits = (record.get("retrieval") or {}).get("hits") or []
    return [(h.get("document") or {}).get("doc_id") for h in hits]


def generation(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("generation") or {}


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ra, rb = by_claim(a), by_claim(b)
    common = sorted(ra.keys() & rb.keys(), key=str)
    same_retrieval = same_answer = same_cf = with_cf = 0
    answer_diffs = []
    for cid in common:
        x, y = ra[cid], rb[cid]
        retrieval_equal = doc_ids(x) == doc_ids(y)
        same_retrieval += retrieval_equal
        answer_equal = generation(x).get("answer") == generation(y).get("answer")
        same_answer += answer_equal
        if not answer_equal:
            answer_diffs.append((cid, retrieval_equal))
        cfx = (generation(x).get("metadata") or {}).get("counterfactual_answer")
        cfy = (generation(y).get("metadata") or {}).get("counterfactual_answer")
        if cfx is not None and cfy is not None:
            with_cf += 1
            same_cf += cfx == cfy
    return {
        "only_a": len(ra.keys() - rb.keys()),
        "only_b": len(rb.keys() - ra.keys()),
        "common": len(common),
        "same_retrieval": same_retrieval,
        "same_answer": same_answer,
        "with_cf": with_cf,
        "same_cf": same_cf,
        "answer_diffs": answer_diffs,
        "commits": (a.get("git_commit", "")[:7], b.get("git_commit", "")[:7]),
        "platforms": (
            (a.get("environment") or {}).get("platform", ""),
            (b.get("environment") or {}).get("platform", ""),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--show", type=int, default=5, help="claim ids with a different answer")
    args = parser.parse_args()
    res = compare(load(args.run_a), load(args.run_b))
    n = res["common"] or 1
    print(f"commit: {res['commits'][0]} / {res['commits'][1]}")
    print(f"claim in comune: {res['common']} (solo A: {res['only_a']}, solo B: {res['only_b']})")
    print(f"stesso retrieval:  {res['same_retrieval']} ({100 * res['same_retrieval'] / n:.1f}%)")
    print(f"stessa risposta:   {res['same_answer']} ({100 * res['same_answer'] / n:.1f}%)")
    if res["with_cf"]:
        print(f"stessa risposta controfattuale: {res['same_cf']} su {res['with_cf']}")
    diffs = res["answer_diffs"]
    if diffs:
        same_ret = sum(1 for _, r in diffs if r)
        print(f"risposte diverse con lo stesso retrieval: {same_ret} su {len(diffs)}")
        for cid, r in diffs[: args.show]:
            print(f"  {cid}  retrieval {'uguale' if r else 'diverso'}")


if __name__ == "__main__":
    main()
