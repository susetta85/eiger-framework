"""
Generate the multi-element variants file (data/multi_element/variants.csv).

Rule-based, no language model: for every verified-true English claim of the
official corpus, change k = 1, 2, 3 of its numeric elements (numbers, years)
with one fixed rule per kind. See eiger/datasets/multi_element.py for the rules
and the ethics boundaries.

The output is git-ignored (data/) and must not be published. It is
deterministic: every machine regenerates it from the versioned corpus and
compares the SHA-256 printed at the end (record it in docs/EXPERIMENTS.md).

Usage:
    python scripts/make_multi_element_variants.py
    python scripts/make_multi_element_variants.py --show 5     # print a few examples
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from eiger.datasets.corpus_claim_official import CorpusClaimOfficialUnreviewedDataset
from eiger.datasets.multi_element import (
    DEFAULT_MULTI_ELEMENT_FILE,
    GENERATOR_VERSION,
    make_variants,
    write_variants,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_MULTI_ELEMENT_FILE)
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--generator-seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=0, help="print this many example rows")
    args = parser.parse_args(argv)

    claims = CorpusClaimOfficialUnreviewedDataset().load()
    rows = []
    for claim in claims:
        rows += make_variants(claim.claim_id, claim.original_fact, args.max_k, args.generator_seed)
    digest = write_variants(rows, args.out)

    per_claim = Counter(r["claim_id"] for r in rows)
    print(f"Claims read: {len(claims)}; with >= 1 numeric element: {len(per_claim)}")
    for k in range(1, args.max_k + 1):
        print(f"  claims with a k={k} variant: {sum(1 for r in rows if int(r['k']) == k)}")
    for row in rows[: args.show]:
        print(f"\n[{row['claim_id']} k={row['k']}] {row['changed_element']}\n  {row['claim_text']}")
    print(f"\nWritten {len(rows)} rows to {args.out}")
    print(f"generator_version={GENERATOR_VERSION} generator_seed={args.generator_seed}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
