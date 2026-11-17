"""
Generate the multi-element variants file (data/multi_element/variants.csv).

Rule-based, no language model: for every verified-true English claim of the
official corpus, change k = 1, 2, 3 of its elements with one rule per kind.
Families (eiger/datasets/multi_element.py has the rules and the ethics
boundaries): num (numbers and years, fixed rule; the default, file
variants.csv), numrand (same elements, random magnitude), mix (numbers plus
US states, countries, months, scale and direction words, fixed rule),
mixrand (same elements as mix, random replacement).

The output is git-ignored (data/) and must not be published. It is
deterministic: every machine regenerates it from the versioned corpus and
compares the SHA-256 printed at the end (record it in docs/EXPERIMENTS.md).

Usage:
    python scripts/make_multi_element_variants.py
    python scripts/make_multi_element_variants.py --show 5     # print a few examples
    python scripts/make_multi_element_variants.py --family mix
    python scripts/make_multi_element_variants.py --all        # every family, one SHA-256 each
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from eiger.datasets.corpus_claim_official import CorpusClaimOfficialUnreviewedDataset
from eiger.datasets.multi_element import FAMILIES, family_file, make_variants, write_variants


def generate(family: str, out: Path, max_k: int, seed: int, show: int, claims) -> str:
    rows = []
    for claim in claims:
        rows += make_variants(claim.claim_id, claim.original_fact, max_k, seed, family)
    digest = write_variants(rows, out)
    per_claim = Counter(r["claim_id"] for r in rows)
    print(f"\n[{family}] claims with >= 1 element: {len(per_claim)}")
    for k in range(1, max_k + 1):
        print(f"  claims with a k={k} variant: {sum(1 for r in rows if int(r['k']) == k)}")
    for row in rows[:show]:
        print(f"\n[{row['claim_id']} k={row['k']}] {row['changed_element']}\n  {row['claim_text']}")
    spec = FAMILIES[family]
    print(f"Written {len(rows)} rows to {out}")
    print(f"generator_version={spec.version} generator_seed={seed} mode={spec.mode}")
    print(f"SHA-256 {family}: {digest}")
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", choices=sorted(FAMILIES), default="num")
    parser.add_argument("--all", action="store_true", help="generate every family")
    parser.add_argument("--out", type=Path, help="default: data/multi_element/<family file>")
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--generator-seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=0, help="print this many example rows")
    args = parser.parse_args(argv)
    if args.all and args.out:
        parser.error("--out works with a single --family")

    claims = CorpusClaimOfficialUnreviewedDataset().load()
    print(f"Claims read: {len(claims)}")
    for family in sorted(FAMILIES) if args.all else [args.family]:
        out = args.out or family_file(family)
        generate(family, out, args.max_k, args.generator_seed, args.show, claims)
    return 0


if __name__ == "__main__":
    sys.exit(main())
