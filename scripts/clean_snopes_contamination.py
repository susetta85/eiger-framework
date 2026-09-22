"""
scripts/clean_snopes_contamination.py

One-off cleanup script for an already-generated data/snopes/snopes_enriched.json
that was produced *before* the Sprint 4 fix to filter_and_dedupe() in
scripts/enrich_snopes_claims.py.

Why this script exists
-----------------------
A Sprint 4 code-review audit found that enrich_snopes_claims.py trusted the
raw export's ``normalised_rating`` column alone to decide "verified true",
without cross-checking it against the same row's own ``original_verdict``
(which the script already carries into every output entry's ``notes``
field, but never validates). A spot-check of the live
data/snopes/snopes_enriched.json found 458 of 3,400 claims (13.5%) whose
original_verdict directly contradicts "verified true" — e.g.
original_verdict=Fake/Unproven/Research In Progress/Mixture/Miscaptioned/
Outdated/Misattributed/Altered/Scam/Legend/Labeled Satire/False — despite
normalised_rating being True for every one of them. Concrete example
pulled from the live file:

    claim_id: SNOPES_80319
    original_fact: "A video posted by TikTok user @iceman_fox1 ...
                     authentically showed an unidentified drone spotted
                     flying at low altitude ... over New Jersey."
    notes: "original_verdict=Fake; ..."

The claim text asserts the claim as true; Snopes' own verdict for the same
row says it's fake. This directly violates the project's core invariant
that every ``Claim.original_fact`` loaded as ground truth must be a
verified-true statement (see docs/DATASETS.md §1) — any FFR/ERS/Source
Integrity numbers computed from the contaminated rows are not measuring
what the project's metrics are documented to measure.

enrich_snopes_claims.py's filter_and_dedupe() was fixed to reject these
rows going forward (this script's parsing logic — see
_is_verdict_consistent_with_true() below — is a copy of the same check, so
that a full re-run is not required just to fix an already-generated file).
This script re-applies that same check retroactively to an existing output
file, without needing Ollama or a full re-run (which would take hours).

What this script does
----------------------
1. Reads an existing enriched JSON file (default:
   data/snopes/snopes_enriched.json).
2. Parses each entry's ``notes`` field (format:
   "original_verdict=<X>; date_published=<Y>") to recover the original
   verdict string.
3. Keeps only entries whose recovered original_verdict is in
   {"true", "correct attribution"} (case/whitespace-insensitive) — the
   same allowlist as enrich_snopes_claims.py's
   _VERIFIED_TRUE_ORIGINAL_VERDICTS.
4. Writes the cleaned list back to the input file (in place) and a
   companion ``<name>.removed.json`` file listing every removed entry in
   full, for manual review before anyone treats them as permanently gone.
5. Prints a summary: how many entries were kept vs. removed, and a
   breakdown of removed entries by their original_verdict value.

This script does NOT:
  - Re-run the LLM enrichment or re-download anything.
  - Modify scripts/enrich_snopes_claims.py's own behavior (already fixed
    separately).
  - Guess at a verdict for entries whose notes field doesn't match the
    expected "original_verdict=...; date_published=..." format — such
    entries are treated conservatively as REMOVED (logged separately as
    "unparseable"), since a research paper's ground truth must never
    include an entry whose verification status could not be confirmed.

Usage
-----
    python scripts/clean_snopes_contamination.py
    python scripts/clean_snopes_contamination.py data/snopes/snopes_enriched.json
    python scripts/clean_snopes_contamination.py --dry-run   # report only, no write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path("data/snopes/snopes_enriched.json")

# Must match scripts/enrich_snopes_claims.py's _VERIFIED_TRUE_ORIGINAL_VERDICTS
# exactly — kept as a separate literal here (rather than importing that
# module) so this script has zero dependency on the rest of the package and
# can run standalone against a JSON file with nothing else installed.
_VERIFIED_TRUE_ORIGINAL_VERDICTS = frozenset({"true", "correct attribution"})

_NOTES_VERDICT_RE = re.compile(r"original_verdict=(.*?);\s*date_published=")


def _extract_original_verdict(notes: str) -> str | None:
    """
    Recover the original_verdict value from an entry's ``notes`` field.

    Returns:
        The verdict string, or None if ``notes`` doesn't match the
        expected "original_verdict=<X>; date_published=<Y>" format.
    """
    match = _NOTES_VERDICT_RE.match(notes)
    return match.group(1) if match else None


def _is_verdict_consistent_with_true(original_verdict: str | None) -> bool:
    """Mirrors enrich_snopes_claims.py's _is_verdict_consistent_with_true()."""
    if original_verdict is None:
        return False
    return original_verdict.strip().lower() in _VERIFIED_TRUE_ORIGINAL_VERDICTS


def clean_entries(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split entries into (kept, removed) based on original_verdict consistency.

    Args:
        entries: The full list of enriched-claim dicts as loaded from JSON.

    Returns:
        (kept, removed) — two lists partitioning the input; every input
        entry appears in exactly one of the two.
    """
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for entry in entries:
        verdict = _extract_original_verdict(str(entry.get("notes", "")))
        if _is_verdict_consistent_with_true(verdict):
            kept.append(entry)
        else:
            removed.append(entry)
    return kept, removed


def _summarize_removed(removed: list[dict[str, Any]]) -> str:
    """Build a human-readable breakdown of removed entries by verdict value."""
    counts = Counter(
        _extract_original_verdict(str(e.get("notes", ""))) or "(unparseable notes field)"
        for e in removed
    )
    lines = [f"  {verdict!r}: {count}" for verdict, count in counts.most_common()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path",
        nargs="?",
        default=str(_DEFAULT_PATH),
        help=f"Path to the enriched JSON file (default: {_DEFAULT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without writing any files.",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"Error: '{path}' does not exist.", file=sys.stderr)
        return 1

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        print(f"Error: '{path}' does not contain a JSON array at the top level.", file=sys.stderr)
        return 1

    kept, removed = clean_entries(entries)

    print(f"Read {len(entries)} entries from {path}")
    print(f"  Kept:    {len(kept)}")
    print(f"  Removed: {len(removed)}")
    if removed:
        print("Removed entries by original_verdict:")
        print(_summarize_removed(removed))

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return 0

    if removed:
        removed_path = path.with_suffix(".removed.json")
        removed_path.write_text(json.dumps(removed, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {len(removed)} removed entries to {removed_path} for manual review.")

    path.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(kept)} cleaned entries back to {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
