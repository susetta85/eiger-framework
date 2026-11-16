"""
Multi-element variants: controlled, rule-based changes to k numbers or years.

Research question
-----------------
Does the number of changed elements in a manipulated claim (k = 1, 2, 3) change
how often the variant is retrieved, whether it outranks the true document, and
whether the answer adopts it? The official corpus cannot answer this: almost
every variant changes a single element (docs/EXPERIMENTS.md, section 4.13).

What the generator does
-----------------------
For each verified-true claim of the official corpus it finds the numeric
elements (integers, decimals, years; identifiers glued to letters such as
"COVID-19" or "F-35" are skipped), shuffles their order with a fixed generator
seed, and changes the first k of them with one fixed rule per kind:

- year (1800-2099): +1 or -1;
- integer: +1 or -1 below 10, otherwise +/-10% rounded (at least 1);
- decimal: +/-10% at the same number of decimals (at least one unit in the
  last place).

The variants are nested: variant k=2 contains the change of k=1 plus one more,
so k is compared within the same claim. The magnitudes mirror the official
corpus's own changes (e.g. "2020 -> 2021", "9 -> 10").

Boundaries (ethics protocol)
----------------------------
- No language model, no rewording: only the numeric tokens change.
- Nothing is tuned for plausibility or for evading detection; the rule is the
  same for every claim and does not look at the result.
- Output goes to data/multi_element/ (git-ignored, not published). It is fully
  reproducible from the versioned corpus, so it never needs to be shared as a
  file: each machine regenerates it and checks the SHA-256.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.models import Claim
from eiger.datasets.corpus_claim_official import CorpusClaimOfficialDataset
from eiger.utils.query_leak import numeric_leak_tokens

GENERATOR_VERSION = "1"
DEFAULT_MULTI_ELEMENT_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "multi_element" / "variants.csv"
)
MANIPULATION_CATEGORY = "MULTI_NUM"
FIELDNAMES = (
    "claim_id",
    "k",
    "n_elements",
    "claim_original_reference",
    "claim_text",
    "changed_element",
    "manipulation_category",
    "generator_version",
    "generator_seed",
)

# A number not glued to a letter, dot, hyphen or slash on its left, and not
# followed by a letter, slash, another digit group or a "-digit" on its right.
NUMBER_RE = re.compile(r"(?<![\w.\-/])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w/]|\.\d|-\d)")


@dataclass(frozen=True)
class Element:
    value: str
    kind: str  # "year" | "integer" | "decimal"


def element_kind(value: str) -> str:
    if "." in value:
        return "decimal"
    if "," not in value and len(value) == 4 and 1800 <= int(value) <= 2099:
        return "year"
    return "integer"


def extract_elements(text: str) -> list[Element]:
    """Distinct numeric elements in order of first appearance."""
    seen: dict[str, Element] = {}
    for match in NUMBER_RE.finditer(text):
        value = match.group(1)
        if value not in seen:
            seen[value] = Element(value, element_kind(value))
    return list(seen.values())


def _format_int(value: int, like: str) -> str:
    return f"{value:,}" if "," in like else str(value)


def perturb(element: Element, rng: random.Random) -> str:
    """Apply the fixed rule for the element's kind (see module docstring)."""
    sign = rng.choice((-1, 1))
    if element.kind == "year":
        return str(int(element.value) + sign)
    if element.kind == "decimal":
        decimals = len(element.value.split(".")[1])
        value = float(element.value)
        unit = 10.0**-decimals
        delta = max(unit, round(abs(value) * 0.1, decimals))
        new = round(value + sign * delta, decimals)
        if new <= 0:
            new = round(value + delta, decimals)
        return f"{new:.{decimals}f}"
    value = int(element.value.replace(",", ""))
    if value < 10:
        new_int = value + 1 if value <= 1 else value + sign
    else:
        new_int = value + sign * max(1, round(value * 0.1))
    return _format_int(new_int, element.value)


def _rng_for(claim_id: str, generator_seed: int) -> random.Random:
    digest = hashlib.sha256(f"{generator_seed}:{claim_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def apply_changes(text: str, changes: dict[str, str]) -> str:
    """Replace every occurrence of each changed element (same token boundaries as extraction)."""
    return NUMBER_RE.sub(lambda m: changes.get(m.group(1), m.group(1)), text)


def make_variants(
    claim_id: str, text: str, max_k: int = 3, generator_seed: int = 0
) -> list[dict[str, Any]]:
    """
    Nested variants for one claim: one row per k = 1 .. min(max_k, n_elements).
    Returns [] for a claim without numeric elements.
    """
    elements = extract_elements(text)
    rng = _rng_for(claim_id, generator_seed)
    order = list(range(len(elements)))
    rng.shuffle(order)
    taken = {e.value for e in elements}
    new_values: dict[str, str] = {}
    for idx in order[:max_k]:
        element = elements[idx]
        candidate = perturb(element, rng)
        # Never collide with another element of the same claim (that would
        # merge two elements into one) or reproduce the original.
        while candidate in taken or candidate == element.value:
            candidate = perturb(Element(candidate, element.kind), rng)
        new_values[element.value] = candidate
        taken.add(candidate)
    rows = []
    for k in range(1, min(max_k, len(elements)) + 1):
        chosen = [elements[i].value for i in order[:k]]
        changes = {value: new_values[value] for value in chosen}
        rows.append(
            {
                "claim_id": claim_id,
                "k": k,
                "n_elements": len(elements),
                "claim_original_reference": text,
                "claim_text": apply_changes(text, changes),
                "changed_element": "; ".join(f"{a} -> {b}" for a, b in changes.items()),
                "manipulation_category": MANIPULATION_CATEGORY,
                "generator_version": GENERATOR_VERSION,
                "generator_seed": generator_seed,
            }
        )
    return rows


def write_variants(rows: list[dict[str, Any]], path: Path) -> str:
    """Write rows as CSV (stable order) and return the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["claim_id"], int(r["k"])))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_multi_element_index(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Rows keyed by (claim_id, k). Raises IngestionError if the file is missing."""
    if not Path(path).is_file():
        raise IngestionError(
            f"Multi-element variants file not found: {path}. "
            "Generate it with: python scripts/make_multi_element_variants.py"
        )
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return {(row["claim_id"], int(row["k"])): row for row in csv.DictReader(fh)}


class CorpusClaimOfficialMultiElementDataset(CorpusClaimOfficialDataset):
    """
    Official corpus claims (all rows, generated v2 queries) that have at least
    ``min_elements`` numeric elements, so every k up to ``min_elements`` exists
    for the SAME claims. A claim is dropped if its query repeats any number
    changed by the largest variant (numeric leak, same policy as genq).
    """

    name: str = "corpus_claim_official_multi_genq"
    description: str = (
        "ENGINEERING ONLY: official corpus, claims with >= N numeric elements, "
        "multi-element variants."
    )
    include_unreviewed: bool = True
    use_generated_queries: bool = True
    keep_numeric_leaks: bool = True  # leaks are checked below against OUR variants
    min_elements: int = 2

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        query_file: str | Path | None = None,
        variants_file: str | Path | None = None,
    ) -> None:
        super().__init__(corpus_dir, query_file)
        self.variants_file = (
            Path(variants_file) if variants_file is not None else DEFAULT_MULTI_ELEMENT_FILE
        )

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        index = load_multi_element_index(self.variants_file)
        kept: list[Claim] = []
        for claim in super().load(split, max_claims=None):
            row = index.get((claim.claim_id, self.min_elements))
            if row is None or row["claim_original_reference"] != claim.original_fact:
                continue
            if numeric_leak_tokens(claim.context_query, claim.original_fact, row["claim_text"]):
                continue
            claim.metadata["n_numeric_elements"] = int(row["n_elements"])
            claim.metadata["query_numeric_leak"] = False
            kept.append(claim)
        if max_claims is not None:
            kept = kept[:max_claims]
        self._loaded_claims = kept
        return kept


class CorpusClaimOfficialMulti2Dataset(CorpusClaimOfficialMultiElementDataset):
    name: str = "corpus_claim_official_multi2_genq"
    description: str = (
        "ENGINEERING ONLY: official-corpus claims with >= 2 numeric elements (k = 1, 2)."
    )
    min_elements: int = 2


class CorpusClaimOfficialMulti3Dataset(CorpusClaimOfficialMultiElementDataset):
    name: str = "corpus_claim_official_multi3_genq"
    description: str = (
        "ENGINEERING ONLY: official-corpus claims with >= 3 numeric elements (k = 1, 2, 3)."
    )
    min_elements: int = 3
