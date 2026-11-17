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

Families (one variants file each)
---------------------------------
- ``num``: the rule above (fixed magnitude), numbers and years only. File
  ``variants.csv``, generator version 1. This is the file behind the first
  multi2/multi3 sweeps and must not change.
- ``numrand``: same elements, chosen in the same order as ``num`` (so the two
  families are paired within claim), but the magnitude is drawn at random:
  year +/-1..10, integer +/-1..5 below 10 else +/-5..50%, decimal +/-5..50%.
- ``mix``: numbers plus four non-numeric kinds, each with a fixed rule:
  ``us_state`` and ``country`` (next entry of the same list, alphabetical,
  cyclic), ``month`` (next month), ``scale`` (thousand -> million -> billion
  -> trillion, trillion -> billion), ``direction`` (antonym: increased <->
  decreased, rose <-> fell, higher <-> lower, more than <-> less than, ...).
- ``mixrand``: same elements as ``mix``, random replacement: any other entry
  of the same list / month / scale word; numbers as in ``numrand``; direction
  has a single antonym, so it is the same as in ``mix``.
The new families write the kind of every change in ``changed_element``
(``Texas -> Utah [us_state]``) and use generator version 2.

Person names and sources are NOT a kind: changing who said or did something is
the attribution switch (M03), excluded from the benchmark by the ethics
protocol. Ambiguous words are left out of the lists on purpose ("Washington",
"Jordan", "Chad", "May"; "up"/"down"; "rise"/"fall").

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
from eiger.utils.query_leak import numeric_leak_tokens, tokens

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

NUMERIC_KINDS = frozenset({"year", "integer", "decimal"})
ENTITY_KINDS = ("us_state", "country", "month", "scale", "direction")

US_STATES = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)  # "Washington" left out: also a city and a surname.
COUNTRIES = (
    "Afghanistan",
    "Argentina",
    "Australia",
    "Austria",
    "Bangladesh",
    "Belgium",
    "Brazil",
    "Canada",
    "Chile",
    "China",
    "Colombia",
    "Cuba",
    "Denmark",
    "Egypt",
    "Ethiopia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Haiti",
    "Honduras",
    "Hungary",
    "India",
    "Indonesia",
    "Iran",
    "Iraq",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Kenya",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Nigeria",
    "North Korea",
    "Norway",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Russia",
    "Saudi Arabia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Syria",
    "Taiwan",
    "Turkey",
    "Ukraine",
    "United Kingdom",
    "Venezuela",
    "Vietnam",
)  # "Jordan", "Chad", "Georgia" (the state is listed) left out: names or ambiguous.
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTHS_DETECTED = tuple(m for m in MONTHS if m != "May")  # "May" is also a verb.
SCALES = ("thousand", "million", "billion", "trillion")
_DIRECTION_PAIRS = (
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("increasing", "decreasing"),
    ("rose", "fell"),
    ("risen", "fallen"),
    ("higher", "lower"),
    ("more than", "less than"),
)
ANTONYMS = {a: b for a, b in _DIRECTION_PAIRS} | {b: a for a, b in _DIRECTION_PAIRS}


@dataclass(frozen=True)
class Family:
    name: str
    mode: str  # "fixed" | "random"
    kinds: frozenset[str]
    filename: str
    category: str
    version: str


FAMILIES = {
    "num": Family("num", "fixed", NUMERIC_KINDS, "variants.csv", "MULTI_NUM", "1"),
    "numrand": Family(
        "numrand", "random", NUMERIC_KINDS, "variants_numrand.csv", "MULTI_NUM_RAND", "2"
    ),
    "mix": Family(
        "mix", "fixed", NUMERIC_KINDS | set(ENTITY_KINDS), "variants_mix.csv", "MULTI_MIX", "2"
    ),
    "mixrand": Family(
        "mixrand",
        "random",
        NUMERIC_KINDS | set(ENTITY_KINDS),
        "variants_mixrand.csv",
        "MULTI_MIX_RAND",
        "2",
    ),
}


def family_file(family: str) -> Path:
    """Default variants file of a family (data/multi_element/<file>)."""
    if family not in FAMILIES:
        raise IngestionError(
            f"Unknown multi-element family '{family}' (expected {sorted(FAMILIES)})."
        )
    return DEFAULT_MULTI_ELEMENT_FILE.parent / FAMILIES[family].filename


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


def _alternation(words: tuple[str, ...] | list[str]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_ENTITY_WORDS: dict[str, tuple[str, ...]] = {
    "us_state": US_STATES,
    "country": COUNTRIES,
    "month": MONTHS_DETECTED,
    "scale": SCALES + tuple(f"{w}s" for w in SCALES),
    "direction": tuple(ANTONYMS),
}
_ENTITY_KIND = {w: kind for kind, words in _ENTITY_WORDS.items() for w in words}
# Case-sensitive for names (a capital letter is part of the cue); scale and
# direction words also match with a capital first letter (sentence start).
ENTITY_RE = re.compile(
    r"\b("
    + _alternation(
        list(US_STATES + COUNTRIES + MONTHS_DETECTED)
        + [
            v
            for w in _ENTITY_WORDS["scale"] + _ENTITY_WORDS["direction"]
            for v in (w, w.capitalize())
        ]
    )
    + r")\b"
)


def _entity_kind(value: str) -> str:
    return _ENTITY_KIND.get(value) or _ENTITY_KIND[value[0].lower() + value[1:]]


def extract_elements(text: str, kinds: frozenset[str] | set[str] = NUMERIC_KINDS) -> list[Element]:
    """Distinct elements of the requested kinds, in order of first appearance."""
    found: list[tuple[int, Element]] = []
    seen: set[str] = set()
    for match in NUMBER_RE.finditer(text):
        value = match.group(1)
        kind = element_kind(value)
        if value not in seen and kind in kinds:
            seen.add(value)
            found.append((match.start(), Element(value, kind)))
    if kinds - NUMERIC_KINDS:
        for match in ENTITY_RE.finditer(text):
            value = match.group(1)
            kind = _entity_kind(value)
            if value not in seen and kind in kinds:
                seen.add(value)
                found.append((match.start(), Element(value, kind)))
    return [element for _, element in sorted(found, key=lambda item: item[0])]


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


def perturb_random(element: Element, rng: random.Random) -> str:
    """Random magnitude for a numeric element (family numrand/mixrand)."""
    sign = rng.choice((-1, 1))
    if element.kind == "year":
        return str(int(element.value) + sign * rng.randint(1, 10))
    if element.kind == "decimal":
        decimals = len(element.value.split(".")[1])
        value = float(element.value)
        unit = 10.0**-decimals
        delta = max(unit, round(abs(value) * rng.uniform(0.05, 0.5), decimals))
        new = round(value + sign * delta, decimals)
        if new <= 0:
            new = round(value + delta, decimals)
        return f"{new:.{decimals}f}"
    value = int(element.value.replace(",", ""))
    if value < 10:
        delta = rng.randint(1, 5)
    else:
        delta = max(1, round(value * rng.uniform(0.05, 0.5)))
    new_int = value + sign * delta
    if new_int <= 0:
        new_int = value + delta
    return _format_int(new_int, element.value)


def _match_case(new: str, like: str) -> str:
    return new[0].upper() + new[1:] if like[0].isupper() and not new[0].isupper() else new


def _plural_scale(value: str) -> tuple[str, bool]:
    base = value.lower()
    return (base[:-1], True) if base.endswith("s") and base[:-1] in SCALES else (base, False)


def perturb_entity(element: Element, rng: random.Random, mode: str, step: int = 1) -> str:
    """
    Replacement for a non-numeric element. ``fixed``: next entry of the list
    (``step`` entries ahead, used to skip collisions); ``random``: uniform
    choice among the other entries. Direction words have one antonym.
    """
    if element.kind == "direction":
        key = element.value[0].lower() + element.value[1:]
        return _match_case(ANTONYMS[key], element.value)
    if element.kind == "scale":
        base, plural = _plural_scale(element.value)
        if mode == "fixed":
            idx = SCALES.index(base)
            new = SCALES[idx + 1] if idx + 1 < len(SCALES) else SCALES[idx - 1]
            if step > 1:
                others = [w for w in SCALES if w != base]
                new = others[(others.index(new) + step - 1) % len(others)]
        else:
            new = rng.choice([w for w in SCALES if w != base])
        return _match_case(new + ("s" if plural else ""), element.value)
    words = {"us_state": US_STATES, "country": COUNTRIES, "month": MONTHS}[element.kind]
    if mode == "fixed":
        ordered = sorted(words) if element.kind != "month" else list(words)
        return ordered[(ordered.index(element.value) + step) % len(ordered)]
    return rng.choice([w for w in words if w != element.value])


def _rng_for(claim_id: str, generator_seed: int) -> random.Random:
    digest = hashlib.sha256(f"{generator_seed}:{claim_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def apply_changes(text: str, changes: dict[str, str]) -> str:
    """Replace every occurrence of each changed element (same token boundaries as extraction)."""
    text = NUMBER_RE.sub(lambda m: changes.get(m.group(1), m.group(1)), text)
    if any(not value[0].isdigit() for value in changes):
        text = ENTITY_RE.sub(lambda m: changes.get(m.group(1), m.group(1)), text)
    return text


def make_variants(
    claim_id: str, text: str, max_k: int = 3, generator_seed: int = 0, family: str = "num"
) -> list[dict[str, Any]]:
    """
    Nested variants for one claim: one row per k = 1 .. min(max_k, n_elements).
    Returns [] for a claim without elements of the family's kinds.
    """
    spec = FAMILIES[family]
    elements = extract_elements(text, spec.kinds)
    present = {e.value.lower() for e in elements}
    # A direction word whose antonym is already in the claim cannot be changed
    # without merging the two: it is not an element (only one antonym exists).
    elements = [
        e for e in elements if e.kind != "direction" or ANTONYMS[e.value.lower()] not in present
    ]
    rng = _rng_for(claim_id, generator_seed)
    order = list(range(len(elements)))
    rng.shuffle(order)
    # The fixed numeric rule keeps drawing from the same generator as the order
    # (family "num", version 1, must stay byte-identical). Random families draw
    # magnitudes from a second generator, so the chosen elements and their
    # order are the same as in the fixed family of the same kinds.
    value_rng = rng if spec.mode == "fixed" else _rng_for(f"random:{claim_id}", generator_seed)
    taken = {e.value for e in elements}
    new_values: dict[str, str] = {}
    for idx in order[:max_k]:
        element = elements[idx]
        if element.kind in NUMERIC_KINDS:
            rule = perturb if spec.mode == "fixed" else perturb_random
            candidate = rule(element, value_rng)
            # Never collide with another element of the same claim (that would
            # merge two elements into one) or reproduce the original.
            while candidate in taken or candidate == element.value:
                candidate = rule(Element(candidate, element.kind), value_rng)
        else:
            step = 1
            candidate = perturb_entity(element, value_rng, spec.mode, step)
            while candidate in taken or candidate == element.value:
                step += 1
                if step > 200:
                    raise IngestionError(
                        f"Claim '{claim_id}': no free replacement for '{element.value}'."
                    )
                candidate = perturb_entity(element, value_rng, spec.mode, step)
        new_values[element.value] = candidate
        taken.add(candidate)
    tagged = family != "num"
    kind_of = {e.value: e.kind for e in elements}
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
                "changed_element": "; ".join(
                    f"{a} -> {b}" + (f" [{kind_of[a]}]" if tagged else "")
                    for a, b in changes.items()
                ),
                "manipulation_category": spec.category,
                "generator_version": spec.version,
                "generator_seed": generator_seed,
            }
        )
    return rows


CHANGE_RE = re.compile(r"(?P<old>.+?) -> (?P<new>.+?)(?: \[(?P<kind>\w+)\])?$")


def parse_changes(changed_element: str) -> list[tuple[str, str, str | None]]:
    """``"a -> b [kind]; c -> d"`` -> [(a, b, kind), (c, d, None)]."""
    out = []
    for part in filter(None, (p.strip() for p in changed_element.split(";"))):
        match = CHANGE_RE.match(part)
        if match:
            out.append((match["old"], match["new"], match["kind"]))
    return out


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


def _query_leaks(family: str, query: str, original: str, row: dict[str, str]) -> bool:
    """
    The query repeats a changed element (old or new value). Family ``num``
    keeps the original numeric check; the other families check the tokens of
    every changed element, numbers and words alike (a query naming the true
    state favours the true document exactly as a repeated year does).
    """
    if family == "num":
        return bool(numeric_leak_tokens(query, original, row["claim_text"]))
    words = tokens(query)
    return any(
        words & (tokens(old) | tokens(new)) for old, new, _ in parse_changes(row["changed_element"])
    )


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
    family: str = "num"

    def __init__(
        self,
        corpus_dir: str | Path | None = None,
        query_file: str | Path | None = None,
        variants_file: str | Path | None = None,
    ) -> None:
        super().__init__(corpus_dir, query_file)
        self.variants_file = (
            Path(variants_file) if variants_file is not None else family_file(self.family)
        )

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        index = load_multi_element_index(self.variants_file)
        kept: list[Claim] = []
        for claim in super().load(split, max_claims=None):
            row = index.get((claim.claim_id, self.min_elements))
            if row is None or row["claim_original_reference"] != claim.original_fact:
                continue
            if _query_leaks(self.family, claim.context_query, claim.original_fact, row):
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


class CorpusClaimOfficialMulti2NumRandDataset(CorpusClaimOfficialMultiElementDataset):
    name: str = "corpus_claim_official_multi2_numrand_genq"
    description: str = (
        "ENGINEERING ONLY: same claims as multi2, numbers changed with a random magnitude."
    )
    min_elements: int = 2
    family: str = "numrand"


class CorpusClaimOfficialMulti2MixDataset(CorpusClaimOfficialMultiElementDataset):
    name: str = "corpus_claim_official_multi2_mix_genq"
    description: str = (
        "ENGINEERING ONLY: claims with >= 2 elements among numbers, places, months, "
        "scale and direction words; fixed rule."
    )
    min_elements: int = 2
    family: str = "mix"


class CorpusClaimOfficialMulti2MixRandDataset(CorpusClaimOfficialMultiElementDataset):
    name: str = "corpus_claim_official_multi2_mixrand_genq"
    description: str = (
        "ENGINEERING ONLY: same claims and elements as multi2_mix, random replacement."
    )
    min_elements: int = 2
    family: str = "mixrand"
