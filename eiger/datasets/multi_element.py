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
    # Graded families (E10): ONE element, four increasing intensities; the
    # "k" column holds the level (1..4), see make_graded_variants().
    "grad": Family("grad", "graded", NUMERIC_KINDS, "variants_grad.csv", "GRADED_NUM", "3"),
    "grad_noel": Family(
        "grad_noel", "graded", NUMERIC_KINDS, "variants_grad_noel.csv", "GRADED_NUM_NOEL", "3"
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
    Returns [] for a claim without elements of the family's kinds. Graded
    families ignore ``max_k`` and return the four levels of one element.
    """
    spec = FAMILIES[family]
    if spec.mode == "graded":
        return make_graded_variants(
            claim_id, text, generator_seed, include_elections=family == "grad"
        )
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


# ─── Graded families (E10): one element, four increasing intensities ─────────
#
# Question: how large must a change be before the answer follows the variant?
# Each claim gets four variants (levels L1..L4) that change the SAME element in
# the SAME direction with increasing intensity, following the graded-threshold
# prompt agreed with the collaborator (docs/EXPERIMENTS.md, section 4.16), but
# with fixed rules instead of a language model:
#
# - quantity (integer, decimal, percentage change): always upwards, +15%, +40%,
#   +100%, +250% (targets 10-20%, 25-50%, 75-150%, >= 200%); at least one unit
#   more than the previous level, so small counts still grow at every level.
# - bounded percentage (a share 0-100%): +5, +10, +20, +40 percentage points,
#   downwards when there are fewer than 40 points of room upwards (one side
#   always has at least 50). Never below 0 or above 100.
# - year (no election in sight): +-1, 3, 10, 25 years, never after
#   LATEST_YEAR nor before 1800; direction drawn once per claim.
# - election year (family "grad" only): presidential general elections move by
#   1-4 presidential cycles (4 years each, never after LATEST_PRESIDENTIAL),
#   House/midterm elections by 1-4 two-year cycles. Senate, primary, runoff,
#   special, state, local, gubernatorial and unspecified elections are not
#   changed (their valid years cannot be established from the claim alone),
#   and neither are elections of other countries (a country name or a word in
#   _FOREIGN_WORDS anywhere in the claim); the generator takes the claim's
#   next element instead. A year counts as an election year when the claim
#   has an election word anywhere (claims are one sentence: the whole claim
#   is the context, which is the conservative choice). Family "grad_noel"
#   never changes a year in a claim with an election word.
# - a year keeps its order with the claim's other years ("announced in 2021
#   ... the 2022 election": 2021 cannot move past 2022).
# - endpoints of a range ("65 to 70 percent", "between 3 and 5") are never
#   changed: changing one of them could reverse the range.
# - numbers that work as identifiers (Route 66, Section 8, Title 42,
#   Proposition 8, Model 3, ...: the word before is in IDENTIFIER_WORDS) are
#   never changed; glued identifiers (COVID-19, F-35, 9/11) are already
#   excluded by NUMBER_RE.
#
# The element is the first of the claim's numeric elements, in the same
# shuffled order as family "num" (so, when allowed, it is the element that
# num k=1 changed by +-1 / +-10%), that admits all four levels. Known limits of
# the rules: political roles around a shifted year are not checked ("former
# President Trump ... the 2004 election"), and a share introduced by a growth
# word ("debt up to 80 percent of GDP") is treated as a percentage change.

GRADED_LEVELS = 4
GRADED_RELATIVE = (0.15, 0.40, 1.00, 2.50)
GRADED_POINTS = (5, 10, 20, 40)
GRADED_YEARS = (1, 3, 10, 25)
LATEST_YEAR = 2026
LATEST_PRESIDENTIAL = 2024
EARLIEST_YEAR = 1800

IDENTIFIER_WORDS = frozenset(
    {
        "act", "amendment", "apollo", "apartment", "area", "article", "bill", "boeing",
        "box", "building", "category", "channel", "chapter", "class", "code", "district",
        "division", "episode", "exhibit", "fleet", "flight", "floor", "form", "fox",
        "gate", "grade", "group", "highway", "hr", "interstate", "issue", "item", "level",
        "lot", "model", "no", "number", "page", "part", "phase", "pier", "precinct",
        "prop", "proposition", "psalm", "region", "room", "route", "rule", "sb", "hb",
        "season", "section", "sector", "stage", "station", "suite", "terminal", "tier",
        "title", "type", "unit", "verse", "version", "volume", "ward", "zone",
    }
)
_ELECTION_WORDS = (
    "election", "elections", "elected", "reelected", "re-elected", "reelection",
    "re-election", "midterm", "midterms", "primary", "primaries", "caucus", "caucuses",
    "runoff", "ballot", "ballots", "vote", "voted", "voters", "voting", "campaign",
    "nominee", "nomination", "inauguration", "inaugurated", "race",
)
_NOT_CHANGEABLE_ELECTIONS = (
    "senate", "senator", "primary", "primaries", "caucus", "caucuses", "runoff",
    "special", "governor", "gubernatorial", "mayor", "mayoral", "state", "local",
    "recall", "county", "city", "municipal", "legislative", "legislature",
)
_FOREIGN_WORDS = frozenset(
    {w.lower() for w in COUNTRIES}
    | {
        "russian", "british", "canadian", "mexican", "french", "german", "israeli",
        "ukrainian", "indian", "brazilian", "venezuelan", "italian", "chinese",
        "japanese", "european", "uk", "parliament", "parliamentary", "prime", "putin",
    }
)
_GROWTH_WORDS = (
    "increase", "increased", "increases", "rise", "rose", "risen", "grew", "grow",
    "growth", "jump", "jumped", "up", "decline", "declined", "drop", "dropped",
    "fell", "fall", "down", "cut", "raise", "raised", "higher", "lower", "more", "less",
)
_WORD_RE = re.compile(r"[a-z][a-z.\-]*")
_RANGE_AFTER = re.compile(r"\s*(?:%|percent)?\s*(?:to|-|–|—|and)\s*\$?\d", re.IGNORECASE)
_RANGE_BEFORE = re.compile(
    r"(?:\d\s*(?:%|percent)?\s*(?:to|-|–|—|and)|\bbetween|\bfrom)\s*\$?$", re.IGNORECASE
)


def _words(text: str) -> list[str]:
    # Trailing dots are sentence ends ("election."), not part of the word.
    return [w.rstrip(".") for w in _WORD_RE.findall(text.lower())]


def _first_span(text: str, value: str) -> tuple[int, int] | None:
    for match in NUMBER_RE.finditer(text):
        if match.group(1) == value:
            return match.start(1), match.end(1)
    return None  # pragma: no cover - value always comes from extract_elements(text)


def graded_kind(text: str, element: Element, include_elections: bool) -> str | None:
    """
    Kind used by the graded rules for this element, or None if the element
    must not be changed (identifier, unchangeable election year).
    """
    start, end = _first_span(text, element.value) or (0, 0)
    before = _words(text[max(0, start - 40) : start])
    after = text[end : end + 12].lower()
    if before and before[-1].replace(".", "") in IDENTIFIER_WORDS:
        return None
    if _RANGE_AFTER.match(text, end) or _RANGE_BEFORE.search(text[max(0, start - 30) : start]):
        return None  # endpoint of a range ("65 to 70 percent"): changing it could reverse it
    if element.kind == "year":
        window = set(_words(text))
        if not window & set(_ELECTION_WORDS):
            return "year"
        foreign = window & _FOREIGN_WORDS or any(
            " " in name and name in text.lower() for name in _FOREIGN_WORDS
        )
        if not include_elections or foreign or window & set(_NOT_CHANGEABLE_ELECTIONS):
            return None
        if window & {"presidential", "president", "white"}:
            return "election_year_presidential" if int(element.value) % 4 == 0 else None
        if window & {"house", "midterm", "midterms", "congressional", "congress"}:
            return "election_year_house" if int(element.value) % 2 == 0 else None
        return None
    is_percent = after.startswith("%") or after.lstrip().startswith(("percent", "per cent"))
    if is_percent:
        value = float(element.value.replace(",", ""))
        if value <= 100 and not set(before[-5:]) & set(_GROWTH_WORDS):
            return "percent_points"
        return "percent_change"
    return element.kind  # "integer" | "decimal"


def _graded_numbers(value_text: str, kind: str, levels: int = GRADED_LEVELS) -> list[str]:
    """Four strictly monotone new values for a quantity or a bounded percentage."""
    decimals = len(value_text.split(".")[1]) if "." in value_text else 0
    unit = 10.0**-decimals
    value = float(value_text.replace(",", ""))
    if kind == "percent_points":
        # A share is at most 100, so one side always has >= 50 points of room.
        sign = 1 if 100 - value >= GRADED_POINTS[-1] else -1
        targets = [value + sign * step for step in GRADED_POINTS[:levels]]
    else:
        sign = 1
        targets = [value * (1 + p) for p in GRADED_RELATIVE[:levels]]
    out: list[float] = []
    previous = value
    for target in targets:
        new = round(target, decimals)
        if sign * (new - previous) < unit - 1e-9:
            new = round(previous + sign * unit, decimals)
        out.append(new)
        previous = new
    if decimals:
        return [f"{v:.{decimals}f}" for v in out]
    return [_format_int(int(v), value_text) for v in out]


def _graded_years(
    value: int, kind: str, sign: int, other_years: frozenset[int] = frozenset()
) -> list[str] | None:
    """
    Four years in one direction (sign), or the other one if the first does not
    fit. Every new year stays on the same side of the claim's other years, so
    the order of the events in the claim does not change.
    """
    if kind == "election_year_presidential":
        steps, latest = [4, 8, 12, 16], LATEST_PRESIDENTIAL
    elif kind == "election_year_house":
        steps, latest = [2, 4, 6, 8], LATEST_YEAR
    else:
        steps, latest = list(GRADED_YEARS), LATEST_YEAR
    for direction in (sign, -sign):
        years = [value + direction * step for step in steps]
        same_side = all((y > o) == (value > o) and y != o for y in years for o in other_years)
        if same_side and all(EARLIEST_YEAR <= y <= latest for y in years):
            return [str(y) for y in years]
    return None


def make_graded_variants(
    claim_id: str, text: str, generator_seed: int = 0, include_elections: bool = True
) -> list[dict[str, Any]]:
    """
    Four variants (k = level 1..4) changing ONE element of the claim with
    increasing intensity (see the comment above). Returns [] when no numeric
    element of the claim admits four levels.
    """
    family = "grad" if include_elections else "grad_noel"
    spec = FAMILIES[family]
    elements = extract_elements(text, NUMERIC_KINDS)
    order = list(range(len(elements)))
    _rng_for(claim_id, generator_seed).shuffle(order)  # same order as family "num"
    direction_rng = _rng_for(f"graded:{claim_id}", generator_seed)
    others = {e.value for e in elements}
    for idx in order:
        element = elements[idx]
        kind = graded_kind(text, element, include_elections)
        if kind is None:
            continue
        if element.kind == "year" and kind not in {"percent_points", "percent_change"}:
            other_years = frozenset(
                int(e.value) for e in elements if e.kind == "year" and e.value != element.value
            )
            new_values = _graded_years(
                int(element.value), kind, direction_rng.choice((-1, 1)), other_years
            )
        else:
            new_values = _graded_numbers(element.value, kind)
        if not new_values or set(new_values) & (others - {element.value}):
            continue
        return [
            {
                "claim_id": claim_id,
                "k": level,
                "n_elements": len(elements),
                "claim_original_reference": text,
                "claim_text": apply_changes(text, {element.value: new}),
                "changed_element": f"{element.value} -> {new} [{kind}]",
                "manipulation_category": spec.category,
                "generator_version": spec.version,
                "generator_seed": generator_seed,
            }
            for level, new in enumerate(new_values, start=1)
        ]
    return []


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
