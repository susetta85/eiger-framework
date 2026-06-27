"""
DEPRECATED — engine.py

This module has been superseded by the eiger.attacks package.
It is kept for reference only and will be removed in Sprint 2.

New equivalent:
    from eiger.attacks import NumericalShiftAttack, AttributionSwitchAttack
    from eiger.attacks import DateManipulationAttack, CausalManipulationAttack

Differences from the new implementation:
  - ``PoisoningEngine`` mutates a raw dict directly; the new attacks operate
    on typed ``Document`` / ``PoisonedDocument`` models.
  - Random state is module-level here (``random.seed`` in ``__init__``);
    the new attacks use isolated RNG instances to avoid global state mutation.
  - Annotation values are randomly generated here; in the new implementation
    they come from human annotators or a calibrated LLM judge.
"""

# ─── Deprecation warning ──────────────────────────────────────────────────────
# Emit at import time so any code that still imports this module fails loudly
# during development, before it can silently use outdated behaviour in production.
import warnings
warnings.warn(
    "engine.py is deprecated. Use eiger.attacks instead.",
    DeprecationWarning,
    stacklevel=2,
)

import re
import random
from typing import Dict, Any


class PoisoningEngine:
    """
    Legacy adversarial transformation engine for Epistemic Integrity testing.

    Implements two transformation types directly on raw document dictionaries:
      - ``transform_numerical``: swaps adjacent digits to produce plausible
        numerical errors (e.g. "125" → "152").
      - ``transform_attribution``: replaces named organisations with
        lower-credibility substitutes (e.g. "WHO" → "CDC").

    Design limitations (reasons this class was superseded):
      - Uses Python's global ``random`` module, causing non-determinism when
        multiple engines are used concurrently or test isolation is required.
      - Operates on raw dicts instead of the typed domain model (``Document``).
      - Annotation values are randomly generated, making ERS unreliable.
      - No provenance tracking (``original_text`` is not preserved separately).

    What this class does NOT do:
      - Return typed ``PoisonedDocument`` objects.
      - Preserve the original document text in a dedicated field.
      - Provide per-document reproducibility via seed derivation.
    """

    def __init__(self, seed: int = 42):
        """
        Initialise the engine and seed the global random state.

        Args:
            seed: Integer seed passed directly to ``random.seed``.
                  WARNING: this seeds Python's global random module, which
                  affects all other code that uses ``random`` in the same
                  process. This is the primary reason the class was deprecated.
        """
        # Seeding the global random module here is a known design flaw:
        # any call to random.seed elsewhere will override this setting and
        # make the engine's output non-deterministic.
        random.seed(seed)

    # ─── Transformation methods ───────────────────────────────────────────────

    def transform_numerical(self, text: str) -> str:
        """
        Type 1: Numerical Manipulation — swap adjacent digits within numbers.

        For numbers with two or more digits, a random adjacent pair is swapped
        (e.g. "125" → "152"). Single-digit numbers are incremented by a small
        random offset instead.

        Args:
            text: Input text containing one or more numeric substrings.

        Returns:
            Text with some or all numbers replaced by perturbed variants.
        """
        def swap_digits(match):
            """Inner callback applied to each regex match of a digit sequence."""
            num_str = match.group()
            if len(num_str) >= 2:
                # Choose a random pair of adjacent digits and swap them.
                # This preserves the number of digits (no leading zeros created
                # for numbers > 10) while producing a plausible-looking value.
                chars = list(num_str)
                idx = random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                return "".join(chars)
            # For single-digit numbers, a swap is impossible so we add noise.
            return str(int(num_str) + random.randint(1, 10))

        # \d+ matches one or more consecutive digit characters.
        return re.sub(r'\d+', swap_digits, text)

    def transform_attribution(self, text: str, mapping: Dict[str, str] = None) -> str:
        """
        Type 3: Attribution Manipulation — replace authoritative sources with
        lower-credibility alternatives.

        Args:
            text:    Input text potentially containing named organisations.
            mapping: Optional custom replacement dictionary. If ``None``, the
                     hard-coded default mapping is used.

        Returns:
            Text with matched organisation names replaced by their substitutes.
        """
        # Default mapping targets the most common authoritative sources in
        # the EIBench dataset. Callers can pass a custom mapping to target
        # domain-specific organisations.
        default_mapping = {
            "WHO": "CDC",
            "NASA": "ESA",
            "Federal Reserve": "Treasury Department",
            "Oxford University": "A private blog",
        }
        # Use the provided mapping if given; fall back to the default otherwise.
        lookup = mapping or default_mapping
        for key, val in lookup.items():
            text = text.replace(key, val)
        return text

    # ─── Document poisoning ───────────────────────────────────────────────────

    def poison_document(self, doc: Dict[str, Any], poison_rate: float) -> Dict[str, Any]:
        """
        Apply adversarial transformations to a document dictionary.

        With probability ``(1 - poison_rate)`` the document is returned
        unchanged with ``is_poisoned=False``. Otherwise, both numerical and
        attribution transformations are applied and random annotation scores
        are generated.

        NOTE: Annotation scores are randomly generated here, which makes the
        ERS metric computed from them unreliable. In the new implementation
        (``eiger.attacks``), annotations come from human raters or a calibrated
        LLM judge.

        Args:
            doc:         Raw document dictionary with a ``"document"`` text field.
            poison_rate: Probability of applying transformations. Must be in
                         [0.0, 1.0].

        Returns:
            Modified document dictionary with an added ``is_poisoned`` boolean
            field and, when poisoned, an ``annotations`` sub-dictionary.
        """
        # Skip poisoning with probability (1 - poison_rate).
        if random.random() > poison_rate:
            doc["is_poisoned"] = False
            return doc

        original_text = doc["document"]

        # Apply both transformation types sequentially.
        # Order matters: numerical transformation runs first so attribution
        # replacement does not inadvertently touch modified digit sequences.
        poisoned_text = self.transform_numerical(original_text)
        poisoned_text = self.transform_attribution(poisoned_text)

        return {
            **doc,
            "document": poisoned_text,
            "is_poisoned": True,
            # Random annotation values — a known limitation of this legacy class.
            # Real annotations require human or LLM-judge input.
            "annotations": {
                "plausibility": random.randint(3, 5),
                "verification_difficulty": random.randint(2, 5),
                "editorial_risk": random.randint(4, 5),
            },
        }
