"""
Numerical Shift Attack.

Perturbs numeric values in a document by swapping adjacent digits
(e.g. 2.1% → 21.% or 125 → 152). The transformation is plausible —
the resulting number is typographically similar to the original —
making it hard to detect without cross-referencing the source.

Why adjacent-digit swap rather than random replacement?
-------------------------------------------------------
A fully random replacement (e.g. changing 2.1% to 7.8%) is immediately
suspicious to a human reader or a simple sanity-check model. Swapping two
neighbouring digits produces a value that looks like an ordinary transcription
error, raising the plausibility score and making the poisoned document harder
to filter from the corpus automatically.

EIBench taxonomy: Type 1 — Numerical Perturbation.
"""

from __future__ import annotations

import re
from typing import Any

from eiger.core.interfaces import BaseAttack
from eiger.core.models import Document, PoisonedDocument, PoisonAnnotation
from eiger.utils.seeding import make_rng, derive_seed


# ─── Attack class ─────────────────────────────────────────────────────────────

class NumericalShiftAttack(BaseAttack):
    """
    Adversarial perturbation that swaps adjacent digits within numeric tokens.

    Every number found in the document text (integers, decimals) is a
    candidate for digit transposition. The specific pair of digits to swap is
    chosen randomly but deterministically via the seeded RNG, so the same
    (document, seed) pair always produces the same poisoned output.

    Responsibilities
    ----------------
    - Locate all numeric substrings via a compiled regex.
    - For each match, randomly select a pair of adjacent digit characters and
      exchange them.
    - Wrap the result in a PoisonedDocument with provenance metadata.

    What this class does NOT do
    ---------------------------
    - It does not reason about units or physical plausibility of the result
      (e.g. it will swap digits in a year like 2024 → 2042 without checking
      whether that year makes contextual sense).
    - It does not modify non-numeric text.
    - It does not attempt to preserve the magnitude order of the number.

    EIBench taxonomy: Type 1.
    """

    # String identifier used as the registry key and stored in PoisonedDocument.
    name: str = "numerical_shift"

    description: str = (
        "Swaps adjacent digits within numeric tokens "
        "(e.g. 2.1% → 21.%, 125 → 152). "
        "Produces plausible but factually incorrect statistics."
    )

    # Compiled once at class definition time for efficiency.
    # Pattern breakdown:
    #   \d+        — one or more digits (integer part)
    #   (?:\.\d+)? — optional decimal part (non-capturing group)
    # The pattern intentionally does NOT capture surrounding whitespace or
    # units (%, kg, etc.) so that only the raw numeric characters are passed
    # to _swap_digits(), leaving the rest of the token intact.
    _NUMBER_RE: re.Pattern[str] = re.compile(r"\d+(?:\.\d+)?")

    # ─── Public interface ─────────────────────────────────────────────────────

    def apply(self, document: Document, seed: int, **kwargs: Any) -> PoisonedDocument:
        """
        Apply numerical shift to all numeric tokens in the document text.

        Each numeric match is processed independently: its digits are shuffled
        in one random adjacent-swap using a seeded RNG derived from the
        document ID and attack name, ensuring full reproducibility.

        Args:
            document: Source ground-truth document containing the original text.
            seed:     Integer seed for the experiment-level RNG. A document-
                      and attack-specific sub-seed is derived from this value
                      so that different attacks on the same document do not
                      share RNG state.
            **kwargs: Accepted but unused — allows uniform call signatures
                      across all attack types in experiment runners.

        Returns:
            PoisonedDocument containing the digit-swapped text, a reference
            to the original text, and a PoisonAnnotation with pre-assessed
            risk scores.
        """
        # Derive a seed that is unique to this (experiment, document, attack)
        # triple. This prevents two attacks from producing identical random
        # sequences when given the same top-level experiment seed.
        rng = make_rng(derive_seed(seed, document.doc_id, self.name))

        # re.sub with a callable replacement: the lambda is invoked once per
        # match, passing the matched numeric string and the shared RNG so that
        # different numbers within the same document get different swaps.
        poisoned_text = self._NUMBER_RE.sub(
            lambda m: self._swap_digits(m.group(), rng),
            document.text,
        )

        # Pre-assessed risk scores for this attack type.
        # plausibility=4.0      : A transposed digit looks like a typo; a human
        #                         reading quickly is unlikely to notice.
        # verification_difficulty=3.5 : Catching the error requires comparing
        #                         the number against the primary source; spell-
        #                         checkers and grammar tools will not flag it.
        # editorial_risk=4.5    : Numbers pass through automated content filters
        #                         because they are syntactically valid.
        annotation = PoisonAnnotation(
            plausibility=4.0,
            verification_difficulty=3.5,
            editorial_risk=4.5,
        )

        return PoisonedDocument(
            doc_id=document.doc_id,
            claim_id=document.claim_id,
            text=poisoned_text,
            attack_name=self.name,
            attack_params=self.describe(),
            original_text=document.text,  # Preserved for diff-based evaluation
            annotation=annotation,
        )

    def describe(self) -> dict[str, Any]:
        """
        Return a serialisable description of this attack's configuration.

        Used to populate the ``attack_params`` field of PoisonedDocument so
        that experiment logs are self-documenting.

        Returns:
            Dict with at minimum 'attack' (name string) and 'method' keys.
        """
        return {"attack": self.name, "method": "adjacent_digit_swap"}

    # ─── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _swap_digits(num_str: str, rng: "random.Random") -> str:
        """
        Swap a randomly selected pair of adjacent digit characters in a string.

        Only positions where both the current character and its right neighbour
        are digits are eligible for swapping. This excludes the decimal point
        in a float (e.g. "3.14" — positions 1 and 2 are '.' and '1', not both
        digits, so only "3"↔"." is skipped; "1"↔"4" is a valid candidate).

        If the string has no eligible pair (e.g. a single-digit number like
        "7"), the original string is returned unchanged to avoid no-op swaps
        that would still be recorded as modifications.

        Args:
            num_str: The matched numeric substring (e.g. "2.1", "125").
            rng:     Seeded random.Random instance — must NOT be the global
                     random module to preserve determinism across parallel runs.

        Returns:
            The digit string after one adjacent-pair swap, or the original
            string if no eligible swap position exists.
        """
        chars = list(num_str)

        # Collect indices where both chars[i] and chars[i+1] are digits.
        # Skipping the decimal point ensures we don't accidentally swap a
        # digit with '.', which would produce malformed output like "3.14" → "31.4"
        # — that swap *is* intentional and handled correctly because '.' is not
        # .isdigit(), so only genuine digit↔digit pairs are selected.
        swappable = [
            i
            for i, c in enumerate(chars)
            if c.isdigit() and i + 1 < len(chars) and chars[i + 1].isdigit()
        ]

        if not swappable:
            # Nothing to swap; return the original to avoid producing a
            # PoisonedDocument that is identical to the source on this token.
            return num_str

        # Pick one eligible position uniformly at random and swap in place.
        idx = rng.choice(swappable)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
