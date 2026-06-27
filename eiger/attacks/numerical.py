"""
Numerical Shift Attack.

Perturbs numeric values in a document by swapping adjacent digits
(e.g. 2.1% → 21.% or 125 → 152). The transformation is plausible —
the resulting number is typographically similar to the original —
making it hard to detect without cross-referencing the source.
"""

from __future__ import annotations

import re
from typing import Any

from eiger.core.interfaces import BaseAttack
from eiger.core.models import Document, PoisonedDocument, PoisonAnnotation
from eiger.utils.seeding import make_rng, derive_seed


class NumericalShiftAttack(BaseAttack):
    """
    Adversarial perturbation: swaps adjacent digits within numeric tokens.

    Type 1 in the EIBench taxonomy.
    """

    name: str = "numerical_shift"
    description: str = (
        "Swaps adjacent digits within numeric tokens "
        "(e.g. 2.1% → 21.%, 125 → 152). "
        "Produces plausible but factually incorrect statistics."
    )

    # Regex matches integer or decimal numbers, optionally followed by % or unit
    _NUMBER_RE: re.Pattern[str] = re.compile(r"\d+(?:\.\d+)?")

    def apply(self, document: Document, seed: int, **kwargs: Any) -> PoisonedDocument:
        """
        Apply numerical shift to all numeric tokens in the document text.

        Args:
            document: Source ground-truth document.
            seed:     Deterministic seed (isolated RNG, no global state mutation).
            **kwargs: Unused.

        Returns:
            PoisonedDocument with swapped digits and full provenance.
        """
        rng = make_rng(derive_seed(seed, document.doc_id, self.name))
        poisoned_text = self._NUMBER_RE.sub(
            lambda m: self._swap_digits(m.group(), rng),
            document.text,
        )

        annotation = PoisonAnnotation(
            plausibility=4.0,           # High: number looks like a typo
            verification_difficulty=3.5, # Moderate: needs source cross-check
            editorial_risk=4.5,          # High: numbers pass spell-check
        )

        return PoisonedDocument(
            doc_id=document.doc_id,
            claim_id=document.claim_id,
            text=poisoned_text,
            attack_name=self.name,
            attack_params=self.describe(),
            original_text=document.text,
            annotation=annotation,
        )

    def describe(self) -> dict[str, Any]:
        return {"attack": self.name, "method": "adjacent_digit_swap"}

    @staticmethod
    def _swap_digits(num_str: str, rng: "random.Random") -> str:
        """Swap two adjacent characters within a digit string."""
        chars = list(num_str)
        swappable = [i for i, c in enumerate(chars) if c.isdigit() and i + 1 < len(chars) and chars[i + 1].isdigit()]
        if not swappable:
            return num_str
        idx = rng.choice(swappable)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
