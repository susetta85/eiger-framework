"""
Missing Context Attack.

Removes an entire qualifying/contextual sentence from a document — one that
introduces a caveat, exception, or explanatory note (e.g. "However, this
figure excludes seasonal workers.") — while leaving every other sentence,
including the core statistic, untouched.

Example:
  Original:  "Inflation rose to 3.5% in 2023. However, this figure excludes
              housing costs, which rose independently."
  Poisoned:  "Inflation rose to 3.5% in 2023."

Why sentence-level, unlike CherryPickingAttack's clause-level deletion?
-------------------------------------------------------------------------
CherryPickingAttack (M05) removes a narrow comparative clause embedded
within a sentence that otherwise stays intact. MissingContextAttack (M06) is
a coarser-grained, and typically more consequential, omission: an entire
sentence carrying a caveat, exception, or qualifying detail disappears. The
distortion this introduces is directly measurable by diffing against
`original_text` (the missing sentence is either present or absent), which is
what `docs/CLAIM_AND_RESEARCH_QUESTIONS.md` §5 requires for this class of
attack ("admissible only if the resulting distortion is measurable").

EIBench taxonomy: Type 6 — Missing Context (M06).
"""

from __future__ import annotations

import re
from typing import Any

from eiger.attacks.causal import _split_sentences
from eiger.core.interfaces import BaseAttack
from eiger.core.models import Document, PoisonedDocument, PoisonAnnotation
from eiger.utils.seeding import make_rng, derive_seed


# ─── Module-level constants ───────────────────────────────────────────────────

# Matches common markers that introduce a caveat, exception, or explanatory
# aside within a sentence — the linguistic signature of "context that
# qualifies the preceding claim". Word-boundary anchored and case-insensitive
# so it matches at any position within a sentence (start, e.g. "However, ...",
# or mid-sentence, e.g. "...figure, however, excludes...").
_CONTEXT_MARKER_RE: re.Pattern[str] = re.compile(
    r"\b(?:"
    r"however|"
    r"note that|"
    r"it (?:is|should be) (?:worth noting|noted)|"
    r"for context|"
    r"meanwhile|"
    r"on average|"
    r"in general|"
    r"excludes|"
    r"does not (?:include|account for)"
    r")\b",
    re.IGNORECASE,
)


# ─── Attack class ─────────────────────────────────────────────────────────────

class MissingContextAttack(BaseAttack):
    """
    Adversarial perturbation that deletes an entire qualifying/contextual
    sentence, leaving the rest of the document — including the core
    statistic — byte-for-byte unchanged.

    Responsibilities
    ----------------
    - Split the document into sentence fragments via `_split_sentences`
      (reused from `eiger.attacks.causal`, which already handles the
      decimal-point-vs-sentence-boundary edge case).
    - Identify sentences containing a caveat/context marker
      (`_CONTEXT_MARKER_RE`).
    - Randomly select up to `omit_count` of them (seeded, deterministic) and
      remove them from the text.
    - Wrap the result in a PoisonedDocument with provenance metadata.

    What this class does NOT do
    ----------------------------
    - It does not touch any sentence that lacks a recognized context marker,
      including the sentence carrying the core statistic.
    - It does not attempt to summarize or paraphrase what was removed — the
      sentence disappears entirely, with nothing left in its place.
    - It does not guarantee the document text actually changes: a document
      with no sentence containing a context marker (e.g. a single short
      statement, mirroring CausalManipulationAttack's degenerate case)
      produces byte-identical output. This is recorded via
      ``attack_params["no_op"]`` rather than hidden — inventing a caveat
      sentence just so it can be deleted would fabricate content that was
      never there, which is a different (and unrequested) attack design.

    EIBench taxonomy: Type 6 (M06).
    """

    name: str = "missing_context"

    description: str = (
        "Deletes an entire qualifying/contextual sentence (a caveat, "
        "exception, or explanatory note), leaving every other sentence — "
        "including the core statistic — unchanged."
    )

    # ─── Public interface ─────────────────────────────────────────────────────

    def apply(
        self,
        document: Document,
        seed: int,
        omit_count: int = 1,
        context_marker_pattern: re.Pattern[str] | None = None,
        **kwargs: Any,
    ) -> PoisonedDocument:
        """
        Delete up to `omit_count` qualifying/contextual sentences from the text.

        Args:
            document:                Source ground-truth document.
            seed:                    Top-level experiment seed. A document-
                                     and attack-specific sub-seed is derived
                                     for full reproducibility.
            omit_count:              Maximum number of matching sentences to
                                     delete. The actual number may be lower if
                                     the document contains fewer eligible
                                     sentences than requested.
            context_marker_pattern:  Custom compiled regex to use instead of
                                     `_CONTEXT_MARKER_RE`. When provided,
                                     completely replaces the default pattern
                                     for this call.
            **kwargs:                Accepted but unused — maintains a
                                     uniform call signature with other attack
                                     classes.

        Returns:
            PoisonedDocument with the selected contextual sentence(s)
            removed, plus a PoisonAnnotation and omitted_count recorded in
            attack_params.
        """
        rng = make_rng(derive_seed(seed, document.doc_id, self.name))
        marker_re = context_marker_pattern or _CONTEXT_MARKER_RE

        sentences = _split_sentences(document.text)
        eligible = [s for s in sentences if marker_re.search(s)]

        if not eligible:
            # No fabricated fallback (see class docstring): if no sentence
            # carries a recognized caveat/context marker, there is nothing
            # honest to delete.
            poisoned_text = document.text
            omitted_count = 0
        else:
            targets = rng.sample(eligible, min(omit_count, len(eligible)))
            poisoned_text = document.text
            for target in targets:
                # `target` is a literal substring of document.text (extracted
                # by _split_sentences via regex over that same text), so a
                # direct replace always finds and removes it exactly as it
                # appears — whether or not it happens to include a leading
                # space (a middle/final sentence's fragment does, per how
                # _SENTENCE_END_RE.findall() resumes scanning right after the
                # previous match; the first sentence's fragment does not,
                # since nothing precedes it). Any resulting double space is
                # collapsed below.
                poisoned_text = poisoned_text.replace(target, "", 1)
            poisoned_text = re.sub(r"\s{2,}", " ", poisoned_text).strip()
            omitted_count = len(targets)

        no_op = poisoned_text == document.text

        # Pre-assessed risk scores for this attack type.
        # plausibility=4.5      : The surviving sentences read as complete,
        #                         well-formed prose; nothing marks the gap.
        # verification_difficulty=5.0 : Detecting a missing caveat requires
        #                         knowing the caveat existed in the first
        #                         place — the single hardest omission class
        #                         to catch without the original source.
        # editorial_risk=4.0    : Automated filters see only well-formed,
        #                         factually correct text with nothing to flag.
        annotation = PoisonAnnotation(
            plausibility=4.5,
            verification_difficulty=5.0,
            editorial_risk=4.0,
        )

        return PoisonedDocument(
            doc_id=document.doc_id,
            claim_id=document.claim_id,
            text=poisoned_text,
            attack_name=self.name,
            # "no_op" records whether any contextual sentence was actually
            # found and removed; "omitted_count" records how many were.
            attack_params=self.describe() | {"no_op": no_op, "omitted_count": omitted_count},
            original_text=document.text,
            annotation=annotation,
            # Propagate the source document's ethics/threat-model
            # classification (docs/ETHICS_AND_THREAT_MODEL.md §5/§7) — a
            # poisoned variant of a classified document is not automatically
            # less sensitive than its source.
            sensitivity_class=document.sensitivity_class,
            risk_level=document.risk_level,
            ground_truth_label=document.ground_truth_label,
        )

    def describe(self) -> dict[str, Any]:
        """
        Return a serialisable description of this attack's static configuration.

        Returns:
            Dict with 'attack' and 'method' keys. Callers of ``apply()``
            additionally merge in per-call 'no_op'/'omitted_count' keys.
        """
        return {"attack": self.name, "method": "contextual_sentence_deletion"}
