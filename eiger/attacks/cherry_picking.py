"""
Cherry-Picking Attack.

Removes a comparative baseline or reference-period clause (e.g. "compared to
2019", "since last quarter", "year-over-year") from a document, leaving the
headline statistic intact but stripping the context needed to judge whether
that statistic is actually favorable or unfavorable.

Example:
  Original:  "Unemployment fell to 4.1%, compared to 6.8% in 2020."
  Poisoned:  "Unemployment fell to 4.1%."

This is a classic cherry-picking pattern: the number itself is not altered,
but the baseline against which it should be interpreted is silently dropped,
letting the reader draw whatever conclusion the (now-missing) comparison
would have contradicted or complicated.

Why omission rather than substitution?
---------------------------------------
The four original attacks (numerical_shift, date_manipulation,
attribution_switch, causal_manipulation) all substitute one value for
another while preserving the document's overall structure. Cherry-picking is
a structurally different manipulation: the poisoned text is a strict subset
of the original (nothing is added or changed, something is deleted). This
tests a different RAG failure mode — whether a retrieval/generation pipeline
notices that supporting context has been silently removed, rather than
whether it propagates a substituted value.

EIBench taxonomy: Type 5 — Cherry-Picking (M05).
"""

from __future__ import annotations

import re
from typing import Any

from eiger.core.interfaces import BaseAttack
from eiger.core.models import Document, PoisonedDocument, PoisonAnnotation
from eiger.utils.seeding import make_rng, derive_seed


# ─── Module-level constants ───────────────────────────────────────────────────

# Matches a comparative/baseline clause: a connector phrase (compared to,
# relative to, versus, since <year>, over the past/last <period>,
# year-over-year) followed by the rest of the clause up to the next comma or
# terminal punctuation. Any leading AND trailing comma/whitespace is
# consumed too (see the ",?" bookending the pattern below), so removing the
# match does not leave a dangling ", ." or a stray orphaned "," behind —
# whether the clause sits mid-sentence (bounded by commas on both sides) or
# opens the sentence (no leading comma, but a trailing one separating it
# from the rest of the sentence).
#
# The clause "tail" (everything after the connector phrase) is built from
# _CLAUSE_TAIL rather than a plain `[^,.;]*`, to avoid the exact decimal-point
# bug already found and fixed once in causal.py's _SENTENCE_END_RE: without
# it, a value like "6.8%" inside the clause would be treated as ending at the
# "." between "6" and "8", truncating the match mid-number and leaving a
# corrupted fragment (e.g. ".8% in 2020.") behind in the poisoned text. Each
# iteration of _CLAUSE_TAIL first tries to consume a whole "<digits>.<digits>"
# token atomically, and only falls back to matching a single non-terminator,
# non-comma character when that fails. "!" and "?" are excluded alongside
# "." so a clause followed by either (e.g. "...compared to 6.8% in 2020!
# Great news.") stops there rather than bleeding into the next sentence.
#
# Design decision: this is a single alternation pattern (not a dict like
# AttributionSwitchAttack's entity map) because, unlike a source substitution,
# there is no natural "replacement" value for a baseline clause — the whole
# point is that it disappears without a trace.
_CLAUSE_TAIL: str = r"(?:\d+\.\d+|[^,.;!?])*"

_BASELINE_CLAUSE_RE: re.Pattern[str] = re.compile(
    r",?\s*(?:"
    r"compared (?:to|with)|"
    r"relative to|"
    r"versus|"
    r"since \d{4}|"
    r"over the (?:past|last) [\w\s]+?|"
    r"year[- ]over[- ]year"
    r")" + _CLAUSE_TAIL + r",?",
    re.IGNORECASE,
)


# ─── Attack class ─────────────────────────────────────────────────────────────

class CherryPickingAttack(BaseAttack):
    """
    Adversarial perturbation that deletes a comparative baseline clause,
    leaving the headline statistic technically true but stripped of the
    context needed to interpret it correctly.

    Responsibilities
    ----------------
    - Locate all baseline/comparison clauses via `_BASELINE_CLAUSE_RE`.
    - Randomly select up to `omit_count` matches (seeded, deterministic).
    - Delete the selected spans from the text (right-to-left, so earlier
      match offsets are not invalidated by later deletions).
    - Wrap the result in a PoisonedDocument with provenance metadata.

    What this class does NOT do
    ----------------------------
    - It does not alter the surviving numeric/factual content in any way —
      the headline statistic is left byte-for-byte identical.
    - It does not attempt to infer *which* baseline would have been most
      damaging to omit; among multiple matches it selects uniformly at
      random via the seeded RNG.
    - It does not guarantee the document text actually changes: a document
      with no comparative/baseline clause at all produces byte-identical
      output. This is recorded via ``attack_params["no_op"]``, following the
      same conservative convention as AttributionSwitchAttack — there is no
      safe generic fallback for "invent a baseline clause to then delete".

    EIBench taxonomy: Type 5 (M05).

    Excluded from the EIB paper's published benchmark corpus
    ------------------------------------------------------------
    The paper draft (Imperatrice & Putortì, "Epistemic Integrity Benchmark")
    lists M05 in its theoretical taxonomy but states it "does not occur in
    the final set of instances" (§6.0.2) — the Mistral-generated corpus
    (Corpus_claim_RAG_Mistral_output_v3.xlsx) contains zero M05 rows.
    This class still exists and is fully functional for engineering
    validation/ablation purposes, but ``excluded_from_benchmark = True``
    makes ``CorpusBuilder`` refuse to run it unless the caller explicitly
    opts in via ``allow_non_benchmark_attacks=True`` — see that class's own
    docstring. Do not use this attack to produce results reported as part
    of the EIB benchmark without first re-confirming with the ethics/
    threat-model protocol that governs the published corpus.
    """

    name: str = "cherry_picking"
    excluded_from_benchmark: bool = True

    description: str = (
        "Deletes a comparative baseline/reference-period clause "
        "(e.g. 'compared to 2019', 'year-over-year'), leaving the headline "
        "statistic intact but stripped of the context needed to interpret it."
    )

    # ─── Public interface ─────────────────────────────────────────────────────

    def apply(
        self,
        document: Document,
        seed: int,
        omit_count: int = 1,
        baseline_pattern: re.Pattern[str] | None = None,
        **kwargs: Any,
    ) -> PoisonedDocument:
        """
        Delete up to `omit_count` baseline/comparison clauses from the text.

        Args:
            document:         Source ground-truth document.
            seed:              Top-level experiment seed. A document- and
                               attack-specific sub-seed is derived for full
                               reproducibility.
            omit_count:        Maximum number of matching clauses to delete.
                               The actual number may be lower if the document
                               contains fewer eligible clauses than requested.
            baseline_pattern:  Custom compiled regex to use instead of
                               `_BASELINE_CLAUSE_RE`. When provided, completely
                               replaces the default pattern for this call.
            **kwargs:          Accepted but unused — maintains a uniform call
                               signature with other attack classes.

        Returns:
            PoisonedDocument with the selected baseline clause(s) removed,
            plus a PoisonAnnotation and omitted_count recorded in
            attack_params.
        """
        rng = make_rng(derive_seed(seed, document.doc_id, self.name))
        pattern = baseline_pattern or _BASELINE_CLAUSE_RE

        matches = list(pattern.finditer(document.text))

        if not matches:
            # Bug-fix parallel to AttributionSwitchAttack (Sprint 4 audit):
            # there is no safe generic fallback here either — fabricating a
            # baseline clause just so it can then be deleted would be a
            # materially different, more invasive attack design. Record the
            # no-op explicitly instead of hiding it.
            poisoned_text = document.text
            omitted_count = 0
        else:
            targets = rng.sample(matches, min(omit_count, len(matches)))
            # Delete right-to-left so that earlier matches' start/end offsets
            # remain valid even after later (higher-offset) spans are removed.
            targets_desc = sorted(targets, key=lambda m: m.start(), reverse=True)
            poisoned_text = document.text
            for m in targets_desc:
                poisoned_text = poisoned_text[: m.start()] + poisoned_text[m.end() :]
            # Collapse any double spaces left behind (e.g. "4.1%  ." -> "4.1%.")
            # and normalize a stray space before terminal punctuation.
            poisoned_text = re.sub(r"\s{2,}", " ", poisoned_text)
            poisoned_text = re.sub(r"\s+([.!?])", r"\1", poisoned_text).strip()
            omitted_count = len(targets_desc)

        no_op = poisoned_text == document.text

        # Pre-assessed risk scores for this attack type.
        # plausibility=4.5      : The surviving text is a strict subset of the
        #                         original and reads as a perfectly normal,
        #                         complete sentence — nothing looks edited.
        # verification_difficulty=4.5 : Detecting an *omission* requires
        #                         knowing the original had more context in
        #                         the first place; there is no textual
        #                         artifact to flag, unlike a substitution.
        # editorial_risk=4.0    : Automated filters have nothing anomalous to
        #                         catch — the output is well-formed, factually
        #                         correct prose.
        annotation = PoisonAnnotation(
            plausibility=4.5,
            verification_difficulty=4.5,
            editorial_risk=4.0,
        )

        return PoisonedDocument(
            doc_id=document.doc_id,
            claim_id=document.claim_id,
            text=poisoned_text,
            attack_name=self.name,
            # "no_op" records whether any baseline clause was actually found
            # and removed; "omitted_count" records how many were deleted.
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
        return {"attack": self.name, "method": "baseline_clause_deletion"}
