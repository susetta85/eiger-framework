"""
CorpusBuilder: converts Claims into a mixed corpus of ground-truth
and adversarially poisoned Documents, ready for vector store ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from eiger.core.interfaces import BaseAttack
from eiger.core.models import Claim, Document, PoisonedDocument, AttackConfig
from eiger.utils.logging import get_logger
from eiger.utils.seeding import derive_seed

log = get_logger(__name__)


@dataclass
class CorpusBuilderResult:
    ground_truth_docs: list[Document] = field(default_factory=list)
    poisoned_docs: list[PoisonedDocument] = field(default_factory=list)

    @property
    def all_documents(self) -> list[Document]:
        return self.ground_truth_docs + self.poisoned_docs  # type: ignore[return-value]

    @property
    def poison_ratio(self) -> float:
        total = len(self.all_documents)
        return len(self.poisoned_docs) / total if total > 0 else 0.0


class CorpusBuilder:
    """
    Builds a mixed (ground-truth + poisoned) corpus from a list of Claims.

    Each claim generates exactly one ground-truth document.
    For each attack in the attack list, it is applied to each claim
    according to the configured poison_rate.
    """

    def __init__(self, attacks: list[tuple[BaseAttack, AttackConfig]], seed: int = 42) -> None:
        self.attacks = attacks
        self.seed = seed

    def build(self, claims: list[Claim]) -> CorpusBuilderResult:
        """
        Build the full corpus from a list of claims.

        Args:
            claims: List of source claims (loaded from a dataset).

        Returns:
            CorpusBuilderResult with ground-truth and poisoned documents separated.
        """
        result = CorpusBuilderResult()
        log.info("corpus_builder.start", n_claims=len(claims), n_attacks=len(self.attacks))

        from eiger.utils.seeding import make_rng

        for claim in claims:
            # Always add the ground-truth document
            gt_doc = Document(
                doc_id=f"gt_{claim.claim_id}",
                claim_id=claim.claim_id,
                text=claim.original_fact,
                doc_type="ground_truth",
                metadata={"source_dataset": claim.source_dataset},
            )
            result.ground_truth_docs.append(gt_doc)

            # Apply each configured attack with its poison_rate
            for attack, attack_cfg in self.attacks:
                # Per-document RNG: deterministic but independent per (claim, attack)
                rng = make_rng(derive_seed(self.seed, claim.claim_id, attack.name))
                if rng.random() <= attack_cfg.poison_rate:
                    doc_seed = derive_seed(self.seed, claim.claim_id, attack.name, "apply")
                    poisoned = attack.apply(gt_doc, seed=doc_seed, **attack_cfg.params)
                    result.poisoned_docs.append(poisoned)

        log.info(
            "corpus_builder.complete",
            n_ground_truth=len(result.ground_truth_docs),
            n_poisoned=len(result.poisoned_docs),
            poison_ratio=f"{result.poison_ratio:.2%}",
        )
        return result
