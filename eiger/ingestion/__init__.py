"""
Corpus ingestion: load claims, apply attacks, upsert to vector store.

This package orchestrates the three-phase corpus preparation pipeline:

  Phase 1 — Dataset loading:
      A BaseDataset implementation reads raw claim files and returns a
      list of Claim objects.

  Phase 2 — Corpus building (this package's primary responsibility):
      CorpusBuilder takes the claims, creates one ground-truth Document
      per claim, and applies configured adversarial attacks at the
      specified poison_rate to generate PoisonedDocument objects.

  Phase 3 — Vector store ingestion:
      A BaseVectorStore implementation receives the combined corpus
      (ground-truth + poisoned documents), embeds each document, and
      upserts the vectors for subsequent retrieval.

Public API:
  from eiger.ingestion import CorpusBuilder, CorpusBuilderResult
"""

# The CorpusBuilder and its result dataclass are the primary public API
# of this package. They are re-exported here so callers don't need to
# know the internal module layout.
from eiger.ingestion.corpus_builder import CorpusBuilder, CorpusBuilderResult

__all__ = ["CorpusBuilder", "CorpusBuilderResult"]
