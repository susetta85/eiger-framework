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

  Phase 3 — Vector store ingestion (IngestionPipeline):
      Takes the combined corpus (ground-truth + poisoned documents) from
      CorpusBuilderResult, embeds each document via a BaseEmbedder, and
      upserts the vectors into a BaseVectorStore collection for
      subsequent retrieval by DenseRetriever.

Public API:
  from eiger.ingestion import (
      CorpusBuilder, CorpusBuilderResult,
      IngestionPipeline, IngestionResult,
  )
"""

# CorpusBuilder/CorpusBuilderResult (Phase 2) and IngestionPipeline/
# IngestionResult (Phase 3) are the primary public API of this package.
# They are re-exported here so callers don't need to know the internal
# module layout.
from eiger.ingestion.corpus_builder import CorpusBuilder, CorpusBuilderResult
from eiger.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "CorpusBuilder",
    "CorpusBuilderResult",
    "IngestionPipeline",
    "IngestionResult",
]
