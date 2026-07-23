"""eiger.retrieval — embedder and dense retriever implementations."""

from eiger.retrieval.embedder import SentenceTransformerEmbedder
from eiger.retrieval.retriever import DenseRetriever

__all__ = ["SentenceTransformerEmbedder", "DenseRetriever"]
