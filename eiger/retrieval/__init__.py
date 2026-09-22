"""eiger.retrieval — embedder and retriever implementations (dense + sparse + hybrid)."""

from eiger.retrieval.embedder import SentenceTransformerEmbedder
from eiger.retrieval.retriever import DenseRetriever
from eiger.retrieval.sparse_retriever import SparseRetriever
from eiger.retrieval.hybrid_retriever import HybridRetriever

__all__ = ["SentenceTransformerEmbedder", "DenseRetriever", "SparseRetriever", "HybridRetriever"]
