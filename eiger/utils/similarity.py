"""
Shared cosine-similarity helpers.

Factored out of ``eiger.metrics.heuristic_scorer`` (Sprint 2) when
``eiger.metrics.pcs`` (Sprint 5+) needed the exact same "embed two strings,
rescale cosine similarity to [0, 1]" computation — rather than duplicate it a
second time, both call this module. Pure math, no infrastructure dependency
(BaseEmbedder is passed in by the caller), consistent with the rest of
``eiger.utils`` being infrastructure-free.
"""

from __future__ import annotations

import math

from eiger.core.interfaces import BaseEmbedder


def raw_cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute the raw cosine similarity between two equal-length vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector, same length as vec_a.

    Returns:
        float in [-1.0, 1.0], or 0.0 if either vector has zero norm (avoids
        a ZeroDivisionError on degenerate/empty-string embeddings).
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_cosine_similarity_01(embedder: BaseEmbedder, text_a: str, text_b: str) -> float:
    """
    Embed two strings and return their cosine similarity, rescaled to [0, 1].

    Args:
        embedder: Any BaseEmbedder implementation.
        text_a:   First string to compare.
        text_b:   Second string to compare.

    Returns:
        float in [0.0, 1.0]. 0.0 if either embedding is a zero vector
        (degenerate case, e.g. an empty string with some embedders).
    """
    vec_a, vec_b = embedder.encode([text_a, text_b])
    raw = raw_cosine(vec_a, vec_b)
    # Rescale [-1, 1] -> [0, 1], the same convention DenseRetriever uses for
    # vector-store similarity scores (see DenseRetriever._normalize_score).
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))
