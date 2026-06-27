"""
SentenceTransformerEmbedder: dense text embedding via sentence-transformers.

This module provides a concrete implementation of BaseEmbedder backed by the
``sentence-transformers`` library (which wraps HuggingFace models).  It is
used at two points in the EIGER pipeline:

  1. **Ingestion** — embed every Document in the corpus before upserting to
     the vector store.
  2. **Retrieval** — embed the query string before issuing a similarity search.

Using the *same* embedder instance (or at least the same model) at both
points is a strict requirement for meaningful similarity scores.

Design decisions
----------------
- **Lazy loading**: the underlying SentenceTransformer model is not loaded
  at construction time.  The first call to ``encode()`` triggers the load.
  This keeps import time fast and avoids downloading a model when the class
  is instantiated in a test or configuration context.
- **Batch encoding**: the model is called once per batch (not once per text)
  to exploit the vectorised forward pass.  The caller controls the batch
  size via ``batch_size``.
- **Determinism**: ``sentence-transformers`` is deterministic for a given
  model and input when no sampling is involved (it is pure inference), so no
  seed management is needed here.
- **show_progress_bar=False**: suppressed by default to keep experiment logs
  clean; set ``verbose=True`` in the constructor to re-enable.

What this module does NOT do:
- It does not manage vector store connections (see QdrantVectorStore).
- It does not perform retrieval (see DenseRetriever).
- It does not support fine-tuning or training of the embedding model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eiger.core.interfaces import BaseEmbedder
from eiger.utils.logging import get_logger

if TYPE_CHECKING:
    # Only imported for type hints; the real import happens lazily in _load_model().
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)

# Default model: small, fast, and well-understood in the research community.
# 384-dimensional vectors; suitable for semantic similarity on short texts.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Dense text embedder backed by the ``sentence-transformers`` library.

    Args:
        model_name: HuggingFace model identifier or local path.  Defaults to
                    ``all-MiniLM-L6-v2`` (384 dims, ~22 MB).
        batch_size: Number of texts to encode in a single forward pass.
                    Larger values are faster on GPU but use more memory.
        device:     PyTorch device string (e.g. ``"cpu"``, ``"cuda:0"``).
                    ``None`` lets sentence-transformers auto-detect.
        verbose:    If True, show the tqdm progress bar during encoding.

    Attributes:
        model_name: Stored for provenance logging.

    Example::

        embedder = SentenceTransformerEmbedder()
        vectors = embedder.encode(["Hello world", "Another sentence"])
        # vectors: list of two 384-dim float lists
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._verbose = verbose
        # Model is not loaded yet — see _load_model().
        self._model: SentenceTransformer | None = None

    # ─── Lazy loader ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer model on first use.

        Idempotent: calling this method multiple times is safe — it returns
        immediately if the model is already loaded.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
                         Install it with: pip install sentence-transformers
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        log.info("embedder.loading_model", model=self.model_name, device=self._device)
        self._model = SentenceTransformer(self.model_name, device=self._device)
        log.info("embedder.model_ready", model=self.model_name, dim=self.embedding_dim)

    # ─── BaseEmbedder interface ───────────────────────────────────────────────

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of texts into dense embedding vectors.

        Calls ``_load_model()`` on the first invocation, then delegates to
        ``SentenceTransformer.encode()`` with batching and numpy→list conversion.

        Args:
            texts: Strings to embed.  May be empty (returns ``[]``).

        Returns:
            List of float lists, one per input text, each of length
            ``self.embedding_dim``.  Order matches the input.
        """
        if not texts:
            return []

        self._load_model()
        assert self._model is not None  # guaranteed by _load_model()

        log.debug("embedder.encoding", n_texts=len(texts), batch_size=self._batch_size)

        # encode() returns a numpy array of shape (n_texts, dim).
        # We convert to a plain Python list of lists so that the output is
        # JSON-serialisable and vector-store-agnostic.
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=self._verbose,
            convert_to_numpy=True,
        )
        # numpy array → list[list[float]]
        return [v.tolist() for v in vectors]

    @property
    def embedding_dim(self) -> int:
        """
        Dimensionality of the embedding vectors produced by this model.

        Triggers model loading on first access so that the dimension is
        always correct, even if called before ``encode()``.

        Returns:
            Positive integer (e.g. 384 for all-MiniLM-L6-v2).
        """
        self._load_model()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()
