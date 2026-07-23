"""
Unit tests for EmbeddingFaithfulnessScorer (eiger.metrics.heuristic_scorer).

Tests verify:
  - __init__ stores the embedder
  - __init__ logs a warning identifying this as a proxy, not RAGAS
  - __call__ returns a dict with exactly "ragas_faithfulness" and
    "ragas_answer_correctness" keys (the exact keys FFRMetric/
    EvaluationRecord expect)
  - identical embeddings produce a similarity of 1.0 (rescaled from cosine 1.0)
  - opposite embeddings produce a similarity of 0.0 (rescaled from cosine -1.0)
  - orthogonal embeddings produce a similarity of 0.5 (rescaled from cosine 0.0)
  - a blank answer short-circuits both scores to 0.0 without calling encode()
  - empty context_docs short-circuits faithfulness to 0.0 (but not correctness)
  - a blank claim.original_fact short-circuits correctness to 0.0 (but not faithfulness)
  - _raw_cosine returns 0.0 for a zero vector instead of raising ZeroDivisionError
  - _raw_cosine raises on mismatched vector lengths (zip(strict=True)) rather
    than silently truncating

What these tests do NOT cover:
  - A real sentence-transformers model (the embedder is mocked throughout;
    SentenceTransformerEmbedder itself is covered by test_embedder.py).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.interfaces import BaseEmbedder
from eiger.core.models import Claim, GenerationResult
from eiger.metrics.heuristic_scorer import EmbeddingFaithfulnessScorer

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """Patch the module-level logger for every test except the ones that assert on it."""
    with patch("eiger.metrics.heuristic_scorer.log"):
        yield


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_claim(original_fact: str = "Inflation rose to 3.5% in 2023.") -> Claim:
    return Claim(
        claim_id="C1",
        original_fact=original_fact,
        context_query="What happened?",
        source_dataset="test_fixture",
    )


def _make_generation(
    answer: str = "It rose to 3.5%.",
    context_docs: list[str] | None = None,
) -> GenerationResult:
    return GenerationResult(
        claim_id="C1",
        query="What happened?",
        context_docs=context_docs if context_docs is not None else ["Inflation context doc."],
        answer=answer,
        model_name="mock-llm",
    )


def _make_scorer(
    encode_side_effect: Callable[[list[str]], list[list[float]]],
) -> tuple[EmbeddingFaithfulnessScorer, MagicMock]:
    """Return a scorer wired to a mock embedder with a custom encode() behavior."""
    mock_embedder = MagicMock(spec=BaseEmbedder)
    mock_embedder.encode.side_effect = encode_side_effect
    scorer = EmbeddingFaithfulnessScorer(embedder=mock_embedder)
    return scorer, mock_embedder


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestInit:
    """Tests for __init__ attribute storage and the proxy warning."""

    def test_stores_embedder(self) -> None:
        mock_embedder = MagicMock(spec=BaseEmbedder)
        scorer = EmbeddingFaithfulnessScorer(embedder=mock_embedder)
        assert scorer.embedder is mock_embedder

    def test_logs_proxy_warning_on_construction(self) -> None:
        mock_embedder = MagicMock(spec=BaseEmbedder)
        with patch("eiger.metrics.heuristic_scorer.log") as mock_log:
            EmbeddingFaithfulnessScorer(embedder=mock_embedder)
        mock_log.warning.assert_called_once()
        args, kwargs = mock_log.warning.call_args
        assert args[0] == "heuristic_scorer.proxy_in_use"
        assert "PROXY" in kwargs["message"]
        assert "RAGAS" in kwargs["message"]


# ─── __call__ — return shape ───────────────────────────────────────────────────

class TestCallReturnShape:
    """Tests for the dict shape returned by __call__."""

    def test_returns_expected_keys(self) -> None:
        scorer, _ = _make_scorer(lambda texts: [[1.0, 0.0] for _ in texts])
        result = scorer(_make_claim(), _make_generation())
        assert set(result.keys()) == {"ragas_faithfulness", "ragas_answer_correctness"}

    def test_scores_are_floats_in_unit_range(self) -> None:
        scorer, _ = _make_scorer(lambda texts: [[0.3, 0.7] for _ in texts])
        result = scorer(_make_claim(), _make_generation())
        for value in result.values():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0


# ─── __call__ — similarity semantics ───────────────────────────────────────────

class TestCallSimilaritySemantics:
    """Tests that cosine similarity is correctly rescaled to [0, 1]."""

    def test_identical_embeddings_score_one(self) -> None:
        # Every encode() call returns the same vector for both strings compared
        # -> cosine similarity 1.0 -> rescaled to 1.0.
        scorer, _ = _make_scorer(lambda texts: [[1.0, 2.0, 3.0] for _ in texts])
        result = scorer(_make_claim(), _make_generation())
        assert result["ragas_faithfulness"] == pytest.approx(1.0)
        assert result["ragas_answer_correctness"] == pytest.approx(1.0)

    def test_opposite_embeddings_score_zero(self) -> None:
        # First text -> [1, 0], second text -> [-1, 0] => cosine -1.0 -> 0.0.
        # Ground truth is left blank so only the faithfulness comparison runs
        # (one encode() call, consuming exactly these two vectors).
        vectors = iter([[1.0, 0.0], [-1.0, 0.0]])
        scorer, _ = _make_scorer(lambda texts: [next(vectors) for _ in texts])
        claim = _make_claim(original_fact="   ")
        result = scorer(claim, _make_generation(context_docs=["ctx"]))
        assert result["ragas_faithfulness"] == pytest.approx(0.0)

    def test_orthogonal_embeddings_score_half(self) -> None:
        # [1, 0] vs [0, 1] => cosine 0.0 -> rescaled to 0.5.
        # Ground truth is left blank so only the faithfulness comparison runs.
        vectors = iter([[1.0, 0.0], [0.0, 1.0]])
        scorer, _ = _make_scorer(lambda texts: [next(vectors) for _ in texts])
        claim = _make_claim(original_fact="   ")
        result = scorer(claim, _make_generation(context_docs=["ctx"]))
        assert result["ragas_faithfulness"] == pytest.approx(0.5)


# ─── __call__ — blank-input short circuits ────────────────────────────────────

class TestCallBlankInputs:
    """Tests for the empty/blank-string edge cases."""

    def test_blank_answer_short_circuits_both_scores_to_zero(self) -> None:
        scorer, mock_embedder = _make_scorer(lambda texts: [[1.0, 0.0] for _ in texts])
        result = scorer(_make_claim(), _make_generation(answer="   "))
        assert result == {"ragas_faithfulness": 0.0, "ragas_answer_correctness": 0.0}
        mock_embedder.encode.assert_not_called()

    def test_empty_context_docs_short_circuits_faithfulness_only(self) -> None:
        scorer, mock_embedder = _make_scorer(lambda texts: [[1.0, 0.0] for _ in texts])
        result = scorer(_make_claim(), _make_generation(context_docs=[]))
        assert result["ragas_faithfulness"] == 0.0
        # Correctness is still computed (answer vs. ground truth), so encode()
        # is called exactly once (for that comparison only).
        mock_embedder.encode.assert_called_once()
        assert result["ragas_answer_correctness"] == pytest.approx(1.0)

    def test_blank_ground_truth_short_circuits_correctness_only(self) -> None:
        claim = _make_claim(original_fact="   ")
        scorer, mock_embedder = _make_scorer(lambda texts: [[1.0, 0.0] for _ in texts])
        result = scorer(claim, _make_generation())
        assert result["ragas_answer_correctness"] == 0.0
        assert result["ragas_faithfulness"] == pytest.approx(1.0)


# ─── _raw_cosine ────────────────────────────────────────────────────────────────

class TestRawCosine:
    """Tests for the static cosine-similarity helper."""

    def test_zero_vector_returns_zero_not_raising(self) -> None:
        assert EmbeddingFaithfulnessScorer._raw_cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert EmbeddingFaithfulnessScorer._raw_cosine([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingFaithfulnessScorer._raw_cosine([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_known_value(self) -> None:
        # [1, 0] . [1, 1] / (1 * sqrt(2)) = 1/sqrt(2) ~= 0.7071
        assert EmbeddingFaithfulnessScorer._raw_cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(
            0.7071067811865476
        )
