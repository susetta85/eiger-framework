"""
Unit tests for RAGASFaithfulnessScorer (eiger.metrics.ragas_scorer).

Since the pinned `ragas`/`langchain-ollama`/`langchain-community` optional-
dependency group (see that module's own docstring for the exact versions
and why) is not part of this project's core test dependencies, every test
here that exercises the "ragas is installed" path injects fake modules via
``sys.modules`` (mirroring the established pattern in test_source_integrity.py
for the optional `transformers` dependency) rather than requiring the real
packages to be installed in the test environment.

Tests verify:
  - __init__ raises an actionable ImportError if ragas/langchain-ollama are
    not importable
  - __init__ constructs ChatOllama with the given model/host/port/temperature,
    wraps it in LangchainLLMWrapper, and builds Faithfulness/AnswerCorrectness
    with it (plus a LangchainEmbeddingsWrapper-wrapped embedder for the latter)
  - __call__ short-circuits to {0.0, 0.0} without invoking the judge LLM at
    all when the answer or every context doc is blank
  - __call__ builds a SingleTurnSample with the right fields and returns the
    two metrics' single_turn_ascore() results under the expected keys
  - _EmbedderAsLangchainEmbeddings correctly adapts a BaseEmbedder's sync
    encode() into langchain's embed_documents/embed_query/aembed_* methods

What these tests do NOT cover:
  - Real RAGAS scoring quality or actual Ollama server behavior — verifying
    that requires a live Ollama instance, well outside a unit test's scope
    (see the module's own docstring for what was and was not verified).
    See tests/integration/test_ragas_scorer_real.py for tests against the
    actual installed `ragas`/`langchain-ollama` packages (skipped, not
    failed, if the `ragas` extra isn't installed, and further skipped for
    the live-scoring case if Ollama itself isn't reachable).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eiger.core.interfaces import BaseEmbedder
from eiger.core.models import Claim, GenerationResult
from eiger.metrics.ragas_scorer import RAGASFaithfulnessScorer, _EmbedderAsLangchainEmbeddings

# ─── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_claim(original_fact: str = "Inflation rose to 3.5% in 2023.") -> Claim:
    return Claim(
        claim_id="C1",
        original_fact=original_fact,
        context_query="What happened to inflation?",
        source_dataset="test_fixture",
    )


def _make_generation(
    answer: str = "Inflation rose to 3.5%.",
    context_docs: list[str] | None = None,
    query: str = "What happened to inflation?",
) -> GenerationResult:
    return GenerationResult(
        claim_id="C1",
        query=query,
        context_docs=context_docs if context_docs is not None else ["Inflation rose to 3.5% in 2023."],
        answer=answer,
        model_name="mock-llm",
    )


class _FakeEmbedder(BaseEmbedder):
    """Minimal BaseEmbedder for adapter tests — no real model needed."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    @property
    def embedding_dim(self) -> int:
        return 3


def _install_fake_ragas_modules() -> dict[str, MagicMock]:
    """
    Build a dict of fake modules simulating a working `ragas` +
    `langchain-ollama` installation, for use with
    ``patch.dict(sys.modules, ...)``.

    Returns the dict so tests can grab specific mock classes (e.g.
    ``mocks["ragas.metrics"].Faithfulness``) to assert on or configure
    return values.
    """
    mock_chat_ollama_cls = MagicMock(name="ChatOllama")
    mock_langchain_ollama = MagicMock()
    mock_langchain_ollama.ChatOllama = mock_chat_ollama_cls

    mock_llm_wrapper_cls = MagicMock(name="LangchainLLMWrapper")
    mock_ragas_llms = MagicMock()
    mock_ragas_llms.LangchainLLMWrapper = mock_llm_wrapper_cls

    mock_embeddings_wrapper_cls = MagicMock(name="LangchainEmbeddingsWrapper")
    mock_ragas_embeddings = MagicMock()
    mock_ragas_embeddings.LangchainEmbeddingsWrapper = mock_embeddings_wrapper_cls

    mock_faithfulness_cls = MagicMock(name="Faithfulness")
    mock_answer_correctness_cls = MagicMock(name="AnswerCorrectness")
    mock_ragas_metrics = MagicMock()
    mock_ragas_metrics.Faithfulness = mock_faithfulness_cls
    mock_ragas_metrics.AnswerCorrectness = mock_answer_correctness_cls

    mock_single_turn_sample_cls = MagicMock(name="SingleTurnSample")
    mock_ragas_dataset_schema = MagicMock()
    mock_ragas_dataset_schema.SingleTurnSample = mock_single_turn_sample_cls

    mock_ragas = MagicMock()

    return {
        "ragas": mock_ragas,
        "ragas.llms": mock_ragas_llms,
        "ragas.embeddings": mock_ragas_embeddings,
        "ragas.metrics": mock_ragas_metrics,
        "ragas.dataset_schema": mock_ragas_dataset_schema,
        "langchain_ollama": mock_langchain_ollama,
    }


# ─── __init__ — missing dependency ─────────────────────────────────────────────

class TestInitMissingDependency:
    def test_raises_actionable_import_error_when_ragas_missing(self) -> None:
        with (
            patch.dict(sys.modules, {"ragas": None, "ragas.llms": None}),
            pytest.raises(ImportError, match="pip install"),
        ):
            RAGASFaithfulnessScorer(embedder=_FakeEmbedder())

    def test_raises_actionable_import_error_when_langchain_ollama_missing(self) -> None:
        with (
            patch.dict(sys.modules, {"langchain_ollama": None}),
            pytest.raises(ImportError, match="pip install"),
        ):
            RAGASFaithfulnessScorer(embedder=_FakeEmbedder())

    def test_import_error_message_names_verified_versions(self) -> None:
        with (
            patch.dict(sys.modules, {"ragas": None}),
            pytest.raises(ImportError, match="ragas==0.2.15"),
        ):
            RAGASFaithfulnessScorer(embedder=_FakeEmbedder())


# ─── __init__ — successful construction ────────────────────────────────────────

class TestInitConstruction:
    def test_constructs_chat_ollama_with_given_params(self) -> None:
        mocks = _install_fake_ragas_modules()
        with patch.dict(sys.modules, mocks), patch("eiger.metrics.ragas_scorer.log"):
            RAGASFaithfulnessScorer(
                embedder=_FakeEmbedder(),
                model_name="llama3.1:8b",
                host="myhost",
                port=12345,
                temperature=0.2,
            )
        mocks["langchain_ollama"].ChatOllama.assert_called_once_with(
            model="llama3.1:8b", base_url="http://myhost:12345", temperature=0.2
        )

    def test_wraps_chat_model_in_langchain_llm_wrapper(self) -> None:
        mocks = _install_fake_ragas_modules()
        with patch.dict(sys.modules, mocks), patch("eiger.metrics.ragas_scorer.log"):
            RAGASFaithfulnessScorer(embedder=_FakeEmbedder())
        mocks["ragas.llms"].LangchainLLMWrapper.assert_called_once_with(
            mocks["langchain_ollama"].ChatOllama.return_value
        )

    def test_builds_faithfulness_and_answer_correctness_with_wrapped_llm(self) -> None:
        mocks = _install_fake_ragas_modules()
        wrapped_llm = mocks["ragas.llms"].LangchainLLMWrapper.return_value
        with patch.dict(sys.modules, mocks), patch("eiger.metrics.ragas_scorer.log"):
            RAGASFaithfulnessScorer(embedder=_FakeEmbedder())
        mocks["ragas.metrics"].Faithfulness.assert_called_once_with(llm=wrapped_llm)
        mocks["ragas.metrics"].AnswerCorrectness.assert_called_once_with(
            llm=wrapped_llm,
            embeddings=mocks["ragas.embeddings"].LangchainEmbeddingsWrapper.return_value,
        )

    def test_wraps_embedder_in_langchain_embeddings_wrapper(self) -> None:
        mocks = _install_fake_ragas_modules()
        with patch.dict(sys.modules, mocks), patch("eiger.metrics.ragas_scorer.log"):
            RAGASFaithfulnessScorer(embedder=_FakeEmbedder())
        # The adapter passed in is a _EmbedderAsLangchainEmbeddings instance.
        (adapter,), _ = mocks["ragas.embeddings"].LangchainEmbeddingsWrapper.call_args
        assert isinstance(adapter, _EmbedderAsLangchainEmbeddings)

    def test_logs_warning_naming_the_judge_model(self) -> None:
        mocks = _install_fake_ragas_modules()
        with patch.dict(sys.modules, mocks), patch("eiger.metrics.ragas_scorer.log") as mock_log:
            RAGASFaithfulnessScorer(embedder=_FakeEmbedder(), model_name="mistral:7b")
        mock_log.warning.assert_called_once()
        _, kwargs = mock_log.warning.call_args
        assert "mistral:7b" in kwargs["message"]


# ─── __call__ ───────────────────────────────────────────────────────────────────

class TestCall:
    """
    BUG FIXED (Sprint 5 coverage gap — was 89%, missing lines 266-274 and
    292-296): the previous version of this class had a helper,
    ``_make_scorer``, that ``return``ed the constructed scorer *from inside*
    its own ``with patch.dict(sys.modules, mocks): ...`` block. A ``return``
    inside a ``with`` statement still runs that block's ``__exit__`` before
    control reaches the caller, so by the time each test method went on to
    call ``scorer(claim, generation)``, ``sys.modules`` had already been
    restored to its real contents. ``__call__`` does its own *separate*
    lazy import (``from ragas.dataset_schema import SingleTurnSample`` at
    line 264 — independent from the imports ``__init__`` does), so that
    import — and everything after it in ``__call__``/``_score_both`` — ran
    outside the patch, against whatever real ``ragas`` happened to be
    installed (or raised ``ImportError`` if it wasn't), never exercising the
    mocked path the assertions actually intended to test. That's exactly the
    code (lines 266-277, 292-296) coverage reported as never hit.

    Fix: every test in this class now keeps the ``patch.dict(sys.modules,
    mocks)`` context open for the entire construct-and-call lifecycle, via
    ``_scorer_ctx`` below (a small contextmanager) rather than a plain
    helper method that returns early.
    """

    @contextmanager
    def _scorer_ctx(
        self, mocks: dict[str, MagicMock]
    ) -> Iterator[RAGASFaithfulnessScorer]:
        with patch.dict(sys.modules, mocks), patch("eiger.metrics.ragas_scorer.log"):
            yield RAGASFaithfulnessScorer(embedder=_FakeEmbedder())

    def test_empty_answer_short_circuits_without_invoking_judge(self) -> None:
        mocks = _install_fake_ragas_modules()
        with self._scorer_ctx(mocks) as scorer:
            result = scorer(_make_claim(), _make_generation(answer=""))
        assert result == {"ragas_faithfulness": 0.0, "ragas_answer_correctness": 0.0}
        mocks["ragas.dataset_schema"].SingleTurnSample.assert_not_called()

    def test_blank_context_short_circuits_without_invoking_judge(self) -> None:
        mocks = _install_fake_ragas_modules()
        with self._scorer_ctx(mocks) as scorer:
            result = scorer(_make_claim(), _make_generation(context_docs=["", "   "]))
        assert result == {"ragas_faithfulness": 0.0, "ragas_answer_correctness": 0.0}
        mocks["ragas.dataset_schema"].SingleTurnSample.assert_not_called()

    def test_builds_sample_with_expected_fields(self) -> None:
        mocks = _install_fake_ragas_modules()
        # single_turn_ascore must be awaitable — plain MagicMock() is not.
        mocks["ragas.metrics"].Faithfulness.return_value.single_turn_ascore = AsyncMock(
            return_value=0.7
        )
        mocks["ragas.metrics"].AnswerCorrectness.return_value.single_turn_ascore = AsyncMock(
            return_value=0.9
        )

        claim = _make_claim(original_fact="Inflation rose to 3.5% in 2023.")
        generation = _make_generation(
            answer="Inflation rose to 3.5%.",
            context_docs=["doc one", "doc two"],
            query="What happened to inflation?",
        )
        with self._scorer_ctx(mocks) as scorer:
            scorer(claim, generation)

        _, kwargs = mocks["ragas.dataset_schema"].SingleTurnSample.call_args
        assert kwargs["user_input"] == "What happened to inflation?"
        assert kwargs["response"] == "Inflation rose to 3.5%."
        assert kwargs["retrieved_contexts"] == ["doc one", "doc two"]
        assert kwargs["reference"] == "Inflation rose to 3.5% in 2023."

    def test_returns_both_scores_under_expected_keys(self) -> None:
        mocks = _install_fake_ragas_modules()
        mocks["ragas.metrics"].Faithfulness.return_value.single_turn_ascore = AsyncMock(
            return_value=0.65
        )
        mocks["ragas.metrics"].AnswerCorrectness.return_value.single_turn_ascore = AsyncMock(
            return_value=0.42
        )
        with self._scorer_ctx(mocks) as scorer:
            result = scorer(_make_claim(), _make_generation())
        assert result == {"ragas_faithfulness": 0.65, "ragas_answer_correctness": 0.42}


# ─── _EmbedderAsLangchainEmbeddings ────────────────────────────────────────────

class TestEmbedderAdapter:
    def test_embed_documents_delegates_to_encode(self) -> None:
        adapter = _EmbedderAsLangchainEmbeddings(_FakeEmbedder())
        result = adapter.embed_documents(["a", "b"])
        assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    def test_embed_query_returns_single_vector(self) -> None:
        adapter = _EmbedderAsLangchainEmbeddings(_FakeEmbedder())
        assert adapter.embed_query("a") == [0.1, 0.2, 0.3]

    def test_aembed_documents_matches_sync_version(self) -> None:
        # No pytest-asyncio dependency in this project — run the coroutine
        # directly via asyncio.run(), matching how RAGASFaithfulnessScorer
        # itself bridges async ragas calls into a sync __call__.
        adapter = _EmbedderAsLangchainEmbeddings(_FakeEmbedder())
        result = asyncio.run(adapter.aembed_documents(["a", "b"]))
        assert result == adapter.embed_documents(["a", "b"])

    def test_aembed_query_matches_sync_version(self) -> None:
        adapter = _EmbedderAsLangchainEmbeddings(_FakeEmbedder())
        result = asyncio.run(adapter.aembed_query("a"))
        assert result == adapter.embed_query("a")
