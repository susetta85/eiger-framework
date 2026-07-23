"""
End-to-end integration test using lightweight, fully-working fake infra.

Unlike tests/unit/, which mock BaseEmbedder/BaseVectorStore/BaseLLM via
MagicMock and only assert on call arguments, this test wires ExperimentRunner
to small but *functionally real* implementations of the three interfaces —
a deterministic character-frequency embedder, an in-memory cosine-similarity
vector store, and an LLM stub that deterministically echoes back the
first retrieved context document. Every layer of the real pipeline logic
(CorpusBuilder -> IngestionPipeline -> DenseRetriever -> BaseLLM ->
ExperimentRunner -> metrics -> ExperimentResult) executes for real; only the
ML/network boundaries are replaced with cheap, deterministic stand-ins.

This test requires no external services and no heavy ML dependencies
(no sentence-transformers download, no running Qdrant/Ollama), so it runs in
any environment, including CI without Docker. For a test against the real
production stack (SentenceTransformerEmbedder + QdrantVectorStore +
OllamaLLM), see test_pipeline_live_infra.py, which skips automatically when
those services are unreachable.

The FakeGenerationLLM is deliberately "maximally faithful to context, blind
to truth": it echoes the first retrieved document's text verbatim as the
answer. This is exactly the failure mode FFR is designed to catch (a faithful
answer that reproduces a poisoned document's falsehood), which makes it a
good deterministic stand-in for validating that retrieval -> prompt ->
generation -> metrics data actually flows through the whole system correctly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from eiger.core.interfaces import BaseEmbedder, BaseLLM, BaseVectorStore
from eiger.core.models import (
    AttackConfig,
    Claim,
    DatasetConfig,
    Document,
    ExperimentConfig,
    ExperimentResult,
    LLMConfig,
    RetrieverConfig,
)
from eiger.experiments import ExperimentRunner
from eiger.metrics import EmbeddingFaithfulnessScorer

# ─── Fake infrastructure ───────────────────────────────────────────────────────

_DIM = 32


class FakeCharFrequencyEmbedder(BaseEmbedder):
    """
    Deterministic, dependency-free stand-in for a real embedding model.

    Encodes each text as a normalised character-frequency vector. Not
    semantically meaningful (it has no notion of word meaning), but
    deterministic and stable, so texts that share a lot of characters
    (e.g. a claim and its numerically-shifted poisoned variant) end up
    reasonably close in cosine space — enough to exercise real ranking
    logic in DenseRetriever/FakeVectorStore without downloading a model.
    """

    model_name = "fake-char-frequency-embedder"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    @property
    def embedding_dim(self) -> int:
        return _DIM

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vec = [0.0] * _DIM
        for ch in text.lower():
            vec[ord(ch) % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeInMemoryVectorStore(BaseVectorStore):
    """
    Minimal in-memory BaseVectorStore backed by real cosine similarity.

    Stores (Document, vector) pairs per collection in a plain dict and
    computes genuine cosine similarity at search time — no mocking of the
    ranking logic itself, only the storage backend.
    """

    def __init__(self) -> None:
        self._collections: dict[str, list[tuple[Document, list[float]]]] = {}

    def create_collection(self, name: str, dim: int) -> None:
        self._collections[name] = []

    def reset_collection(self, name: str, dim: int) -> None:
        self._collections[name] = []

    def upsert(
        self, collection: str, documents: list[Document], vectors: list[list[float]]
    ) -> None:
        self._collections.setdefault(collection, [])
        self._collections[collection] = list(zip(documents, vectors, strict=True))

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        points = self._collections.get(collection, [])
        scored = [(doc, self._cosine(query_vector, vec)) for doc, vec in points]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            {
                "doc_id": doc.doc_id,
                "score": score,
                "payload": {
                    "doc_id": doc.doc_id,
                    "claim_id": doc.claim_id,
                    "text": doc.text,
                    "doc_type": doc.doc_type,
                },
            }
            for doc, score in scored[:top_k]
        ]

    @staticmethod
    def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
        norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
        return dot / (norm_a * norm_b)


class FakeGenerationLLM(BaseLLM):
    """
    Deterministic LLM stand-in: answers by echoing the first context document.

    This is "perfectly faithful to whatever was retrieved" by construction —
    if a poisoned document ranks first, the answer IS the poisoned text,
    verbatim. That is exactly the faithful-falsehood scenario FFR exists to
    detect, which makes this a meaningful (if crude) stand-in for a real LLM
    in an end-to-end plumbing test.
    """

    model_name = "fake-echo-llm"

    def build_rag_prompt(self, query: str, context_docs: list[str]) -> str:
        joined = "\n".join(context_docs) if context_docs else "(no context)"
        return f"Context:\n{joined}\nQuestion: {query}\nAnswer:"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        marker = "Context:\n"
        after = prompt.split(marker, 1)[1] if marker in prompt else prompt
        return after.split("\n", 1)[0]


# ─── Test ───────────────────────────────────────────────────────────────────────

def test_full_pipeline_runs_end_to_end_with_fake_infra(tmp_path: Path) -> None:
    """
    Corpus -> ingestion -> retrieval -> generation -> metrics -> ExperimentResult,
    exercised for real (no mocked call-argument assertions) against fake but
    functionally working infrastructure.
    """
    claim = Claim(
        claim_id="INTEG_001",
        original_fact="The WHO reported that inflation rose to 3.5% in 2023.",
        context_query="What did the WHO report about 2023 inflation?",
        source_dataset="integration_fixture",
    )

    embedder = FakeCharFrequencyEmbedder()
    config = ExperimentConfig(
        dataset=DatasetConfig(name="integration_fixture"),
        attacks=[AttackConfig(name="numerical_shift", poison_rate=1.0)],
        retriever=RetrieverConfig(collection_name="eiger_integration_fake", top_k=5),
        llm=LLMConfig(model="fake-echo-llm", temperature=0.0, max_tokens=128),
        metrics=["ffr", "ers"],
        output_dir=str(tmp_path),
    )

    runner = ExperimentRunner(
        config=config,
        embedder=embedder,
        vector_store=FakeInMemoryVectorStore(),
        llm=FakeGenerationLLM(),
        faithfulness_scorer=EmbeddingFaithfulnessScorer(embedder),
    )

    result = runner.run([claim])

    # ─── Top-level result shape ────────────────────────────────────────────
    assert isinstance(result, ExperimentResult)
    assert result.experiment_id == config.experiment_id
    assert result.config_hash == config.config_hash
    assert len(result.records) == 1

    # ─── Corpus: poison_rate=1.0 -> exactly 1 ground-truth + 1 poisoned doc ──
    record = result.records[0]
    assert len(record.retrieval.hits) == 2
    assert record.retrieval.contains_poisoned is True
    assert record.retrieval.poison_ratio == 0.5

    # ─── Generation: FakeGenerationLLM echoes the top-ranked context doc ─────
    assert record.generation.answer.strip() != ""
    assert record.generation.answer == record.generation.context_docs[0]

    # ─── Metrics: both configured metrics computed, values in range ─────────
    assert set(result.aggregate_metrics.keys()) == {"ffr", "ers"}
    assert 0.0 <= result.aggregate_metrics["ffr"] <= 1.0
    assert 0.0 <= result.aggregate_metrics["ers"] <= 1.0
    assert "ffr" in record.metrics
    assert "ers" in record.metrics
    # The faithfulness_scorer hook populated its keys before FFR ran.
    assert "ragas_faithfulness" in record.metrics
    assert "ragas_answer_correctness" in record.metrics

    # ─── Persistence: results.json written and round-trips as valid JSON ────
    result_path = tmp_path / "results.json"
    assert result_path.exists()
    on_disk = json.loads(result_path.read_text())
    assert on_disk["experiment_id"] == config.experiment_id
    assert len(on_disk["records"]) == 1
    assert on_disk["aggregate_metrics"]["ffr"] == result.aggregate_metrics["ffr"]


def test_full_pipeline_with_no_attacks_and_no_faithfulness_scorer(tmp_path: Path) -> None:
    """
    A minimal run with no attacks and no faithfulness_scorer must still
    complete successfully: FFR trivially 0.0 (documented limitation), corpus
    contains only the ground-truth document.
    """
    claim = Claim(
        claim_id="INTEG_002",
        original_fact="NASA confirmed the Mars mission launched in July 2020.",
        context_query="When did NASA launch the Mars mission?",
        source_dataset="integration_fixture",
    )

    config = ExperimentConfig(
        dataset=DatasetConfig(name="integration_fixture"),
        attacks=[],
        retriever=RetrieverConfig(collection_name="eiger_integration_fake_2", top_k=5),
        llm=LLMConfig(model="fake-echo-llm"),
        metrics=["ffr"],
        output_dir=str(tmp_path),
    )

    runner = ExperimentRunner(
        config=config,
        embedder=FakeCharFrequencyEmbedder(),
        vector_store=FakeInMemoryVectorStore(),
        llm=FakeGenerationLLM(),
    )

    result = runner.run([claim], save=False)

    assert len(result.records[0].retrieval.hits) == 1
    assert result.records[0].retrieval.contains_poisoned is False
    assert result.aggregate_metrics["ffr"] == 0.0
    assert not (tmp_path / "results.json").exists()
