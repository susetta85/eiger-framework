"""
EmbeddingFaithfulnessScorer: LLM-judge-free proxy for the faithfulness /
answer-correctness signals that FFRMetric needs.

Background — why this exists
-----------------------------
FFRMetric (eiger.metrics.ffr) reads ``EvaluationRecord.faithfulness_score``
and ``.factual_correctness_score``, which are read from
``record.metrics["ragas_faithfulness"]`` / ``["ragas_answer_correctness"]``.
Historically these were meant to come from RAGAS (an LLM-judge-based RAG
evaluation library, ``ragas>=0.1.0`` in pyproject.toml), but no RAGAS wrapper
exists yet in EIGER, and wiring one up is a substantial undertaking:
  - RAGAS's faithfulness/answer_correctness metrics both require an LLM
    judge, wrapped via ``ragas.llms.LangchainLLMWrapper`` — meaning a new,
    heavy ``langchain`` + ``langchain-ollama`` dependency, not just ``ragas``.
  - Ollama-as-judge configurations have documented reliability issues
    upstream (see e.g. explodinggradients/ragas issues #1120, #1246).
  - RAGAS's public API has changed materially across versions (function-
    style metrics vs. class-based ``Faithfulness()``/``AnswerCorrectness()``
    with async ``ascore()``), and pyproject.toml pins no upper bound.

Given that, this module provides a lightweight, fully local, LLM-judge-free
*proxy* using only the BaseEmbedder abstraction already in the project
(SentenceTransformerEmbedder). It makes FFR computable today, with no new
dependencies and no external judge call, at the cost of being a much
coarser signal than a real NLI/LLM judge.

WHAT THIS SCORER IS NOT — read before using in any published result
---------------------------------------------------------------------
This is NOT RAGAS. It does not verify logical entailment, does not reason
about individual statements in the answer, and can be fooled by superficial
lexical/semantic similarity (e.g. an answer that repeats a poisoned
document's wording will score as "faithful" to that context even though a
real NLI/LLM judge might reason about it differently — which, for EIBench's
purposes, is arguably still the right call, since faithfulness to
*whatever was retrieved* is exactly what FFR wants to measure. It is
"factual correctness vs. ground truth" where a coarse embedding-similarity
proxy is the bigger approximation: it cannot detect negation, sign flips,
or numeric perturbations as reliably as a real judge would).

Results computed with this scorer MUST be reported as such, e.g.
"FFR (embedding-similarity proxy)", not "FFR (RAGAS)". A warning is logged
once per scorer instance to make this impossible to miss in run logs.

Swapping in a real RAGAS-based scorer later requires no changes to
ExperimentRunner: any callable with signature
``(Claim, GenerationResult) -> dict[str, float]`` works via the
``faithfulness_scorer`` constructor argument (see eiger.experiments.runner).

Score semantics
---------------
- ``ragas_faithfulness`` (proxy): cosine similarity between the generated
  answer and the concatenated retrieved context, rescaled from [-1, 1] to
  [0, 1]. High = the answer looks embedding-similar to what was retrieved
  (whether that context is ground-truth or poisoned — faithfulness to
  context is exactly what this is supposed to capture).
- ``ragas_answer_correctness`` (proxy): cosine similarity between the
  generated answer and ``claim.original_fact`` (the ground truth), rescaled
  to [0, 1]. High = the answer looks embedding-similar to the actual
  ground truth.

Both keys populate exactly the fields FFRMetric reads.
"""

from __future__ import annotations

from eiger.core.interfaces import BaseEmbedder
from eiger.core.models import Claim, GenerationResult
from eiger.utils.logging import get_logger
from eiger.utils.similarity import embed_cosine_similarity_01

log = get_logger(__name__)


class EmbeddingFaithfulnessScorer:
    """
    Cosine-similarity proxy for RAGAS-style faithfulness/answer-correctness.

    Args:
        embedder: Any BaseEmbedder implementation. Does not need to be the
                  same instance used for retrieval/ingestion (no vectors are
                  compared against stored ones — everything is computed
                  fresh on the two short strings being compared), but using
                  the same model is a reasonable default for consistency.

    Example::

        scorer = EmbeddingFaithfulnessScorer(embedder=SentenceTransformerEmbedder())
        runner = ExperimentRunner(
            config=config,
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
            faithfulness_scorer=scorer,
        )
        result = runner.run(claims)
        # result.aggregate_metrics["ffr"] is now a real (if coarse) signal,
        # not trivially 0.0.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        # Logged once at construction (not per-call) so it appears exactly
        # once per experiment run in the logs, impossible to miss but not
        # spammy across hundreds of claims.
        log.warning(
            "heuristic_scorer.proxy_in_use",
            message=(
                "EmbeddingFaithfulnessScorer is a coarse cosine-similarity "
                "PROXY for faithfulness/answer_correctness, NOT the RAGAS "
                "LLM-judge metrics. Report resulting FFR as "
                "'FFR (embedding-similarity proxy)', not 'FFR (RAGAS)'."
            ),
        )

    def __call__(self, claim: Claim, generation: GenerationResult) -> dict[str, float]:
        """
        Compute proxy faithfulness/correctness scores for one generation.

        Args:
            claim:      Source claim; provides original_fact as ground truth.
            generation: RAG generation result (answer + context_docs).

        Returns:
            Dict with "ragas_faithfulness" and "ragas_answer_correctness"
            keys, each in [0.0, 1.0]. Both default to 0.0 if the answer,
            context, or ground truth text is empty/blank (nothing to
            meaningfully compare).
        """
        context_text = "\n".join(generation.context_docs)
        answer = generation.answer

        faithfulness = (
            embed_cosine_similarity_01(self.embedder, answer, context_text)
            if answer.strip() and context_text.strip()
            else 0.0
        )
        correctness = (
            embed_cosine_similarity_01(self.embedder, answer, claim.original_fact)
            if answer.strip() and claim.original_fact.strip()
            else 0.0
        )

        return {
            "ragas_faithfulness": faithfulness,
            "ragas_answer_correctness": correctness,
        }
