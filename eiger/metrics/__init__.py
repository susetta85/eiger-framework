"""
Evaluation metrics for epistemic integrity in RAG systems.

This package exposes the three built-in metrics used by EIBench:

  - FFRMetric          — Faithful Falsehood Rate: fraction of RAG answers that
                         are internally consistent with (possibly poisoned) context
                         yet factually wrong against ground truth.
  - ERSMetric          — Epistemic Risk Score: weighted combination of human/LLM
                         annotation dimensions (plausibility, verification
                         difficulty, editorial risk) into a single [0,1] scalar.
  - SourceIntegrityMetric — NLI-based measure of factual consistency between
                            retrieved documents and ground-truth claims.

Importing this module is the only action required to activate the metrics: the
three classes are automatically registered in the metric registry so they can
be retrieved by name via ``get_metric("ffr")`` etc.

This package also exposes ``EmbeddingFaithfulnessScorer``, a lightweight,
LLM-judge-free proxy for the RAGAS-style faithfulness/answer-correctness
signal that FFRMetric needs (see eiger.metrics.heuristic_scorer for the full
rationale and its documented limitations). It is NOT a BaseMetric and is
deliberately NOT registered in the metric registry — it produces raw scores
to merge into EvaluationRecord.metrics *before* metrics are computed, rather
than a reportable MetricScore itself. Pass an instance of it as
ExperimentRunner's ``faithfulness_scorer`` argument.

What this package does NOT do:
  - Run the RAG pipeline (see eiger.ingestion / eiger.retrieval).
  - Compute real RAGAS (LLM-judge) faithfulness or answer-correctness scores;
    EmbeddingFaithfulnessScorer is an explicitly-labeled proxy, not RAGAS.
  - Manage experiment orchestration (see eiger.experiments.ExperimentRunner).
"""

# ─── Public API re-exports ────────────────────────────────────────────────────

# Registry helpers: used by the CLI and experiment runner to resolve metrics by name.
from eiger.metrics.ers import ERSMetric

# Concrete metric implementations.
from eiger.metrics.ffr import FFRMetric

# Faithfulness proxy scorer (not a BaseMetric — see module docstring above).
from eiger.metrics.heuristic_scorer import EmbeddingFaithfulnessScorer
from eiger.metrics.registry import get_metric, list_metrics, register_metric
from eiger.metrics.source_integrity import SourceIntegrityMetric

# ─── Auto-register built-in metrics ──────────────────────────────────────────
# Registration happens at import time so that any code that does
#   ``from eiger.metrics import ...``
# can immediately call ``get_metric("ffr")`` without explicit registration.
# This mirrors the pattern used by eiger.attacks.__init__.
register_metric(FFRMetric)
register_metric(ERSMetric)
register_metric(SourceIntegrityMetric)

# ─── Explicit public surface ──────────────────────────────────────────────────
# Only symbols listed here are considered stable public API.
__all__ = [
    "register_metric",
    "get_metric",
    "list_metrics",
    "FFRMetric",
    "ERSMetric",
    "SourceIntegrityMetric",
    "EmbeddingFaithfulnessScorer",
]
