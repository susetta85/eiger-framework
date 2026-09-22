"""
Evaluation metrics for epistemic integrity in RAG systems.

This package exposes the built-in metrics used by EIBench:

  - FFRMetric          — Faithful Falsehood Rate: fraction of RAG answers that
                         are internally consistent with (possibly poisoned) context
                         yet factually wrong against ground truth.
  - ERSMetric          — Epistemic Risk Score: weighted combination of human/LLM
                         annotation dimensions (plausibility, verification
                         difficulty, editorial risk) into a single [0,1] scalar.
  - SourceIntegrityMetric — NLI-based measure of factual consistency between
                            retrieved documents and ground-truth claims.
  - PRRMetric          — Poisoned Retrieval Rate at top-k: fraction of queries
                         whose retrieved top-k documents include at least one
                         poisoned document (added Sprint 4; see eiger.metrics.prr).
  - PRDMetric          — Poisoned Rank-1 Dominance: fraction of queries whose
                         single rank-1 retrieved document is a poisoned
                         document (added Sprint 4; see eiger.metrics.prd).
  - PCSMetric          — Poisoned Context Sensitivity: embedding-space
                         distance between the real answer and a counterfactual
                         answer generated with poisoned context removed
                         (added Sprint 5+; see eiger.metrics.pcs). Requires
                         "pcs" to be in ExperimentConfig.metrics so
                         ExperimentRunner actually runs the counterfactual
                         generation each record needs.

Importing this module is the only action required to activate the metrics:
all six classes are automatically registered in the metric registry so they
can be retrieved by name via ``get_metric("ffr")`` etc — with one exception:
``PCSMetric`` needs an ``embedder`` to construct (see its own docstring),
which the registry's ``get_metric`` cannot supply (by design, it always
instantiates with zero arguments — see ``eiger.metrics.registry``'s own
design note). ``get_metric("pcs")`` therefore raises ``TypeError``;
``ExperimentRunner`` resolves "pcs" specially instead (see
``eiger/experiments/runner.py``'s ``_resolve_metric``), constructing
``PCSMetric(embedder=self.embedder)`` directly. ``PCSMetric`` is still
registered here so ``list_metrics()``/``eiger list-metrics`` lists it
alongside the others.

This package also exposes two ``faithfulness_scorer`` implementations for
``ExperimentRunner`` — neither is a ``BaseMetric``, and neither is
registered in the metric registry; both produce raw scores merged into
``EvaluationRecord.metrics`` *before* metrics are computed, rather than a
reportable ``MetricScore`` itself:

  - ``EmbeddingFaithfulnessScorer`` — a lightweight, LLM-judge-free proxy
    using cosine similarity (see eiger.metrics.heuristic_scorer for the full
    rationale and its documented limitations). The CLI's default.
  - ``RAGASFaithfulnessScorer`` — real RAGAS scoring via an Ollama LLM judge
    (added Sprint 5; see eiger.metrics.ragas_scorer for the exact pinned
    dependency versions this requires and why). Opt-in via
    ``ExperimentConfig.faithfulness_scorer: "ragas"``.

What this package does NOT do:
  - Run the RAG pipeline (see eiger.ingestion / eiger.retrieval).
  - Manage experiment orchestration (see eiger.experiments.ExperimentRunner).
"""

# ─── Public API re-exports ────────────────────────────────────────────────────

# Registry helpers: used by the CLI and experiment runner to resolve metrics by name.
from eiger.metrics.ers import ERSMetric

# Concrete metric implementations.
from eiger.metrics.ffr import FFRMetric

# Faithfulness scorers (not BaseMetric — see module docstring above).
from eiger.metrics.heuristic_scorer import EmbeddingFaithfulnessScorer
from eiger.metrics.pcs import PCSMetric
from eiger.metrics.prd import PRDMetric
from eiger.metrics.prr import PRRMetric
from eiger.metrics.ragas_scorer import RAGASFaithfulnessScorer
from eiger.metrics.registry import get_metric, list_metrics, register_metric
from eiger.metrics.source_integrity import SourceIntegrityMetric

# ─── Auto-register built-in metrics ──────────────────────────────────────────
# Registration happens at import time so that any code that does
#   ``from eiger.metrics import ...``
# can immediately call ``get_metric("ffr")`` without explicit registration.
# This mirrors the pattern used by eiger.attacks.__init__.
# PCSMetric is registered too (for list_metrics() visibility), but
# get_metric("pcs") alone will raise TypeError — see module docstring above
# and eiger/experiments/runner.py's _resolve_metric for why.
register_metric(FFRMetric)
register_metric(ERSMetric)
register_metric(SourceIntegrityMetric)
register_metric(PRRMetric)
register_metric(PRDMetric)
register_metric(PCSMetric)

# ─── Explicit public surface ──────────────────────────────────────────────────
# Only symbols listed here are considered stable public API.
__all__ = [
    "register_metric",
    "get_metric",
    "list_metrics",
    "FFRMetric",
    "ERSMetric",
    "SourceIntegrityMetric",
    "PRRMetric",
    "PRDMetric",
    "PCSMetric",
    "EmbeddingFaithfulnessScorer",
    "RAGASFaithfulnessScorer",
]
