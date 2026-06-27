"""Evaluation metrics for epistemic integrity in RAG systems."""

from eiger.metrics.registry import register_metric, get_metric, list_metrics
from eiger.metrics.ffr import FFRMetric
from eiger.metrics.ers import ERSMetric
from eiger.metrics.source_integrity import SourceIntegrityMetric

# Auto-register built-in metrics
register_metric(FFRMetric)
register_metric(ERSMetric)
register_metric(SourceIntegrityMetric)

__all__ = [
    "register_metric",
    "get_metric",
    "list_metrics",
    "FFRMetric",
    "ERSMetric",
    "SourceIntegrityMetric",
]
