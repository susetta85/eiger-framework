"""
Metric registry — analogous to the attack registry in eiger.attacks.registry.

Provides a central dictionary that maps metric name strings (e.g. "ffr") to
their concrete ``BaseMetric`` subclass.  Built-in metrics are registered
automatically when ``eiger.metrics`` is imported; third-party metrics can be
added by calling ``register_metric`` with a custom class.

Design note: the registry stores *classes*, not instances, so that each call
to ``get_metric`` returns a freshly instantiated object with default parameters.
This avoids accidental shared state between experiment runs.

What this module does NOT do:
  - Execute any metric computation.
  - Validate metric configuration (handled by each metric's ``__init__``).
  - Persist or serialise the registry across processes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eiger.core.exceptions import MetricNotFoundError

# TYPE_CHECKING guard keeps the import out of the runtime critical path;
# BaseMetric is only needed for type annotations here.
if TYPE_CHECKING:
    from eiger.core.interfaces import BaseMetric

# ─── Internal registry store ──────────────────────────────────────────────────
# Module-level dict: metric_name -> metric class.
# Using a plain dict (not a class) keeps the implementation minimal and avoids
# thread-safety complications (registration happens at import time, before any
# concurrent experiment runners are spawned).
_REGISTRY: dict[str, type[BaseMetric]] = {}


# ─── Registry operations ──────────────────────────────────────────────────────

def register_metric(cls: type[BaseMetric]) -> type[BaseMetric]:
    """
    Register a metric class under its ``name`` attribute.

    Designed to be used both as a plain function call and as a class decorator::

        @register_metric
        class MyMetric(BaseMetric):
            name = "my_metric"
            ...

    Registering a name that already exists silently overwrites the previous
    entry, which allows downstream packages to override built-in metrics.

    Args:
        cls: A ``BaseMetric`` subclass with a non-empty ``name`` class attribute.

    Returns:
        The same class unchanged, enabling decorator usage.
    """
    # Store the class (not an instance) so get_metric can instantiate on demand.
    _REGISTRY[cls.name] = cls
    return cls


def get_metric(name: str) -> BaseMetric:
    """
    Retrieve and instantiate a metric by its registered name.

    Args:
        name: The string key used when the metric was registered (e.g. "ffr").

    Returns:
        A new instance of the corresponding metric class, initialised with
        its default parameters.

    Raises:
        MetricNotFoundError: If ``name`` is not present in the registry.
            The error message includes the list of registered names to help
            callers correct typos.
    """
    if name not in _REGISTRY:
        # Pass available keys so the error message is actionable.
        raise MetricNotFoundError(name, list(_REGISTRY.keys()))
    # Always instantiate fresh to avoid shared mutable state across calls.
    return _REGISTRY[name]()


def list_metrics() -> list[str]:
    """
    Return a sorted list of all registered metric names.

    Sorting ensures deterministic ordering in CLI output and documentation,
    regardless of the order in which metrics were registered.

    Returns:
        Alphabetically sorted list of metric name strings.
    """
    return sorted(_REGISTRY.keys())
