"""Metric registry — analogous to the attack registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eiger.core.exceptions import MetricNotFoundError

if TYPE_CHECKING:
    from eiger.core.interfaces import BaseMetric

_REGISTRY: dict[str, type[BaseMetric]] = {}


def register_metric(cls: type[BaseMetric]) -> type[BaseMetric]:
    _REGISTRY[cls.name] = cls
    return cls


def get_metric(name: str) -> BaseMetric:
    if name not in _REGISTRY:
        raise MetricNotFoundError(name, list(_REGISTRY.keys()))
    return _REGISTRY[name]()


def list_metrics() -> list[str]:
    return sorted(_REGISTRY.keys())
