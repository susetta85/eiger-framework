"""
Dataset registry — maps string names to BaseDataset classes.

Mirrors eiger.attacks.registry / eiger.metrics.registry exactly, for the
same reasons: a single module-level dict keeps discovery trivial and
patchable in tests, and storing classes (not instances) means every
get_dataset() call returns a fresh, stateless loader.

Built-in datasets are registered automatically when eiger.datasets is
imported. Third-party datasets can register the same way third-party
attacks/metrics would (see eiger.attacks.registry's docstring for the
entry-point convention this project intends to follow).

Responsibilities of this module
--------------------------------
- Maintain a module-level dictionary (_REGISTRY) that maps the string
  ``name`` attribute of each dataset class to the class itself.
- Provide three thin functions — register_dataset, get_dataset,
  list_datasets — as the complete public interface for discovery and
  instantiation of datasets.

What this module does NOT do
-----------------------------
- It does not import or instantiate dataset classes itself; callers
  (eiger/datasets/__init__.py) are responsible for passing classes in
  via register_dataset().
- It does not call .load() or .download() on anything; it only resolves
  names to fresh instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eiger.core.exceptions import DatasetNotFoundError

# TYPE_CHECKING guard: BaseDataset is only imported for type annotations,
# not at runtime, breaking the circular import that would otherwise arise.
if TYPE_CHECKING:
    from eiger.core.interfaces import BaseDataset

# ─── Registry store ───────────────────────────────────────────────────────────

# Module-level singleton dict: { dataset_name: DatasetClass }.
_REGISTRY: dict[str, type[BaseDataset]] = {}


# ─── Public API ───────────────────────────────────────────────────────────────

def register_dataset(cls: type[BaseDataset]) -> type[BaseDataset]:
    """
    Register a dataset class under its ``name`` attribute.

    Idempotent: re-registering the same class overwrites the previous
    entry with an identical value and returns the class unchanged, so
    importing eiger.datasets multiple times in the same process (e.g.
    across test modules) is harmless.

    Design note: returning ``cls`` allows this function to be used as a
    class decorator:
        @register_dataset
        class MyDataset(BaseDataset): ...

    Args:
        cls: A concrete subclass of BaseDataset exposing a non-empty
             ``name`` class attribute used as the registry key.

    Returns:
        The same class that was passed in (facilitates decorator usage).
    """
    _REGISTRY[cls.name] = cls
    return cls


def get_dataset(name: str) -> BaseDataset:
    """
    Instantiate and return a registered dataset by its string name.

    A new instance is created on every call using the class's default
    constructor arguments. Datasets that need a non-default constructor
    argument (e.g. a custom ``path``) should be instantiated directly
    rather than through this lookup — get_dataset() is the convenience
    path used when a DatasetConfig.name is resolved with no overrides.

    Args:
        name: The string identifier of the dataset (must match the
              ``name`` class attribute used during registration).

    Returns:
        A freshly instantiated BaseDataset object ready to call .load() on.

    Raises:
        DatasetNotFoundError: If ``name`` is not present in the registry.
                              The exception includes the list of valid
                              names to help callers recover gracefully.
    """
    if name not in _REGISTRY:
        raise DatasetNotFoundError(name, list(_REGISTRY.keys()))
    return _REGISTRY[name]()


def list_datasets() -> list[str]:
    """
    Return a sorted list of all registered dataset names.

    Sorting is alphabetical to give deterministic output regardless of
    the order in which datasets were registered.

    Returns:
        Sorted list of dataset name strings, e.g. ['json_fixture'].
    """
    return sorted(_REGISTRY.keys())
