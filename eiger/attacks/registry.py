"""
Attack registry — maps string names to BaseAttack classes.

Built-in attacks are registered automatically when the eiger.attacks
package is imported. Third-party attacks can register via entry points:

    [project.entry-points."eiger.attacks"]
    my_attack = "my_package.attacks:MyAttack"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eiger.core.exceptions import AttackNotFoundError

if TYPE_CHECKING:
    from eiger.core.interfaces import BaseAttack

_REGISTRY: dict[str, type[BaseAttack]] = {}


def register_attack(cls: type[BaseAttack]) -> type[BaseAttack]:
    """Register an attack class. Idempotent — re-registering the same class is safe."""
    _REGISTRY[cls.name] = cls
    return cls


def get_attack(name: str) -> BaseAttack:
    """Instantiate and return a registered attack by name."""
    if name not in _REGISTRY:
        raise AttackNotFoundError(name, list(_REGISTRY.keys()))
    return _REGISTRY[name]()


def list_attacks() -> list[str]:
    """Return sorted list of registered attack names."""
    return sorted(_REGISTRY.keys())
