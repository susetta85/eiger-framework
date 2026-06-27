"""
Attack registry — maps string names to BaseAttack classes.

Built-in attacks are registered automatically when the eiger.attacks
package is imported. Third-party attacks can register via entry points:

    [project.entry-points."eiger.attacks"]
    my_attack = "my_package.attacks:MyAttack"

Responsibilities of this module
--------------------------------
- Maintain a module-level dictionary (_REGISTRY) that maps the
  string ``name`` attribute of each attack class to the class itself.
- Provide three thin functions — register_attack, get_attack,
  list_attacks — as the complete public interface for discovery and
  instantiation of attacks.

What this module does NOT do
-----------------------------
- It does not import or instantiate attack classes itself; callers
  (including eiger/attacks/__init__.py) are responsible for passing
  classes in via register_attack().
- It does not validate whether a class conforms to BaseAttack beyond
  the static type hint on register_attack's parameter. Runtime checks
  are the responsibility of BaseAttack's own __init_subclass__ hook
  (if any).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eiger.core.exceptions import AttackNotFoundError

# TYPE_CHECKING guard: BaseAttack is only imported for type annotations,
# not at runtime. This breaks the circular dependency that would arise
# if eiger.core.interfaces imported from eiger.attacks and vice versa.
if TYPE_CHECKING:
    from eiger.core.interfaces import BaseAttack

# ─── Registry store ───────────────────────────────────────────────────────────

# Module-level singleton dict: { attack_name: AttackClass }.
# Using a plain dict rather than a class-level registry keeps the state
# truly global and makes it trivially inspectable and patchable in tests.
_REGISTRY: dict[str, type[BaseAttack]] = {}


# ─── Public API ───────────────────────────────────────────────────────────────

def register_attack(cls: type[BaseAttack]) -> type[BaseAttack]:
    """
    Register an attack class under its ``name`` attribute.

    The function is idempotent: re-registering the same class (e.g. because
    the package is imported multiple times in the same process) silently
    overwrites the previous entry with an identical value and returns the
    class unchanged.

    Design note: returning ``cls`` allows this function to be used as a
    class decorator in third-party code:
        @register_attack
        class MyAttack(BaseAttack): ...

    Args:
        cls: A concrete subclass of BaseAttack that exposes a non-empty
             ``name`` class attribute used as the registry key.

    Returns:
        The same class that was passed in (facilitates decorator usage).
    """
    # Store the class itself, not an instance. Instances are created on demand
    # inside get_attack() so that each call gets a fresh, stateless object.
    _REGISTRY[cls.name] = cls
    return cls


def get_attack(name: str) -> BaseAttack:
    """
    Instantiate and return a registered attack by its string name.

    A new instance is created on every call. Attack classes are expected
    to be stateless (all mutable state lives on PoisonedDocument), so
    fresh instantiation is cheap and avoids cross-call contamination.

    Args:
        name: The string identifier of the attack (must match the
              ``name`` class attribute used during registration).

    Returns:
        A freshly instantiated BaseAttack object ready to call .apply() on.

    Raises:
        AttackNotFoundError: If ``name`` is not present in the registry.
                             The exception includes the list of valid names
                             to help callers recover gracefully.
    """
    if name not in _REGISTRY:
        # Pass the full list of registered names so the error message can
        # suggest alternatives — this is more helpful than a bare KeyError.
        raise AttackNotFoundError(name, list(_REGISTRY.keys()))
    return _REGISTRY[name]()


def list_attacks() -> list[str]:
    """
    Return a sorted list of all registered attack names.

    Sorting is alphabetical to give deterministic output regardless of
    the order in which attacks were registered (import order can vary
    across Python versions and environments).

    Returns:
        Sorted list of attack name strings, e.g.
        ['attribution_switch', 'causal_manipulation', 'date_manipulation',
         'numerical_shift'].
    """
    return sorted(_REGISTRY.keys())
