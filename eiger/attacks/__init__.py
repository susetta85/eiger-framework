"""Adversarial poisoning attacks for RAG corpus contamination."""

from eiger.attacks.registry import register_attack, get_attack, list_attacks
from eiger.attacks.numerical import NumericalShiftAttack
from eiger.attacks.attribution import AttributionSwitchAttack
from eiger.attacks.temporal import DateManipulationAttack
from eiger.attacks.causal import CausalManipulationAttack

# Auto-register built-in attacks
register_attack(NumericalShiftAttack)
register_attack(AttributionSwitchAttack)
register_attack(DateManipulationAttack)
register_attack(CausalManipulationAttack)

__all__ = [
    "register_attack",
    "get_attack",
    "list_attacks",
    "NumericalShiftAttack",
    "AttributionSwitchAttack",
    "DateManipulationAttack",
    "CausalManipulationAttack",
]
