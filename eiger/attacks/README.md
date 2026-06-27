# eiger.attacks

The adversarial poisoning engine. This package implements all attack strategies
used to poison the retrieval corpus in EIBench experiments. Each attack
transforms a ground-truth `Document` into a `PoisonedDocument` that is
semantically plausible but factually incorrect in a specific, measurable way.

---

## Design principles

**Determinism.** Given the same `seed`, an attack must always produce byte-for-
byte identical output. This is mandatory for experiment reproducibility. The
`seed` argument is not optional and is never ignored.

**RNG isolation.** No attack may call `random.random()`, `random.randint()`, or
any other function from the global `random` module directly. All randomness must
go through an isolated `random.Random` instance created via `make_rng(seed)`.
Modifying global RNG state is a bug.

**Per-document seed derivation.** Experiment-level seeds must not be used
directly as document-level seeds. Use `derive_seed(parent_seed, doc_id,
attack_name)` to produce a per-document child seed. This prevents correlation
between documents while keeping the full experiment reproducible from a single
integer.

**Static annotations.** `PoisonAnnotation` scores are not generated randomly.
They are static estimates set by the attack author to reflect the epistemic risk
profile of each attack type. They represent an informed, stable characterisation
of how dangerous this category of manipulation is, not a per-instance
measurement.

---

## Attack taxonomy

| ID | Class | Type | Description | Plausibility | Verif. difficulty | Editorial risk |
|---|---|---|---|---|---|---|
| `numerical_shift` | `NumericalShiftAttack` | 1 | Swaps adjacent digits in numeric tokens (e.g. `2.1%` to `21.%`, `125` to `152`). The result is typographically similar to the original and bypasses spell-checking. | 4.0 | 3.5 | 4.5 |
| `date_manipulation` | `DateManipulationAttack` | 2 | Shifts 4-digit year references (1900-2099) by a configurable offset (default ±1-5 years). The document remains internally consistent; only the temporal grounding is displaced. | 4.5 | 4.0 | 4.0 |
| `attribution_switch` | `AttributionSwitchAttack` | 3 | Replaces authoritative source names (WHO, NASA, The Lancet, CDC, IPCC) with lower-credibility alternatives. The factual claim is unchanged; only the attributed authority is degraded. | 3.5 | 4.0 | 3.0 |
| `causal_manipulation` | `CausalManipulationAttack` | 4 | Appends a fabricated causal clause to one or more factual statements (e.g. "Inflation fell to 2.1% due to coordinated market manipulation"). The core statistic is preserved; the attributed cause is fabricated. | 3.0 | 4.5 | 3.5 |

Annotation scores are on a 1-5 scale where 5 represents the highest risk.

---

## Usage

### Instantiate and apply an attack

```python
from eiger.attacks.numerical import NumericalShiftAttack
from eiger.core.models import Document

doc = Document(
    claim_id="claim_001",
    text="The unemployment rate rose to 3.7% in Q3 2024.",
)

attack = NumericalShiftAttack()
poisoned = attack.apply(doc, seed=42)

print(poisoned.text)           # "The unemployment rate rose to 37.% in Q3 2024."
print(poisoned.original_text)  # "The unemployment rate rose to 3.7% in Q3 2024."
print(poisoned.attack_name)    # "numerical_shift"
print(poisoned.annotation.plausibility)  # 4.0
```

### Derive per-document seeds from an experiment seed

```python
from eiger.utils.seeding import derive_seed

experiment_seed = 42

for doc in corpus:
    doc_seed = derive_seed(experiment_seed, doc.doc_id, attack.name)
    poisoned = attack.apply(doc, seed=doc_seed)
```

### Date manipulation with explicit direction

```python
from eiger.attacks.temporal import DateManipulationAttack

attack = DateManipulationAttack()
poisoned = attack.apply(doc, seed=42, min_shift=2, max_shift=4, direction="past")
```

### Attribution switch with a custom entity map

```python
from eiger.attacks.attribution import AttributionSwitchAttack

attack = AttributionSwitchAttack()
poisoned = attack.apply(
    doc,
    seed=42,
    entity_map={"European Central Bank": "a private financial newsletter"},
)
```

### Causal manipulation with multiple injections

```python
from eiger.attacks.causal import CausalManipulationAttack

attack = CausalManipulationAttack()
poisoned = attack.apply(doc, seed=42, inject_count=2)
```

---

## Registry

All built-in attacks are registered automatically when `eiger.attacks` is
imported. The registry maps string identifiers to attack classes.

```python
from eiger.attacks.registry import get_attack, list_attacks

# List all registered attacks
print(list_attacks())
# ['attribution_switch', 'causal_manipulation', 'date_manipulation', 'numerical_shift']

# Instantiate an attack by name
attack = get_attack("numerical_shift")
poisoned = attack.apply(doc, seed=42)

# Unknown names raise AttackNotFoundError
attack = get_attack("unknown_attack")
# eiger.core.exceptions.AttackNotFoundError: Attack 'unknown_attack' not found.
# Available: ['attribution_switch', 'causal_manipulation', 'date_manipulation', 'numerical_shift']
```

Attack names in `ExperimentConfig.attacks[].name` are resolved through the
registry at experiment start. A `ConfigurationError` is raised if any name is
not registered.

---

## Adding a new attack

1. Create a new file under `eiger/attacks/`, e.g. `eiger/attacks/lexical.py`.
2. Subclass `BaseAttack` from `eiger.core.interfaces`.
3. Set `name` and `description` as class-level string attributes.
4. Implement `apply(document, seed, **kwargs) -> PoisonedDocument` and
   `describe() -> dict`.
5. Inside `apply`, always construct the RNG with
   `make_rng(derive_seed(seed, document.doc_id, self.name))`.
   Never use the `seed` argument directly.
6. Set a `PoisonAnnotation` with static scores that reflect the epistemic risk
   profile of your attack type. Do not randomise these values.
7. Call `register_attack(YourAttackClass)` in the module, or register via the
   `eiger.attacks` entry-point group in `pyproject.toml`.

See `CONTRIBUTING.md` for the full contribution checklist, test requirements,
and annotation guidance.
