# eiger.utils

This package provides two cross-cutting utilities used throughout the EIGER framework:
structured logging (`logging.py`) and deterministic random-number generation
(`seeding.py`). Both are designed to be lightweight and to impose no global side effects
unless explicitly requested.

---

## logging.py

Wraps [structlog](https://www.structlog.org/) to provide consistent, context-rich log
output across all EIGER components.

### Functions

- `configure_logging(level: str) -> None` — Configures structlog processors and sets the
  log level. Call once at application startup (e.g., inside `seed_everything` or the
  main entry point). The `level` argument accepts standard Python level names:
  `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`.

- `get_logger(name: str) -> structlog.BoundLogger` — Returns a bound logger namespaced
  to `name`. Typically called at module level with `__name__`.

### Usage

```python
from eiger.utils.logging import configure_logging, get_logger

configure_logging("INFO")
log = get_logger(__name__)

log.info("corpus built", n_docs=120, poison_ratio=0.15)
log.warning("attack skipped", attack="numerical_shift", claim_id="claim_042")
```

Structlog emits structured key-value output by default. In development the renderer
produces human-readable lines; in production it can be switched to JSON by adjusting
the processor chain inside `configure_logging`.

Example console output:

```
2026-06-27 10:04:12 [info     ] corpus built   n_docs=120 poison_ratio=0.15
2026-06-27 10:04:12 [warning  ] attack skipped attack=numerical_shift claim_id=claim_042
```

---

## seeding.py

Provides deterministic, reproducible random-number generation for EIGER experiments.
The central design principle is that each component receives its own isolated RNG
instance derived from a root seed, so no component can accidentally affect the random
state of another.

### Why isolated RNG matters

Python's `random` module maintains a single global state. If one part of the code calls
`random.choice()` an extra time, every subsequent call in the entire process produces a
different result — making experiments non-reproducible across code changes. EIGER avoids
this by giving each component a `random.Random` instance seeded from a deterministic
child seed derived from the root seed and a context string. Components never touch the
global `random` state.

### Functions

#### `make_rng(seed: int) -> random.Random`

Returns a new, isolated `random.Random` instance seeded with `seed`. This instance is
completely independent of the global random state.

```python
from eiger.utils.seeding import make_rng
import random

rng = make_rng(42)
print(rng.randint(0, 100))   # deterministic, isolated

# The global state is unaffected:
before = random.random()
_ = rng.randint(0, 100)
after = random.random()
assert before != after  # global state advanced only by its own calls, not by rng
```

#### `derive_seed(parent_seed: int, *context: str) -> int`

Produces a deterministic child seed by hashing the parent seed together with one or
more context strings using SHA-256. The result is a non-negative integer suitable for
seeding any RNG.

```python
from eiger.utils.seeding import derive_seed

seed_a = derive_seed(42, "claim_001", "numerical_shift")
seed_b = derive_seed(42, "claim_001", "negation_attack")
seed_c = derive_seed(42, "claim_002", "numerical_shift")

# seed_a, seed_b, seed_c are all different and fully deterministic
```

This is the primary mechanism used by `CorpusBuilder` to assign a unique, reproducible
seed to every (claim, attack) pair without requiring a seed registry.

#### `seed_everything(seed: int) -> None`

Seeds the global random state for the three libraries used by EIGER:

- `random` (Python standard library)
- `numpy.random`
- `torch` (CPU and, if available, CUDA)

Call this once at the start of an experiment before any computation begins.

```python
from eiger.utils.seeding import seed_everything

seed_everything(42)
```

After this call, results that depend on global random state (e.g., dataset shuffles
using numpy) are reproducible across runs with the same seed.

---

## Design summary

| Function            | Touches global state | Use case                                      |
|---------------------|----------------------|-----------------------------------------------|
| `make_rng(seed)`    | No                   | Per-component or per-document randomness      |
| `derive_seed(...)`  | No                   | Generating child seeds from a root seed       |
| `seed_everything()` | Yes (intentionally)  | One-time experiment startup                   |
