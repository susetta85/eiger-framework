# Contributing to EIGER / EIBench

Thank you for your interest in extending the EIGER framework. This guide covers the
workflow and standards for contributing new attacks, metrics, datasets, and LLM
backends, as well as general code quality requirements.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Code Style and Quality](#code-style-and-quality)
3. [How to Add a New Attack](#how-to-add-a-new-attack)
4. [How to Add a New Metric](#how-to-add-a-new-metric)
5. [How to Add a New Dataset](#how-to-add-a-new-dataset)
6. [How to Add a New LLM Backend](#how-to-add-a-new-llm-backend)
7. [Testing Requirements](#testing-requirements)
8. [Pull Request Checklist](#pull-request-checklist)

---

## Getting Started

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/eiger-framework.git
cd eiger-framework
git remote add upstream https://github.com/your-org/eiger-framework.git
```

### 2. Create a Branch

Always work on a feature branch, never directly on `main`:

```bash
git checkout -b feature/my-new-attack
```

### 3. Set Up the Development Environment

```bash
make setup
source venv/bin/activate
```

### 4. Verify the Baseline

Before making any changes, confirm that all existing tests pass:

```bash
make test-unit
```

Expected output: `63 passed`. If any test fails on a clean checkout, open an issue
before proceeding.

---

## Code Style and Quality

### Linting and Formatting

The project uses **Ruff** for both linting and code formatting. All submitted code must
pass without errors or warnings.

```bash
make lint        # check for linting errors
make format      # auto-format the codebase in-place
```

Run `make format` before every commit. CI will reject PRs that fail `make lint`.

### Type Checking

The project uses **Mypy** in strict mode. All public functions and methods must carry
complete type annotations.

```bash
make type-check
```

Fix all reported errors before submitting a PR. The use of `# type: ignore` is
permitted only in exceptional cases and must be accompanied by a comment explaining
why the suppression is necessary.

### Docstrings

Every public class and every public method or function must have a docstring. Use
Google-style docstrings:

```python
def apply(self, documents: list[str], rng: np.random.Generator) -> list[str]:
    """Apply the attack to a list of documents.

    Args:
        documents: The original knowledge-base documents.
        rng: Seeded random number generator. Do not replace with any other
            source of randomness.

    Returns:
        A new list of documents with the attack applied. The input list is
        not modified.
    """
```

### Other Conventions

- **Named constants over magic numbers.** Do not embed numeric literals in logic.
  Define them as module-level constants or expose them as config fields.
- **Comments in English only.** All inline comments, docstrings, commit messages, and
  PR descriptions must be written in English.
- **No hardcoded credentials or host addresses.** All external addresses must come from
  the Pydantic Settings object (environment variables prefixed `EIGER_`).
- **No global state mutation.** Functions and classes must not modify module-level or
  class-level mutable state. This is especially important for randomness — see the
  seeding rules below.

---

## How to Add a New Attack

### Step 1 — Create the Attack Module

Create a new file `eiger/attacks/my_attack.py`. Implement the `BaseAttack` abstract
base class:

```python
"""My attack: one-line summary of what it does."""

from __future__ import annotations

import numpy as np

from eiger.attacks.base import BaseAttack
from eiger.registry import register

# Named constant — never embed this value directly in logic
DEFAULT_NOISE_FRACTION: float = 0.1


@register("attacks")
class MyAttack(BaseAttack):
    """One-paragraph description of the attack strategy.

    This attack works by <explanation>. It is intended to simulate
    <threat model>. Cite the relevant paper or technique if applicable.

    Attributes:
        noise_fraction: Fraction of tokens to perturb per document.
    """

    name: str = "my_attack"
    description: str = "Short human-readable label used in result files."

    def __init__(self, noise_fraction: float = DEFAULT_NOISE_FRACTION) -> None:
        """Initialise the attack.

        Args:
            noise_fraction: Fraction of tokens to perturb. Must be in [0, 1].
        """
        if not (0.0 <= noise_fraction <= 1.0):
            raise ValueError(f"noise_fraction must be in [0, 1], got {noise_fraction}")
        self.noise_fraction = noise_fraction

    def apply(self, documents: list[str], rng: np.random.Generator) -> list[str]:
        """Apply the attack to a list of documents.

        Args:
            documents: The original knowledge-base documents (not mutated).
            rng: Seeded random number generator provided by the framework.
                Do not use any other source of randomness.

        Returns:
            A new list of documents with the attack applied.
        """
        result: list[str] = []
        for doc in documents:
            # implement perturbation logic here using rng
            result.append(doc)  # replace with actual logic
        return result

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable description of this attack's configuration.

        Returns:
            A dict suitable for inclusion in the experiment provenance record.
        """
        return {
            "name": self.name,
            "description": self.description,
            "noise_fraction": self.noise_fraction,
        }
```

Key rules:

- **Always use `rng`** (the injected `numpy.random.Generator`) for randomness. Never
  call `random.random()`, `np.random.rand()`, or any other global random function.
- **Never mutate the input list or its strings.** Return a new list.
- `describe()` must return a dict that is serialisable by `json.dumps` with no
  additional arguments.

### Step 2 — Register the Attack

Open `eiger/attacks/__init__.py` and import the new module so that the `@register`
decorator runs at import time:

```python
# eiger/attacks/__init__.py
from eiger.attacks import (
    baseline_attack,
    existing_attack,
    my_attack,          # add this line
)
```

### Step 3 — Write Unit Tests

Create `tests/unit/attacks/test_my_attack.py`. The following four tests are
**mandatory**. Add more tests as appropriate for your attack's specific logic.

```python
"""Unit tests for MyAttack."""

import numpy as np
import pytest

from eiger.attacks import MyAttack
from eiger.registry import get_registry


def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


SAMPLE_DOCS = ["The capital of France is Paris.", "Water boils at 100 °C."]


class TestMyAttackDeterminism:
    """Same seed must produce bitwise-identical output."""

    def test_identical_output_same_seed(self) -> None:
        attack = MyAttack()
        result_a = attack.apply(SAMPLE_DOCS, make_rng(42))
        result_b = attack.apply(SAMPLE_DOCS, make_rng(42))
        assert result_a == result_b

    def test_different_output_different_seed(self) -> None:
        attack = MyAttack()
        result_a = attack.apply(SAMPLE_DOCS, make_rng(1))
        result_b = attack.apply(SAMPLE_DOCS, make_rng(2))
        # For most non-trivial attacks these should differ; adjust if deterministic
        # by design (e.g., a zero-noise attack).
        assert isinstance(result_a, list)
        assert isinstance(result_b, list)


class TestMyAttackNoGlobalStateMutation:
    """Applying the attack must not mutate global or class-level state."""

    def test_input_documents_not_mutated(self) -> None:
        attack = MyAttack()
        original = list(SAMPLE_DOCS)
        attack.apply(SAMPLE_DOCS, make_rng(42))
        assert SAMPLE_DOCS == original

    def test_sequential_calls_are_independent(self) -> None:
        attack = MyAttack()
        # Run with seed 42, then again — second run must not be influenced by first
        first = attack.apply(SAMPLE_DOCS, make_rng(42))
        second = attack.apply(SAMPLE_DOCS, make_rng(42))
        assert first == second


class TestMyAttackAnnotations:
    """All public methods must carry type annotations."""

    def test_apply_has_annotations(self) -> None:
        import inspect
        sig = inspect.signature(MyAttack.apply)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert param.annotation is not inspect.Parameter.empty, (
                f"Parameter '{param_name}' of apply() is missing a type annotation"
            )
        assert sig.return_annotation is not inspect.Parameter.empty


class TestMyAttackRegistry:
    """The attack must be discoverable via the registry."""

    def test_registered_by_name(self) -> None:
        registry = get_registry("attacks")
        assert "my_attack" in registry, (
            "MyAttack is not registered. Check eiger/attacks/__init__.py."
        )
```

Run the new tests:

```bash
make test-unit
```

All 63 pre-existing tests plus your new tests must pass.

### Step 4 — Add a YAML Experiment Example

Create `experiments/example_my_attack.yaml` to demonstrate how to use the new attack:

```yaml
# experiments/example_my_attack.yaml
experiment:
  name: "example_my_attack"
  seed: 42

dataset:
  name: "triviaqa_subset"
  split: "validation"
  max_samples: 100

attack:
  name: "my_attack"
  noise_fraction: 0.15

metrics:
  - ffr
  - si
  - ers

llm:
  backend: "ollama"
  model: "llama3.1:8b"
  batch_size: 8
```

---

## How to Add a New Metric

### Step 1 — Create the Metric Module

Create `eiger/metrics/my_metric.py`:

```python
"""MyMetric: one-line summary."""

from __future__ import annotations

from eiger.metrics.base import BaseMetric
from eiger.registry import register


@register("metrics")
class MyMetric(BaseMetric):
    """Description of what this metric measures and how it is computed.

    Range: describe the expected value range (e.g., [0, 1], higher is better).
    """

    name: str = "my_metric"
    description: str = "Human-readable label."

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        """Compute the metric over a batch of predictions and references.

        Args:
            predictions: Model-generated answers.
            references: Ground-truth answers.

        Returns:
            A scalar score. Higher values indicate better performance.
        """
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable description of this metric's configuration."""
        return {"name": self.name, "description": self.description}
```

### Step 2 — Register the Metric

Add an import to `eiger/metrics/__init__.py`:

```python
from eiger.metrics import (
    existing_metric,
    my_metric,          # add this line
)
```

### Step 3 — Write Unit Tests

Create `tests/unit/metrics/test_my_metric.py`. Required tests:

- **Correct range**: assert the output is in the documented range for edge-case inputs
  (empty lists, identical predictions and references, completely wrong predictions).
- **Type annotation test**: same pattern as the attack annotation test above.
- **Registry test**: confirm `"my_metric"` appears in `get_registry("metrics")`.
- **`describe()` is serialisable**: `json.dumps(metric.describe())` must not raise.

---

## How to Add a New Dataset

See [DATASETS.md](DATASETS.md) for the full guide, including how to implement
`BaseDataset`, where to place raw data files, and how to register content hashes
for provenance tracking.

The short version:

1. Create `eiger/datasets/my_dataset.py` implementing `BaseDataset`.
2. Add an import to `eiger/datasets/__init__.py`.
3. Add at least three unit tests: schema validation, reproducible sampling, and registry
   registration.

---

## How to Add a New LLM Backend

Create `eiger/llms/my_llm.py` implementing `BaseLLM`:

```python
"""MyLLM: one-line summary."""

from __future__ import annotations

from eiger.llms.base import BaseLLM
from eiger.registry import register


@register("llms")
class MyLLM(BaseLLM):
    """Description of the LLM backend.

    All network calls must use the host/port from settings, never hardcoded values.
    """

    name: str = "my_llm"

    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Backend-specific generation parameters
                (e.g., temperature, max_tokens).

        Returns:
            The generated text string.
        """
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        """Return a JSON-serialisable description of this backend's configuration."""
        return {"name": self.name}
```

Add an import to `eiger/llms/__init__.py` and write unit tests that mock the network
layer (do not make real HTTP calls in unit tests).

---

## Testing Requirements

- Every new component (attack, metric, dataset, LLM) **must** have a corresponding
  unit test file under `tests/unit/`.
- **Determinism test is mandatory for all attacks.** Running `apply()` twice with the
  same seed must produce identical output.
- **No global state mutation test is mandatory for all attacks.** The input documents
  must be unchanged after `apply()` returns.
- All tests must pass without network access. Mock external services where needed.
- **Coverage must not decrease.** Check coverage before and after your change:

  ```bash
  make test-unit         # runs pytest with coverage
  ```

  If the overall coverage percentage drops, add tests to compensate before submitting.

- Tests must be runnable in isolation (no dependency on test ordering):

  ```bash
  pytest tests/unit/attacks/test_my_attack.py -v
  ```

---

## Pull Request Checklist

Before requesting a review, confirm every item below. Reviewers will not approve PRs
with unchecked items.

- [ ] All tests pass: `make test-unit` exits with zero errors
- [ ] No coverage regression compared to the base branch
- [ ] Type annotations present on all public methods and functions
- [ ] Docstrings present on all public classes and methods (Google style)
- [ ] All comments and docstrings written in English
- [ ] No hardcoded credentials, host addresses, or port numbers
- [ ] New component decorated with `@register(...)` and imported in the relevant
      `__init__.py`
- [ ] YAML experiment example added to `experiments/` if a new component was added
- [ ] `describe()` method returns a `json.dumps`-serialisable dict
- [ ] `make lint` passes with no errors or warnings
- [ ] `make type-check` passes with no errors
- [ ] Determinism test included for any new attack
- [ ] No global state mutation test included for any new attack
- [ ] PR description explains the motivation, the approach, and any known limitations
