"""
Reproducible seeding utilities.

EIGER treats reproducibility as a first-class requirement: given the same
experiment configuration and seed, every run must produce bit-identical
results. This module enforces three rules to achieve that:

  1. Never modify the global random.random() state from within framework
     code — use isolated random.Random instances instead.
  2. Every stochastic operation receives an explicit seed argument; no
     function silently draws from ambient global state.
  3. Child seeds are derived deterministically from a parent seed plus
     a string context, so per-document and per-attack seeds are unique
     and stable without requiring a shared counter or database.

Why SHA-256 for seed derivation (rather than, say, hash()):
  - Python's built-in hash() is non-deterministic across processes
    (PYTHONHASHSEED randomisation) and across Python versions.
  - SHA-256 is deterministic across all platforms, Python versions, and
    operating systems, so derived seeds are stable in CI and on different
    developers' machines.
  - Truncating to 4 bytes gives a 32-bit integer which is the standard
    seed type accepted by random.Random, numpy, and torch.
"""

from __future__ import annotations

import hashlib
import random


def make_rng(seed: int) -> random.Random:
    """
    Create an isolated Random instance that does NOT affect global state.

    Why isolated: using random.Random(seed) creates an independent PRNG
    with its own internal state, completely separate from the module-level
    random.random() sequence. This is critical because third-party libraries
    (e.g. HuggingFace datasets, sklearn) may draw from the global RNG, and
    we don't want their calls to affect our experiment's random draws or
    vice versa.

    Args:
        seed: Integer seed. Same seed always produces the same sequence.

    Returns:
        An independent random.Random instance, seeded and ready to use.
    """
    return random.Random(seed)


def derive_seed(parent_seed: int, *context: str | int) -> int:
    """
    Derive a deterministic child seed from a parent seed and string context.

    Useful for producing per-document or per-attack seeds from a single
    experiment-level seed without collision between different (document, attack)
    pairs. The derived seed is fully determined by its inputs, so it is
    stable across runs.

    Why hashing instead of simple arithmetic (e.g. parent + hash(context)):
      - Simple arithmetic can produce seed collisions for different inputs.
      - SHA-256 provides uniform distribution and collision resistance.
      - The colon-separated key format ensures that ("ab", "c") and ("a", "bc")
        produce different hashes, avoiding context concatenation ambiguities.

    Example:
        # Each (claim, attack) pair gets a unique, reproducible seed.
        doc_seed = derive_seed(experiment_seed, claim_id, attack_name)

    Args:
        parent_seed: The top-level experiment seed (integer).
        *context:    Additional string or integer context values that make
                     the derived seed unique (e.g. claim_id, attack_name).

    Returns:
        A 32-bit unsigned integer suitable for seeding any standard RNG.
    """
    # Assemble a unique string key from the parent seed and all context pieces.
    # Colon separators prevent "1:23" == "12:3" ambiguity.
    key = f"{parent_seed}:" + ":".join(str(c) for c in context)
    digest = hashlib.sha256(key.encode()).digest()
    # Use first 4 bytes as a 32-bit unsigned integer seed.
    # "big" byte order is conventional and consistent across platforms.
    return int.from_bytes(digest[:4], "big")


def seed_everything(seed: int) -> None:
    """
    Seed all relevant global RNGs for full reproducibility.

    Call once at the start of an experiment run, before any stochastic
    operations. This function seeds:
      - Python's random module (always)
      - numpy (if installed)
      - PyTorch (if installed, including CUDA seeds and cuDNN determinism)

    Why seed global RNGs at all, given make_rng() for local operations:
      Third-party libraries (sentence-transformers, RAGAS, HuggingFace
      tokenizers) draw from the global RNG or from numpy/torch global state
      internally. seeding them globally is the only way to make those
      library calls reproducible without patching each library.

    Why numpy and torch are optional (try/except ImportError):
      EIGER supports environments without GPU frameworks. Tests and minimal
      installs should not require torch or numpy to be present. The try/except
      pattern means seeding degrades gracefully rather than raising ImportError.

    Args:
        seed: Integer seed to apply to all global RNGs.
    """
    # Always seed the stdlib random module — this affects random.random(),
    # random.choice(), random.shuffle(), etc. used by framework code.
    random.seed(seed)

    # Seed numpy if present. numpy.random.seed() affects both the legacy
    # RandomState API and (via legacy compatibility) some HuggingFace code.
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        # numpy is not installed — skip silently.
        pass

    # Seed PyTorch if present. torch.cuda.manual_seed_all() seeds all GPU
    # devices, not just the current one, which matters in multi-GPU setups.
    # cuDNN determinism settings prevent non-deterministic algorithm selection
    # in convolution operations.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # Seed all CUDA devices, not just the default one.
            torch.cuda.manual_seed_all(seed)
        # deterministic=True forces cuDNN to use deterministic algorithms;
        # benchmark=False disables auto-tuning which would pick different
        # algorithms on different runs based on hardware timing.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is not installed — skip silently.
        pass
