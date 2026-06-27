"""
Reproducible seeding utilities.

Rules enforced here:
  - Never modify the global random.random() state.
  - Every stochastic operation receives an explicit seed.
  - Seed derivation is deterministic: child seeds are hashed from parent + context.
"""

from __future__ import annotations

import hashlib
import random


def make_rng(seed: int) -> random.Random:
    """
    Create an isolated Random instance that does NOT affect global state.

    Args:
        seed: Integer seed. Same seed always produces the same sequence.

    Returns:
        An independent random.Random instance.
    """
    return random.Random(seed)


def derive_seed(parent_seed: int, *context: str | int) -> int:
    """
    Derive a deterministic child seed from a parent seed and string context.

    Useful for producing per-document or per-attack seeds from an
    experiment-level seed without collision.

    Example:
        doc_seed = derive_seed(experiment_seed, claim_id, attack_name)
    """
    key = f"{parent_seed}:" + ":".join(str(c) for c in context)
    digest = hashlib.sha256(key.encode()).digest()
    # Use first 4 bytes as a 32-bit unsigned integer seed
    return int.from_bytes(digest[:4], "big")


def seed_everything(seed: int) -> None:
    """
    Seed all relevant global RNGs for full reproducibility.

    Call once at the start of an experiment run. Covers:
      - Python's random module
      - numpy (if available)
      - torch (if available)
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
