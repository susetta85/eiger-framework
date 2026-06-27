"""
EIGER — Epistemic Integrity Benchmark for RAG systems.

This is the top-level package for the EIGER framework. It exposes
only version metadata at this level; all public domain objects are
imported from sub-packages (eiger.core, eiger.config, eiger.utils, etc.).

What this file does NOT do:
  - It does not import any heavy dependencies at import time (no models,
    no torch, no sentence-transformers). This keeps `import eiger` fast.
  - It does not configure logging or settings; callers are responsible
    for calling configure_logging() and get_settings() explicitly.
"""

# Semantic version following PEP 440 / SemVer conventions.
# Kept in __init__.py so tools like importlib.metadata and
# pip editable installs can discover it without importing sub-modules.
__version__ = "0.1.0"

# Identifies the team in package metadata (pyproject.toml mirrors this).
__author__ = "EIGER Research Team"
