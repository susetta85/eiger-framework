"""
Configuration management — settings, environment variables, YAML loading.

This package provides the single entry point for all runtime configuration.
Callers should import get_settings() and EigerSettings from here rather
than from the sub-module directly, so that the import path remains stable
if the internal module structure is reorganized.

Usage:
    from eiger.config import get_settings
    settings = get_settings()
    print(settings.qdrant_url)
"""

# Re-export the two public symbols from the settings sub-module.
# get_settings() is the primary API; EigerSettings is exported for
# type annotations and for tests that need to construct custom instances.
from eiger.config.settings import EigerSettings, get_settings

# Explicit __all__ ensures that `from eiger.config import *` only
# surfaces the intended public interface, not internal helpers.
__all__ = ["EigerSettings", "get_settings"]
