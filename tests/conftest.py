"""Test-process defaults."""

from __future__ import annotations

import os

# Numerical unit tests must be deterministic and runnable on non-GPU CI hosts.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
