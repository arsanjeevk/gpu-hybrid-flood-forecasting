"""Pure JAX ghost-cell boundary handling for structured SWE grids."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

REFLECTIVE = 0
OPEN = 1


class BoundaryConfig(NamedTuple):
    """Boundary codes ordered as left, right, bottom, top."""

    types: jnp.ndarray


def all_reflective() -> BoundaryConfig:
    """Return the policy currently matching the ANUGA baseline."""
    return BoundaryConfig(jnp.full((4,), REFLECTIVE, dtype=jnp.int32))


def all_open() -> BoundaryConfig:
    """Return zero-gradient transmissive boundaries on every side."""
    return BoundaryConfig(jnp.full((4,), OPEN, dtype=jnp.int32))


def apply_ghost_cells(
    h: jnp.ndarray,
    hu: jnp.ndarray,
    hv: jnp.ndarray,
    boundary_types: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pad one ghost layer and reflect only the normal momentum component."""
    h_ghost = jnp.pad(h, ((1, 1), (1, 1)), mode="edge")
    hu_ghost = jnp.pad(hu, ((1, 1), (1, 1)), mode="edge")
    hv_ghost = jnp.pad(hv, ((1, 1), (1, 1)), mode="edge")

    hu_ghost = hu_ghost.at[:, 0].set(
        jnp.where(boundary_types[0] == REFLECTIVE, -hu_ghost[:, 1], hu_ghost[:, 1])
    )
    hu_ghost = hu_ghost.at[:, -1].set(
        jnp.where(boundary_types[1] == REFLECTIVE, -hu_ghost[:, -2], hu_ghost[:, -2])
    )
    hv_ghost = hv_ghost.at[0, :].set(
        jnp.where(boundary_types[2] == REFLECTIVE, -hv_ghost[1, :], hv_ghost[1, :])
    )
    hv_ghost = hv_ghost.at[-1, :].set(
        jnp.where(boundary_types[3] == REFLECTIVE, -hv_ghost[-2, :], hv_ghost[-2, :])
    )
    return h_ghost, hu_ghost, hv_ghost
