"""Pure JAX finite-volume numerics for the two-dimensional shallow-water equations.

The implementation uses an HLL approximate Riemann solver and Audusse-style
hydrostatic reconstruction. The paired face-source corrections preserve a
lake at rest over variable bathymetry, while a positivity/dry-state filter
supports wetting and drying. All integration is expressed through
``jax.lax.scan``; no Python loop advances physical time.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hybrid_flood.jax_solver.boundary_conditions import apply_ghost_cells


class SWEState(NamedTuple):
    """Cell-average conserved variables."""

    h: jnp.ndarray
    hu: jnp.ndarray
    hv: jnp.ndarray


class SWEParams(NamedTuple):
    """Static fields and scalar controls consumed by the pure step function."""

    bed: jnp.ndarray
    manning_n: jnp.ndarray
    domain_mask: jnp.ndarray
    rainfall_multiplier: jnp.ndarray
    boundary_types: jnp.ndarray
    rainfall_times_s: jnp.ndarray
    rainfall_rates_m_s: jnp.ndarray
    default_rainfall_m_s: jnp.ndarray
    dx: jnp.ndarray
    dy: jnp.ndarray
    gravity: jnp.ndarray
    cfl: jnp.ndarray
    dry_tolerance: jnp.ndarray
    max_dt: jnp.ndarray


class ScanResult(NamedTuple):
    """Fixed-output simulation result returned by the compiled scan."""

    states: SWEState
    output_times_s: jnp.ndarray
    final_time_s: jnp.ndarray
    outputs_written: jnp.ndarray
    steps_used: jnp.ndarray


def _safe_velocity(momentum: jnp.ndarray, depth: jnp.ndarray, dry: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(depth > dry, momentum / jnp.maximum(depth, dry), 0.0)


def _hll_x(
    h_left: jnp.ndarray,
    hu_left: jnp.ndarray,
    hv_left: jnp.ndarray,
    h_right: jnp.ndarray,
    hu_right: jnp.ndarray,
    hv_right: jnp.ndarray,
    gravity: jnp.ndarray,
    dry: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    u_left = _safe_velocity(hu_left, h_left, dry)
    v_left = _safe_velocity(hv_left, h_left, dry)
    u_right = _safe_velocity(hu_right, h_right, dry)
    v_right = _safe_velocity(hv_right, h_right, dry)
    c_left = jnp.sqrt(gravity * jnp.maximum(h_left, 0.0))
    c_right = jnp.sqrt(gravity * jnp.maximum(h_right, 0.0))
    speed_left = jnp.minimum(u_left - c_left, u_right - c_right)
    speed_right = jnp.maximum(u_left + c_left, u_right + c_right)

    flux_left = jnp.stack(
        (
            hu_left,
            hu_left * u_left + 0.5 * gravity * h_left**2,
            hu_left * v_left,
        )
    )
    flux_right = jnp.stack(
        (
            hu_right,
            hu_right * u_right + 0.5 * gravity * h_right**2,
            hu_right * v_right,
        )
    )
    state_left = jnp.stack((h_left, hu_left, hv_left))
    state_right = jnp.stack((h_right, hu_right, hv_right))
    denominator = jnp.where(
        speed_right - speed_left > 1.0e-12,
        speed_right - speed_left,
        1.0,
    )
    middle = (
        speed_right[None] * flux_left
        - speed_left[None] * flux_right
        + (speed_left * speed_right)[None] * (state_right - state_left)
    ) / denominator[None]
    flux = jnp.where(
        (speed_left >= 0.0)[None],
        flux_left,
        jnp.where((speed_right <= 0.0)[None], flux_right, middle),
    )
    return flux[0], flux[1], flux[2]


def _hll_y(
    h_bottom: jnp.ndarray,
    hu_bottom: jnp.ndarray,
    hv_bottom: jnp.ndarray,
    h_top: jnp.ndarray,
    hu_top: jnp.ndarray,
    hv_top: jnp.ndarray,
    gravity: jnp.ndarray,
    dry: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    v_bottom = _safe_velocity(hv_bottom, h_bottom, dry)
    v_top = _safe_velocity(hv_top, h_top, dry)
    c_bottom = jnp.sqrt(gravity * jnp.maximum(h_bottom, 0.0))
    c_top = jnp.sqrt(gravity * jnp.maximum(h_top, 0.0))
    speed_bottom = jnp.minimum(v_bottom - c_bottom, v_top - c_top)
    speed_top = jnp.maximum(v_bottom + c_bottom, v_top + c_top)

    flux_bottom = jnp.stack(
        (
            hv_bottom,
            hu_bottom * v_bottom,
            hv_bottom * v_bottom + 0.5 * gravity * h_bottom**2,
        )
    )
    flux_top = jnp.stack(
        (
            hv_top,
            hu_top * v_top,
            hv_top * v_top + 0.5 * gravity * h_top**2,
        )
    )
    state_bottom = jnp.stack((h_bottom, hu_bottom, hv_bottom))
    state_top = jnp.stack((h_top, hu_top, hv_top))
    denominator = jnp.where(speed_top - speed_bottom > 1.0e-12, speed_top - speed_bottom, 1.0)
    middle = (
        speed_top[None] * flux_bottom
        - speed_bottom[None] * flux_top
        + (speed_bottom * speed_top)[None] * (state_top - state_bottom)
    ) / denominator[None]
    flux = jnp.where(
        (speed_bottom >= 0.0)[None],
        flux_bottom,
        jnp.where((speed_top <= 0.0)[None], flux_top, middle),
    )
    return flux[0], flux[1], flux[2]


def _x_interface_fluxes(
    h: jnp.ndarray,
    hu: jnp.ndarray,
    hv: jnp.ndarray,
    bed: jnp.ndarray,
    active: jnp.ndarray,
    gravity: jnp.ndarray,
    dry: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    h_left, h_right = h[:, :-1], h[:, 1:]
    hu_left, hu_right = hu[:, :-1], hu[:, 1:]
    hv_left, hv_right = hv[:, :-1], hv[:, 1:]
    bed_left, bed_right = bed[:, :-1], bed[:, 1:]
    active_left, active_right = active[:, :-1], active[:, 1:]

    # An active/inactive interface is a solid wall on the rasterized domain
    # perimeter. Mirror the active state and reverse normal momentum so the
    # HLL flux retains the hydrostatic pressure force while cancelling mass
    # flux. Simply zeroing the whole face flux creates a false pressure
    # imbalance and destroys lake-at-rest well-balancedness near irregular
    # masks.
    left_wall = active_left & ~active_right
    right_wall = ~active_left & active_right
    h_right = jnp.where(left_wall, h_left, h_right)
    hu_right = jnp.where(left_wall, -hu_left, hu_right)
    hv_right = jnp.where(left_wall, hv_left, hv_right)
    bed_right = jnp.where(left_wall, bed_left, bed_right)
    h_left = jnp.where(right_wall, h_right, h_left)
    hu_left = jnp.where(right_wall, -hu_right, hu_left)
    hv_left = jnp.where(right_wall, hv_right, hv_left)
    bed_left = jnp.where(right_wall, bed_right, bed_left)

    eta_left, eta_right = h_left + bed_left, h_right + bed_right
    bed_face = jnp.maximum(bed_left, bed_right)
    reconstructed_left = jnp.maximum(eta_left - bed_face, 0.0)
    reconstructed_right = jnp.maximum(eta_right - bed_face, 0.0)
    u_left = _safe_velocity(hu_left, h_left, dry)
    v_left = _safe_velocity(hv_left, h_left, dry)
    u_right = _safe_velocity(hu_right, h_right, dry)
    v_right = _safe_velocity(hv_right, h_right, dry)
    flux = jnp.stack(
        _hll_x(
            reconstructed_left,
            reconstructed_left * u_left,
            reconstructed_left * v_left,
            reconstructed_right,
            reconstructed_right * u_right,
            reconstructed_right * v_right,
            gravity,
            dry,
        )
    )
    face_relevant = active_left | active_right
    flux = jnp.where(face_relevant[None], flux, 0.0)
    correction_left = (
        jnp.zeros_like(flux).at[1].set(0.5 * gravity * (h_left**2 - reconstructed_left**2))
    )
    correction_right = (
        jnp.zeros_like(flux).at[1].set(0.5 * gravity * (h_right**2 - reconstructed_right**2))
    )
    return flux + jnp.where(face_relevant[None], correction_left, 0.0), flux + jnp.where(
        face_relevant[None], correction_right, 0.0
    )


def _y_interface_fluxes(
    h: jnp.ndarray,
    hu: jnp.ndarray,
    hv: jnp.ndarray,
    bed: jnp.ndarray,
    active: jnp.ndarray,
    gravity: jnp.ndarray,
    dry: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    h_bottom, h_top = h[:-1, :], h[1:, :]
    hu_bottom, hu_top = hu[:-1, :], hu[1:, :]
    hv_bottom, hv_top = hv[:-1, :], hv[1:, :]
    bed_bottom, bed_top = bed[:-1, :], bed[1:, :]
    active_bottom, active_top = active[:-1, :], active[1:, :]

    bottom_wall = active_bottom & ~active_top
    top_wall = ~active_bottom & active_top
    h_top = jnp.where(bottom_wall, h_bottom, h_top)
    hu_top = jnp.where(bottom_wall, hu_bottom, hu_top)
    hv_top = jnp.where(bottom_wall, -hv_bottom, hv_top)
    bed_top = jnp.where(bottom_wall, bed_bottom, bed_top)
    h_bottom = jnp.where(top_wall, h_top, h_bottom)
    hu_bottom = jnp.where(top_wall, hu_top, hu_bottom)
    hv_bottom = jnp.where(top_wall, -hv_top, hv_bottom)
    bed_bottom = jnp.where(top_wall, bed_top, bed_bottom)

    eta_bottom, eta_top = h_bottom + bed_bottom, h_top + bed_top
    bed_face = jnp.maximum(bed_bottom, bed_top)
    reconstructed_bottom = jnp.maximum(eta_bottom - bed_face, 0.0)
    reconstructed_top = jnp.maximum(eta_top - bed_face, 0.0)
    u_bottom = _safe_velocity(hu_bottom, h_bottom, dry)
    v_bottom = _safe_velocity(hv_bottom, h_bottom, dry)
    u_top = _safe_velocity(hu_top, h_top, dry)
    v_top = _safe_velocity(hv_top, h_top, dry)
    flux = jnp.stack(
        _hll_y(
            reconstructed_bottom,
            reconstructed_bottom * u_bottom,
            reconstructed_bottom * v_bottom,
            reconstructed_top,
            reconstructed_top * u_top,
            reconstructed_top * v_top,
            gravity,
            dry,
        )
    )
    face_relevant = active_bottom | active_top
    flux = jnp.where(face_relevant[None], flux, 0.0)
    correction_bottom = (
        jnp.zeros_like(flux).at[2].set(0.5 * gravity * (h_bottom**2 - reconstructed_bottom**2))
    )
    correction_top = (
        jnp.zeros_like(flux).at[2].set(0.5 * gravity * (h_top**2 - reconstructed_top**2))
    )
    return flux + jnp.where(face_relevant[None], correction_bottom, 0.0), flux + jnp.where(
        face_relevant[None], correction_top, 0.0
    )


def rainfall_rate(params: SWEParams, time_s: jnp.ndarray) -> jnp.ndarray:
    """Linearly interpolate the uniform rainfall forcing in metres per second."""
    return jnp.interp(
        time_s,
        params.rainfall_times_s,
        params.rainfall_rates_m_s,
        left=params.default_rainfall_m_s,
        right=params.default_rainfall_m_s,
    )


def compute_cfl_dt(state: SWEState, params: SWEParams) -> jnp.ndarray:
    """Return an unsplit two-dimensional CFL timestep."""
    u = _safe_velocity(state.hu, state.h, params.dry_tolerance)
    v = _safe_velocity(state.hv, state.h, params.dry_tolerance)
    wave_speed = jnp.sqrt(params.gravity * jnp.maximum(state.h, 0.0))
    inverse_dt = (jnp.abs(u) + wave_speed) / params.dx + (jnp.abs(v) + wave_speed) / params.dy
    inverse_dt = jnp.where(params.domain_mask, inverse_dt, 0.0)
    maximum_inverse_dt = jnp.max(inverse_dt)
    return jnp.where(
        maximum_inverse_dt > 0.0,
        jnp.minimum(params.cfl / maximum_inverse_dt, params.max_dt),
        params.max_dt,
    )


def finite_volume_step(
    state: SWEState,
    params: SWEParams,
    dt: jnp.ndarray,
    time_s: jnp.ndarray,
) -> SWEState:
    """Advance one conservative, well-balanced HLL step as a pure function."""
    h_ghost, hu_ghost, hv_ghost = apply_ghost_cells(
        state.h,
        state.hu,
        state.hv,
        params.boundary_types,
    )
    bed_ghost = jnp.pad(params.bed, ((1, 1), (1, 1)), mode="edge")
    mask_ghost = jnp.pad(params.domain_mask, ((1, 1), (1, 1)), mode="edge")

    x_left, x_right = _x_interface_fluxes(
        h_ghost[1:-1],
        hu_ghost[1:-1],
        hv_ghost[1:-1],
        bed_ghost[1:-1],
        mask_ghost[1:-1],
        params.gravity,
        params.dry_tolerance,
    )
    y_bottom, y_top = _y_interface_fluxes(
        h_ghost[:, 1:-1],
        hu_ghost[:, 1:-1],
        hv_ghost[:, 1:-1],
        bed_ghost[:, 1:-1],
        mask_ghost[:, 1:-1],
        params.gravity,
        params.dry_tolerance,
    )

    state_array = jnp.stack((state.h, state.hu, state.hv))
    divergence_x = (x_left[:, :, 1:] - x_right[:, :, :-1]) / params.dx
    divergence_y = (y_bottom[:, 1:, :] - y_top[:, :-1, :]) / params.dy
    updated = state_array - dt * (divergence_x + divergence_y)
    updated = updated.at[0].add(dt * rainfall_rate(params, time_s) * params.rainfall_multiplier)

    h = jnp.maximum(updated[0], 0.0)
    hu, hv = updated[1], updated[2]
    momentum_norm = jnp.sqrt(hu**2 + hv**2)
    friction_denominator = 1.0 + (
        dt
        * params.gravity
        * params.manning_n**2
        * momentum_norm
        / jnp.maximum(h, params.dry_tolerance) ** (7.0 / 3.0)
    )
    hu = hu / friction_denominator
    hv = hv / friction_denominator
    wet = params.domain_mask & (h > params.dry_tolerance)
    h = jnp.where(params.domain_mask, h, 0.0)
    hu = jnp.where(wet, hu, 0.0)
    hv = jnp.where(wet, hv, 0.0)
    return SWEState(h, hu, hv)


@partial(jax.jit, static_argnames=("max_steps",))
def integrate_to_outputs(
    initial_state: SWEState,
    params: SWEParams,
    output_times_s: jnp.ndarray,
    *,
    max_steps: int,
) -> ScanResult:
    """Integrate adaptively to fixed output times using one compiled ``lax.scan``."""
    output_h = jnp.zeros((output_times_s.shape[0], *initial_state.h.shape), initial_state.h.dtype)
    output_hu = jnp.zeros_like(output_h)
    output_hv = jnp.zeros_like(output_h)
    output_h = output_h.at[0].set(initial_state.h)
    output_hu = output_hu.at[0].set(initial_state.hu)
    output_hv = output_hv.at[0].set(initial_state.hv)

    carry = (
        initial_state,
        output_times_s[0],
        jnp.array(1, dtype=jnp.int32),
        SWEState(output_h, output_hu, output_hv),
        jnp.array(0, dtype=jnp.int32),
    )

    def scan_body(carry_value, _):
        state, time_s, output_index, outputs, steps_used = carry_value
        active = output_index < output_times_s.shape[0]

        def advance(active_carry):
            active_state, active_time, active_index, active_outputs, active_steps = active_carry
            safe_index = jnp.minimum(active_index, output_times_s.shape[0] - 1)
            target_time = output_times_s[safe_index]
            dt = jnp.minimum(compute_cfl_dt(active_state, params), target_time - active_time)
            next_state = finite_volume_step(active_state, params, dt, active_time)
            next_time = active_time + dt
            reached = next_time >= target_time - 1.0e-7

            def record(output_state):
                return SWEState(
                    output_state.h.at[safe_index].set(next_state.h),
                    output_state.hu.at[safe_index].set(next_state.hu),
                    output_state.hv.at[safe_index].set(next_state.hv),
                )

            next_outputs = jax.lax.cond(reached, record, lambda value: value, active_outputs)
            next_index = active_index + reached.astype(jnp.int32)
            return next_state, next_time, next_index, next_outputs, active_steps + 1

        next_carry = jax.lax.cond(active, advance, lambda value: value, carry_value)
        return next_carry, jnp.array(0, dtype=jnp.int8)

    final_carry, _ = jax.lax.scan(scan_body, carry, xs=None, length=max_steps)
    final_state, final_time, outputs_written, output_states, steps_used = final_carry
    del final_state
    return ScanResult(
        states=output_states,
        output_times_s=output_times_s,
        final_time_s=final_time,
        outputs_written=outputs_written,
        steps_used=steps_used,
    )
