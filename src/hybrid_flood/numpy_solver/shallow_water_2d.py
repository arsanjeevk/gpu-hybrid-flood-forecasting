"""NumPy baseline matching the JAX structured shallow-water discretization.

This is the V1 reference implementation of the fast structured forecast. It
uses the same HLL flux, hydrostatic reconstruction, Manning friction,
wetting/drying threshold, rainfall interpolation, CFL rule, grid, and
all-reflective boundary policy as V2. Only the array execution backend and
time-loop implementation differ. ANUGA remains the independent unstructured
hydrodynamic reference for both versions.

The explicit Python time loop is intentional: V1 represents conventional
NumPy/SciPy execution against which the compiled ``jax.lax.scan`` V2 is
measured. No JAX package is imported by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class NumpyState(NamedTuple):
    """Cell-average conserved variables ``(h, hu, hv)``."""

    h: np.ndarray
    hu: np.ndarray
    hv: np.ndarray


class NumpyResult(NamedTuple):
    """States recorded at fixed output times."""

    states: NumpyState
    output_times_s: np.ndarray
    final_time_s: float
    outputs_written: int
    steps_used: int


@dataclass(frozen=True)
class NumpyParams:
    """Static arrays and scalar numerical controls."""

    bed: np.ndarray
    manning_n: np.ndarray
    domain_mask: np.ndarray
    rainfall_multiplier: np.ndarray
    rainfall_times_s: np.ndarray
    rainfall_rates_m_s: np.ndarray
    default_rainfall_m_s: float
    dx: float
    dy: float
    gravity: float
    cfl: float
    dry_tolerance: float
    max_dt: float


def _safe_velocity(momentum: np.ndarray, depth: np.ndarray, dry: float) -> np.ndarray:
    result = np.zeros_like(momentum)
    np.divide(momentum, np.maximum(depth, dry), out=result, where=depth > dry)
    return result


def _hll_x(
    h_left: np.ndarray,
    hu_left: np.ndarray,
    hv_left: np.ndarray,
    h_right: np.ndarray,
    hu_right: np.ndarray,
    hv_right: np.ndarray,
    gravity: float,
    dry: float,
) -> np.ndarray:
    u_left = _safe_velocity(hu_left, h_left, dry)
    v_left = _safe_velocity(hv_left, h_left, dry)
    u_right = _safe_velocity(hu_right, h_right, dry)
    v_right = _safe_velocity(hv_right, h_right, dry)
    c_left = np.sqrt(gravity * np.maximum(h_left, 0.0))
    c_right = np.sqrt(gravity * np.maximum(h_right, 0.0))
    speed_left = np.minimum(u_left - c_left, u_right - c_right)
    speed_right = np.maximum(u_left + c_left, u_right + c_right)
    flux_left = np.stack((hu_left, hu_left * u_left + 0.5 * gravity * h_left**2, hu_left * v_left))
    flux_right = np.stack(
        (hu_right, hu_right * u_right + 0.5 * gravity * h_right**2, hu_right * v_right)
    )
    state_left = np.stack((h_left, hu_left, hv_left))
    state_right = np.stack((h_right, hu_right, hv_right))
    denominator = np.where(speed_right - speed_left > 1.0e-12, speed_right - speed_left, 1.0)
    middle = (
        speed_right[None] * flux_left
        - speed_left[None] * flux_right
        + (speed_left * speed_right)[None] * (state_right - state_left)
    ) / denominator[None]
    return np.where(
        (speed_left >= 0.0)[None],
        flux_left,
        np.where((speed_right <= 0.0)[None], flux_right, middle),
    )


def _hll_y(
    h_bottom: np.ndarray,
    hu_bottom: np.ndarray,
    hv_bottom: np.ndarray,
    h_top: np.ndarray,
    hu_top: np.ndarray,
    hv_top: np.ndarray,
    gravity: float,
    dry: float,
) -> np.ndarray:
    v_bottom = _safe_velocity(hv_bottom, h_bottom, dry)
    v_top = _safe_velocity(hv_top, h_top, dry)
    c_bottom = np.sqrt(gravity * np.maximum(h_bottom, 0.0))
    c_top = np.sqrt(gravity * np.maximum(h_top, 0.0))
    speed_bottom = np.minimum(v_bottom - c_bottom, v_top - c_top)
    speed_top = np.maximum(v_bottom + c_bottom, v_top + c_top)
    flux_bottom = np.stack(
        (hv_bottom, hu_bottom * v_bottom, hv_bottom * v_bottom + 0.5 * gravity * h_bottom**2)
    )
    flux_top = np.stack((hv_top, hu_top * v_top, hv_top * v_top + 0.5 * gravity * h_top**2))
    state_bottom = np.stack((h_bottom, hu_bottom, hv_bottom))
    state_top = np.stack((h_top, hu_top, hv_top))
    denominator = np.where(speed_top - speed_bottom > 1.0e-12, speed_top - speed_bottom, 1.0)
    middle = (
        speed_top[None] * flux_bottom
        - speed_bottom[None] * flux_top
        + (speed_bottom * speed_top)[None] * (state_top - state_bottom)
    ) / denominator[None]
    return np.where(
        (speed_bottom >= 0.0)[None],
        flux_bottom,
        np.where((speed_top <= 0.0)[None], flux_top, middle),
    )


def _x_interface_fluxes(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    bed: np.ndarray,
    active: np.ndarray,
    gravity: float,
    dry: float,
) -> tuple[np.ndarray, np.ndarray]:
    h_left, h_right = h[:, :-1].copy(), h[:, 1:].copy()
    hu_left, hu_right = hu[:, :-1].copy(), hu[:, 1:].copy()
    hv_left, hv_right = hv[:, :-1].copy(), hv[:, 1:].copy()
    bed_left, bed_right = bed[:, :-1].copy(), bed[:, 1:].copy()
    active_left, active_right = active[:, :-1], active[:, 1:]
    left_wall = active_left & ~active_right
    right_wall = ~active_left & active_right
    h_right[left_wall] = h_left[left_wall]
    hu_right[left_wall] = -hu_left[left_wall]
    hv_right[left_wall] = hv_left[left_wall]
    bed_right[left_wall] = bed_left[left_wall]
    h_left[right_wall] = h_right[right_wall]
    hu_left[right_wall] = -hu_right[right_wall]
    hv_left[right_wall] = hv_right[right_wall]
    bed_left[right_wall] = bed_right[right_wall]
    bed_face = np.maximum(bed_left, bed_right)
    reconstructed_left = np.maximum(h_left + bed_left - bed_face, 0.0)
    reconstructed_right = np.maximum(h_right + bed_right - bed_face, 0.0)
    u_left = _safe_velocity(hu_left, h_left, dry)
    v_left = _safe_velocity(hv_left, h_left, dry)
    u_right = _safe_velocity(hu_right, h_right, dry)
    v_right = _safe_velocity(hv_right, h_right, dry)
    flux = _hll_x(
        reconstructed_left,
        reconstructed_left * u_left,
        reconstructed_left * v_left,
        reconstructed_right,
        reconstructed_right * u_right,
        reconstructed_right * v_right,
        gravity,
        dry,
    )
    relevant = active_left | active_right
    flux = np.where(relevant[None], flux, 0.0)
    correction_left = np.zeros_like(flux)
    correction_right = np.zeros_like(flux)
    correction_left[1] = 0.5 * gravity * (h_left**2 - reconstructed_left**2)
    correction_right[1] = 0.5 * gravity * (h_right**2 - reconstructed_right**2)
    return flux + np.where(relevant[None], correction_left, 0.0), flux + np.where(
        relevant[None], correction_right, 0.0
    )


def _y_interface_fluxes(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    bed: np.ndarray,
    active: np.ndarray,
    gravity: float,
    dry: float,
) -> tuple[np.ndarray, np.ndarray]:
    h_bottom, h_top = h[:-1].copy(), h[1:].copy()
    hu_bottom, hu_top = hu[:-1].copy(), hu[1:].copy()
    hv_bottom, hv_top = hv[:-1].copy(), hv[1:].copy()
    bed_bottom, bed_top = bed[:-1].copy(), bed[1:].copy()
    active_bottom, active_top = active[:-1], active[1:]
    bottom_wall = active_bottom & ~active_top
    top_wall = ~active_bottom & active_top
    h_top[bottom_wall] = h_bottom[bottom_wall]
    hu_top[bottom_wall] = hu_bottom[bottom_wall]
    hv_top[bottom_wall] = -hv_bottom[bottom_wall]
    bed_top[bottom_wall] = bed_bottom[bottom_wall]
    h_bottom[top_wall] = h_top[top_wall]
    hu_bottom[top_wall] = hu_top[top_wall]
    hv_bottom[top_wall] = -hv_top[top_wall]
    bed_bottom[top_wall] = bed_top[top_wall]
    bed_face = np.maximum(bed_bottom, bed_top)
    reconstructed_bottom = np.maximum(h_bottom + bed_bottom - bed_face, 0.0)
    reconstructed_top = np.maximum(h_top + bed_top - bed_face, 0.0)
    u_bottom = _safe_velocity(hu_bottom, h_bottom, dry)
    v_bottom = _safe_velocity(hv_bottom, h_bottom, dry)
    u_top = _safe_velocity(hu_top, h_top, dry)
    v_top = _safe_velocity(hv_top, h_top, dry)
    flux = _hll_y(
        reconstructed_bottom,
        reconstructed_bottom * u_bottom,
        reconstructed_bottom * v_bottom,
        reconstructed_top,
        reconstructed_top * u_top,
        reconstructed_top * v_top,
        gravity,
        dry,
    )
    relevant = active_bottom | active_top
    flux = np.where(relevant[None], flux, 0.0)
    correction_bottom = np.zeros_like(flux)
    correction_top = np.zeros_like(flux)
    correction_bottom[2] = 0.5 * gravity * (h_bottom**2 - reconstructed_bottom**2)
    correction_top[2] = 0.5 * gravity * (h_top**2 - reconstructed_top**2)
    return flux + np.where(relevant[None], correction_bottom, 0.0), flux + np.where(
        relevant[None], correction_top, 0.0
    )


def apply_reflective_ghost_cells(state: NumpyState) -> NumpyState:
    """Add one ghost layer and reverse the normal wall momentum."""
    h = np.pad(state.h, 1, mode="edge")
    hu = np.pad(state.hu, 1, mode="edge")
    hv = np.pad(state.hv, 1, mode="edge")
    hu[:, 0] = -hu[:, 1]
    hu[:, -1] = -hu[:, -2]
    hv[0] = -hv[1]
    hv[-1] = -hv[-2]
    return NumpyState(h, hu, hv)


def rainfall_rate(params: NumpyParams, time_s: float) -> float:
    """Linearly interpolate uniform rainfall in metres per second."""
    return float(
        np.interp(
            time_s,
            params.rainfall_times_s,
            params.rainfall_rates_m_s,
            left=params.default_rainfall_m_s,
            right=params.default_rainfall_m_s,
        )
    )


def compute_cfl_dt(state: NumpyState, params: NumpyParams) -> float:
    """Return the same unsplit two-dimensional CFL step used by V2."""
    u = _safe_velocity(state.hu, state.h, params.dry_tolerance)
    v = _safe_velocity(state.hv, state.h, params.dry_tolerance)
    wave = np.sqrt(params.gravity * np.maximum(state.h, 0.0))
    inverse = (np.abs(u) + wave) / params.dx + (np.abs(v) + wave) / params.dy
    maximum = float(np.max(np.where(params.domain_mask, inverse, 0.0)))
    return min(params.cfl / maximum, params.max_dt) if maximum > 0.0 else params.max_dt


def finite_volume_step(
    state: NumpyState,
    params: NumpyParams,
    dt: float,
    time_s: float,
) -> NumpyState:
    """Advance one conservative, well-balanced HLL step."""
    ghost = apply_reflective_ghost_cells(state)
    bed_ghost = np.pad(params.bed, 1, mode="edge")
    mask_ghost = np.pad(params.domain_mask, 1, mode="edge")
    x_left, x_right = _x_interface_fluxes(
        ghost.h[1:-1],
        ghost.hu[1:-1],
        ghost.hv[1:-1],
        bed_ghost[1:-1],
        mask_ghost[1:-1],
        params.gravity,
        params.dry_tolerance,
    )
    y_bottom, y_top = _y_interface_fluxes(
        ghost.h[:, 1:-1],
        ghost.hu[:, 1:-1],
        ghost.hv[:, 1:-1],
        bed_ghost[:, 1:-1],
        mask_ghost[:, 1:-1],
        params.gravity,
        params.dry_tolerance,
    )
    state_array = np.stack(state)
    divergence_x = (x_left[:, :, 1:] - x_right[:, :, :-1]) / params.dx
    divergence_y = (y_bottom[:, 1:] - y_top[:, :-1]) / params.dy
    updated = state_array - dt * (divergence_x + divergence_y)
    updated[0] += dt * rainfall_rate(params, time_s) * params.rainfall_multiplier
    h = np.maximum(updated[0], 0.0)
    hu, hv = updated[1], updated[2]
    momentum_norm = np.hypot(hu, hv)
    denominator = 1.0 + (
        dt
        * params.gravity
        * params.manning_n**2
        * momentum_norm
        / np.maximum(h, params.dry_tolerance) ** (7.0 / 3.0)
    )
    hu = hu / denominator
    hv = hv / denominator
    wet = params.domain_mask & (h > params.dry_tolerance)
    return NumpyState(
        np.where(params.domain_mask, h, 0.0).astype(np.float32),
        np.where(wet, hu, 0.0).astype(np.float32),
        np.where(wet, hv, 0.0).astype(np.float32),
    )


class NumpyShallowWaterSolver:
    """V1 structured solver with no JAX dependency."""

    def __init__(
        self,
        bed: np.ndarray,
        manning_n: np.ndarray,
        domain_mask: np.ndarray,
        *,
        resolution_m: float,
        rainfall_times_s: np.ndarray,
        rainfall_rates_m_s: np.ndarray,
        default_rainfall_m_s: float = 0.0,
        gravity_m_s2: float = 9.81,
        cfl: float = 0.45,
        dry_tolerance_m: float = 1.0e-4,
        max_dt_s: float = 10.0,
    ) -> None:
        shape = np.asarray(bed).shape
        if np.asarray(manning_n).shape != shape or np.asarray(domain_mask).shape != shape:
            raise ValueError("Bed, roughness, and domain mask must have identical shapes.")
        self.params = NumpyParams(
            bed=np.asarray(bed, dtype=np.float32),
            manning_n=np.asarray(manning_n, dtype=np.float32),
            domain_mask=np.asarray(domain_mask, dtype=bool),
            rainfall_multiplier=np.asarray(domain_mask, dtype=np.float32),
            rainfall_times_s=np.asarray(rainfall_times_s, dtype=np.float32),
            rainfall_rates_m_s=np.asarray(rainfall_rates_m_s, dtype=np.float32),
            default_rainfall_m_s=float(default_rainfall_m_s),
            dx=float(resolution_m),
            dy=float(resolution_m),
            gravity=float(gravity_m_s2),
            cfl=float(cfl),
            dry_tolerance=float(dry_tolerance_m),
            max_dt=float(max_dt_s),
        )

    def initial_state(
        self,
        *,
        depth_m: float | np.ndarray = 0.0,
        x_velocity_m_s: float | np.ndarray = 0.0,
        y_velocity_m_s: float | np.ndarray = 0.0,
    ) -> NumpyState:
        """Create masked conserved arrays from primitive fields."""
        shape = self.params.bed.shape
        depth = np.broadcast_to(np.asarray(depth_m, dtype=np.float32), shape).copy()
        u = np.broadcast_to(np.asarray(x_velocity_m_s, dtype=np.float32), shape)
        v = np.broadcast_to(np.asarray(y_velocity_m_s, dtype=np.float32), shape)
        depth = np.where(self.params.domain_mask, np.maximum(depth, 0.0), 0.0)
        return NumpyState(depth, depth * u, depth * v)

    def run(
        self,
        initial_state: NumpyState,
        output_times_s: np.ndarray,
        *,
        max_steps: int = 20_000,
    ) -> NumpyResult:
        """Integrate adaptively using the explicit V1 Python loop."""
        times = np.asarray(output_times_s, dtype=np.float32)
        if times.ndim != 1 or not len(times) or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("Output times must be one-dimensional, start at zero, and increase.")
        shape = (len(times), *initial_state.h.shape)
        output_h = np.zeros(shape, dtype=np.float32)
        output_hu = np.zeros(shape, dtype=np.float32)
        output_hv = np.zeros(shape, dtype=np.float32)
        output_h[0], output_hu[0], output_hv[0] = initial_state
        state = initial_state
        time_s = float(times[0])
        output_index = 1
        steps = 0
        while output_index < len(times) and steps < max_steps:
            target = float(times[output_index])
            dt = min(compute_cfl_dt(state, self.params), target - time_s)
            state = finite_volume_step(state, self.params, dt, time_s)
            time_s += dt
            steps += 1
            if time_s >= target - 1.0e-7:
                output_h[output_index] = state.h
                output_hu[output_index] = state.hu
                output_hv[output_index] = state.hv
                output_index += 1
        if output_index != len(times):
            raise RuntimeError(
                f"max_steps={max_steps} wrote {output_index}/{len(times)} outputs; "
                f"last time={time_s:.3f} s."
            )
        return NumpyResult(
            NumpyState(output_h, output_hu, output_hv),
            times,
            time_s,
            output_index,
            steps,
        )
