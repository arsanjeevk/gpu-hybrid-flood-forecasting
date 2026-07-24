"""Pure-JAX coupling of finite-volume physics and a learned residual corrector.

The network was trained on transitions between solver output times (currently
60 seconds), not on individual adaptive CFL substeps.  One outer scan step
therefore advances physics through as many internal substeps as needed to reach
the next correction time, predicts the learned one-step residual from the
pre-advance state and forcing, applies it to the advanced state, and feeds the
corrected conserved state back to the next outer step.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hybrid_flood.jax_solver.numerics import (
    SWEParams,
    SWEState,
    compute_cfl_dt,
    finite_volume_step,
    rainfall_rate,
)

M_S_TO_MM_HR = 3.6e6


class HybridInputs(NamedTuple):
    """Static normalized CNN inputs and training statistics."""

    terrain_standardized: jax.Array
    roughness_standardized: jax.Array
    state_mean: jax.Array
    state_std: jax.Array
    rainfall_scale_mm_hr: jax.Array
    residual_std: jax.Array


class CoupledRolloutResult(NamedTuple):
    """Corrected conserved states at requested output times."""

    states: SWEState
    output_times_s: jax.Array
    physics_steps_used: jax.Array


def _velocity(momentum: jax.Array, depth: jax.Array, dry: jax.Array) -> jax.Array:
    return jnp.where(depth > dry, momentum / jnp.maximum(depth, dry), 0.0)


def network_features(
    state: SWEState,
    params: SWEParams,
    hybrid_inputs: HybridInputs,
    time_s: jax.Array,
) -> jax.Array:
    """Assemble the six normalized channels used during Phase 5 training."""
    velocity_x = _velocity(state.hu, state.h, params.dry_tolerance)
    velocity_y = _velocity(state.hv, state.h, params.dry_tolerance)
    dynamic = jnp.stack((state.h, velocity_x, velocity_y), axis=-1)
    dynamic = (dynamic - hybrid_inputs.state_mean) / hybrid_inputs.state_std
    dynamic = jnp.where(params.domain_mask[..., None], dynamic, 0.0)
    rainfall_mm_hr = rainfall_rate(params, time_s) * M_S_TO_MM_HR
    rainfall_channel = jnp.full_like(
        state.h,
        rainfall_mm_hr / hybrid_inputs.rainfall_scale_mm_hr,
    )
    features = jnp.concatenate(
        (
            dynamic,
            rainfall_channel[..., None],
            hybrid_inputs.terrain_standardized[..., None],
            hybrid_inputs.roughness_standardized[..., None],
        ),
        axis=-1,
    )
    return features[None, ...]


def apply_residual_correction(
    predicted: SWEState,
    residual: jax.Array,
    params: SWEParams,
    *,
    correction_relaxation: jax.Array,
) -> SWEState:
    """Apply a residual while preserving positivity and physics-step water volume.

    The neural network may redistribute water spatially, but it must not act
    as an unmodelled source or sink. After clipping negative provisional
    depths, a positive global scale projects the corrected field back to the
    total water volume produced by the finite-volume physics step.
    """
    predicted_u = _velocity(predicted.hu, predicted.h, params.dry_tolerance)
    predicted_v = _velocity(predicted.hv, predicted.h, params.dry_tolerance)
    provisional_h = jnp.where(
        params.domain_mask,
        jnp.maximum(predicted.h + correction_relaxation * residual[..., 0], 0.0),
        0.0,
    )
    target_volume_cells = jnp.sum(jnp.where(params.domain_mask, predicted.h, 0.0))
    provisional_volume_cells = jnp.sum(provisional_h)
    volume_scale = jnp.where(
        provisional_volume_cells > jnp.finfo(provisional_h.dtype).eps,
        target_volume_cells / provisional_volume_cells,
        1.0,
    )
    corrected_h = jnp.where(
        target_volume_cells > jnp.finfo(provisional_h.dtype).eps,
        provisional_h * volume_scale,
        predicted.h,
    )
    corrected_u = predicted_u + correction_relaxation * residual[..., 1]
    corrected_v = predicted_v + correction_relaxation * residual[..., 2]
    wet = params.domain_mask & (corrected_h > params.dry_tolerance)
    corrected_hu = jnp.where(wet, corrected_h * corrected_u, 0.0)
    corrected_hv = jnp.where(wet, corrected_h * corrected_v, 0.0)
    return SWEState(corrected_h, corrected_hu, corrected_hv)


def advance_physics_to_time(
    state: SWEState,
    params: SWEParams,
    start_time_s: jax.Array,
    target_time_s: jax.Array,
    *,
    max_substeps: int,
) -> tuple[SWEState, jax.Array]:
    """Advance adaptively to one correction time with a fixed-shape scan."""
    carry = (state, start_time_s, jnp.asarray(0, dtype=jnp.int32))

    def substep(carry_value, _):
        current_state, current_time, steps = carry_value
        active = current_time < target_time_s - 1.0e-7

        def advance(active_carry):
            active_state, active_time, active_steps = active_carry
            dt = jnp.minimum(
                compute_cfl_dt(active_state, params),
                target_time_s - active_time,
            )
            next_state = finite_volume_step(active_state, params, dt, active_time)
            return next_state, active_time + dt, active_steps + 1

        next_carry = jax.lax.cond(active, advance, lambda value: value, carry_value)
        return next_carry, None

    (final_state, final_time, steps), _ = jax.lax.scan(
        substep,
        carry,
        xs=None,
        length=max_substeps,
    )
    reached = final_time >= target_time_s - 1.0e-6
    final_state = jax.tree.map(
        lambda value: jnp.where(reached, value, jnp.full_like(value, jnp.nan)),
        final_state,
    )
    return final_state, steps


@partial(
    jax.jit,
    static_argnames=("apply_fn", "max_substeps_per_interval"),
)
def run_coupled_forecast(
    initial_state: SWEState,
    params: SWEParams,
    model_params: Any,
    apply_fn: Any,
    hybrid_inputs: HybridInputs,
    output_times_s: jax.Array,
    *,
    max_substeps_per_interval: int,
    correction_relaxation: float = 0.05,
    correction_clip_sigma: float = 5.0,
) -> CoupledRolloutResult:
    """Run an autoregressive, fully device-resident physics-AI forecast."""
    if max_substeps_per_interval < 1:
        raise ValueError("At least one physics substep per correction interval is required.")
    if output_times_s.shape[0] < 2:
        raise ValueError("At least two output times are required.")
    relaxation = jnp.asarray(correction_relaxation, dtype=initial_state.h.dtype)
    clip_sigma = jnp.asarray(correction_clip_sigma, dtype=initial_state.h.dtype)

    def coupled_step(carry, target_time):
        state, current_time, total_steps = carry
        features = network_features(state, params, hybrid_inputs, current_time)
        advanced, steps = advance_physics_to_time(
            state,
            params,
            current_time,
            target_time,
            max_substeps=max_substeps_per_interval,
        )
        residual = apply_fn({"params": model_params}, features)[0]
        limits = clip_sigma * hybrid_inputs.residual_std
        residual = jnp.clip(residual, -limits, limits)
        corrected = apply_residual_correction(
            advanced,
            residual,
            params,
            correction_relaxation=relaxation,
        )
        return (corrected, target_time, total_steps + steps), corrected

    initial_carry = (
        initial_state,
        output_times_s[0],
        jnp.asarray(0, dtype=jnp.int32),
    )
    (final_state, final_time, steps), scanned = jax.lax.scan(
        coupled_step,
        initial_carry,
        output_times_s[1:],
    )
    del final_state, final_time
    states = SWEState(
        jnp.concatenate((initial_state.h[None], scanned.h), axis=0),
        jnp.concatenate((initial_state.hu[None], scanned.hu), axis=0),
        jnp.concatenate((initial_state.hv[None], scanned.hv), axis=0),
    )
    return CoupledRolloutResult(states, output_times_s, steps)
