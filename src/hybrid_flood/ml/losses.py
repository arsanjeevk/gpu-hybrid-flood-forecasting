"""Masked data losses and a non-negative-depth physical penalty.

For predicted residual ``r_hat`` and ANUGA-minus-JAX target ``r``, the data
term is ``w_mse MSE((r_hat-r)/s) + w_mae MAE((r_hat-r)/s)``, where ``s`` is
the training-set standard deviation of each output channel.  This prevents
velocity units and variance from overwhelming depth learning.  Permanently
dry cells are excluded to keep the dry background from dominating.  The
physical term ``w_neg mean((ReLU(-(h_jax+r_hat_h))/s_h)**2)`` penalizes
corrections that would produce negative depth.  Predictions remain in physical
units; scaling is used only inside the objective.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LossComponents(NamedTuple):
    """Scalar components returned for logging."""

    total: jax.Array
    mse: jax.Array
    mae: jax.Array
    negative_depth_penalty: jax.Array


def _broadcast_mask(mask: jax.Array, values: jax.Array) -> jax.Array:
    mask = jnp.asarray(mask, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask[..., None]
    return jnp.broadcast_to(mask, values.shape)


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    """Return a safe mean over a broadcast spatial mask."""
    weights = _broadcast_mask(mask, values)
    denominator = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(values * weights) / denominator


def residual_correction_loss(
    predicted_residual: jax.Array,
    target_residual: jax.Array,
    raw_depth_t_plus_1: jax.Array,
    mask: jax.Array,
    *,
    mse_weight: float = 1.0,
    mae_weight: float = 0.0,
    negative_depth_weight: float = 1.0,
    channel_scales: jax.Array | None = None,
) -> LossComponents:
    """Calculate data-fit and physically motivated non-negative-depth losses."""
    if predicted_residual.shape != target_residual.shape:
        raise ValueError("Predicted and target residual arrays must have identical shapes.")
    if predicted_residual.shape[-1] < 1:
        raise ValueError("Residual predictions must include a depth channel.")
    scales = (
        jnp.ones((predicted_residual.shape[-1],), dtype=predicted_residual.dtype)
        if channel_scales is None
        else jnp.asarray(channel_scales, dtype=predicted_residual.dtype)
    )
    if scales.shape != (predicted_residual.shape[-1],):
        raise ValueError("Channel scales must contain one value per residual channel.")
    scales = jnp.maximum(scales, jnp.finfo(predicted_residual.dtype).eps)
    error = (predicted_residual - target_residual) / scales
    mse = masked_mean(jnp.square(error), mask)
    mae = masked_mean(jnp.abs(error), mask)
    corrected_depth = raw_depth_t_plus_1 + predicted_residual[..., 0]
    negative_penalty = masked_mean(
        jnp.square(jax.nn.relu(-corrected_depth) / scales[0]),
        mask,
    )
    total = mse_weight * mse + mae_weight * mae + negative_depth_weight * negative_penalty
    return LossComponents(total, mse, mae, negative_penalty)
