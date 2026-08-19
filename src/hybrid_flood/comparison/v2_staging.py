"""Large compiled JAX stages with one DLPack exchange per direction."""

from __future__ import annotations

from time import perf_counter

import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils import dlpack as torch_dlpack

from hybrid_flood.ml.dataset import ResidualDataset


@jax.jit
def _build_features(
    state: jax.Array,
    rainfall: jax.Array,
    terrain: jax.Array,
    roughness: jax.Array,
    domain_mask: jax.Array,
    state_mean: jax.Array,
    state_std: jax.Array,
    rainfall_scale: jax.Array,
) -> jax.Array:
    """Normalize every timestep with vmap and assemble NHWC features."""

    def normalize(one_state):
        return (one_state - state_mean) / state_std

    dynamic = jax.vmap(normalize)(state)
    dynamic = jnp.where(domain_mask[None, ..., None], dynamic, 0.0)
    count, height, width, _ = dynamic.shape
    rain = jnp.broadcast_to(
        (rainfall / rainfall_scale)[:, None, None, None], (count, height, width, 1)
    )
    static_terrain = jnp.broadcast_to(terrain[None, ..., None], (count, height, width, 1))
    static_roughness = jnp.broadcast_to(roughness[None, ..., None], (count, height, width, 1))
    return jnp.concatenate((dynamic, rain, static_terrain, static_roughness), axis=-1)


@jax.jit
def _correct_depth(raw_depth: jax.Array, residual_nhwc: jax.Array) -> jax.Array:
    """Apply non-negative corrections to all timesteps in one compiled stage."""
    return jax.vmap(lambda depth, residual: jnp.maximum(depth + residual[..., 0], 0.0))(
        raw_depth, residual_nhwc
    )


def v2_features_to_torch(
    dataset: ResidualDataset,
) -> tuple[torch.Tensor, float, float]:
    """Compile/execute feature assembly and transfer once through GPU DLPack."""
    normalization = dataset.metadata["normalization"]
    arguments = (
        jnp.asarray(dataset.state_t),
        jnp.asarray(dataset.rainfall_t_mm_hr),
        jnp.asarray(dataset.terrain_standardized),
        jnp.asarray(dataset.roughness_standardized),
        jnp.asarray(dataset.domain_mask),
        jnp.asarray(normalization["state_mean"], dtype=jnp.float32),
        jnp.asarray(normalization["state_std"], dtype=jnp.float32),
        jnp.asarray(normalization["rainfall_scale_mm_hr"], dtype=jnp.float32),
    )
    start = perf_counter()
    executable = _build_features.lower(*arguments).compile()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    features = executable(*arguments)
    jax.block_until_ready(features)
    execution_seconds = perf_counter() - start
    # DLPack shares the device allocation; it does not stage through CPU/NumPy.
    torch_nhwc = torch_dlpack.from_dlpack(features)
    return torch_nhwc.permute(0, 3, 1, 2), compile_seconds, execution_seconds


def corrections_to_jax(
    raw_depth: np.ndarray,
    predictions_nchw: torch.Tensor,
) -> tuple[np.ndarray, float, float]:
    """Transfer predictions once with DLPack, then compile/execute correction."""
    predictions_nhwc = predictions_nchw.permute(0, 2, 3, 1).contiguous()
    residual = jax.dlpack.from_dlpack(predictions_nhwc)
    depth = jnp.asarray(raw_depth)
    start = perf_counter()
    executable = _correct_depth.lower(depth, residual).compile()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    corrected = executable(depth, residual)
    jax.block_until_ready(corrected)
    execution_seconds = perf_counter() - start
    return np.asarray(corrected), compile_seconds, execution_seconds
