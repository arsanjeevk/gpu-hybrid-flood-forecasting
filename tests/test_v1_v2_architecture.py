"""Regression tests for the matched NumPy/JAX/PyTorch version design."""

from __future__ import annotations

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import torch
from omegaconf import OmegaConf

from hybrid_flood.jax_solver.numerics import (
    SWEParams,
    SWEState,
)
from hybrid_flood.jax_solver.numerics import (
    finite_volume_step as jax_step,
)
from hybrid_flood.ml.torch_losses import residual_correction_loss
from hybrid_flood.ml.torch_model import PyTorchResidualUNet
from hybrid_flood.numpy_solver.shallow_water_2d import (
    NumpyShallowWaterSolver,
    NumpyState,
)
from hybrid_flood.numpy_solver.shallow_water_2d import (
    finite_volume_step as numpy_step,
)


def _solvers(shape=(7, 9), *, bed=None, rainfall=0.0):
    bed_array = np.zeros(shape, dtype=np.float32) if bed is None else np.asarray(bed, np.float32)
    roughness = np.full(shape, 0.03, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    rain_times = np.asarray([0.0, 1000.0], dtype=np.float32)
    rain_rates = np.asarray([rainfall, rainfall], dtype=np.float32)
    numpy_solver = NumpyShallowWaterSolver(
        bed_array,
        roughness,
        mask,
        resolution_m=20.0,
        rainfall_times_s=rain_times,
        rainfall_rates_m_s=rain_rates,
        max_dt_s=2.0,
    )
    jax_params = SWEParams(
        jnp.asarray(bed_array),
        jnp.asarray(roughness),
        jnp.asarray(mask),
        jnp.asarray(mask, dtype=jnp.float32),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.asarray(rain_times),
        jnp.asarray(rain_rates),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(20.0, dtype=jnp.float32),
        jnp.asarray(20.0, dtype=jnp.float32),
        jnp.asarray(9.81, dtype=jnp.float32),
        jnp.asarray(0.45, dtype=jnp.float32),
        jnp.asarray(1.0e-4, dtype=jnp.float32),
        jnp.asarray(2.0, dtype=jnp.float32),
    )
    return numpy_solver, jax_params


def test_v1_module_imports_no_jax() -> None:
    """A fresh V1 process must not load JAX transitively."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hybrid_flood.numpy_solver; "
            "assert not any(x == 'jax' or x.startswith('jax.') for x in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_numpy_and_jax_steps_are_numerically_matched() -> None:
    """Only roundoff-level differences are allowed for one common HLL step."""
    rng = np.random.default_rng(7)
    bed = rng.uniform(0.0, 0.2, size=(7, 9)).astype(np.float32)
    numpy_solver, jax_params = _solvers(bed=bed, rainfall=1.2e-6)
    depth = np.full(bed.shape, 0.1, dtype=np.float32)
    state = NumpyState(
        depth,
        rng.normal(0.0, 1.0e-3, bed.shape).astype(np.float32),
        rng.normal(0.0, 1.0e-3, bed.shape).astype(np.float32),
    )
    numpy_result = numpy_step(state, numpy_solver.params, 0.75, 20.0)
    jax_result = jax_step(
        SWEState(*(jnp.asarray(value) for value in state)),
        jax_params,
        jnp.asarray(0.75, dtype=jnp.float32),
        jnp.asarray(20.0, dtype=jnp.float32),
    )
    for baseline, optimized in zip(numpy_result, jax_result, strict=True):
        np.testing.assert_allclose(baseline, np.asarray(optimized), rtol=2.0e-5, atol=2.0e-7)


def test_numpy_lake_at_rest_is_well_balanced() -> None:
    """V1 must preserve a constant free surface over variable bathymetry."""
    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    bed = 0.2 * np.outer(np.sin(np.pi * x) ** 2, np.sin(np.pi * x) ** 2)
    solver, _ = _solvers(shape=bed.shape, bed=bed)
    surface = np.float32(0.35)
    initial = NumpyState(surface - bed, np.zeros_like(bed), np.zeros_like(bed))
    result = solver.run(initial, np.asarray([0.0, 20.0], np.float32), max_steps=100)
    np.testing.assert_allclose(result.states.h[-1], initial.h, atol=2.0e-6)
    assert np.max(np.abs(result.states.hu[-1])) < 2.0e-6
    assert np.max(np.abs(result.states.hv[-1])) < 2.0e-6


def test_pytorch_training_step_reduces_tiny_loss() -> None:
    """The active AI backend must perform a real differentiable update."""
    torch.manual_seed(3)
    model = PyTorchResidualUNet(
        input_channels=6,
        depth=2,
        channels=(4, 8),
        activation="gelu",
        output_channels=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    inputs = torch.randn(2, 6, 16, 16)
    target = torch.full((2, 3, 16, 16), 0.05)
    depth = torch.full((2, 16, 16), 0.1)
    mask = torch.ones((2, 16, 16), dtype=torch.bool)
    scales = torch.ones(3)

    def loss():
        return residual_correction_loss(
            model(inputs),
            target,
            depth,
            mask,
            channel_scales=scales,
            mse_weight=1.0,
            mae_weight=0.0,
            negative_depth_weight=1.0,
        ).total

    before = float(loss().detach())
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        optimizer.step()
    assert float(loss().detach()) < before


def test_active_configuration_targets_t4_and_shared_pytorch() -> None:
    comparison = OmegaConf.load("config/comparison/v1_v2_t4.yaml")
    platform = OmegaConf.load("config/platform/colab_t4.yaml")
    model = OmegaConf.load("config/model/residual_cnn.yaml")
    assert platform.required_gpu_name == "T4"
    assert comparison.v1.physics_backend == "numpy"
    assert comparison.v2.physics_backend == "jax"
    assert model.framework == "pytorch"
    assert int(model.training.batch_size) == 2
    assert float(comparison.common.duration_s) == 10_800.0
