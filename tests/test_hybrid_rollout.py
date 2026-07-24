"""Held-out autoregressive physics-AI rollout regression test."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_flood.hybrid.coupled_forecast import apply_residual_correction
from hybrid_flood.hybrid.rollout import run_comparison_rollout
from hybrid_flood.jax_solver.boundary_conditions import all_reflective
from hybrid_flood.jax_solver.numerics import SWEState
from hybrid_flood.jax_solver.shallow_water_2d import (
    ShallowWaterSolver,
    load_rainfall_series,
    load_structured_grid,
)
from hybrid_flood.ml.dataset import load_residual_dataset
from hybrid_flood.ml.residual_net import ResidualUNet
from hybrid_flood.ml.train import load_checkpoint


def test_neural_correction_preserves_physics_water_volume() -> None:
    """The learned correction may redistribute water but cannot create/remove it."""
    grid = load_structured_grid(
        Path(__file__).resolve().parents[1] / "data/synthetic_dem/dem.tif",
        Path(__file__).resolve().parents[1] / "data/interim/roughness_epsg32643.gpkg",
        Path(__file__).resolve().parents[1] / "data/interim/domain_epsg32643.gpkg",
        resolution_m=50.0,
    )
    solver = ShallowWaterSolver(
        grid,
        rainfall_times_s=jnp.asarray([0.0, 1.0]),
        rainfall_rates_m_s=jnp.asarray([0.0, 0.0]),
    )
    depth = jnp.where(solver.params.domain_mask, 0.02, 0.0)
    predicted = SWEState(depth, jnp.zeros_like(depth), jnp.zeros_like(depth))
    residual = jnp.zeros((*depth.shape, 3), dtype=depth.dtype)
    residual = residual.at[..., 0].set(jnp.where(solver.params.domain_mask, -0.01, 0.0))
    corrected = apply_residual_correction(
        predicted,
        residual,
        solver.params,
        correction_relaxation=jnp.asarray(1.0),
    )
    np.testing.assert_allclose(
        np.asarray(corrected.h).sum(),
        np.asarray(predicted.h).sum(),
        rtol=2.0e-6,
    )


def test_hybrid_reduces_error_on_held_out_time_block(tmp_path) -> None:
    """Autoregressive correction must beat raw physics on unseen late times."""
    project_root = Path(__file__).resolve().parents[1]
    grid = load_structured_grid(
        project_root / "data/synthetic_dem/dem.tif",
        project_root / "data/interim/roughness_epsg32643.gpkg",
        project_root / "data/interim/domain_epsg32643.gpkg",
        resolution_m=50.0,
    )
    rainfall_times, rainfall_rates, default_rate = load_rainfall_series(
        project_root / "data/interim/rainfall_clean.parquet",
        scenario="45rp_rain",
    )
    solver = ShallowWaterSolver(
        grid,
        rainfall_times_s=rainfall_times,
        rainfall_rates_m_s=rainfall_rates,
        default_rainfall_m_s=default_rate,
        boundaries=all_reflective(),
        max_dt_s=10.0,
    )
    residual_dataset = load_residual_dataset(
        project_root / "data/processed/residual_training_dataset.npz"
    )
    model = ResidualUNet(
        depth=3,
        channels=(16, 32, 64),
        activation="gelu",
        kernel_size=3,
        output_channels=3,
    )
    template = model.init(
        jax.random.PRNGKey(42),
        jnp.zeros((1, 128, 128, residual_dataset.input_channels), dtype=jnp.float32),
    )["params"]
    model_params = load_checkpoint(
        template,
        project_root / "data/outputs/jax_runs/residual_net_best.msgpack",
    )
    first_test = int(residual_dataset.test_indices[0])
    last_test = int(residual_dataset.test_indices[-1])

    result = run_comparison_rollout(
        grid=grid,
        params=solver.params,
        residual_dataset=residual_dataset,
        model=model,
        model_params=model_params,
        raw_jax_path=project_root / "data/processed/jax_solver_raw.nc",
        anuga_path=project_root / "data/processed/anuga_baseline.nc",
        output_path=tmp_path / "held_out_comparison.nc",
        start_time_s=float(residual_dataset.input_time_s[first_test]),
        end_time_s=float(residual_dataset.target_time_s[last_test]),
        max_physics_steps=20_000,
        max_substeps_per_interval=64,
        correction_relaxation=0.05,
    )

    assert result.dataset.sizes["time"] == len(residual_dataset.test_indices) + 1
    assert set(result.dataset.source.values.tolist()) == {"anuga", "jax", "hybrid"}
    assert result.metrics["hybrid_depth_rmse"] < result.metrics["jax_depth_rmse"]
    assert result.metrics["hybrid_depth_rmse_skill_percent"] > 0.0
