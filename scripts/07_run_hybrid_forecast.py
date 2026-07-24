"""Run the coupled physics-AI forecast and save a three-source comparison."""

# ruff: noqa: E402

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

import jax
import jax.numpy as jnp

from hybrid_flood.hybrid.rollout import run_comparison_rollout
from hybrid_flood.jax_solver.boundary_conditions import all_reflective
from hybrid_flood.jax_solver.shallow_water_2d import (
    ShallowWaterSolver,
    load_rainfall_series,
    load_structured_grid,
)
from hybrid_flood.ml.dataset import load_residual_dataset
from hybrid_flood.ml.residual_net import model_from_config
from hybrid_flood.ml.train import load_checkpoint

LOGGER = logging.getLogger(__name__)


def _path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Build shared inputs, restore the CNN, and run both forecast branches."""
    root = Path(get_original_cwd()).resolve()
    hybrid_cfg = cfg.hybrid
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if hybrid_cfg.execution.require_gpu and not gpu_devices:
        raise RuntimeError(
            "GPU hybrid execution was required, but JAX found no GPU device. "
            f"{JAX_RUNTIME['cuda_driver']['reason']}"
        )
    if gpu_devices:
        LOGGER.info("Hybrid rollout on JAX GPU device(s): %s", ", ".join(map(str, gpu_devices)))
    else:
        LOGGER.warning(
            "No JAX GPU is available; hybrid rollout will run on CPU. %s",
            JAX_RUNTIME["cuda_driver"]["reason"],
        )

    solver_cfg = cfg.jax_solver
    grid = load_structured_grid(
        _path(root, solver_cfg.inputs.dem),
        _path(root, solver_cfg.inputs.roughness),
        _path(root, solver_cfg.inputs.domain),
        resolution_m=float(solver_cfg.resolution_m),
    )
    rainfall_times, rainfall_rates, default_rate = load_rainfall_series(
        _path(root, solver_cfg.inputs.rainfall),
        scenario=solver_cfg.rainfall.scenario,
        default_rate_mm_hr=float(solver_cfg.rainfall.default_rate_mm_hr),
    )
    solver = ShallowWaterSolver(
        grid,
        rainfall_times_s=rainfall_times,
        rainfall_rates_m_s=rainfall_rates,
        default_rainfall_m_s=default_rate,
        boundaries=all_reflective(),
        gravity_m_s2=float(solver_cfg.gravity_m_s2),
        cfl=float(solver_cfg.cfl),
        dry_tolerance_m=float(solver_cfg.dry_tolerance_m),
        max_dt_s=float(solver_cfg.max_dt_s),
    )
    residual_dataset = load_residual_dataset(_path(root, hybrid_cfg.inputs.residual_dataset))
    if residual_dataset.spatial_shape != grid.bed.shape:
        raise ValueError("Residual dataset and solver grid shapes do not match.")
    model = model_from_config(cfg.model.architecture)
    template = model.init(
        jax.random.PRNGKey(int(cfg.project.seed)),
        jnp.zeros(
            (
                1,
                int(cfg.model.training.patch_size),
                int(cfg.model.training.patch_size),
                residual_dataset.input_channels,
            ),
            dtype=jnp.float32,
        ),
    )["params"]
    model_params = load_checkpoint(
        template,
        _path(root, hybrid_cfg.inputs.checkpoint),
    )
    LOGGER.info("Running pure-JAX and autoregressive hybrid forecast branches")
    result = run_comparison_rollout(
        grid=grid,
        params=solver.params,
        residual_dataset=residual_dataset,
        model=model,
        model_params=model_params,
        raw_jax_path=_path(root, hybrid_cfg.inputs.raw_jax),
        anuga_path=_path(root, hybrid_cfg.inputs.anuga),
        output_path=_path(root, hybrid_cfg.output),
        start_time_s=hybrid_cfg.rollout.start_time_s,
        end_time_s=hybrid_cfg.rollout.end_time_s,
        max_physics_steps=int(hybrid_cfg.rollout.max_physics_steps),
        max_substeps_per_interval=int(hybrid_cfg.rollout.max_substeps_per_interval),
        correction_relaxation=float(hybrid_cfg.rollout.correction_relaxation),
        correction_clip_sigma=float(hybrid_cfg.rollout.correction_clip_sigma),
    )
    LOGGER.info(
        "Saved %s; hybrid depth RMSE %.6g m versus raw JAX %.6g m (skill %.2f%%)",
        result.output_path,
        result.metrics["hybrid_depth_rmse"],
        result.metrics["jax_depth_rmse"],
        result.metrics["hybrid_depth_rmse_skill_percent"],
    )


if __name__ == "__main__":
    main()
