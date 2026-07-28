"""Run the coupled physics-AI forecast and save a three-source comparison."""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from hybrid_flood.hybrid.rollout import (
    comparison_metrics,
    run_comparison_rollout,
    select_relaxation_by_validation,
)
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
    expected_duration_s = float(solver_cfg.duration_s)
    if not np.isclose(
        float(residual_dataset.target_time_s[-1]),
        expected_duration_s,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Residual dataset is stale or incomplete: it does not reach the configured "
            f"{expected_duration_s} s horizon."
        )
    for source_name, configured_path in (
        ("raw JAX", hybrid_cfg.inputs.raw_jax),
        ("ANUGA", hybrid_cfg.inputs.anuga),
    ):
        with xr.open_dataset(_path(root, configured_path)) as source:
            if not np.isclose(
                float(source.time.values[-1]),
                expected_duration_s,
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError(
                    f"{source_name} artifact is stale or incomplete: it ends at "
                    f"{float(source.time.values[-1])} s, expected {expected_duration_s} s."
                )
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
    selection_cfg = hybrid_cfg.rollout.relaxation_selection
    selection_record: dict[str, object]
    if bool(selection_cfg.enabled):
        validation_indices = np.asarray(residual_dataset.val_indices, dtype=np.int64)
        if validation_indices.size == 0:
            raise ValueError("Relaxation selection requires a non-empty validation block.")
        validation_times = residual_dataset.target_time_s[validation_indices]
        candidates = sorted({float(value) for value in selection_cfg.candidates})
        if any(value < 0.0 for value in candidates):
            raise ValueError("Correction-relaxation candidates must be non-negative.")
        candidate_metrics: dict[float, float] = {}
        LOGGER.info(
            "Selecting correction relaxation on %d validation times; test times are excluded",
            validation_indices.size,
        )
        with tempfile.TemporaryDirectory(prefix="hybrid-relaxation-") as temporary:
            for candidate in candidates:
                candidate_result = run_comparison_rollout(
                    grid=grid,
                    params=solver.params,
                    residual_dataset=residual_dataset,
                    model=model,
                    model_params=model_params,
                    raw_jax_path=_path(root, hybrid_cfg.inputs.raw_jax),
                    anuga_path=_path(root, hybrid_cfg.inputs.anuga),
                    output_path=Path(temporary) / "candidate.nc",
                    start_time_s=0.0,
                    end_time_s=float(validation_times[-1]),
                    max_physics_steps=int(hybrid_cfg.rollout.max_physics_steps),
                    max_substeps_per_interval=int(hybrid_cfg.rollout.max_substeps_per_interval),
                    correction_relaxation=candidate,
                    correction_clip_sigma=float(hybrid_cfg.rollout.correction_clip_sigma),
                )
                validation_dataset = candidate_result.dataset.sel(time=validation_times)
                score = float(comparison_metrics(validation_dataset)["hybrid_depth_rmse"])
                candidate_metrics[candidate] = score
                candidate_result.dataset.close()
                LOGGER.info(
                    "Validation relaxation %.6g: depth RMSE %.6g m",
                    candidate,
                    score,
                )
        selected_relaxation, selected_score = select_relaxation_by_validation(candidate_metrics)
        selection_record = {
            "enabled": True,
            "selection_partition": "validation",
            "test_partition_used_for_selection": False,
            "metric": str(selection_cfg.metric),
            "validation_time_start_s": float(validation_times[0]),
            "validation_time_end_s": float(validation_times[-1]),
            "candidate_validation_depth_rmse_m": {
                str(candidate): score for candidate, score in candidate_metrics.items()
            },
            "selected_relaxation": selected_relaxation,
            "selected_validation_depth_rmse_m": selected_score,
        }
    else:
        selected_relaxation = float(hybrid_cfg.rollout.correction_relaxation)
        selection_record = {
            "enabled": False,
            "selected_relaxation": selected_relaxation,
        }

    LOGGER.info(
        "Running final pure-JAX and autoregressive hybrid branches with relaxation %.6g",
        selected_relaxation,
    )
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
        correction_relaxation=selected_relaxation,
        correction_clip_sigma=float(hybrid_cfg.rollout.correction_clip_sigma),
    )
    metadata_path = result.output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "configuration": OmegaConf.to_container(hybrid_cfg, resolve=True),
                "jax_solver_configuration": OmegaConf.to_container(
                    solver_cfg,
                    resolve=True,
                ),
                "execution": {
                    "devices": [str(device) for device in devices],
                    "default_backend": jax.default_backend(),
                    "gpu_accelerated": bool(gpu_devices),
                    "runtime_probe": JAX_RUNTIME,
                },
                "output_netcdf": str(result.output_path),
                "metrics_json": str(result.output_path.with_suffix(".json")),
                "relaxation_selection": selection_record,
                "metrics": result.metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
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
