"""Run the configured pure-JAX shallow-water baseline."""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# This must run before importing any module that imports JAX.  It prevents a
# CUDA-plugin traceback on CPU-only machines while retaining GPU use when a
# working NVIDIA driver is exposed.
from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

import jax

from hybrid_flood.data.rainfall_window import summarize_rainfall_window
from hybrid_flood.jax_solver.benchmarking import benchmark_compiled_steps
from hybrid_flood.jax_solver.boundary_conditions import all_reflective
from hybrid_flood.jax_solver.shallow_water_2d import (
    ShallowWaterSolver,
    load_rainfall_series,
    load_structured_grid,
    result_to_xarray,
    save_result_netcdf,
)

LOGGER = logging.getLogger(__name__)


def _path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Load geospatial inputs, compile/run JAX, benchmark, and save NetCDF."""
    root = Path(get_original_cwd()).resolve()
    solver_cfg = cfg.jax_solver
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if solver_cfg.execution.require_gpu and not gpu_devices:
        raise RuntimeError(
            "GPU execution was required, but JAX found no GPU device. "
            f"{JAX_RUNTIME['cuda_driver']['reason']} "
            "Run `nvidia-smi` and verify that an NVIDIA driver/device is exposed."
        )
    if gpu_devices:
        LOGGER.info("JAX GPU execution enabled: %s", ", ".join(map(str, gpu_devices)))
    else:
        LOGGER.warning(
            "No JAX GPU is available; this run and its benchmark will use CPU only. %s "
            "Use `jax_solver.execution.require_gpu=true` to reject CPU fallback.",
            JAX_RUNTIME["cuda_driver"]["reason"],
        )
    if solver_cfg.boundary.policy != "all_reflective":
        raise ValueError("Only the ANUGA-comparable all_reflective policy is configured.")
    if solver_cfg.flux.lower() != "hll":
        raise ValueError("Only the implemented HLL numerical flux is supported.")

    LOGGER.info("Loading structured %.1f m grid", solver_cfg.resolution_m)
    grid = load_structured_grid(
        _path(root, solver_cfg.inputs.dem),
        _path(root, solver_cfg.inputs.roughness),
        _path(root, solver_cfg.inputs.domain),
        resolution_m=solver_cfg.resolution_m,
    )
    rainfall_times, rainfall_rates, default_rate = load_rainfall_series(
        _path(root, solver_cfg.inputs.rainfall),
        scenario=solver_cfg.rainfall.scenario,
        default_rate_mm_hr=solver_cfg.rainfall.default_rate_mm_hr,
    )
    rainfall_window = summarize_rainfall_window(
        rainfall_times,
        rainfall_rates,
        float(solver_cfg.duration_s),
        default_rate_m_s=default_rate,
    )
    if not rainfall_window["peak_in_simulation_window"]:
        LOGGER.warning(
            "The %.2f h simulation excludes the selected rainfall peak at %.2f h "
            "and covers only %.2f%% of the rainfall record.",
            float(solver_cfg.duration_s) / 3600.0,
            rainfall_window["peak_time_s"] / 3600.0,
            100.0 * rainfall_window["record_fraction_covered"],
        )
    solver = ShallowWaterSolver(
        grid,
        rainfall_times_s=rainfall_times,
        rainfall_rates_m_s=rainfall_rates,
        default_rainfall_m_s=default_rate,
        boundaries=all_reflective(),
        gravity_m_s2=solver_cfg.gravity_m_s2,
        cfl=solver_cfg.cfl,
        dry_tolerance_m=solver_cfg.dry_tolerance_m,
        max_dt_s=solver_cfg.max_dt_s,
    )
    initial = solver.initial_state(**OmegaConf.to_container(solver_cfg.initial_condition))
    output_times = np.arange(
        0.0,
        solver_cfg.duration_s + 0.5 * solver_cfg.output_interval_s,
        solver_cfg.output_interval_s,
        dtype=np.float32,
    )

    LOGGER.info("Compiling and running %d requested output times", len(output_times))
    start = perf_counter()
    result = solver.run(initial, output_times, max_steps=solver_cfg.max_steps)
    runtime = perf_counter() - start
    dataset = result_to_xarray(
        result,
        grid,
        dry_tolerance_m=solver_cfg.dry_tolerance_m,
    )
    output_path = save_result_netcdf(
        dataset,
        _path(root, solver_cfg.outputs.netcdf),
    )

    benchmark_report = None
    if solver_cfg.benchmark.enabled:
        LOGGER.info("Benchmarking compiled steps after warmup")
        benchmark_report = benchmark_compiled_steps(
            initial,
            solver.params,
            iterations=solver_cfg.benchmark.iterations,
            dt_s=min(1.0, solver_cfg.max_dt_s),
            output_directory=_path(root, solver_cfg.outputs.run_directory),
        )

    run_directory = _path(root, solver_cfg.outputs.run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    metadata_path = run_directory / f"run_{timestamp}.json"
    valid_depth = dataset["depth"].values[:, grid.domain_mask]
    metadata = {
        "configuration": OmegaConf.to_container(solver_cfg, resolve=True),
        "execution": {
            "devices": [str(device) for device in devices],
            "default_backend": jax.default_backend(),
            "gpu_accelerated": bool(gpu_devices),
            "runtime_probe": JAX_RUNTIME,
        },
        "runtime_seconds_including_compile": runtime,
        "steps_used": int(np.asarray(result.steps_used)),
        "output_netcdf": str(output_path),
        "grid_shape": list(grid.bed.shape),
        "domain_cell_count": int(grid.domain_mask.sum()),
        "minimum_depth_m": float(np.min(valid_depth)),
        "maximum_depth_m": float(np.max(valid_depth)),
        "rainfall_simulation_window": rainfall_window,
        "benchmark": benchmark_report,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "JAX run complete in %.3fs (%d adaptive steps): %s",
        runtime,
        metadata["steps_used"],
        output_path,
    )


if __name__ == "__main__":
    main()
