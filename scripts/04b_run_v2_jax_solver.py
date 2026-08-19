"""Run the V2 JAX forecast and record compilation separately from execution."""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

import jax
import jax.numpy as jnp

from hybrid_flood.jax_solver.boundary_conditions import all_reflective
from hybrid_flood.jax_solver.numerics import integrate_to_outputs
from hybrid_flood.jax_solver.shallow_water_2d import ShallowWaterSolver
from hybrid_flood.structured_io import (
    conserved_to_xarray,
    load_rainfall_series,
    load_structured_grid,
    save_netcdf,
)

LOGGER = logging.getLogger(__name__)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Compile once, execute once, and preserve both timings for V2."""
    root = Path(get_original_cwd()).resolve()
    common = cfg.comparison.common
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    allow_cpu = bool(cfg.comparison.hardware.allow_cpu_smoke)
    if not gpu_devices and not allow_cpu:
        raise RuntimeError("V2 requires the configured T4; CPU fallback is disabled.")
    if bool(cfg.comparison.hardware.require_t4) and gpu_devices:
        names = [str(getattr(device, "device_kind", device)) for device in gpu_devices]
        if not any("T4" in name.upper() for name in names):
            raise RuntimeError(f"The active comparison requires an NVIDIA T4, found {names}.")
    grid = load_structured_grid(
        _path(root, common.inputs.dem),
        _path(root, common.inputs.roughness),
        _path(root, common.inputs.domain),
        resolution_m=float(common.resolution_m),
    )
    rainfall_times, rainfall_rates, default_rate = load_rainfall_series(
        _path(root, common.inputs.rainfall), scenario=str(common.inputs.rainfall_scenario)
    )
    solver = ShallowWaterSolver(
        grid,
        rainfall_times_s=rainfall_times,
        rainfall_rates_m_s=rainfall_rates,
        default_rainfall_m_s=default_rate,
        boundaries=all_reflective(),
        gravity_m_s2=float(common.gravity_m_s2),
        cfl=float(common.cfl),
        dry_tolerance_m=float(common.dry_tolerance_m),
        max_dt_s=float(common.max_dt_s),
    )
    initial = solver.initial_state(**OmegaConf.to_container(common.initial_condition))
    times = jnp.arange(
        0.0,
        float(common.duration_s) + 0.5 * float(common.output_interval_s),
        float(common.output_interval_s),
        dtype=jnp.float32,
    )
    start = perf_counter()
    executable = integrate_to_outputs.lower(
        initial, solver.params, times, max_steps=int(common.max_steps)
    ).compile()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    result = executable(initial, solver.params, times)
    jax.block_until_ready(result.states.h)
    execution_seconds = perf_counter() - start
    if int(np.asarray(result.outputs_written)) != len(times):
        raise RuntimeError("V2 exhausted max_steps before all outputs were written.")
    dataset = conserved_to_xarray(
        np.asarray(result.states.h),
        np.asarray(result.states.hu),
        np.asarray(result.states.hv),
        np.asarray(result.output_times_s),
        grid,
        dry_tolerance_m=float(common.dry_tolerance_m),
        title="V2 JAX finite-volume shallow-water forecast",
        backend="jax",
        steps_used=int(np.asarray(result.steps_used)),
    )
    output = save_netcdf(dataset, _path(root, cfg.comparison.v2.output))
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "version": "V2",
        "architecture": "ANUGA reference + JAX SWE + PyTorch CUDA AI",
        "configuration": OmegaConf.to_container(common, resolve=True),
        "execution": {
            "devices": [str(device) for device in jax.devices()],
            "gpu_accelerated": bool(gpu_devices),
            "runtime_probe": JAX_RUNTIME,
            "jax_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "jax_compilation_seconds": compile_seconds,
        "jax_execution_seconds": execution_seconds,
        "cold_start_seconds": compile_seconds + execution_seconds,
        "steps_used": int(np.asarray(result.steps_used)),
        "output": str(output),
        "grid_shape": list(grid.bed.shape),
        "domain_cell_count": int(grid.domain_mask.sum()),
    }
    metadata_path = _path(root, cfg.comparison.v2.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info(
        "V2 compiled in %.3f s and executed in %.3f s: %s",
        compile_seconds,
        execution_seconds,
        output,
    )


if __name__ == "__main__":
    main()
