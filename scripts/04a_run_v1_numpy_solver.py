"""Run the V1 NumPy structured forecast without importing JAX."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.numpy_solver.shallow_water_2d import NumpyShallowWaterSolver
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
    """Execute the conventional NumPy V1 and persist common-schema output."""
    if any(name == "jax" or name.startswith("jax.") for name in sys.modules):
        raise RuntimeError("V1 imported JAX; the baseline must remain JAX-free.")
    root = Path(get_original_cwd()).resolve()
    common = cfg.comparison.common
    grid = load_structured_grid(
        _path(root, common.inputs.dem),
        _path(root, common.inputs.roughness),
        _path(root, common.inputs.domain),
        resolution_m=float(common.resolution_m),
    )
    rainfall_times, rainfall_rates, default_rate = load_rainfall_series(
        _path(root, common.inputs.rainfall), scenario=str(common.inputs.rainfall_scenario)
    )
    solver = NumpyShallowWaterSolver(
        grid.bed,
        grid.manning_n,
        grid.domain_mask,
        resolution_m=grid.resolution_m,
        rainfall_times_s=rainfall_times,
        rainfall_rates_m_s=rainfall_rates,
        default_rainfall_m_s=default_rate,
        gravity_m_s2=float(common.gravity_m_s2),
        cfl=float(common.cfl),
        dry_tolerance_m=float(common.dry_tolerance_m),
        max_dt_s=float(common.max_dt_s),
    )
    initial = solver.initial_state(**OmegaConf.to_container(common.initial_condition))
    times = np.arange(
        0.0,
        float(common.duration_s) + 0.5 * float(common.output_interval_s),
        float(common.output_interval_s),
        dtype=np.float32,
    )
    start = perf_counter()
    result = solver.run(initial, times, max_steps=int(common.max_steps))
    runtime = perf_counter() - start
    dataset = conserved_to_xarray(
        *result.states,
        result.output_times_s,
        grid,
        dry_tolerance_m=float(common.dry_tolerance_m),
        title="V1 NumPy finite-volume shallow-water forecast",
        backend="numpy",
        steps_used=result.steps_used,
    )
    output = save_netcdf(dataset, _path(root, cfg.comparison.v1.output))
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "version": "V1",
        "architecture": "ANUGA reference + NumPy SWE + PyTorch CUDA AI",
        "jax_imported": False,
        "configuration": OmegaConf.to_container(common, resolve=True),
        "runtime_seconds": runtime,
        "steps_used": result.steps_used,
        "output": str(output),
        "grid_shape": list(grid.bed.shape),
        "domain_cell_count": int(grid.domain_mask.sum()),
    }
    metadata_path = _path(root, cfg.comparison.v1.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("V1 completed in %.3f s (%d steps): %s", runtime, result.steps_used, output)


if __name__ == "__main__":
    main()
