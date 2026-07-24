"""Run and postprocess the configured ANUGA physics baseline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.anuga_solver.boundary_conditions import configure_boundary_conditions
from hybrid_flood.anuga_solver.build_domain import build_domain
from hybrid_flood.anuga_solver.postprocess_sww import postprocess_sww
from hybrid_flood.anuga_solver.rainfall_forcing import apply_uniform_rainfall
from hybrid_flood.anuga_solver.run_simulation import run_simulation
from hybrid_flood.data.rainfall_window import summarize_rainfall_window

LOGGER = logging.getLogger(__name__)


def _path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Build, run, and regular-grid the ANUGA baseline."""
    root = Path(get_original_cwd()).resolve()
    anuga_cfg = cfg.anuga
    domain, build_report = build_domain(
        _path(root, anuga_cfg.inputs.mesh_elements),
        _path(root, anuga_cfg.inputs.boundary_lines),
        _path(root, anuga_cfg.inputs.roughness_lulc),
        _path(root, anuga_cfg.inputs.dem),
        initial_depth_m=anuga_cfg.initial_depth_m,
        boundary_match_tolerance_m=anuga_cfg.boundary.match_tolerance_m,
    )
    boundary_report = configure_boundary_conditions(
        domain,
        policy=anuga_cfg.boundary.policy,
    )
    rainfall_operator, rainfall_series, rainfall_report = apply_uniform_rainfall(
        domain,
        _path(root, anuga_cfg.inputs.rainfall),
        scenario=anuga_cfg.rainfall.scenario,
        default_rate_mm_hr=anuga_cfg.rainfall.default_rate_mm_hr,
    )
    rainfall_window = summarize_rainfall_window(
        rainfall_series.elapsed_seconds,
        rainfall_series.rate_m_s,
        float(anuga_cfg.duration_s),
        default_rate_m_s=rainfall_series.default_rate_m_s,
    )
    rainfall_report["simulation_window"] = rainfall_window
    if not rainfall_window["peak_in_simulation_window"]:
        LOGGER.warning(
            "The %.2f h simulation excludes the selected rainfall peak at %.2f h "
            "and covers only %.2f%% of the rainfall record.",
            float(anuga_cfg.duration_s) / 3600.0,
            rainfall_window["peak_time_s"] / 3600.0,
            100.0 * rainfall_window["record_fraction_covered"],
        )
    # Keep the operator alive and make its role explicit; ANUGA registers it
    # with the domain during construction.
    assert rainfall_operator.domain is domain

    simulation = run_simulation(
        domain,
        _path(root, anuga_cfg.outputs.run_directory),
        duration_s=anuga_cfg.duration_s,
        yieldstep_s=anuga_cfg.yieldstep_s,
        outputstep_s=anuga_cfg.outputstep_s,
        flow_algorithm=anuga_cfg.flow_algorithm,
        cfl=anuga_cfg.cfl,
        minimum_storable_height_m=anuga_cfg.minimum_storable_height_m,
    )
    _, postprocess_report = postprocess_sww(
        simulation.sww_path,
        _path(root, anuga_cfg.outputs.processed_netcdf),
        grid_resolution_m=anuga_cfg.outputs.postprocess_grid_resolution_m,
        dry_tolerance_m=anuga_cfg.outputs.dry_tolerance_m,
    )
    metadata = {
        "configuration": OmegaConf.to_container(anuga_cfg, resolve=True),
        "build": build_report,
        "boundaries": boundary_report,
        "rainfall": rainfall_report,
        "simulation": {
            "run_name": simulation.run_name,
            "sww_path": str(simulation.sww_path),
            "metrics_path": str(simulation.metrics_path),
            "runtime_seconds": simulation.runtime_seconds,
            "maximum_absolute_mass_error_m3": simulation.maximum_absolute_mass_error_m3,
            "maximum_relative_mass_error": simulation.maximum_relative_mass_error,
            "final_mass_balance_error_m3": simulation.final_mass_balance_error_m3,
        },
        "postprocessing": postprocess_report,
    }
    metadata_path = simulation.sww_path.with_name(f"{simulation.run_name}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("ANUGA baseline metadata written to %s", metadata_path)


if __name__ == "__main__":
    main()
