"""Run raw GIS and rainfall ingestion from the Hydra project configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.data.load_rainfall import clean_rainfall_workbook
from hybrid_flood.data.load_shapefiles import (
    SchemaValidationError,
    load_all_shapefiles,
    write_validation_report,
)
from hybrid_flood.data.reproject import reproject_all_layers
from hybrid_flood.data.validate_mesh import validate_mesh_topology

LOGGER = logging.getLogger(__name__)


def _resolve_from_project(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def run_pipeline(cfg: DictConfig, project_root: Path) -> dict[str, Any]:
    """Execute validation, reprojection, rainfall cleaning, and mesh checks."""
    raw_dir = _resolve_from_project(project_root, cfg.project.raw_data_dir)
    interim_dir = _resolve_from_project(project_root, cfg.project.interim_data_dir)
    report_path = interim_dir / cfg.data.validation_report
    interim_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Validating raw vector layers in %s", raw_dir)
    layers, report = load_all_shapefiles(raw_dir, strict=False)

    LOGGER.info("Checking mesh topology without modifying geometry")
    mesh_report = validate_mesh_topology(
        layers["mesh_elements"],
        layers["domain"],
        degenerate_area_tolerance_m2=cfg.data.mesh.degenerate_area_tolerance_m2,
        gap_relative_tolerance=cfg.data.mesh.gap_relative_tolerance,
    )
    report["mesh_topology"] = mesh_report

    vector_critical = report["summary"]["critical_errors"]
    if vector_critical or mesh_report["critical_errors"]:
        report["summary"]["critical_errors"] = vector_critical + mesh_report["critical_errors"]
        report["summary"]["warnings"] += mesh_report["warnings"]
        write_validation_report(report, report_path)
        raise SchemaValidationError(
            f"GIS validation found {report['summary']['critical_errors']} critical error(s); "
            f"inspect {report_path}."
        )

    LOGGER.info("Reprojecting vector layers to %s", cfg.data.target_crs)
    projected_paths = reproject_all_layers(
        raw_dir,
        interim_dir,
        target_crs=cfg.data.target_crs,
        layers=layers,
    )
    report["reprojected_layers"] = {
        name: str(path.relative_to(project_root)) for name, path in projected_paths.items()
    }

    rainfall_cfg = cfg.data.rainfall
    rainfall_input = raw_dir / rainfall_cfg.filename
    rainfall_output = interim_dir / rainfall_cfg.output_filename
    LOGGER.info("Cleaning rainfall workbook %s", rainfall_input)
    rainfall, rainfall_xarray = clean_rainfall_workbook(
        rainfall_input,
        rainfall_output,
        sheet_name=rainfall_cfg.sheet_name,
        reference_start=rainfall_cfg.reference_start,
        frequency=rainfall_cfg.frequency,
    )
    report["rainfall"] = {
        "source": str(rainfall_input.relative_to(project_root)),
        "output": str(rainfall_output.relative_to(project_root)),
        "row_count": len(rainfall),
        "timestamp_count": int(rainfall["timestamp"].nunique()),
        "scenario_count": int(rainfall["scenario"].nunique()),
        "missing_timestamp_rows": int(rainfall["is_missing_timestamp"].sum()),
        "units": rainfall_xarray["rainfall_mm_hr"].attrs["units"],
        "critical_errors": 0,
        "warnings": 0,
    }
    report["configuration"] = OmegaConf.to_container(cfg.data, resolve=True)
    report["summary"]["critical_errors"] = vector_critical + mesh_report["critical_errors"]
    report["summary"]["warnings"] += mesh_report["warnings"]
    write_validation_report(report, report_path)
    LOGGER.info("Preparation complete; validation report: %s", report_path)
    return report


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the complete ingestion pipeline."""
    run_pipeline(cfg, Path(get_original_cwd()).resolve())


if __name__ == "__main__":
    main()
