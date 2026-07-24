"""Generate the reproducible synthetic Gurugram DEM from Hydra configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.dem.export_geotiff import export_dem, write_generation_metadata
from hybrid_flood.dem.hydrological_conditioning import (
    condition_hydrology,
    flow_accumulation,
)
from hybrid_flood.dem.synthetic_terrain import generate_base_terrain
from hybrid_flood.dem.urban_conditioning import condition_urban_terrain
from hybrid_flood.viz.static_figures import plot_dem_diagnostics

LOGGER = logging.getLogger(__name__)


def _project_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def run_generation(cfg: DictConfig, project_root: Path) -> dict[str, Any]:
    """Run all synthetic DEM generation stages and return generation metadata."""
    dem_cfg = cfg.dem
    domain_path = _project_path(project_root, dem_cfg.inputs.domain)
    roughness_path = _project_path(project_root, dem_cfg.inputs.roughness)
    if not domain_path.is_file() or not roughness_path.is_file():
        raise FileNotFoundError(
            "Reprojected domain/roughness inputs are missing. Run scripts/01_prepare_data.py first."
        )

    domain = gpd.read_file(domain_path)
    roughness = gpd.read_file(roughness_path)
    if domain.crs is None:
        raise ValueError("Domain layer has no declared CRS.")
    if domain.crs.to_string() != dem_cfg.crs:
        domain = domain.to_crs(dem_cfg.crs)
    if roughness.crs is None:
        raise ValueError("Roughness layer has no declared CRS.")
    roughness = roughness if roughness.crs == domain.crs else roughness.to_crs(domain.crs)

    LOGGER.info("Generating deterministic Perlin terrain at %.2f m", dem_cfg.resolution_m)
    terrain = generate_base_terrain(
        domain,
        resolution_m=dem_cfg.resolution_m,
        seed=dem_cfg.seed,
        **OmegaConf.to_container(dem_cfg.terrain, resolve=True),
    )
    LOGGER.info("Applying documented LULC-based urban conditioning")
    urban = condition_urban_terrain(
        terrain.elevation,
        terrain.domain_mask,
        roughness,
        transform=terrain.transform,
        crs=terrain.crs,
        resolution_m=dem_cfg.resolution_m,
        seed=dem_cfg.seed,
        **OmegaConf.to_container(dem_cfg.urban, resolve=True),
    )
    LOGGER.info("Applying %s hydrological conditioning", dem_cfg.hydrology.aggressiveness)
    hydrology = condition_hydrology(
        urban.elevation,
        terrain.domain_mask,
        intentional_depression_mask=urban.intentional_depression_mask,
        **OmegaConf.to_container(dem_cfg.hydrology, resolve=True),
    )
    if hydrology.metadata["unexpected_remaining_sink_cells"]:
        raise RuntimeError(
            "Hydrological conditioning left "
            f"{hydrology.metadata['unexpected_remaining_sink_cells']} unexpected sink cells."
        )

    dem_path = _project_path(project_root, dem_cfg.outputs.dem)
    metadata_path = _project_path(project_root, dem_cfg.outputs.metadata)
    diagnostics_path = _project_path(project_root, dem_cfg.outputs.diagnostics)
    export_dem(
        hydrology.elevation,
        dem_path,
        transform=terrain.transform,
        crs=terrain.crs,
        domain_mask=terrain.domain_mask,
    )

    LOGGER.info("Calculating D8 flow accumulation and diagnostic figure")
    accumulation = flow_accumulation(
        hydrology.elevation,
        terrain.domain_mask,
        terrain.transform,
    )
    plot_dem_diagnostics(
        hydrology.elevation,
        terrain.domain_mask,
        accumulation,
        transform=terrain.transform,
        output_path=diagnostics_path,
    )

    valid_elevation = hydrology.elevation[terrain.domain_mask]
    metadata = {
        "project": "hybrid-flood-model",
        "product": "synthetic Gurugram DEM",
        "vertical_datum_note": (
            "Synthetic elevations are expressed as nominal metres AMSL for scenario "
            "modelling; they are not tied to a surveyed vertical datum."
        ),
        "inputs": {
            "domain": _display_path(domain_path, project_root),
            "roughness_lulc": _display_path(roughness_path, project_root),
        },
        "outputs": {
            "dem": _display_path(dem_path, project_root),
            "diagnostics": _display_path(diagnostics_path, project_root),
        },
        "configuration": OmegaConf.to_container(dem_cfg, resolve=True),
        "terrain": terrain.metadata,
        "urban_conditioning": urban.metadata,
        "hydrological_conditioning": hydrology.metadata,
        "statistics": {
            "valid_cell_count": int(terrain.domain_mask.sum()),
            "minimum_elevation_m": float(np.min(valid_elevation)),
            "maximum_elevation_m": float(np.max(valid_elevation)),
            "mean_elevation_m": float(np.mean(valid_elevation)),
            "standard_deviation_m": float(np.std(valid_elevation)),
        },
    }
    write_generation_metadata(metadata, metadata_path)
    LOGGER.info("Synthetic DEM written to %s", dem_path)
    return metadata


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    run_generation(cfg, Path(get_original_cwd()).resolve())


if __name__ == "__main__":
    main()
