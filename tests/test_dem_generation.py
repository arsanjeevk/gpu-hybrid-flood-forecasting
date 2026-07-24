"""Tests for reproducible, hydrologically conditioned synthetic terrain."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from hybrid_flood.data.load_shapefiles import load_all_shapefiles
from hybrid_flood.data.reproject import reproject_all_layers
from hybrid_flood.dem.export_geotiff import export_dem, write_generation_metadata
from hybrid_flood.dem.hydrological_conditioning import (
    condition_hydrology,
    find_unfilled_sink_mask,
    flow_accumulation,
)
from hybrid_flood.dem.synthetic_terrain import TerrainGrid, generate_base_terrain
from hybrid_flood.dem.urban_conditioning import (
    UrbanConditioningResult,
    condition_urban_terrain,
)
from hybrid_flood.viz.static_figures import plot_dem_diagnostics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


@pytest.fixture(scope="module")
def conditioned_dem() -> tuple[TerrainGrid, UrbanConditioningResult, object]:
    domain_path = INTERIM_DIR / "domain_epsg32643.gpkg"
    roughness_path = INTERIM_DIR / "roughness_epsg32643.gpkg"
    if not domain_path.is_file() or not roughness_path.is_file():
        layers, _ = load_all_shapefiles(PROJECT_ROOT / "data" / "raw", strict=True)
        reproject_all_layers(
            PROJECT_ROOT / "data" / "raw",
            INTERIM_DIR,
            target_crs="EPSG:32643",
            layers=layers,
        )
    domain = gpd.read_file(domain_path)
    roughness = gpd.read_file(roughness_path)
    terrain = generate_base_terrain(
        domain,
        resolution_m=100.0,
        base_elevation_m=225.0,
        relief_amplitude_m=6.0,
        regional_gradient_drop_m=8.0,
        seed=42,
        noise_sample_spacing_m=200.0,
    )
    urban = condition_urban_terrain(
        terrain.elevation,
        terrain.domain_mask,
        roughness,
        transform=terrain.transform,
        crs=terrain.crs,
        resolution_m=100.0,
        intentional_depression_radius_m=150.0,
        intentional_depression_depth_m=1.0,
        seed=42,
    )
    hydrology = condition_hydrology(
        urban.elevation,
        terrain.domain_mask,
        intentional_depression_mask=urban.intentional_depression_mask,
        aggressiveness="selective",
    )
    return terrain, urban, hydrology


def test_dem_is_finite_and_physically_plausible(conditioned_dem) -> None:
    terrain, _, hydrology = conditioned_dem
    valid = hydrology.elevation[terrain.domain_mask]
    assert np.isfinite(valid).all()
    assert 200.0 <= float(valid.min()) < float(valid.max()) <= 250.0
    assert terrain.metadata["seed"] == 42
    assert terrain.metadata["drainage_azimuth_degrees"] == 225.0


def test_only_intentional_sinks_remain(conditioned_dem) -> None:
    terrain, urban, hydrology = conditioned_dem
    remaining = find_unfilled_sink_mask(hydrology.elevation, terrain.domain_mask)
    assert not np.any(remaining & ~urban.intentional_depression_mask)
    assert hydrology.metadata["unexpected_remaining_sink_cells"] == 0
    assert urban.intentional_depression_mask.any()
    assert hydrology.metadata["intentional_remaining_sink_cells"] > 0


def test_generation_is_reproducible(conditioned_dem) -> None:
    terrain, _, _ = conditioned_dem
    domain = gpd.read_file(INTERIM_DIR / "domain_epsg32643.gpkg")
    repeated = generate_base_terrain(
        domain,
        resolution_m=100.0,
        base_elevation_m=225.0,
        relief_amplitude_m=6.0,
        regional_gradient_drop_m=8.0,
        seed=42,
        noise_sample_spacing_m=200.0,
    )
    np.testing.assert_array_equal(terrain.domain_mask, repeated.domain_mask)
    np.testing.assert_allclose(
        terrain.elevation[terrain.domain_mask],
        repeated.elevation[repeated.domain_mask],
        rtol=0,
        atol=0,
    )


def test_geotiff_metadata_and_diagnostics(conditioned_dem, tmp_path: Path) -> None:
    terrain, urban, hydrology = conditioned_dem
    dem_path = export_dem(
        hydrology.elevation,
        tmp_path / "dem.tif",
        transform=terrain.transform,
        crs=terrain.crs,
        domain_mask=terrain.domain_mask,
    )
    metadata_path = write_generation_metadata(
        {
            "terrain": terrain.metadata,
            "urban_conditioning": urban.metadata,
            "hydrological_conditioning": hydrology.metadata,
        },
        tmp_path / "metadata.json",
    )
    with rasterio.open(dem_path) as dataset:
        raster = dataset.read(1, masked=True)
        assert dataset.crs.to_epsg() == 32643
        assert abs(dataset.res[0] - 100.0) < 1.0e-9
        assert np.isfinite(raster.compressed()).all()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["terrain"]["seed"] == 42
    assert metadata["hydrological_conditioning"]["backend"].startswith("pyflwdir")

    accumulation = flow_accumulation(
        hydrology.elevation,
        terrain.domain_mask,
        terrain.transform,
    )
    figure_path = plot_dem_diagnostics(
        hydrology.elevation,
        terrain.domain_mask,
        accumulation,
        transform=terrain.transform,
        output_path=tmp_path / "dem_diagnostics.png",
    )
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0
