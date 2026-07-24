"""Reproject raw vector layers into a common projected CRS."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyproj import CRS

from hybrid_flood.data.load_shapefiles import load_all_shapefiles

TARGET_CRS = "EPSG:32643"


def reproject_layer(gdf: gpd.GeoDataFrame, target_crs: str | CRS = TARGET_CRS) -> gpd.GeoDataFrame:
    """Return a copy of a layer in ``target_crs`` without mutating its source."""
    if gdf.crs is None:
        raise ValueError("Cannot reproject a layer without a declared CRS.")
    target = CRS.from_user_input(target_crs)
    if gdf.crs == target:
        return gdf.copy()
    return gdf.to_crs(target)


def reproject_all_layers(
    raw_data_dir: str | Path,
    interim_data_dir: str | Path,
    *,
    target_crs: str | CRS = TARGET_CRS,
    layers: dict[str, gpd.GeoDataFrame] | None = None,
) -> dict[str, Path]:
    """Reproject all validated layers and write GeoPackage copies."""
    if layers is None:
        layers, _ = load_all_shapefiles(raw_data_dir, strict=True)

    output_dir = Path(interim_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epsg = CRS.from_user_input(target_crs).to_epsg()
    suffix = f"epsg{epsg}" if epsg is not None else "projected"

    outputs: dict[str, Path] = {}
    for layer_name, layer in layers.items():
        projected = reproject_layer(layer, target_crs)
        output_path = output_dir / f"{layer_name}_{suffix}.gpkg"
        projected.to_file(output_path, layer=layer_name, driver="GPKG", index=False)
        outputs[layer_name] = output_path
    return outputs
