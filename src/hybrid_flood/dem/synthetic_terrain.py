"""Generate deterministic, near-flat synthetic terrain for Gurugram.

The regional component rises gently from south-west to north-east, so the
negative elevation gradient points toward the south-west. This approximates
the broad drainage tendency toward the Najafgarh system without claiming that
the synthetic surface reproduces surveyed topography.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, floor, radians, sin
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from affine import Affine
from noise import pnoise2
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import MultiPolygon, Polygon


@dataclass(frozen=True)
class TerrainGrid:
    """A synthetic elevation grid and its geospatial metadata."""

    elevation: np.ndarray
    domain_mask: np.ndarray
    transform: Affine
    crs: Any
    metadata: dict[str, Any]


def domain_to_polygon(domain: gpd.GeoDataFrame) -> Polygon | MultiPolygon:
    """Convert polygon or closed-line domain geometry to a polygon in memory."""
    polygons: list[Polygon | MultiPolygon] = []
    for geometry in domain.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type in {"Polygon", "MultiPolygon"}:
            polygons.append(geometry)
        elif geometry.geom_type == "LineString" and geometry.is_closed:
            polygons.append(Polygon(geometry))
        elif geometry.geom_type == "MultiLineString":
            polygons.extend(shapely.polygonize(geometry.geoms).geoms)
    if not polygons:
        raise ValueError("Domain must contain polygon geometry or a closed line perimeter.")
    merged = shapely.union_all(polygons)
    if not isinstance(merged, (Polygon, MultiPolygon)):
        raise ValueError("Domain geometries could not be converted to a polygon.")
    return merged


def make_grid(
    domain: gpd.GeoDataFrame, resolution_m: float
) -> tuple[np.ndarray, Affine, Polygon | MultiPolygon]:
    """Create an aligned raster mask covering the project domain."""
    if domain.crs is None or not domain.crs.is_projected:
        raise ValueError("Synthetic terrain requires a domain in a projected CRS.")
    if resolution_m <= 0:
        raise ValueError("Raster resolution must be positive.")

    polygon = domain_to_polygon(domain)
    xmin, ymin, xmax, ymax = polygon.bounds
    aligned_xmin = floor(xmin / resolution_m) * resolution_m
    aligned_ymin = floor(ymin / resolution_m) * resolution_m
    aligned_xmax = ceil(xmax / resolution_m) * resolution_m
    aligned_ymax = ceil(ymax / resolution_m) * resolution_m
    width = int(round((aligned_xmax - aligned_xmin) / resolution_m))
    height = int(round((aligned_ymax - aligned_ymin) / resolution_m))
    transform = from_origin(aligned_xmin, aligned_ymax, resolution_m, resolution_m)
    mask = geometry_mask(
        [polygon],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=False,
    )
    return mask, transform, polygon


def _perlin_control_grid(
    height: int,
    width: int,
    *,
    spacing_m: float,
    noise_scale_m: float,
    seed: int,
    octaves: int,
    persistence: float,
    lacunarity: float,
) -> np.ndarray:
    rows = max(2, ceil(height / spacing_m) + 2)
    columns = max(2, ceil(width / spacing_m) + 2)
    values = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        y = row * spacing_m / noise_scale_m
        for column in range(columns):
            x = column * spacing_m / noise_scale_m
            values[row, column] = pnoise2(
                x,
                y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                base=seed,
            )
    return values


def generate_base_terrain(
    domain: gpd.GeoDataFrame,
    *,
    resolution_m: float = 5.0,
    base_elevation_m: float = 225.0,
    relief_amplitude_m: float = 6.0,
    regional_gradient_drop_m: float = 8.0,
    drainage_azimuth_degrees: float = 225.0,
    seed: int = 42,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    noise_scale_m: float = 2500.0,
    noise_sample_spacing_m: float = 100.0,
) -> TerrainGrid:
    """Generate multi-octave Perlin relief plus a regional drainage gradient."""
    if relief_amplitude_m < 0 or regional_gradient_drop_m < 0:
        raise ValueError("Relief amplitude and regional gradient must be non-negative.")
    if octaves < 1 or noise_scale_m <= 0 or noise_sample_spacing_m <= 0:
        raise ValueError("Noise octaves and spatial scales must be positive.")

    mask, transform, polygon = make_grid(domain, resolution_m)
    height, width = mask.shape
    physical_height = height * resolution_m
    physical_width = width * resolution_m

    coarse = _perlin_control_grid(
        physical_height,
        physical_width,
        spacing_m=noise_sample_spacing_m,
        noise_scale_m=noise_scale_m,
        seed=seed,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
    )
    coarse_transform = from_origin(
        transform.c,
        transform.f,
        noise_sample_spacing_m,
        noise_sample_spacing_m,
    )
    noise_surface = np.empty(mask.shape, dtype=np.float32)
    reproject(
        source=coarse,
        destination=noise_surface,
        src_transform=coarse_transform,
        src_crs=domain.crs,
        dst_transform=transform,
        dst_crs=domain.crs,
        resampling=Resampling.cubic,
    )
    valid_noise = noise_surface[mask]
    noise_midpoint = (float(valid_noise.min()) + float(valid_noise.max())) / 2.0
    noise_half_range = max(
        (float(valid_noise.max()) - float(valid_noise.min())) / 2.0,
        np.finfo(np.float32).eps,
    )
    noise_surface = (noise_surface - noise_midpoint) / noise_half_range
    np.clip(noise_surface, -1.0, 1.0, out=noise_surface)

    # The azimuth is the downslope direction; the elevation gradient points opposite.
    downslope_radians = radians(drainage_azimuth_degrees)
    downslope_east = sin(downslope_radians)
    downslope_north = cos(downslope_radians)
    columns = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    east = (columns + 0.5) * resolution_m
    north = physical_height - (rows + 0.5) * resolution_m
    projection = (
        np.multiply.outer(np.ones(height, dtype=np.float32), east) * -downslope_east
        + np.multiply.outer(north, np.ones(width, dtype=np.float32)) * -downslope_north
    )
    valid_projection = projection[mask]
    projection_range = max(
        float(valid_projection.max() - valid_projection.min()), np.finfo(np.float32).eps
    )
    regional_surface = (
        (projection - float(valid_projection.min())) / projection_range - 0.5
    ) * regional_gradient_drop_m

    elevation = (base_elevation_m + relief_amplitude_m * noise_surface + regional_surface).astype(
        np.float32
    )
    elevation[~mask] = np.nan
    metadata = {
        "method": "multi-octave Perlin noise plus planar regional gradient",
        "seed": seed,
        "resolution_m": resolution_m,
        "base_elevation_m": base_elevation_m,
        "relief_amplitude_m": relief_amplitude_m,
        "regional_gradient_drop_m": regional_gradient_drop_m,
        "drainage_azimuth_degrees": drainage_azimuth_degrees,
        "drainage_azimuth_convention": "degrees clockwise from north",
        "octaves": octaves,
        "persistence": persistence,
        "lacunarity": lacunarity,
        "noise_scale_m": noise_scale_m,
        "noise_sample_spacing_m": noise_sample_spacing_m,
        "domain_bounds": [float(value) for value in polygon.bounds],
        "shape": [height, width],
        "crs": domain.crs.to_string(),
    }
    return TerrainGrid(elevation, mask, transform, domain.crs, metadata)
