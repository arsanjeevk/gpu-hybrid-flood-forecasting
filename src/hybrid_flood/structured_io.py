"""Backend-neutral structured-grid, forcing, and NetCDF utilities.

This module deliberately imports neither JAX nor PyTorch. It is shared by V1
and V2 so geospatial preprocessing cannot confound the backend comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt

from hybrid_flood.dem.synthetic_terrain import domain_to_polygon

MM_HR_TO_M_S = 1.0e-3 / 3600.0


@dataclass(frozen=True)
class StructuredGrid:
    """Regular south-to-north grid shared by both forecast versions."""

    bed: np.ndarray
    manning_n: np.ndarray
    domain_mask: np.ndarray
    x: np.ndarray
    y: np.ndarray
    resolution_m: float
    crs: Any
    north_up_transform: Affine


def load_structured_grid(
    dem_path: str | Path,
    roughness_path: str | Path,
    domain_path: str | Path,
    *,
    resolution_m: float = 50.0,
) -> StructuredGrid:
    """Rasterize DEM, Manning roughness, and the domain once for both versions."""
    if resolution_m <= 0:
        raise ValueError("Structured-grid resolution must be positive.")
    domain = gpd.read_file(domain_path)
    roughness = gpd.read_file(roughness_path)
    if domain.crs is None or not domain.crs.is_projected:
        raise ValueError("Domain must use a projected CRS.")
    roughness = roughness if roughness.crs == domain.crs else roughness.to_crs(domain.crs)
    if "roughness" not in roughness.columns:
        raise ValueError("LULC roughness layer must contain the 'roughness' attribute.")
    polygon = domain_to_polygon(domain)
    xmin, ymin, xmax, ymax = polygon.bounds
    aligned_xmin = floor(xmin / resolution_m) * resolution_m
    aligned_ymin = floor(ymin / resolution_m) * resolution_m
    aligned_xmax = ceil(xmax / resolution_m) * resolution_m
    aligned_ymax = ceil(ymax / resolution_m) * resolution_m
    width = int(round((aligned_xmax - aligned_xmin) / resolution_m))
    height = int(round((aligned_ymax - aligned_ymin) / resolution_m))
    transform = from_origin(aligned_xmin, aligned_ymax, resolution_m, resolution_m)
    x = aligned_xmin + (np.arange(width) + 0.5) * resolution_m
    y = aligned_ymin + (np.arange(height) + 0.5) * resolution_m
    shape = (height, width)
    north_mask = geometry_mask(
        [polygon], out_shape=shape, transform=transform, invert=True, all_touched=False
    )
    north_bed = np.full(shape, np.nan, dtype=np.float32)
    north_nearest = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as source:
        if source.crs != domain.crs:
            raise ValueError(f"DEM CRS {source.crs} does not match domain CRS {domain.crs}.")
        common = {
            "source": rasterio.band(source, 1),
            "src_transform": source.transform,
            "src_crs": source.crs,
            "src_nodata": source.nodata,
            "dst_transform": transform,
            "dst_crs": domain.crs,
            "dst_nodata": np.nan,
        }
        reproject(destination=north_bed, resampling=Resampling.bilinear, **common)
        reproject(destination=north_nearest, resampling=Resampling.nearest, **common)
    missing = north_mask & ~np.isfinite(north_bed)
    north_bed[missing] = north_nearest[missing]
    missing = north_mask & ~np.isfinite(north_bed)
    if missing.any():
        finite = np.isfinite(north_bed)
        if not finite.any():
            raise ValueError("DEM resampling produced no finite elevations.")
        nearest = distance_transform_edt(~finite, return_distances=False, return_indices=True)
        north_bed[missing] = north_bed[nearest[0][missing], nearest[1][missing]]
    if not np.isfinite(north_bed[north_mask]).all():
        raise ValueError("DEM resampling left non-finite domain cells.")
    north_roughness = rasterize(
        (
            (geometry, float(value))
            for geometry, value in zip(roughness.geometry, roughness["roughness"], strict=True)
            if geometry is not None and not geometry.is_empty and np.isfinite(value)
        ),
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype=np.float32,
        all_touched=False,
    )
    if not np.isfinite(north_roughness[north_mask]).all():
        missing_count = int((north_mask & ~np.isfinite(north_roughness)).sum())
        raise ValueError(f"Roughness rasterization left {missing_count} domain cells empty.")
    bed = np.flipud(north_bed).copy()
    manning = np.flipud(north_roughness).copy()
    mask = np.flipud(north_mask).copy()
    bed[~mask] = 0.0
    manning[~mask] = 0.0
    return StructuredGrid(
        bed,
        manning,
        mask,
        x.astype(np.float64),
        y.astype(np.float64),
        float(resolution_m),
        domain.crs,
        transform,
    )


def load_rainfall_series(
    rainfall_path: str | Path,
    *,
    scenario: str,
    default_rate_mm_hr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load explicit mm/hr rainfall and return SI arrays."""
    rainfall = pd.read_parquet(rainfall_path)
    required = {"timestamp", "scenario", "rainfall_mm_hr", "is_missing_timestamp", "units"}
    missing = required.difference(rainfall.columns)
    if missing:
        raise ValueError(f"Rainfall table is missing columns: {sorted(missing)}")
    if set(rainfall["units"].dropna().astype(str)) != {"mm/hr"}:
        raise ValueError("Rainfall forcing must use explicit mm/hr units.")
    selected = rainfall.loc[rainfall["scenario"] == scenario].sort_values("timestamp")
    if selected.empty:
        raise ValueError(f"Rainfall scenario {scenario!r} is unavailable.")
    if selected["is_missing_timestamp"].any() or selected["rainfall_mm_hr"].isna().any():
        raise ValueError("Rainfall contains missing timestamps or intensities.")
    timestamps = pd.to_datetime(selected["timestamp"])
    elapsed = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(np.float32)
    rates = selected["rainfall_mm_hr"].to_numpy(np.float32) * MM_HR_TO_M_S
    return elapsed, rates, float(default_rate_mm_hr) * MM_HR_TO_M_S


def conserved_to_xarray(
    h: np.ndarray,
    hu: np.ndarray,
    hv: np.ndarray,
    output_times_s: np.ndarray,
    grid: StructuredGrid,
    *,
    dry_tolerance_m: float,
    title: str,
    backend: str,
    steps_used: int,
) -> xr.Dataset:
    """Convert backend states to the common ANUGA-compatible schema."""
    depth = np.asarray(h, dtype=np.float32).copy()
    momentum_x = np.asarray(hu, dtype=np.float32)
    momentum_y = np.asarray(hv, dtype=np.float32)
    wet = depth > dry_tolerance_m
    velocity_x = np.zeros_like(depth)
    velocity_y = np.zeros_like(depth)
    np.divide(momentum_x, depth, out=velocity_x, where=wet)
    np.divide(momentum_y, depth, out=velocity_y, where=wet)
    speed = np.hypot(velocity_x, velocity_y).astype(np.float32)
    for values in (depth, velocity_x, velocity_y, speed):
        values[:, ~grid.domain_mask] = np.nan
    elevation = grid.bed.astype(np.float32, copy=True)
    elevation[~grid.domain_mask] = np.nan
    dataset = xr.Dataset(
        {
            "elevation": (("y", "x"), elevation),
            "depth": (("time", "y", "x"), depth),
            "x_velocity": (("time", "y", "x"), velocity_x),
            "y_velocity": (("time", "y", "x"), velocity_y),
            "velocity": (("time", "y", "x"), speed),
        },
        coords={"time": np.asarray(output_times_s, dtype=np.float64), "x": grid.x, "y": grid.y},
        attrs={
            "title": title,
            "crs": grid.crs.to_string(),
            "grid_resolution_m": grid.resolution_m,
            "dry_tolerance_m": dry_tolerance_m,
            "numerical_flux": "HLL with hydrostatic reconstruction",
            "execution_backend": backend,
            "steps_used": int(steps_used),
        },
    )
    dataset["elevation"].attrs["units"] = "m"
    dataset["depth"].attrs["units"] = "m"
    for name in ("x_velocity", "y_velocity", "velocity"):
        dataset[name].attrs["units"] = "m s-1"
    return dataset


def save_netcdf(dataset: xr.Dataset, output_path: str | Path) -> Path:
    """Write a compressed common-schema forecast."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, min(dataset.sizes["y"], 256), min(dataset.sizes["x"], 256))
    encoding = {
        "elevation": {"zlib": True, "complevel": 4, "dtype": "float32"},
        **{
            name: {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "chunksizes": chunks,
            }
            for name in ("depth", "x_velocity", "y_velocity", "velocity")
        },
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path
