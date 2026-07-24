"""High-level API and geospatial I/O for the pure JAX SWE solver."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

import geopandas as gpd
import jax
import jax.numpy as jnp
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
from hybrid_flood.jax_solver.boundary_conditions import BoundaryConfig, all_reflective
from hybrid_flood.jax_solver.numerics import (
    ScanResult,
    SWEParams,
    SWEState,
    integrate_to_outputs,
)

MM_HR_TO_M_S = 1.0e-3 / 3600.0


@dataclass(frozen=True)
class StructuredGrid:
    """Regular, south-to-north array grid used by the JAX solver."""

    bed: np.ndarray
    manning_n: np.ndarray
    domain_mask: np.ndarray
    x: np.ndarray
    y: np.ndarray
    resolution_m: float
    crs: Any
    north_up_transform: Affine


def _aligned_grid_geometry(
    domain: gpd.GeoDataFrame,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, Affine, tuple[int, int], Any]:
    polygon = domain_to_polygon(domain)
    xmin, ymin, xmax, ymax = polygon.bounds
    aligned_xmin = floor(xmin / resolution_m) * resolution_m
    aligned_ymin = floor(ymin / resolution_m) * resolution_m
    aligned_xmax = ceil(xmax / resolution_m) * resolution_m
    aligned_ymax = ceil(ymax / resolution_m) * resolution_m
    width = int(round((aligned_xmax - aligned_xmin) / resolution_m))
    height = int(round((aligned_ymax - aligned_ymin) / resolution_m))
    transform = from_origin(
        aligned_xmin,
        aligned_ymax,
        resolution_m,
        resolution_m,
    )
    x = aligned_xmin + (np.arange(width) + 0.5) * resolution_m
    y = aligned_ymin + (np.arange(height) + 0.5) * resolution_m
    return x, y, transform, (height, width), polygon


def load_structured_grid(
    dem_path: str | Path,
    roughness_path: str | Path,
    domain_path: str | Path,
    *,
    resolution_m: float = 50.0,
) -> StructuredGrid:
    """Resample DEM, LULC Manning n, and domain mask onto one aligned grid."""
    if resolution_m <= 0:
        raise ValueError("Structured-grid resolution must be positive.")
    domain = gpd.read_file(domain_path)
    roughness = gpd.read_file(roughness_path)
    if domain.crs is None or not domain.crs.is_projected:
        raise ValueError("Domain must use a projected CRS.")
    roughness = roughness if roughness.crs == domain.crs else roughness.to_crs(domain.crs)
    if "roughness" not in roughness.columns:
        raise ValueError("LULC roughness layer must contain the 'roughness' attribute.")

    x, y, north_transform, shape, polygon = _aligned_grid_geometry(domain, resolution_m)
    north_mask = geometry_mask(
        [polygon],
        out_shape=shape,
        transform=north_transform,
        invert=True,
        all_touched=False,
    )

    north_bed = np.full(shape, np.nan, dtype=np.float32)
    north_nearest = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as source:
        if source.crs != domain.crs:
            raise ValueError(f"DEM CRS {source.crs} does not match domain CRS {domain.crs}.")
        reproject(
            source=rasterio.band(source, 1),
            destination=north_bed,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=north_transform,
            dst_crs=domain.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        reproject(
            source=rasterio.band(source, 1),
            destination=north_nearest,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=north_transform,
            dst_crs=domain.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    missing_bed = north_mask & ~np.isfinite(north_bed)
    north_bed[missing_bed] = north_nearest[missing_bed]
    missing_bed = north_mask & ~np.isfinite(north_bed)
    if missing_bed.any():
        finite_bed = np.isfinite(north_bed)
        if not finite_bed.any():
            raise ValueError("DEM resampling produced no finite elevations.")
        nearest_indices = distance_transform_edt(
            ~finite_bed,
            return_distances=False,
            return_indices=True,
        )
        north_bed[missing_bed] = north_bed[
            nearest_indices[0][missing_bed],
            nearest_indices[1][missing_bed],
        ]
    if not np.isfinite(north_bed[north_mask]).all():
        raise ValueError("DEM resampling left non-finite cells inside the domain.")

    north_roughness = rasterize(
        (
            (geometry, float(value))
            for geometry, value in zip(
                roughness.geometry,
                roughness["roughness"],
                strict=True,
            )
            if geometry is not None and not geometry.is_empty and np.isfinite(value)
        ),
        out_shape=shape,
        transform=north_transform,
        fill=np.nan,
        dtype=np.float32,
        all_touched=False,
    )
    if not np.isfinite(north_roughness[north_mask]).all():
        missing = int((north_mask & ~np.isfinite(north_roughness)).sum())
        raise ValueError(f"Roughness rasterization left {missing} domain cells unassigned.")

    # Rasterio rows run north-to-south. Flip so array row index and y coordinate
    # both increase toward geographic north, matching the ANUGA NetCDF schema.
    bed = np.flipud(north_bed).copy()
    manning = np.flipud(north_roughness).copy()
    mask = np.flipud(north_mask).copy()
    bed[~mask] = 0.0
    manning[~mask] = 0.0
    return StructuredGrid(
        bed=bed,
        manning_n=manning,
        domain_mask=mask,
        x=x.astype(np.float64),
        y=y.astype(np.float64),
        resolution_m=resolution_m,
        crs=domain.crs,
        north_up_transform=north_transform,
    )


def load_rainfall_series(
    rainfall_path: str | Path,
    *,
    scenario: str,
    default_rate_mm_hr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load explicit mm/hr rainfall and convert it to JAX-ready m/s arrays."""
    rainfall = pd.read_parquet(rainfall_path)
    required = {"timestamp", "scenario", "rainfall_mm_hr", "is_missing_timestamp", "units"}
    missing = required.difference(rainfall.columns)
    if missing:
        raise ValueError(f"Rainfall table is missing columns: {sorted(missing)}")
    units = set(rainfall["units"].dropna().astype(str))
    if units != {"mm/hr"}:
        raise ValueError(f"Expected rainfall units 'mm/hr', found {sorted(units)}")
    selected = rainfall.loc[rainfall["scenario"] == scenario].sort_values("timestamp")
    if selected.empty:
        raise ValueError(f"Rainfall scenario {scenario!r} is unavailable.")
    if selected["is_missing_timestamp"].any() or selected["rainfall_mm_hr"].isna().any():
        raise ValueError("Rainfall contains missing timestamps or intensities.")
    timestamps = pd.to_datetime(selected["timestamp"])
    elapsed = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(np.float32)
    rates = selected["rainfall_mm_hr"].to_numpy(np.float32) * MM_HR_TO_M_S
    return elapsed, rates, default_rate_mm_hr * MM_HR_TO_M_S


class ShallowWaterSolver:
    """Convenience wrapper around the pure, JIT-compatible numerical core."""

    def __init__(
        self,
        grid: StructuredGrid,
        *,
        rainfall_times_s: np.ndarray,
        rainfall_rates_m_s: np.ndarray,
        default_rainfall_m_s: float = 0.0,
        rainfall_multiplier: np.ndarray | None = None,
        boundaries: BoundaryConfig | None = None,
        gravity_m_s2: float = 9.81,
        cfl: float = 0.45,
        dry_tolerance_m: float = 1.0e-4,
        max_dt_s: float = 10.0,
        dtype: Any = jnp.float32,
    ) -> None:
        self.grid = grid
        multiplier = (
            grid.domain_mask.astype(np.float32)
            if rainfall_multiplier is None
            else np.asarray(rainfall_multiplier, dtype=np.float32)
        )
        if multiplier.shape != grid.bed.shape:
            raise ValueError("Rainfall multiplier must match the structured-grid shape.")
        boundary_config = boundaries or all_reflective()
        self.params = SWEParams(
            bed=jnp.asarray(grid.bed, dtype=dtype),
            manning_n=jnp.asarray(grid.manning_n, dtype=dtype),
            domain_mask=jnp.asarray(grid.domain_mask, dtype=bool),
            rainfall_multiplier=jnp.asarray(multiplier, dtype=dtype),
            boundary_types=jnp.asarray(boundary_config.types, dtype=jnp.int32),
            rainfall_times_s=jnp.asarray(rainfall_times_s, dtype=dtype),
            rainfall_rates_m_s=jnp.asarray(rainfall_rates_m_s, dtype=dtype),
            default_rainfall_m_s=jnp.asarray(default_rainfall_m_s, dtype=dtype),
            dx=jnp.asarray(grid.resolution_m, dtype=dtype),
            dy=jnp.asarray(grid.resolution_m, dtype=dtype),
            gravity=jnp.asarray(gravity_m_s2, dtype=dtype),
            cfl=jnp.asarray(cfl, dtype=dtype),
            dry_tolerance=jnp.asarray(dry_tolerance_m, dtype=dtype),
            max_dt=jnp.asarray(max_dt_s, dtype=dtype),
        )
        self.dtype = dtype

    def initial_state(
        self,
        *,
        depth_m: float | np.ndarray = 0.0,
        x_velocity_m_s: float | np.ndarray = 0.0,
        y_velocity_m_s: float | np.ndarray = 0.0,
    ) -> SWEState:
        """Create masked conserved fields from depth and velocity inputs."""
        depth = jnp.broadcast_to(
            jnp.asarray(depth_m, dtype=self.dtype),
            self.params.bed.shape,
        )
        x_velocity = jnp.broadcast_to(
            jnp.asarray(x_velocity_m_s, dtype=self.dtype),
            self.params.bed.shape,
        )
        y_velocity = jnp.broadcast_to(
            jnp.asarray(y_velocity_m_s, dtype=self.dtype),
            self.params.bed.shape,
        )
        depth = jnp.where(self.params.domain_mask, jnp.maximum(depth, 0.0), 0.0)
        return SWEState(depth, depth * x_velocity, depth * y_velocity)

    def run(
        self,
        initial_state: SWEState,
        output_times_s: np.ndarray | jnp.ndarray,
        *,
        max_steps: int = 20_000,
    ) -> ScanResult:
        """Run the compiled adaptive scan and verify all requested outputs."""
        times = jnp.asarray(output_times_s, dtype=self.dtype)
        host_times = np.asarray(output_times_s)
        if host_times.ndim != 1 or len(host_times) < 1:
            raise ValueError("Output times must be a non-empty one-dimensional array.")
        if host_times[0] != 0 or np.any(np.diff(host_times) <= 0):
            raise ValueError("Output times must start at zero and increase strictly.")
        result = integrate_to_outputs(
            initial_state,
            self.params,
            times,
            max_steps=max_steps,
        )
        jax.block_until_ready(result.states.h)
        outputs_written = int(np.asarray(result.outputs_written))
        if outputs_written != len(host_times):
            raise RuntimeError(
                f"max_steps={max_steps} was insufficient: wrote "
                f"{outputs_written}/{len(host_times)} outputs and reached "
                f"t={float(np.asarray(result.final_time_s)):.3f}s."
            )
        return result


def result_to_xarray(
    result: ScanResult,
    grid: StructuredGrid,
    *,
    dry_tolerance_m: float,
) -> xr.Dataset:
    """Convert conserved JAX output to the ANUGA-compatible NetCDF schema."""
    depth = np.array(result.states.h, dtype=np.float32, copy=True)
    hu = np.asarray(result.states.hu, dtype=np.float32)
    hv = np.asarray(result.states.hv, dtype=np.float32)
    mask = grid.domain_mask
    wet = depth > dry_tolerance_m
    x_velocity = np.zeros_like(depth)
    y_velocity = np.zeros_like(depth)
    np.divide(hu, depth, out=x_velocity, where=wet)
    np.divide(hv, depth, out=y_velocity, where=wet)
    velocity = np.hypot(x_velocity, y_velocity).astype(np.float32)
    depth[:, ~mask] = np.nan
    x_velocity[:, ~mask] = np.nan
    y_velocity[:, ~mask] = np.nan
    velocity[:, ~mask] = np.nan
    elevation = grid.bed.astype(np.float32, copy=True)
    elevation[~mask] = np.nan
    dataset = xr.Dataset(
        data_vars={
            "elevation": (("y", "x"), elevation),
            "depth": (("time", "y", "x"), depth),
            "x_velocity": (("time", "y", "x"), x_velocity),
            "y_velocity": (("time", "y", "x"), y_velocity),
            "velocity": (("time", "y", "x"), velocity),
        },
        coords={
            "time": np.asarray(result.output_times_s, dtype=np.float64),
            "x": grid.x,
            "y": grid.y,
        },
        attrs={
            "title": "JAX finite-volume shallow-water raw forecast",
            "crs": grid.crs.to_string(),
            "grid_resolution_m": grid.resolution_m,
            "dry_tolerance_m": dry_tolerance_m,
            "numerical_flux": "HLL with hydrostatic reconstruction",
            "steps_used": int(np.asarray(result.steps_used)),
        },
    )
    dataset["time"].attrs["units_description"] = "seconds since simulation start"
    dataset["elevation"].attrs["units"] = "m"
    dataset["depth"].attrs["units"] = "m"
    for name in ("x_velocity", "y_velocity", "velocity"):
        dataset[name].attrs["units"] = "m s-1"
    return dataset


def save_result_netcdf(dataset: xr.Dataset, output_path: str | Path) -> Path:
    """Write compressed float32 JAX output."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (
        1,
        min(dataset.sizes["y"], 256),
        min(dataset.sizes["x"], 256),
    )
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
