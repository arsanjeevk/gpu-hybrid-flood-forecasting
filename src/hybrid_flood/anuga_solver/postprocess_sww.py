"""Convert unstructured ANUGA SWW output to a regular xarray training grid."""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path
from typing import Any

import matplotlib.tri as mtri
import numpy as np
import xarray as xr


def _regular_grid(
    x: np.ndarray,
    y: np.ndarray,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    xmin = floor(float(x.min()) / resolution_m) * resolution_m
    ymin = floor(float(y.min()) / resolution_m) * resolution_m
    xmax = ceil(float(x.max()) / resolution_m) * resolution_m
    ymax = ceil(float(y.max()) / resolution_m) * resolution_m
    grid_x = np.arange(xmin + resolution_m / 2.0, xmax, resolution_m)
    grid_y = np.arange(ymin + resolution_m / 2.0, ymax, resolution_m)
    return grid_x, grid_y


def _barycentric_mapping(
    node_x: np.ndarray,
    node_y: np.ndarray,
    triangles: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = mtri.Triangulation(node_x, node_y, triangles)
    finder = mesh.get_trifinder()
    xx, yy = np.meshgrid(grid_x, grid_y)
    triangle_index = finder(xx.ravel(), yy.ravel()).astype(np.int64)
    valid_flat = np.flatnonzero(triangle_index >= 0)
    selected_triangles = triangles[triangle_index[valid_flat]]
    x0 = node_x[selected_triangles[:, 0]]
    y0 = node_y[selected_triangles[:, 0]]
    x1 = node_x[selected_triangles[:, 1]]
    y1 = node_y[selected_triangles[:, 1]]
    x2 = node_x[selected_triangles[:, 2]]
    y2 = node_y[selected_triangles[:, 2]]
    px = xx.ravel()[valid_flat]
    py = yy.ravel()[valid_flat]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if np.isclose(denominator, 0.0).any():
        raise ValueError("SWW mesh contains degenerate triangles.")
    weight0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denominator
    weight1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denominator
    weights = np.column_stack((weight0, weight1, 1.0 - weight0 - weight1))
    return valid_flat, selected_triangles, weights.astype(np.float32)


def _interpolate_nodes(
    values: np.ndarray,
    valid_flat: np.ndarray,
    selected_triangles: np.ndarray,
    weights: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    output = np.full(output_shape[0] * output_shape[1], np.nan, dtype=np.float32)
    selected_values = values[selected_triangles]
    output[valid_flat] = np.sum(selected_values * weights, axis=1, dtype=np.float32)
    return output.reshape(output_shape)


def postprocess_sww(
    sww_path: str | Path,
    output_path: str | Path,
    *,
    grid_resolution_m: float = 50.0,
    dry_tolerance_m: float = 0.001,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Interpolate SWW node fields to ``(time, y, x)`` and save NetCDF."""
    if grid_resolution_m <= 0 or dry_tolerance_m < 0:
        raise ValueError("Grid resolution must be positive and dry tolerance non-negative.")
    source_path = Path(sww_path)
    with xr.open_dataset(source_path, engine="netcdf4") as source:
        required = {"x", "y", "volumes", "elevation", "stage", "xmomentum", "ymomentum", "time"}
        missing = required.difference(source.variables)
        if missing:
            raise ValueError(f"SWW file is missing variables: {sorted(missing)}")
        x_offset = float(source.attrs.get("xllcorner", 0.0))
        y_offset = float(source.attrs.get("yllcorner", 0.0))
        node_x = source["x"].values.astype(np.float64) + x_offset
        node_y = source["y"].values.astype(np.float64) + y_offset
        triangles = source["volumes"].values.astype(np.int64)
        grid_x, grid_y = _regular_grid(node_x, node_y, grid_resolution_m)
        valid_flat, selected_triangles, weights = _barycentric_mapping(
            node_x,
            node_y,
            triangles,
            grid_x,
            grid_y,
        )
        output_shape = (len(grid_y), len(grid_x))
        elevation = _interpolate_nodes(
            source["elevation"].values.astype(np.float32),
            valid_flat,
            selected_triangles,
            weights,
            output_shape,
        )
        times = source["time"].values.astype(np.float64)
        if len(times) == 0 or not np.isfinite(times).all():
            raise ValueError("SWW time coordinate is empty or non-finite.")
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("SWW time coordinate must be strictly increasing.")
        shape = (len(times), *output_shape)
        depth = np.full(shape, np.nan, dtype=np.float32)
        x_velocity = np.full(shape, np.nan, dtype=np.float32)
        y_velocity = np.full(shape, np.nan, dtype=np.float32)

        for time_index in range(len(times)):
            stage = _interpolate_nodes(
                source["stage"].isel(number_of_timesteps=time_index).values,
                valid_flat,
                selected_triangles,
                weights,
                output_shape,
            )
            x_momentum = _interpolate_nodes(
                source["xmomentum"].isel(number_of_timesteps=time_index).values,
                valid_flat,
                selected_triangles,
                weights,
                output_shape,
            )
            y_momentum = _interpolate_nodes(
                source["ymomentum"].isel(number_of_timesteps=time_index).values,
                valid_flat,
                selected_triangles,
                weights,
                output_shape,
            )
            depth_at_time = np.maximum(stage - elevation, 0.0)
            wet = np.isfinite(depth_at_time) & (depth_at_time > dry_tolerance_m)
            depth[time_index] = depth_at_time
            x_velocity[time_index, wet] = x_momentum[wet] / depth_at_time[wet]
            y_velocity[time_index, wet] = y_momentum[wet] / depth_at_time[wet]
            x_velocity[time_index, ~wet & np.isfinite(depth_at_time)] = 0.0
            y_velocity[time_index, ~wet & np.isfinite(depth_at_time)] = 0.0

        velocity = np.hypot(x_velocity, y_velocity).astype(np.float32)
        inside_mesh = np.isfinite(elevation)
        output_fields = {
            "depth": depth,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "velocity": velocity,
        }
        for field_name, field in output_fields.items():
            nonfinite_inside = int((~np.isfinite(field[:, inside_mesh])).sum())
            if nonfinite_inside:
                raise ValueError(
                    f"Postprocessed {field_name} contains {nonfinite_inside} "
                    "non-finite values inside the mesh."
                )
        if np.any(depth[:, inside_mesh] < 0):
            raise ValueError("Postprocessed water depth contains negative values.")
        zone = int(source.attrs.get("zone", -1))
        hemisphere = str(source.attrs.get("hemisphere", "undefined")).lower()
        if 1 <= zone <= 60:
            epsg = (32700 if hemisphere == "southern" else 32600) + zone
            crs_name = f"EPSG:{epsg}"
        else:
            crs_name = "LOCAL_PROJECTED_UNKNOWN"
        output = xr.Dataset(
            data_vars={
                "elevation": (("y", "x"), elevation),
                "depth": (("time", "y", "x"), depth),
                "x_velocity": (("time", "y", "x"), x_velocity),
                "y_velocity": (("time", "y", "x"), y_velocity),
                "velocity": (("time", "y", "x"), velocity),
            },
            coords={"time": times, "x": grid_x, "y": grid_y},
            attrs={
                "title": "ANUGA baseline regular-grid reference solution",
                "source_sww": str(source_path),
                "crs": crs_name,
                "grid_resolution_m": grid_resolution_m,
                "dry_tolerance_m": dry_tolerance_m,
                "interpolation": "piecewise-linear barycentric interpolation",
            },
        )

    output["time"].attrs.update({"units_description": "seconds since simulation start"})
    output["x"].attrs.update({"units": "m", "standard_name": "projection_x_coordinate"})
    output["y"].attrs.update({"units": "m", "standard_name": "projection_y_coordinate"})
    output["elevation"].attrs.update({"units": "m", "long_name": "bed elevation"})
    output["depth"].attrs.update({"units": "m", "long_name": "water depth"})
    for name in ("x_velocity", "y_velocity", "velocity"):
        output[name].attrs["units"] = "m s-1"

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, min(len(grid_y), 256), min(len(grid_x), 256))
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
    output.to_netcdf(destination, engine="netcdf4", encoding=encoding)
    report = {
        "source_sww": str(source_path),
        "output_netcdf": str(destination),
        "time_steps": len(times),
        "grid_shape": [len(grid_y), len(grid_x)],
        "grid_resolution_m": grid_resolution_m,
        "inside_mesh_cell_count": len(valid_flat),
        "variables": list(output.data_vars),
        "minimum_depth_m": float(np.min(depth[:, inside_mesh])),
        "maximum_depth_m": float(np.max(depth[:, inside_mesh])),
        "maximum_velocity_m_s": float(np.max(velocity[:, inside_mesh])),
        "nonfinite_inside_mesh_count": 0,
    }
    return output, report
