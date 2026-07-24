"""Run and persist comparable ANUGA, raw-JAX, and hybrid rollouts."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

# netCDF4 1.7.x can emit this Cython/NumPy size warning on first import with
# NumPy 2.x even though read/write operations are ABI-compatible and succeed.
# Import once here under the narrow warning filter so xarray cannot trigger it
# lazily in the middle of a rollout or test.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"numpy\.ndarray size changed, may indicate binary incompatibility.*",
        category=RuntimeWarning,
    )
    import netCDF4 as _netcdf4  # noqa: F401

from hybrid_flood.hybrid.coupled_forecast import (
    CoupledRolloutResult,
    HybridInputs,
    run_coupled_forecast,
)
from hybrid_flood.jax_solver.numerics import SWEParams, SWEState, integrate_to_outputs
from hybrid_flood.jax_solver.shallow_water_2d import StructuredGrid
from hybrid_flood.ml.dataset import ResidualDataset
from hybrid_flood.ml.residual_net import ResidualUNet

SOURCE_NAMES = ("anuga", "jax", "hybrid")


@dataclass(frozen=True)
class ComparisonRollout:
    """Saved comparison data and summary errors."""

    dataset: xr.Dataset
    metrics: dict[str, float | int]
    output_path: Path


def hybrid_inputs_from_dataset(dataset: ResidualDataset) -> HybridInputs:
    """Move static training normalization fields into a JAX pytree."""
    normalization = dataset.metadata["normalization"]
    return HybridInputs(
        terrain_standardized=jnp.asarray(dataset.terrain_standardized),
        roughness_standardized=jnp.asarray(dataset.roughness_standardized),
        state_mean=jnp.asarray(normalization["state_mean"], dtype=jnp.float32),
        state_std=jnp.asarray(normalization["state_std"], dtype=jnp.float32),
        rainfall_scale_mm_hr=jnp.asarray(
            normalization["rainfall_scale_mm_hr"],
            dtype=jnp.float32,
        ),
        residual_std=jnp.asarray(
            normalization["target_residual_std"],
            dtype=jnp.float32,
        ),
    )


def state_from_dataset(
    dataset: xr.Dataset,
    index: int,
    domain_mask: np.ndarray,
) -> SWEState:
    """Convert depth/velocity fields to finite conserved variables."""
    depth = np.nan_to_num(dataset["depth"].isel(time=index).values, nan=0.0).astype(np.float32)
    velocity_x = np.nan_to_num(
        dataset["x_velocity"].isel(time=index).values,
        nan=0.0,
    ).astype(np.float32)
    velocity_y = np.nan_to_num(
        dataset["y_velocity"].isel(time=index).values,
        nan=0.0,
    ).astype(np.float32)
    depth[~domain_mask] = 0.0
    return SWEState(
        jnp.asarray(depth),
        jnp.asarray(depth * velocity_x),
        jnp.asarray(depth * velocity_y),
    )


def _primitive_fields(
    states: SWEState,
    domain_mask: np.ndarray,
    dry_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = np.array(states.h, dtype=np.float32, copy=True)
    momentum_x = np.asarray(states.hu, dtype=np.float32)
    momentum_y = np.asarray(states.hv, dtype=np.float32)
    wet = depth > dry_tolerance_m
    velocity_x = np.zeros_like(depth)
    velocity_y = np.zeros_like(depth)
    np.divide(momentum_x, depth, out=velocity_x, where=wet)
    np.divide(momentum_y, depth, out=velocity_y, where=wet)
    speed = np.hypot(velocity_x, velocity_y).astype(np.float32)
    for values in (depth, velocity_x, velocity_y, speed):
        values[:, ~domain_mask] = np.nan
    return depth, velocity_x, velocity_y, speed


def _reference_fields(
    path: str | Path,
    *,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as source:
        source_times = np.asarray(source.time.values)
        time_indices = np.searchsorted(source_times, times)
        exact_time_match = np.all(time_indices < len(source_times)) and np.array_equal(
            source_times[time_indices], times
        )
        exact_grid_match = np.array_equal(source.x.values, x) and np.array_equal(
            source.y.values,
            y,
        )
        if exact_time_match and exact_grid_match:
            aligned = source.isel(time=time_indices)
        else:
            aligned = source.interp(
                time=xr.DataArray(times, dims="time"),
                x=xr.DataArray(x, dims="x"),
                y=xr.DataArray(y, dims="y"),
                method="linear",
            )
        return tuple(
            np.array(aligned[name].values, dtype=np.float32, copy=True)
            for name in ("depth", "x_velocity", "y_velocity", "velocity")
        )


def comparison_metrics(dataset: xr.Dataset) -> dict[str, float | int]:
    """Compute raw and hybrid errors against ANUGA over their common domain."""
    reference = dataset.sel(source="anuga")
    metrics: dict[str, float | int] = {"time_count": int(dataset.sizes["time"])}
    for source in ("jax", "hybrid"):
        for variable in ("depth", "x_velocity", "y_velocity", "velocity"):
            difference = dataset[variable].sel(source=source) - reference[variable]
            values = difference.values
            metrics[f"{source}_{variable}_rmse"] = float(np.sqrt(np.nanmean(np.square(values))))
            metrics[f"{source}_{variable}_mae"] = float(np.nanmean(np.abs(values)))
        reference_wet = np.isfinite(reference["depth"].values) & (
            reference["depth"].values > 1.0e-4
        )
        wet_depth_error = (
            dataset["depth"].sel(source=source).values[reference_wet]
            - reference["depth"].values[reference_wet]
        )
        metrics[f"{source}_reference_wet_depth_rmse"] = float(
            np.sqrt(np.mean(np.square(wet_depth_error)))
        )
        metrics[f"{source}_reference_wet_depth_mae"] = float(np.mean(np.abs(wet_depth_error)))
    raw = float(metrics["jax_depth_rmse"])
    hybrid = float(metrics["hybrid_depth_rmse"])
    metrics["hybrid_depth_rmse_skill_percent"] = float(
        100.0 * (raw - hybrid) / max(raw, np.finfo(np.float32).eps)
    )
    raw_wet = float(metrics["jax_reference_wet_depth_rmse"])
    hybrid_wet = float(metrics["hybrid_reference_wet_depth_rmse"])
    metrics["hybrid_reference_wet_depth_rmse_skill_percent"] = float(
        100.0 * (raw_wet - hybrid_wet) / max(raw_wet, np.finfo(np.float32).eps)
    )
    metrics["reference_wet_threshold_m"] = 1.0e-4
    metrics["reference_wet_cell_samples"] = int(
        np.count_nonzero(
            np.isfinite(reference["depth"].values) & (reference["depth"].values > 1.0e-4)
        )
    )
    if "water_volume" in dataset:
        reference_volume = dataset["water_volume"].sel(source="anuga").values
        for source in ("jax", "hybrid"):
            volume = dataset["water_volume"].sel(source=source).values
            difference = volume - reference_volume
            metrics[f"{source}_water_volume_final_error_m3"] = float(difference[-1])
            metrics[f"{source}_water_volume_rmse_m3"] = float(
                np.sqrt(np.mean(np.square(difference)))
            )
        raw_volume = dataset["water_volume"].sel(source="jax").values
        hybrid_volume = dataset["water_volume"].sel(source="hybrid").values
        correction_volume_difference = hybrid_volume - raw_volume
        metrics["hybrid_minus_jax_water_volume_final_m3"] = float(correction_volume_difference[-1])
        metrics["hybrid_minus_jax_water_volume_max_absolute_m3"] = float(
            np.max(np.abs(correction_volume_difference))
        )
        metrics["hybrid_minus_jax_water_volume_max_relative"] = float(
            np.max(
                np.abs(correction_volume_difference)
                / np.maximum(np.abs(raw_volume), np.finfo(np.float64).eps)
            )
        )
    return metrics


def run_comparison_rollout(
    *,
    grid: StructuredGrid,
    params: SWEParams,
    residual_dataset: ResidualDataset,
    model: ResidualUNet,
    model_params: Any,
    raw_jax_path: str | Path,
    anuga_path: str | Path,
    output_path: str | Path,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
    max_physics_steps: int = 20_000,
    max_substeps_per_interval: int = 64,
    correction_relaxation: float = 0.05,
    correction_clip_sigma: float = 5.0,
) -> ComparisonRollout:
    """Run both forecasts from one initial state and save a three-source NetCDF."""
    with xr.open_dataset(raw_jax_path) as raw_source:
        all_times = np.asarray(raw_source.time.values, dtype=np.float32)
        start = all_times[0] if start_time_s is None else float(start_time_s)
        end = all_times[-1] if end_time_s is None else float(end_time_s)
        selection = (all_times >= start - 1.0e-6) & (all_times <= end + 1.0e-6)
        output_times = all_times[selection]
        if len(output_times) < 2:
            raise ValueError("Selected rollout period must contain at least two output times.")
        initial_index = int(np.flatnonzero(selection)[0])
        initial_state = state_from_dataset(raw_source, initial_index, grid.domain_mask)

    pure_result = integrate_to_outputs(
        initial_state,
        params,
        jnp.asarray(output_times),
        max_steps=max_physics_steps,
    )
    if int(np.asarray(pure_result.outputs_written)) != len(output_times):
        raise RuntimeError("Pure JAX rollout exhausted max_physics_steps before completion.")
    hybrid_result: CoupledRolloutResult = run_coupled_forecast(
        initial_state,
        params,
        model_params,
        model.apply,
        hybrid_inputs_from_dataset(residual_dataset),
        jnp.asarray(output_times),
        max_substeps_per_interval=max_substeps_per_interval,
        correction_relaxation=correction_relaxation,
        correction_clip_sigma=correction_clip_sigma,
    )
    hybrid_depth = np.asarray(hybrid_result.states.h)
    if not np.isfinite(hybrid_depth[:, grid.domain_mask]).all():
        raise RuntimeError(
            "Hybrid rollout became non-finite or exhausted max_substeps_per_interval."
        )

    jax_fields = _primitive_fields(
        pure_result.states,
        grid.domain_mask,
        float(np.asarray(params.dry_tolerance)),
    )
    hybrid_fields = _primitive_fields(
        hybrid_result.states,
        grid.domain_mask,
        float(np.asarray(params.dry_tolerance)),
    )
    anuga_fields = _reference_fields(
        anuga_path,
        times=output_times,
        x=grid.x,
        y=grid.y,
    )
    anuga_fields[0][...] = np.maximum(anuga_fields[0], 0.0)
    common_mask = grid.domain_mask.copy()
    for values in anuga_fields:
        common_mask &= np.isfinite(values).all(axis=0)
    for fields in (anuga_fields, jax_fields, hybrid_fields):
        for values in fields:
            values[:, ~common_mask] = np.nan
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name, anuga_values, jax_values, hybrid_values in zip(
        ("depth", "x_velocity", "y_velocity", "velocity"),
        anuga_fields,
        jax_fields,
        hybrid_fields,
        strict=True,
    ):
        variables[name] = (
            ("source", "time", "y", "x"),
            np.stack((anuga_values, jax_values, hybrid_values)),
        )
    elevation = grid.bed.astype(np.float32, copy=True)
    elevation[~common_mask] = np.nan
    variables["elevation"] = (("y", "x"), elevation)
    cell_area_m2 = float(grid.resolution_m**2)
    source_depth = variables["depth"][1]
    water_volume = np.nansum(source_depth, axis=(-2, -1), dtype=np.float64) * cell_area_m2
    variables["water_volume"] = (("source", "time"), water_volume)
    comparison = xr.Dataset(
        variables,
        coords={
            "source": np.asarray(SOURCE_NAMES, dtype=str),
            "time": output_times.astype(np.float64),
            "x": grid.x,
            "y": grid.y,
        },
        attrs={
            "title": "ANUGA, raw-JAX, and autoregressive hybrid forecast comparison",
            "crs": grid.crs.to_string(),
            "grid_resolution_m": grid.resolution_m,
            "hybrid_correction_interval_s": float(np.median(np.diff(output_times))),
            "hybrid_physics_steps_used": int(np.asarray(hybrid_result.physics_steps_used)),
            "correction_relaxation": correction_relaxation,
            "correction_clip_sigma": correction_clip_sigma,
            "neural_correction_water_volume_policy": (
                "preserve finite-volume physics-step water volume"
            ),
            "common_domain_cell_count": int(common_mask.sum()),
        },
    )
    comparison["depth"].attrs["units"] = "m"
    comparison["elevation"].attrs["units"] = "m"
    comparison["water_volume"].attrs.update(
        {
            "units": "m3",
            "long_name": "water volume integrated on the common comparison grid",
        }
    )
    for name in ("x_velocity", "y_velocity", "velocity"):
        comparison[name].attrs["units"] = "m s-1"
    metrics = comparison_metrics(comparison)
    for split_name, indices in (
        ("validation", residual_dataset.val_indices),
        ("test", residual_dataset.test_indices),
    ):
        split_start = float(residual_dataset.input_time_s[int(indices[0])])
        split_end = float(residual_dataset.target_time_s[int(indices[-1])])
        if output_times[0] <= split_start and output_times[-1] >= split_end:
            split_metrics = comparison_metrics(comparison.sel(time=slice(split_start, split_end)))
            metrics.update({f"{split_name}_{name}": value for name, value in split_metrics.items()})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (1, 1, min(grid.bed.shape[0], 256), min(grid.bed.shape[1], 256))
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
        "water_volume": {"zlib": True, "complevel": 4, "dtype": "float64"},
    }
    comparison.to_netcdf(path, engine="netcdf4", encoding=encoding)
    path.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ComparisonRollout(comparison, metrics, path)
