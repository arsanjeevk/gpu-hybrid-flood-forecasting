"""Publication-ready terrain, forcing, state, error, and hydrograph figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from matplotlib.colors import LightSource, LogNorm, TwoSlopeNorm

from hybrid_flood.viz.style import (
    DEPTH_CMAP,
    DOUBLE_COLUMN_WIDTH,
    RAIN_CMAP,
    RESIDUAL_CMAP,
    ROUGHNESS_CMAP,
    SINGLE_COLUMN_WIDTH,
    SLOPE_CMAP,
    SOLVER_COLORS,
    save_figure_pair,
)


def _display_stride(shape: tuple[int, int], maximum_dimension: int) -> int:
    return max(1, int(np.ceil(max(shape) / maximum_dimension)))


def _extent(dataset: xr.Dataset) -> tuple[float, float, float, float]:
    return (
        float(dataset.x.min()),
        float(dataset.x.max()),
        float(dataset.y.min()),
        float(dataset.y.max()),
    )


def _map_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    axis.ticklabel_format(style="plain", useOffset=False)


def plot_dem_diagnostics(
    elevation: np.ndarray,
    domain_mask: np.ndarray,
    flow_accumulation: np.ndarray,
    *,
    transform: Affine,
    output_path: str | Path,
    maximum_display_dimension: int = 1400,
) -> Path:
    """Plot hillshade, slope magnitude, and D8 contributing-cell accumulation."""
    if not (elevation.shape == domain_mask.shape == flow_accumulation.shape):
        raise ValueError("DEM, mask, and flow accumulation shapes must match.")
    valid = domain_mask & np.isfinite(elevation)
    if not valid.any():
        raise ValueError("DEM contains no valid domain cells.")

    resolution_x = abs(transform.a)
    resolution_y = abs(transform.e)
    working = np.where(valid, elevation, np.nanmedian(elevation[valid]))
    gradient_y, gradient_x = np.gradient(working, resolution_y, resolution_x)
    slope_degrees = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
    hillshade = LightSource(azdeg=315, altdeg=40).hillshade(
        working,
        vert_exag=2.0,
        dx=resolution_x,
        dy=resolution_y,
    )
    hillshade[~valid] = np.nan
    slope_degrees[~valid] = np.nan

    stride = _display_stride(elevation.shape, maximum_display_dimension)
    index = np.s_[::stride, ::stride]
    extent = (
        transform.c,
        transform.c + elevation.shape[1] * transform.a,
        transform.f + elevation.shape[0] * transform.e,
        transform.f,
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.65),
        constrained_layout=True,
    )
    hillshade_image = axes[0].imshow(hillshade[index], cmap="gray", extent=extent)
    slope_image = axes[1].imshow(
        slope_degrees[index],
        cmap=SLOPE_CMAP,
        extent=extent,
        vmin=0,
        vmax=max(1.0, float(np.nanpercentile(slope_degrees[valid], 99))),
    )
    positive = flow_accumulation[valid & (flow_accumulation > 0)]
    accumulation_image = axes[2].imshow(
        flow_accumulation[index],
        cmap=RAIN_CMAP,
        extent=extent,
        norm=LogNorm(
            vmin=max(1.0, float(positive.min(initial=1.0))),
            vmax=max(2.0, float(positive.max(initial=2.0))),
        ),
    )
    for axis, title in zip(
        axes,
        ("DEM hillshade", "Slope", "Flow accumulation"),
        strict=True,
    ):
        _map_axis(axis, title)
    figure.colorbar(hillshade_image, ax=axes[0], label="Relative illumination")
    figure.colorbar(slope_image, ax=axes[1], label="Slope (degrees)")
    figure.colorbar(accumulation_image, ax=axes[2], label="Upstream cells")
    paths = save_figure_pair(figure, output_path)
    requested = Path(output_path)
    return requested if requested.suffix.lower() in {".png", ".pdf"} else paths[1]


def plot_flood_depth_comparison(
    comparison: xr.Dataset,
    key_times_s: Sequence[float],
    output_base: str | Path,
) -> tuple[Path, Path]:
    """Plot synchronized ANUGA, JAX, and hybrid flood-depth maps."""
    sources = ("anuga", "jax", "hybrid")
    selected = comparison.sel(time=list(key_times_s), method="nearest")
    depth = selected["depth"]
    valid = depth.values[np.isfinite(depth.values)]
    vmax = max(1.0e-4, float(np.percentile(valid, 99.5)))
    figure, axes = plt.subplots(
        len(key_times_s),
        len(sources),
        figsize=(DOUBLE_COLUMN_WIDTH, 2.15 * len(key_times_s)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, time_s in enumerate(selected.time.values):
        for column, source in enumerate(sources):
            axis = axes[row, column]
            image = axis.imshow(
                depth.sel(source=source).isel(time=row),
                origin="lower",
                extent=_extent(comparison),
                cmap=DEPTH_CMAP,
                vmin=0.0,
                vmax=vmax,
            )
            _map_axis(axis, f"{source.upper()} — {float(time_s) / 60:.0f} min")
    figure.colorbar(image, ax=axes, label="Flood depth (m)", shrink=0.85)
    return save_figure_pair(figure, output_base)


def plot_spatial_error_maps(
    comparison: xr.Dataset,
    key_times_s: Sequence[float],
    output_base: str | Path,
) -> tuple[Path, Path]:
    """Plot hybrid-minus-ANUGA depth error centered exactly at zero."""
    selected = comparison.sel(time=list(key_times_s), method="nearest")
    error = selected.depth.sel(source="hybrid") - selected.depth.sel(source="anuga")
    finite = np.abs(error.values[np.isfinite(error.values)])
    limit = max(1.0e-5, float(np.percentile(finite, 99)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(
        1,
        len(key_times_s),
        figsize=(DOUBLE_COLUMN_WIDTH, 2.5),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for index, time_s in enumerate(selected.time.values):
        axis = axes[0, index]
        image = axis.imshow(
            error.isel(time=index),
            origin="lower",
            extent=_extent(comparison),
            cmap=RESIDUAL_CMAP,
            norm=norm,
        )
        _map_axis(axis, f"{float(time_s) / 60:.0f} min")
    figure.colorbar(image, ax=axes, label="Hybrid − ANUGA depth (m)", shrink=0.85)
    return save_figure_pair(figure, output_base)


def plot_hydrographs(
    comparison: xr.Dataset,
    monitoring_points: Mapping[str, tuple[float, float]],
    output_base: str | Path,
    *,
    variable: str = "depth",
) -> tuple[Path, Path]:
    """Plot depth or discharge time series at named nearest-grid locations."""
    if variable not in comparison:
        raise ValueError(f"Comparison dataset has no {variable!r} variable.")
    if not monitoring_points:
        raise ValueError("At least one named monitoring point is required.")
    figure, axes = plt.subplots(
        len(monitoring_points),
        1,
        figsize=(DOUBLE_COLUMN_WIDTH, 1.9 * len(monitoring_points)),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    time_minutes = comparison.time.values / 60.0
    for axis, (name, (x, y)) in zip(axes[:, 0], monitoring_points.items(), strict=True):
        point = comparison[variable].sel(x=x, y=y, method="nearest")
        for source in ("anuga", "jax", "hybrid"):
            axis.plot(
                time_minutes,
                point.sel(source=source),
                label=source.upper(),
                color=SOLVER_COLORS[source],
            )
        axis.set_title(
            f"{name} ({float(point.x):.0f} E, {float(point.y):.0f} N)",
            loc="left",
        )
        axis.set_ylabel("Depth (m)" if variable == "depth" else variable)
        axis.grid(True)
    axes[-1, 0].set_xlabel("Simulation time (min)")
    axes[0, 0].legend(ncol=3, frameon=False)
    return save_figure_pair(figure, output_base)


def plot_roughness_map(
    roughness: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    domain_mask: np.ndarray,
    output_base: str | Path,
) -> tuple[Path, Path]:
    """Plot the structured-grid Manning roughness field."""
    values = np.where(domain_mask, roughness, np.nan)
    figure, axis = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, 3.0),
        constrained_layout=True,
    )
    image = axis.imshow(
        values,
        origin="lower",
        extent=(x.min(), x.max(), y.min(), y.max()),
        cmap=ROUGHNESS_CMAP,
    )
    _map_axis(axis, "Manning roughness")
    figure.colorbar(image, ax=axis, label="Manning $n$ (s m$^{-1/3}$)")
    return save_figure_pair(figure, output_base)


def plot_rainfall_hyetograph(
    rainfall: pd.DataFrame,
    output_base: str | Path,
    *,
    scenario: str,
) -> tuple[Path, Path]:
    """Plot the cleaned rainfall intensity series with explicit units."""
    selected = rainfall.loc[rainfall["scenario"] == scenario].sort_values("timestamp")
    if selected.empty:
        raise ValueError(f"Rainfall scenario {scenario!r} was not found.")
    if set(selected["units"].dropna().astype(str)) != {"mm/hr"}:
        raise ValueError("Rainfall hyetograph requires units of mm/hr.")
    timestamp = pd.to_datetime(selected["timestamp"])
    hours = (timestamp - timestamp.iloc[0]).dt.total_seconds() / 3600.0
    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH, 2.4),
        constrained_layout=True,
    )
    axis.step(
        hours,
        selected["rainfall_mm_hr"],
        where="post",
        color="#0072B2",
        linewidth=1.4,
    )
    axis.fill_between(
        hours,
        selected["rainfall_mm_hr"],
        step="post",
        color="#56B4E9",
        alpha=0.45,
    )
    axis.set(
        title=f"Rainfall hyetograph — {scenario}",
        xlabel="Elapsed time (h)",
        ylabel="Rainfall intensity (mm h$^{-1}$)",
    )
    axis.set_ylim(bottom=0)
    axis.grid(True)
    return save_figure_pair(figure, output_base)


def plot_framework_benchmark(
    results: pd.DataFrame,
    output_base: str | Path,
) -> tuple[Path, Path]:
    """Plot repeated JAX/PyTorch timings with one-standard-deviation bars."""
    required = {
        "framework",
        "operation",
        "batch_size",
        "mean_ms",
        "std_ms",
        "device_type",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Benchmark table is missing columns: {sorted(missing)}")
    operations = ("forward", "forward_backward_adamw")
    labels = {
        "forward": "Forward pass",
        "forward_backward_adamw": "Forward + backward + AdamW",
    }
    colors = {"JAX": "#D55E00", "PyTorch": "#0072B2"}
    frameworks = ("JAX", "PyTorch")
    batches = sorted(int(value) for value in results["batch_size"].unique())
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COLUMN_WIDTH, 2.8),
        constrained_layout=True,
        sharey=False,
    )
    x_positions = np.arange(len(batches), dtype=float)
    bar_width = 0.36
    for axis, operation in zip(axes, operations, strict=True):
        subset = results.loc[results["operation"] == operation]
        for framework_index, framework in enumerate(frameworks):
            framework_rows = (
                subset.loc[subset["framework"] == framework]
                .set_index("batch_size")
                .reindex(batches)
            )
            if framework_rows["mean_ms"].isna().any():
                raise ValueError(f"Missing {framework} measurements for {operation}.")
            positions = x_positions + (framework_index - 0.5) * bar_width
            axis.bar(
                positions,
                framework_rows["mean_ms"],
                bar_width,
                yerr=framework_rows["std_ms"],
                capsize=2.5,
                label=framework,
                color=colors[framework],
                alpha=0.9,
            )
        axis.set_title(labels[operation])
        axis.set_xticks(x_positions, labels=batches)
        axis.set_xlabel("Batch size")
        axis.set_ylabel("Wall-clock time (ms)")
        axis.set_ylim(bottom=0)
        axis.grid(True, axis="y")
    device_types = sorted(set(results["device_type"].astype(str)))
    figure.suptitle(f"Matched residual U-Net performance ({'/'.join(device_types).upper()})")
    axes[0].legend(frameon=False)
    return save_figure_pair(figure, output_base)
