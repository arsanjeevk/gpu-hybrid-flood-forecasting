"""Synchronized H.264/GIF animations generated directly from comparison NetCDF."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import PowerNorm, SymLogNorm

from hybrid_flood.viz.style import DEPTH_CMAP, RESIDUAL_CMAP


def _paths(output_base: str | Path) -> tuple[Path, Path]:
    base = Path(output_base)
    if base.suffix.lower() in {".mp4", ".gif"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    return base.with_suffix(".mp4"), base.with_suffix(".gif")


def _frame(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def _writers(mp4_path: Path, gif_path: Path, fps: float):
    mp4 = imageio.get_writer(
        mp4_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=2,
        ffmpeg_log_level="error",
    )
    gif = imageio.get_writer(
        gif_path,
        format="GIF-PIL",
        mode="I",
        fps=fps,
        loop=0,
    )
    return mp4, gif


def animate_flood_depth_comparison(
    comparison: xr.Dataset,
    output_base: str | Path,
    *,
    fps: float = 6.0,
    frame_stride: int = 1,
) -> tuple[Path, Path]:
    """Export synchronized ANUGA/JAX/hybrid flood-depth animations."""
    if fps <= 0 or frame_stride < 1:
        raise ValueError("Animation FPS and frame stride must be positive.")
    sources = ("anuga", "jax", "hybrid")
    depth = comparison.depth
    finite = depth.values[np.isfinite(depth.values)]
    vmax = max(1.0e-4, float(np.percentile(finite, 99.5)))
    norm = PowerNorm(gamma=0.4, vmin=0.0, vmax=vmax)
    cmap = DEPTH_CMAP.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    extent = (
        float(comparison.x.min()),
        float(comparison.x.max()),
        float(comparison.y.min()),
        float(comparison.y.max()),
    )
    domain_mask = np.isfinite(depth.sel(source="anuga").isel(time=-1).values)

    def visible(values: xr.DataArray) -> np.ma.MaskedArray:
        array = np.asarray(values)
        return np.ma.masked_where(~np.isfinite(array) | (array < 1.0e-4), array)

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    images = [
        axis.imshow(
            visible(depth.sel(source=source).isel(time=0)),
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
        )
        for axis, source in zip(axes, sources, strict=True)
    ]
    for axis in axes:
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
        axis.set_aspect("equal")
        axis.ticklabel_format(style="plain", useOffset=False)
        axis.contour(
            domain_mask.astype(float),
            levels=[0.5],
            colors="#666666",
            linewidths=0.6,
            origin="lower",
            extent=extent,
        )
    figure.colorbar(images[0], ax=axes, label="Flood depth (m)", shrink=0.82)
    mp4_path, gif_path = _paths(output_base)
    mp4_writer, gif_writer = _writers(mp4_path, gif_path, fps)
    try:
        for time_index in range(0, comparison.sizes["time"], frame_stride):
            time_minutes = float(comparison.time.values[time_index]) / 60.0
            for axis, image, source in zip(axes, images, sources, strict=True):
                image.set_data(visible(depth.sel(source=source).isel(time=time_index)))
                axis.set_title(f"{source.upper()} — $t$ = {time_minutes:.0f} min")
            rendered = _frame(figure)
            mp4_writer.append_data(rendered)
            gif_writer.append_data(rendered)
    finally:
        mp4_writer.close()
        gif_writer.close()
        plt.close(figure)
    return mp4_path, gif_path


def animate_residual_correction(
    comparison: xr.Dataset,
    output_base: str | Path,
    *,
    fps: float = 6.0,
    frame_stride: int = 1,
) -> tuple[Path, Path]:
    """Animate the accumulated hybrid-minus-raw-JAX depth correction field."""
    if fps <= 0 or frame_stride < 1:
        raise ValueError("Animation FPS and frame stride must be positive.")
    residual = comparison.depth.sel(source="hybrid") - comparison.depth.sel(source="jax")
    finite = np.abs(residual.values[np.isfinite(residual.values)])
    limit = max(1.0e-5, float(np.percentile(finite, 99)))
    norm = SymLogNorm(
        linthresh=max(limit * 0.02, 1.0e-6),
        linscale=0.8,
        vmin=-limit,
        vmax=limit,
        base=10,
    )
    cmap = RESIDUAL_CMAP.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    extent = (
        float(comparison.x.min()),
        float(comparison.x.max()),
        float(comparison.y.min()),
        float(comparison.y.max()),
    )
    domain_mask = np.isfinite(residual.isel(time=-1).values)
    figure, axis = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    image = axis.imshow(
        residual.isel(time=0),
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
    )
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.set_aspect("equal")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.contour(
        domain_mask.astype(float),
        levels=[0.5],
        colors="#666666",
        linewidths=0.6,
        origin="lower",
        extent=extent,
    )
    figure.colorbar(image, ax=axis, label="Hybrid − raw JAX depth (m)")
    mp4_path, gif_path = _paths(output_base)
    mp4_writer, gif_writer = _writers(mp4_path, gif_path, fps)
    try:
        for time_index in range(0, comparison.sizes["time"], frame_stride):
            time_minutes = float(comparison.time.values[time_index]) / 60.0
            image.set_data(residual.isel(time=time_index))
            axis.set_title(f"Residual correction — $t$ = {time_minutes:.0f} min")
            rendered = _frame(figure)
            mp4_writer.append_data(rendered)
            gif_writer.append_data(rendered)
    finally:
        mp4_writer.close()
        gif_writer.close()
        plt.close(figure)
    return mp4_path, gif_path
