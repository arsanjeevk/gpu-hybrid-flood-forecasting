"""Regression tests for encoded animation raster updates."""

from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
import xarray as xr

from hybrid_flood.viz.animations import (
    animate_flood_depth_comparison,
    animate_residual_correction,
)

pytestmark = pytest.mark.filterwarnings(r"ignore:os\.fork\(\) was called.*:RuntimeWarning")


def _tiny_comparison() -> xr.Dataset:
    sources = ("anuga", "jax", "hybrid")
    depth = np.zeros((3, 2, 16, 16), dtype=np.float32)
    depth[0, 1, 2:-2, 2:-2] = 0.002
    depth[1, 1, 2:-2, 2:-2] = 0.006
    depth[2, 1, 2:-2, 2:-2] = 0.003
    depth[:, :, :2] = np.nan
    depth[:, :, -2:] = np.nan
    depth[:, :, :, :2] = np.nan
    depth[:, :, :, -2:] = np.nan
    return xr.Dataset(
        {"depth": (("source", "time", "y", "x"), depth)},
        coords={
            "source": np.asarray(sources, dtype=str),
            "time": [0.0, 60.0],
            "x": np.arange(16, dtype=float),
            "y": np.arange(16, dtype=float),
        },
    )


def _encoded_frame_difference(path) -> tuple[int, float]:
    reader = imageio.get_reader(path)
    first = reader.get_data(0).astype(np.int16)
    second = reader.get_data(1).astype(np.int16)
    reader.close()
    difference = np.abs(second - first)
    return int(np.any(difference > 3, axis=-1).sum()), float(difference.mean())


def test_flood_animation_encodes_updated_map_pixels(tmp_path) -> None:
    """The encoded second frame must contain raster changes, not only new text."""
    mp4, gif = animate_flood_depth_comparison(
        _tiny_comparison(),
        tmp_path / "flood",
        fps=2,
    )

    changed_pixels, mean_difference = _encoded_frame_difference(mp4)

    assert mp4.is_file() and gif.is_file()
    assert changed_pixels > 5_000
    assert mean_difference > 5.0


def test_residual_animation_encodes_updated_map_pixels(tmp_path) -> None:
    """Residual fields must be redrawn before every captured frame."""
    mp4, gif = animate_residual_correction(
        _tiny_comparison(),
        tmp_path / "residual",
        fps=2,
    )

    changed_pixels, mean_difference = _encoded_frame_difference(mp4)

    assert mp4.is_file() and gif.is_file()
    assert changed_pixels > 5_000
    assert mean_difference > 5.0
