"""Regenerate every report figure and animation from saved model outputs."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from hybrid_flood.dem.hydrological_conditioning import flow_accumulation
from hybrid_flood.viz.animations import (
    animate_flood_depth_comparison,
    animate_residual_correction,
)
from hybrid_flood.viz.static_figures import (
    plot_dem_diagnostics,
    plot_flood_depth_comparison,
    plot_hydrographs,
    plot_rainfall_hyetograph,
    plot_roughness_map,
    plot_spatial_error_maps,
)
from hybrid_flood.viz.training_curves import plot_training_curves

LOGGER = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)


def _path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _monitoring_points(comparison: xr.Dataset) -> dict[str, tuple[float, float]]:
    depth = comparison.depth.sel(source="anuga").values
    peak_flat = int(np.nanargmax(depth))
    _, peak_y, peak_x = np.unravel_index(peak_flat, depth.shape)
    final_error = np.abs(
        comparison.depth.sel(source="v2").isel(time=-1).values
        - comparison.depth.sel(source="anuga").isel(time=-1).values
    )
    error_y, error_x = np.unravel_index(int(np.nanargmax(final_error)), final_error.shape)
    return {
        "Peak ANUGA depth": (
            float(comparison.x.values[peak_x]),
            float(comparison.y.values[peak_y]),
        ),
        "Largest final V2 error": (
            float(comparison.x.values[error_x]),
            float(comparison.y.values[error_y]),
        ),
    }


def _roughness_from_dataset(
    archive_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(archive_path, allow_pickle=False) as archive:
        standardized = archive["roughness_standardized"]
        mask = archive["domain_mask"]
        x = archive["x"]
        y = archive["y"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    normalization = metadata["normalization"]
    roughness = standardized * float(normalization["roughness_std"]) + float(
        normalization["roughness_mean"]
    )
    roughness[~mask] = 0.0
    return roughness, x, y, mask


def _copy_to_report(artifacts: list[Path], report_directory: Path) -> list[Path]:
    report_directory.mkdir(parents=True, exist_ok=True)
    copies: list[Path] = []
    for artifact in artifacts:
        destination = report_directory / artifact.name
        shutil.copy2(artifact, destination)
        copies.append(destination)
    return copies


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Generate static and animated publication products and a manifest."""
    root = Path(get_original_cwd()).resolve()
    viz_cfg = cfg.viz
    anuga_duration_s = float(cfg.anuga.duration_s)
    forecast_duration_s = float(cfg.comparison.common.duration_s)
    if not np.isclose(anuga_duration_s, forecast_duration_s, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            "ANUGA and JAX simulation durations must match before report figures "
            f"are generated ({anuga_duration_s} != {forecast_duration_s})."
        )
    if any(
        float(time_s) < 0.0 or float(time_s) > anuga_duration_s for time_s in viz_cfg.key_times_s
    ):
        raise ValueError("Every configured figure time must lie inside the simulation window.")
    inputs = viz_cfg.inputs
    data_directory = _path(root, viz_cfg.outputs.data_directory)
    report_directory = _path(root, viz_cfg.outputs.report_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    LOGGER.info("Loading comparison rollout")
    with xr.open_dataset(_path(root, inputs.comparison)) as source:
        comparison = source[["depth"]].load()
    if not np.isclose(
        float(comparison.time.values[-1]),
        anuga_duration_s,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Comparison rollout is stale or incomplete: it ends at "
            f"{float(comparison.time.values[-1])} s, expected {anuga_duration_s} s."
        )

    LOGGER.info("Generating solver comparison maps, errors, and hydrographs")
    artifacts.extend(
        plot_flood_depth_comparison(
            comparison,
            list(viz_cfg.key_times_s),
            data_directory / "flood_depth_comparison",
        )
    )
    artifacts.extend(
        plot_spatial_error_maps(
            comparison,
            list(viz_cfg.key_times_s),
            data_directory / "hybrid_depth_error",
        )
    )
    artifacts.extend(
        plot_hydrographs(
            comparison,
            _monitoring_points(comparison),
            data_directory / "monitoring_point_hydrographs",
        )
    )

    LOGGER.info("Generating terrain, roughness, rainfall, and training diagnostics")
    with rasterio.open(_path(root, inputs.dem)) as dem_source:
        elevation = dem_source.read(1).astype(np.float32)
        domain_mask = dem_source.read_masks(1) > 0
        elevation[~domain_mask] = np.nan
        transform = dem_source.transform
    accumulation = flow_accumulation(elevation, domain_mask, transform)
    dem_output = data_directory / "dem_diagnostics.png"
    plot_dem_diagnostics(
        elevation,
        domain_mask,
        accumulation,
        transform=transform,
        output_path=dem_output,
    )
    artifacts.extend((dem_output.with_suffix(".pdf"), dem_output))

    roughness, x, y, roughness_mask = _roughness_from_dataset(
        _path(root, inputs.residual_dataset),
        _path(root, inputs.residual_metadata),
    )
    artifacts.extend(
        plot_roughness_map(
            roughness,
            x,
            y,
            roughness_mask,
            data_directory / "manning_roughness",
        )
    )
    artifacts.extend(
        plot_rainfall_hyetograph(
            pd.read_parquet(_path(root, inputs.rainfall)),
            data_directory / "rainfall_hyetograph",
            scenario=str(viz_cfg.rainfall_scenario),
            simulation_duration_s=anuga_duration_s,
        )
    )
    artifacts.extend(
        plot_training_curves(
            _path(root, inputs.training_log),
            data_directory / "residual_net_training",
        )
    )

    LOGGER.info("Rendering synchronized MP4 and GIF animations")
    artifacts.extend(
        animate_flood_depth_comparison(
            comparison,
            data_directory / "flood_depth_evolution",
            fps=float(viz_cfg.animation.fps),
            frame_stride=int(viz_cfg.animation.frame_stride),
        )
    )
    artifacts.extend(
        animate_residual_correction(
            comparison,
            data_directory / "residual_correction_evolution",
            fps=float(viz_cfg.animation.fps),
            frame_stride=int(viz_cfg.animation.frame_stride),
        )
    )

    copies = _copy_to_report(artifacts, report_directory)
    manifest = {
        "source_inputs": {name: str(_path(root, value)) for name, value in dict(inputs).items()},
        "artifacts": [str(path) for path in artifacts],
        "report_copies": [str(path) for path in copies],
        "key_times_s": list(viz_cfg.key_times_s),
        "simulation_duration_s": anuga_duration_s,
        "animation": {
            "fps": float(viz_cfg.animation.fps),
            "frame_stride": int(viz_cfg.animation.frame_stride),
            "frame_count": int(
                np.ceil(comparison.sizes["time"] / int(viz_cfg.animation.frame_stride))
            ),
        },
    }
    manifest_path = data_directory / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copy2(manifest_path, report_directory / manifest_path.name)
    LOGGER.info(
        "Generated %d artifacts and copied them to %s",
        len(artifacts),
        report_directory,
    )


if __name__ == "__main__":
    main()
