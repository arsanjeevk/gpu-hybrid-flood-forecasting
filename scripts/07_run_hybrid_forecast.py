"""Apply one shared PyTorch corrector to V1 and V2 forecast states."""

# ruff: noqa: E402

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.utils import get_original_cwd
from netCDF4 import Dataset
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

LOGGER = logging.getLogger(__name__)

from hybrid_flood.comparison.inference import (
    dataset_for_forecast,
    predict_device_features,
    v1_features_to_torch,
)
from hybrid_flood.ml.dataset import load_residual_dataset
from hybrid_flood.ml.evaluate import compute_test_metrics
from hybrid_flood.ml.torch_model import PyTorchResidualUNet
from hybrid_flood.ml.torch_train import load_checkpoint


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _model(cfg: DictConfig, input_channels: int) -> PyTorchResidualUNet:
    architecture = cfg.model.architecture
    return PyTorchResidualUNet(
        input_channels=input_channels,
        depth=int(architecture.depth),
        channels=tuple(int(value) for value in architecture.channels),
        activation=str(architecture.activation),
        kernel_size=int(architecture.kernel_size),
        output_channels=int(architecture.output_channels),
    )


def _metrics(dataset, prediction: np.ndarray, threshold: float) -> dict[str, float | int]:
    indices = dataset.test_indices
    return compute_test_metrics(
        prediction[indices],
        dataset.target_residual[indices],
        dataset.raw_depth_t_plus_1[indices],
        dataset.domain_mask,
        flood_threshold_m=threshold,
    )


def _require_finite_metrics(name: str, metrics: dict[str, float | int]) -> None:
    """Reject incomplete evaluations instead of serializing NaN results."""
    invalid = [
        key
        for key, value in metrics.items()
        if isinstance(value, (float, np.floating)) and not np.isfinite(value)
    ]
    if invalid:
        raise RuntimeError(f"{name} produced non-finite metrics: {invalid}")


VARIABLES = ("depth", "x_velocity", "y_velocity", "velocity")


def _create_streaming_output(
    output: Path,
    coordinate_path: Path,
    *,
    duration_s: float,
) -> Path:
    """Create a temporary chunked NetCDF without allocating the result cube."""
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with xr.open_dataset(coordinate_path) as coordinates, Dataset(
        temporary, "w", format="NETCDF4"
    ) as destination:
        sizes = coordinates.sizes
        destination.createDimension("source", 3)
        destination.createDimension("time", sizes["time"])
        destination.createDimension("y", sizes["y"])
        destination.createDimension("x", sizes["x"])
        source = destination.createVariable("source", str, ("source",))
        source[:] = np.asarray(("anuga", "v1", "v2"), dtype=object)
        for name in ("time", "y", "x"):
            values = np.asarray(coordinates[name].values)
            variable = destination.createVariable(name, values.dtype, (name,))
            variable[:] = values
            for key, value in coordinates[name].attrs.items():
                variable.setncattr(key, value)
        chunks = (1, 1, min(256, sizes["y"]), min(256, sizes["x"]))
        for name in VARIABLES:
            variable = destination.createVariable(
                name,
                "f4",
                ("source", "time", "y", "x"),
                fill_value=np.float32(np.nan),
                zlib=True,
                complevel=1,
                shuffle=True,
                chunksizes=chunks,
            )
            variable.setncattr("units", "m" if name == "depth" else "m s-1")
        destination.setncattr(
            "title", "ANUGA reference and matched V1/V2 PyTorch-corrected forecasts"
        )
        destination.setncattr("crs", str(coordinates.attrs.get("crs", "EPSG:32643")))
        destination.setncattr("simulation_duration_s", duration_s)
        destination.setncattr("ai_checkpoint_shared", 1)
    return temporary


def _write_reference(
    temporary: Path,
    reference_path: Path,
    coordinate_path: Path,
) -> None:
    """Interpolate and write one ANUGA variable at a time."""
    with xr.open_dataset(reference_path) as reference, xr.open_dataset(
        coordinate_path
    ) as coordinates, Dataset(temporary, "r+") as destination:
        target = {name: coordinates[name] for name in ("time", "x", "y")}
        same_grid = all(
            reference[name].shape == coordinates[name].shape
            and np.allclose(reference[name].values, coordinates[name].values)
            for name in target
        )
        for name in VARIABLES:
            field = (
                reference[name]
                if same_grid
                else reference[name].interp(**target, method="linear")
            )
            values = np.asarray(field.values, dtype=np.float32)
            destination.variables[name][0, :, :, :] = values
            del values, field


def _write_corrected(
    temporary: Path,
    source_index: int,
    raw_path: Path,
    prediction: np.ndarray,
    domain_mask: np.ndarray,
) -> None:
    """Correct and write fields sequentially to cap host-memory use."""
    with xr.open_dataset(raw_path) as raw, Dataset(temporary, "r+") as destination:
        depth = np.asarray(raw.depth.values, dtype=np.float32).copy()
        depth[1:] = np.maximum(depth[1:] + prediction[..., 0], 0.0)
        depth[:, ~domain_mask] = np.nan
        destination.variables["depth"][source_index, :, :, :] = depth
        del depth

        velocity_x = np.asarray(raw.x_velocity.values, dtype=np.float32).copy()
        velocity_x[1:] += prediction[..., 1]
        velocity_x[:, ~domain_mask] = np.nan
        destination.variables["x_velocity"][source_index, :, :, :] = velocity_x

        velocity_y = np.asarray(raw.y_velocity.values, dtype=np.float32).copy()
        velocity_y[1:] += prediction[..., 2]
        velocity_y[:, ~domain_mask] = np.nan
        destination.variables["y_velocity"][source_index, :, :, :] = velocity_y
        speed = np.hypot(velocity_x, velocity_y, dtype=np.float32)
        destination.variables["velocity"][source_index, :, :, :] = speed
        del velocity_x, velocity_y, speed


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Produce comparable corrected V1/V2 fields and held-out metrics."""
    root = Path(get_original_cwd()).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if bool(cfg.model.execution.require_gpu) and device.type != "cuda":
        raise RuntimeError("The publication comparison requires PyTorch CUDA.")
    if bool(cfg.comparison.hardware.require_t4) and device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        if "T4" not in name.upper():
            raise RuntimeError(f"Expected NVIDIA T4, found {name!r}.")
    LOGGER.info("Loading the chronological residual dataset")
    template = load_residual_dataset(_path(root, cfg.model.dataset.output))
    model = _model(cfg, template.input_channels).to(device)
    load_checkpoint(model, _path(root, cfg.hybrid.inputs.checkpoint), device=device)
    patch_size = int(cfg.model.training.patch_size)
    batch_size = int(cfg.model.training.batch_size)
    output = _path(root, cfg.hybrid.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _create_streaming_output(
        output,
        _path(root, cfg.hybrid.inputs.v1_raw),
        duration_s=float(cfg.comparison.common.duration_s),
    )
    LOGGER.info("Writing the ANUGA reference to the comparison file")
    _write_reference(
        temporary,
        _path(root, cfg.hybrid.inputs.anuga),
        _path(root, cfg.hybrid.inputs.v1_raw),
    )

    from hybrid_flood.comparison.v2_staging import (
        corrections_to_jax,
        v2_features_to_torch,
    )

    # Establish one finite support mask before either model inference. Applying
    # it only during scoring would allow different dry-edge inputs to influence
    # neighbouring CNN predictions through the convolutional receptive field.
    LOGGER.info("Loading V2 and defining the common finite inference domain")
    v2_dataset = dataset_for_forecast(
        template,
        _path(root, cfg.hybrid.inputs.v2_raw),
        _path(root, cfg.hybrid.inputs.anuga),
    )
    common_mask = v2_dataset.domain_mask
    excluded_cells = int(template.domain_mask.sum() - common_mask.sum())
    if excluded_cells:
        LOGGER.warning(
            "Excluding %d cells lacking finite V2 support from both V1 and V2",
            excluded_cells,
        )
    v1_evaluation = replace(
        template,
        domain_mask=common_mask,
        loss_mask=template.loss_mask & common_mask,
    )

    # The stored training dataset was constructed from V1. Reusing its arrays
    # under the common mask avoids another full-domain host copy.
    LOGGER.info("Running V1 PyTorch correction inference")
    v1_features, v1_feature_seconds = v1_features_to_torch(v1_evaluation, device)
    v1_prediction_device, v1_ai_seconds = predict_device_features(
        model, v1_features, patch_size=patch_size, batch_size=batch_size
    )
    del v1_features
    start = perf_counter()
    v1_prediction = v1_prediction_device.permute(0, 2, 3, 1).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    v1_post_seconds = perf_counter() - start
    del v1_prediction_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    threshold = float(cfg.comparison.accuracy.flood_threshold_m)
    v1_metrics = _metrics(v1_evaluation, v1_prediction, threshold)
    _require_finite_metrics("V1", v1_metrics)
    _write_corrected(
        temporary,
        1,
        _path(root, cfg.hybrid.inputs.v1_raw),
        v1_prediction,
        common_mask,
    )
    del v1_prediction, v1_evaluation
    gc.collect()

    LOGGER.info("Assembling V2 features with JAX")
    v2_features, v2_feature_compile, v2_feature_seconds = v2_features_to_torch(v2_dataset)
    LOGGER.info("Running V2 PyTorch correction inference")
    v2_prediction_device, v2_ai_seconds = predict_device_features(
        model, v2_features, patch_size=patch_size, batch_size=batch_size
    )
    del v2_features
    corrected_depth, v2_post_compile, v2_post_seconds = corrections_to_jax(
        v2_dataset.raw_depth_t_plus_1, v2_prediction_device
    )
    del corrected_depth
    v2_prediction = v2_prediction_device.permute(0, 2, 3, 1).cpu().numpy()
    del v2_prediction_device
    if device.type == "cuda":
        torch.cuda.empty_cache()

    v2_metrics = _metrics(v2_dataset, v2_prediction, threshold)
    _require_finite_metrics("V2", v2_metrics)
    LOGGER.info("Writing V2 corrected fields")
    _write_corrected(
        temporary,
        2,
        _path(root, cfg.hybrid.inputs.v2_raw),
        v2_prediction,
        common_mask,
    )
    del v2_prediction, v2_dataset
    gc.collect()
    os.replace(temporary, output)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "hardware": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "jax_runtime_probe": JAX_RUNTIME,
        "configuration": OmegaConf.to_container(cfg.comparison, resolve=True),
        "shared_checkpoint": str(_path(root, cfg.hybrid.inputs.checkpoint)),
        "evaluation_domain": {
            "common_finite_cell_count": int(common_mask.sum()),
            "cells_excluded_for_nonfinite_v2_support": excluded_cells,
            "policy": "intersection of finite ANUGA, V1, and V2 support",
        },
        "transfers": {
            "v1": "one bulk NumPy-to-PyTorch copy and one result copy",
            "v2": "one JAX-to-PyTorch DLPack exchange and one PyTorch-to-JAX DLPack exchange",
        },
        "timings_seconds": {
            "v1_feature_assembly_and_transfer": v1_feature_seconds,
            "v1_ai_inference": v1_ai_seconds,
            "v1_postprocess": v1_post_seconds,
            "v2_jax_feature_compile": v2_feature_compile,
            "v2_jax_feature_execution": v2_feature_seconds,
            "v2_ai_inference": v2_ai_seconds,
            "v2_jax_postprocess_compile": v2_post_compile,
            "v2_jax_postprocess_execution": v2_post_seconds,
        },
        "v1_test_metrics": v1_metrics,
        "v2_test_metrics": v2_metrics,
        "output": str(output),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    LOGGER.info("Hybrid V1/V2 comparison complete: %s", output)


if __name__ == "__main__":
    main()
