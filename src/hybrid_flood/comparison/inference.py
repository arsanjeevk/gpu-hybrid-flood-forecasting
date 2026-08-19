"""Shared PyTorch inference and version-specific feature staging."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import xarray as xr

from hybrid_flood.ml.dataset import ResidualDataset
from hybrid_flood.ml.torch_model import PyTorchResidualUNet
from hybrid_flood.ml.torch_train import spatial_patch_origins


def dataset_for_forecast(
    template: ResidualDataset,
    forecast_path: str | Path,
    anuga_path: str | Path,
) -> ResidualDataset:
    """Apply V1 normalization and split definitions to another backend forecast."""
    with xr.open_dataset(forecast_path) as forecast, xr.open_dataset(anuga_path) as anuga:
        aligned = anuga.interp(time=forecast.time, x=forecast.x, y=forecast.y, method="linear")
        state = np.stack(
            (
                forecast.depth.values,
                forecast.x_velocity.values,
                forecast.y_velocity.values,
            ),
            axis=-1,
        ).astype(np.float32)
        reference = np.stack(
            (aligned.depth.values, aligned.x_velocity.values, aligned.y_velocity.values),
            axis=-1,
        ).astype(np.float32)
        times = np.asarray(forecast.time.values, dtype=np.float64)
    if state.shape[1:3] != template.spatial_shape or len(times) - 1 != len(template.input_time_s):
        raise ValueError("V2 forecast shape/time coordinates differ from the V1 training task.")
    if not np.allclose(times[:-1], template.input_time_s) or not np.allclose(
        times[1:], template.target_time_s
    ):
        raise ValueError("V1 and V2 output times are not identical.")
    finite_dynamic = np.isfinite(state).all(axis=(0, 3)) & np.isfinite(reference).all(
        axis=(0, 3)
    )
    common_mask = template.domain_mask & finite_dynamic
    if not common_mask.any():
        raise ValueError("Forecast and reference have no common finite evaluation cells.")
    state[:, ~common_mask] = 0.0
    reference[:, ~common_mask] = 0.0
    metadata = dict(template.metadata)
    metadata["evaluation_forecast"] = str(Path(forecast_path).resolve())
    metadata["common_evaluation_domain_cell_count"] = int(common_mask.sum())
    metadata["cells_excluded_for_nonfinite_forecast"] = int(
        template.domain_mask.sum() - common_mask.sum()
    )
    return replace(
        template,
        state_t=state[:-1],
        target_residual=reference[1:] - state[1:],
        raw_depth_t_plus_1=state[1:, :, :, 0],
        domain_mask=common_mask,
        loss_mask=template.loss_mask & common_mask,
        metadata=metadata,
    )


def predict_device_features(
    model: PyTorchResidualUNet,
    features_nchw: torch.Tensor,
    *,
    patch_size: int,
    batch_size: int,
) -> tuple[torch.Tensor, float]:
    """Infer all time slices while keeping features and predictions on one GPU."""
    if features_nchw.ndim != 4:
        raise ValueError("Feature tensor must be NCHW.")
    count, _, height, width = features_nchw.shape
    predictions = torch.zeros((count, 3, height, width), device=features_nchw.device)
    overlap = torch.zeros((height, width), device=features_nchw.device)
    model.eval()
    if features_nchw.device.type == "cuda":
        torch.cuda.synchronize(features_nchw.device)
    start_time = perf_counter()
    with torch.inference_mode():
        for row, col in spatial_patch_origins(height, width, patch_size):
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                patch = features_nchw[start:stop, :, row : row + patch_size, col : col + patch_size]
                predictions[start:stop, :, row : row + patch_size, col : col + patch_size] += model(
                    patch
                )
            overlap[row : row + patch_size, col : col + patch_size] += 1.0
    predictions /= overlap[None, None]
    if features_nchw.device.type == "cuda":
        torch.cuda.synchronize(features_nchw.device)
    return predictions, perf_counter() - start_time


def v1_features_to_torch(
    dataset: ResidualDataset,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Build all V1 features with NumPy and perform one bulk host-to-GPU copy."""
    start = perf_counter()
    host = np.ascontiguousarray(
        dataset.inputs(np.arange(len(dataset.input_time_s))).transpose(0, 3, 1, 2)
    )
    features = torch.from_numpy(host).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return features, perf_counter() - start
