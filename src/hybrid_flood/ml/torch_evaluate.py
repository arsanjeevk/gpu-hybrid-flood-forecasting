"""Patchwise PyTorch inference and held-out flood metrics."""

from __future__ import annotations

import numpy as np
import torch

from hybrid_flood.ml.dataset import ResidualDataset
from hybrid_flood.ml.evaluate import compute_test_metrics
from hybrid_flood.ml.torch_model import PyTorchResidualUNet
from hybrid_flood.ml.torch_train import spatial_patch_origins


def predict_residuals(
    model: PyTorchResidualUNet,
    dataset: ResidualDataset,
    indices: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Infer full fields, transferring each input batch to CUDA only once."""
    selected = np.asarray(indices, dtype=np.int32)
    inputs = dataset.inputs(selected)
    height, width = dataset.spatial_shape
    output = np.zeros((len(selected), height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for row, col in spatial_patch_origins(height, width, patch_size):
            selection = np.s_[row : row + patch_size, col : col + patch_size]
            for start in range(0, len(selected), batch_size):
                stop = min(start + batch_size, len(selected))
                host = np.ascontiguousarray(
                    inputs[start:stop, selection[0], selection[1]].transpose(0, 3, 1, 2)
                )
                prediction = model(torch.from_numpy(host).to(device, non_blocking=True))
                output[start:stop, selection[0], selection[1]] += (
                    prediction.permute(0, 2, 3, 1).cpu().numpy()
                )
            counts[selection] += 1.0
    output /= counts[None, :, :, None]
    output[:, ~dataset.domain_mask] = 0.0
    return output


def evaluate_model(
    model: PyTorchResidualUNet,
    dataset: ResidualDataset,
    *,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    flood_threshold_m: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Evaluate the frozen chronological test block."""
    prediction = predict_residuals(
        model,
        dataset,
        dataset.test_indices,
        patch_size=patch_size,
        batch_size=batch_size,
        device=device,
    )
    indices = dataset.test_indices
    metrics = compute_test_metrics(
        prediction,
        dataset.target_residual[indices],
        dataset.raw_depth_t_plus_1[indices],
        dataset.domain_mask,
        flood_threshold_m=flood_threshold_m,
    )
    metrics["test_sample_count"] = int(len(indices))
    return prediction, metrics
