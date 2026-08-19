"""Masked physical residual loss implemented exclusively in PyTorch."""

from __future__ import annotations

from typing import NamedTuple

import torch


class TorchLossComponents(NamedTuple):
    """Differentiable scalar loss components."""

    total: torch.Tensor
    mse: torch.Tensor
    mae: torch.Tensor
    negative_depth_penalty: torch.Tensor


def residual_correction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    raw_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    channel_scales: torch.Tensor,
    mse_weight: float,
    mae_weight: float,
    negative_depth_weight: float,
) -> TorchLossComponents:
    """Evaluate normalized masked data error and a non-negative-depth penalty.

    Tensors use native PyTorch NCHW layout. Predictions remain in physical
    units; training-set channel standard deviations scale only the objective.
    """
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("Predicted and target residuals must be equal NCHW tensors.")
    scales = channel_scales.reshape(1, -1, 1, 1).clamp_min(torch.finfo(predicted.dtype).eps)
    weights = mask[:, None].to(dtype=predicted.dtype).expand_as(predicted)
    denominator = weights.sum().clamp_min(1.0)
    error = (predicted - target) / scales
    mse = (error.square() * weights).sum() / denominator
    mae = (error.abs() * weights).sum() / denominator
    negative = torch.relu(-(raw_depth + predicted[:, 0])) / scales[:, 0]
    depth_weights = mask.to(dtype=predicted.dtype)
    negative_penalty = (negative.square() * depth_weights).sum() / depth_weights.sum().clamp_min(
        1.0
    )
    total = (
        float(mse_weight) * mse
        + float(mae_weight) * mae
        + float(negative_depth_weight) * negative_penalty
    )
    return TorchLossComponents(total, mse, mae, negative_penalty)
