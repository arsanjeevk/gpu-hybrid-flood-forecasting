"""Test-set residual, corrected-state, and flood-extent skill metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_flood.ml.dataset import ResidualDataset
from hybrid_flood.ml.residual_net import ResidualUNet
from hybrid_flood.ml.train import spatial_patch_origins


def predict_residuals(
    model: ResidualUNet,
    params: Any,
    dataset: ResidualDataset,
    indices: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
) -> np.ndarray:
    """Predict complete fields by averaging overlapping edge-covering patches."""
    selected = np.asarray(indices, dtype=np.int32)
    height, width = dataset.spatial_shape
    predictions = np.zeros((len(selected), height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    full_inputs = dataset.inputs(selected)

    @jax.jit
    def apply(inputs):
        return model.apply({"params": params}, inputs)

    for row, column in spatial_patch_origins(height, width, patch_size):
        selection = np.s_[row : row + patch_size, column : column + patch_size]
        for start in range(0, len(selected), batch_size):
            stop = min(start + batch_size, len(selected))
            patch = full_inputs[start:stop, selection[0], selection[1], :]
            output = np.asarray(apply(jnp.asarray(patch)))
            predictions[start:stop, selection[0], selection[1], :] += output
        counts[selection] += 1.0
    predictions /= counts[None, :, :, None]
    predictions[:, ~dataset.domain_mask, :] = 0.0
    return predictions


def compute_test_metrics(
    predicted_residual: np.ndarray,
    target_residual: np.ndarray,
    raw_depth_t_plus_1: np.ndarray,
    domain_mask: np.ndarray,
    *,
    flood_threshold_m: float = 0.05,
    wet_evaluation_threshold_m: float = 1.0e-4,
) -> dict[str, float | int]:
    """Compute RMSE, MAE, and flood-extent critical success index (CSI).

    CSI is ``TP / (TP + FP + FN)`` after classifying corrected and reference
    depth at the configured threshold.  Correct negatives are intentionally
    excluded, as is conventional for event-focused flood-extent skill.
    """
    if flood_threshold_m < 0 or wet_evaluation_threshold_m < 0:
        raise ValueError("Flood and wet-evaluation thresholds cannot be negative.")
    if predicted_residual.shape != target_residual.shape:
        raise ValueError("Predicted and target residual shapes differ.")
    if predicted_residual.shape[:-1] != raw_depth_t_plus_1.shape:
        raise ValueError("Raw depth shape must match residual batch and spatial dimensions.")

    expanded_mask = np.broadcast_to(
        domain_mask[None, :, :, None],
        predicted_residual.shape,
    )
    finite_residuals = np.isfinite(predicted_residual) & np.isfinite(target_residual)
    finite_depth = np.broadcast_to(
        np.isfinite(raw_depth_t_plus_1)[..., None], predicted_residual.shape
    )
    invalid_count = int(np.count_nonzero(expanded_mask & ~(finite_residuals & finite_depth)))
    if invalid_count:
        raise ValueError(
            f"Evaluation domain contains {invalid_count} non-finite predicted, target, "
            "or raw-depth values. Use a common finite V1/V2/reference mask."
        )
    error = predicted_residual - target_residual
    valid_error = error[expanded_mask].reshape(-1, predicted_residual.shape[-1])
    metrics: dict[str, float | int] = {
        "rmse_all_channels": float(np.sqrt(np.mean(np.square(valid_error)))),
        "mae_all_channels": float(np.mean(np.abs(valid_error))),
    }
    names = ("depth", "x_velocity", "y_velocity")
    for channel, name in enumerate(names):
        values = valid_error[:, channel]
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(np.square(values))))
        metrics[f"{name}_mae"] = float(np.mean(np.abs(values)))

    corrected_depth = np.maximum(raw_depth_t_plus_1 + predicted_residual[..., 0], 0.0)
    reference_depth = raw_depth_t_plus_1 + target_residual[..., 0]
    mask = np.broadcast_to(domain_mask, corrected_depth.shape)
    reference_wet = mask & (reference_depth > wet_evaluation_threshold_m)
    wet_depth_error = corrected_depth[reference_wet] - reference_depth[reference_wet]
    metrics.update(
        {
            "wet_evaluation_threshold_m": float(wet_evaluation_threshold_m),
            "reference_wet_cell_samples": int(reference_wet.sum()),
            "reference_wet_depth_rmse": (
                float(np.sqrt(np.mean(np.square(wet_depth_error))))
                if wet_depth_error.size
                else float("nan")
            ),
            "reference_wet_depth_mae": (
                float(np.mean(np.abs(wet_depth_error))) if wet_depth_error.size else float("nan")
            ),
        }
    )
    predicted_flood = corrected_depth >= flood_threshold_m
    reference_flood = reference_depth >= flood_threshold_m
    true_positive = int(np.count_nonzero(mask & predicted_flood & reference_flood))
    false_positive = int(np.count_nonzero(mask & predicted_flood & ~reference_flood))
    false_negative = int(np.count_nonzero(mask & ~predicted_flood & reference_flood))
    denominator = true_positive + false_positive + false_negative
    metrics.update(
        {
            "flood_threshold_m": float(flood_threshold_m),
            "critical_success_index": (float(true_positive / denominator) if denominator else 1.0),
            "true_positive_cells": true_positive,
            "false_positive_cells": false_positive,
            "false_negative_cells": false_negative,
        }
    )
    return metrics


def evaluate_model(
    model: ResidualUNet,
    params: Any,
    dataset: ResidualDataset,
    *,
    patch_size: int,
    batch_size: int,
    flood_threshold_m: float,
    output_path: str | Path | None = None,
) -> dict[str, float | int]:
    """Predict the held-out chronological test block and optionally save metrics."""
    prediction = predict_residuals(
        model,
        params,
        dataset,
        dataset.test_indices,
        patch_size=patch_size,
        batch_size=batch_size,
    )
    indices = dataset.test_indices
    metrics = compute_test_metrics(
        prediction,
        dataset.target_residual[indices],
        dataset.raw_depth_t_plus_1[indices],
        dataset.domain_mask,
        flood_threshold_m=flood_threshold_m,
    )
    raw_jax_metrics = compute_test_metrics(
        np.zeros_like(prediction),
        dataset.target_residual[indices],
        dataset.raw_depth_t_plus_1[indices],
        dataset.domain_mask,
        flood_threshold_m=flood_threshold_m,
    )
    for name, value in raw_jax_metrics.items():
        if name not in {"flood_threshold_m", "wet_evaluation_threshold_m"}:
            metrics[f"raw_jax_{name}"] = value
    metrics["depth_rmse_skill_percent"] = float(
        100.0
        * (raw_jax_metrics["depth_rmse"] - metrics["depth_rmse"])
        / max(float(raw_jax_metrics["depth_rmse"]), np.finfo(np.float32).eps)
    )
    metrics["all_channel_rmse_skill_percent"] = float(
        100.0
        * (raw_jax_metrics["rmse_all_channels"] - metrics["rmse_all_channels"])
        / max(float(raw_jax_metrics["rmse_all_channels"]), np.finfo(np.float32).eps)
    )
    metrics["critical_success_index_change"] = float(
        metrics["critical_success_index"] - raw_jax_metrics["critical_success_index"]
    )
    metrics["reference_wet_depth_rmse_skill_percent"] = float(
        100.0
        * (raw_jax_metrics["reference_wet_depth_rmse"] - metrics["reference_wet_depth_rmse"])
        / max(
            float(raw_jax_metrics["reference_wet_depth_rmse"]),
            np.finfo(np.float32).eps,
        )
    )
    metrics["test_sample_count"] = int(len(indices))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics
