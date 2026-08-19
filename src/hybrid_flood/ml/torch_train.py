"""T4-conscious PyTorch training with chronological validation and checkpoints."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hybrid_flood.ml.dataset import ResidualDataset
from hybrid_flood.ml.torch_losses import residual_correction_loss
from hybrid_flood.ml.torch_model import PyTorchResidualUNet


@dataclass(frozen=True)
class TorchTrainingResult:
    """Best model and logged training artifacts."""

    model: PyTorchResidualUNet
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    checkpoint_path: Path
    history_path: Path


def spatial_patch_origins(height: int, width: int, patch_size: int) -> list[tuple[int, int]]:
    """Cover the complete domain including east/north edge patches."""
    if patch_size < 1 or patch_size > min(height, width):
        raise ValueError("Patch size must fit both spatial dimensions.")

    def starts(length: int) -> list[int]:
        values = list(range(0, length - patch_size + 1, patch_size))
        final = length - patch_size
        if not values or values[-1] != final:
            values.append(final)
        return values

    return [(row, col) for row in starts(height) for col in starts(width)]


def _records(
    indices: np.ndarray,
    shape: tuple[int, int],
    patch_size: int,
    *,
    training: bool,
    patches_per_sample: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    height, width = shape
    if training:
        result = [
            (
                int(index),
                int(rng.integers(0, height - patch_size + 1)),
                int(rng.integers(0, width - patch_size + 1)),
            )
            for index in indices
            for _ in range(patches_per_sample)
        ]
        rng.shuffle(result)
        return result
    return [
        (int(index), row, col)
        for index in indices
        for row, col in spatial_patch_origins(height, width, patch_size)
    ]


def _batch(
    dataset: ResidualDataset,
    records: list[tuple[int, int, int]],
    patch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    unique = np.unique([record[0] for record in records])
    inputs = dataset.inputs(unique)
    lookup = {int(index): position for position, index in enumerate(unique)}
    input_patches, targets, depths, masks = [], [], [], []
    for index, row, col in records:
        selection = np.s_[row : row + patch_size, col : col + patch_size]
        input_patches.append(inputs[lookup[index]][selection])
        targets.append(dataset.target_residual[index][selection])
        depths.append(dataset.raw_depth_t_plus_1[index][selection])
        masks.append(dataset.loss_mask[selection])
    return {
        "inputs": torch.from_numpy(np.stack(input_patches).transpose(0, 3, 1, 2)).to(device),
        "target": torch.from_numpy(np.stack(targets).transpose(0, 3, 1, 2)).to(device),
        "raw_depth": torch.from_numpy(np.stack(depths)).to(device),
        "mask": torch.from_numpy(np.stack(masks)).to(device),
    }


def _epoch(
    model: PyTorchResidualUNet,
    dataset: ResidualDataset,
    indices: np.ndarray,
    config: Any,
    device: torch.device,
    channel_scales: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    records = _records(
        indices,
        dataset.spatial_shape,
        int(config.patch_size),
        training=training,
        patches_per_sample=int(config.patches_per_sample) if training else 1,
        rng=rng,
    )
    totals: list[list[float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(records), int(config.batch_size)):
            batch = _batch(
                dataset,
                records[start : start + int(config.batch_size)],
                int(config.patch_size),
                device,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["inputs"])
            components = residual_correction_loss(
                prediction,
                batch["target"],
                batch["raw_depth"],
                batch["mask"],
                channel_scales=channel_scales,
                mse_weight=float(config.loss.mse_weight),
                mae_weight=float(config.loss.mae_weight),
                negative_depth_weight=float(config.loss.negative_depth_weight),
            )
            if optimizer is not None:
                components.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip_norm))
                optimizer.step()
            totals.append([float(value.detach().cpu()) for value in components])
    means = np.mean(np.asarray(totals, dtype=np.float64), axis=0)
    return dict(zip(("total", "mse", "mae", "negative_depth_penalty"), means, strict=True))


def save_checkpoint(
    model: PyTorchResidualUNet,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> Path:
    """Save portable state_dict weights and JSON provenance."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the binary restricted to tensors so PyTorch's safe weights-only
    # loader never needs to allowlist Python or NumPy object types.
    torch.save(model.state_dict(), destination)
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def load_checkpoint(
    model: PyTorchResidualUNet,
    path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Load weights using safe weights-only deserialization."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    state_dict = torch.load(source, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    metadata_path = source.with_suffix(".json")
    return (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )


def train_model(
    model: PyTorchResidualUNet,
    dataset: ResidualDataset,
    config: Any,
    *,
    seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
    history_path: str | Path,
) -> TorchTrainingResult:
    """Train the shared V1/V2 model with early stopping on validation loss."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    total_epochs = int(config.epochs)
    warmup = int(config.warmup_epochs)

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return max((epoch + 1) / max(warmup, 1), 0.1)
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scales = torch.tensor(
        dataset.metadata["normalization"]["target_residual_std"],
        dtype=torch.float32,
        device=device,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    best = float("inf")
    best_epoch = 0
    patience_reference = float("inf")
    stale = 0
    for epoch in range(1, total_epochs + 1):
        train = _epoch(
            model,
            dataset,
            dataset.train_indices,
            config,
            device,
            scales,
            optimizer=optimizer,
            rng=rng,
        )
        validation = _epoch(
            model,
            dataset,
            dataset.val_indices,
            config,
            device,
            scales,
            optimizer=None,
            rng=rng,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        validation_loss = validation["total"]
        if validation_loss < best:
            best = validation_loss
            best_epoch = epoch
            save_checkpoint(
                model,
                checkpoint_path,
                metadata={
                    "framework": "pytorch",
                    "epoch": epoch,
                    "validation_loss": best,
                    "input_channels": dataset.input_channels,
                    "output_channels": 3,
                },
            )
        if validation_loss < patience_reference - float(config.early_stopping.min_delta):
            patience_reference = validation_loss
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= int(config.early_stopping.patience):
            break
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with history_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    load_checkpoint(model, checkpoint_path, device=device)
    return TorchTrainingResult(
        model, best_epoch, best, len(history), Path(checkpoint_path), history_file
    )
