"""Optax training, early stopping, CSV logging, and Flax checkpointing."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from hybrid_flood.ml.dataset import ResidualDataset
from hybrid_flood.ml.losses import LossComponents, residual_correction_loss
from hybrid_flood.ml.residual_net import ResidualUNet


@dataclass(frozen=True)
class TrainingResult:
    """Completed training artifacts."""

    state: train_state.TrainState
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    history: list[dict[str, float | int]]
    checkpoint_path: Path
    history_path: Path


def spatial_patch_origins(height: int, width: int, patch_size: int) -> list[tuple[int, int]]:
    """Cover a grid with deterministic patches, including its south/east edges."""
    if patch_size < 1 or patch_size > min(height, width):
        raise ValueError("Patch size must be positive and no larger than either spatial dimension.")

    def starts(length: int) -> list[int]:
        result = list(range(0, length - patch_size + 1, patch_size))
        final = length - patch_size
        if not result or result[-1] != final:
            result.append(final)
        return result

    return [(row, column) for row in starts(height) for column in starts(width)]


def _training_records(
    indices: np.ndarray,
    shape: tuple[int, int],
    patch_size: int,
    patches_per_sample: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    height, width = shape
    records = [
        (
            int(index),
            int(rng.integers(0, height - patch_size + 1)),
            int(rng.integers(0, width - patch_size + 1)),
        )
        for index in indices
        for _ in range(patches_per_sample)
    ]
    rng.shuffle(records)
    return records


def _evaluation_records(
    indices: np.ndarray,
    shape: tuple[int, int],
    patch_size: int,
) -> list[tuple[int, int, int]]:
    return [
        (int(index), row, column)
        for index in indices
        for row, column in spatial_patch_origins(*shape, patch_size)
    ]


def _batch_from_records(
    dataset: ResidualDataset,
    records: list[tuple[int, int, int]],
    patch_size: int,
) -> dict[str, np.ndarray]:
    sample_indices = np.asarray([record[0] for record in records], dtype=np.int32)
    full_inputs = dataset.inputs(np.unique(sample_indices))
    input_lookup = {
        int(index): position for position, index in enumerate(np.unique(sample_indices))
    }
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    raw_depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index, row, column in records:
        selection = np.s_[row : row + patch_size, column : column + patch_size]
        inputs.append(full_inputs[input_lookup[index]][selection])
        targets.append(dataset.target_residual[index][selection])
        raw_depths.append(dataset.raw_depth_t_plus_1[index][selection])
        masks.append(dataset.loss_mask[selection])
    return {
        "inputs": np.stack(inputs).astype(np.float32),
        "target": np.stack(targets).astype(np.float32),
        "raw_depth": np.stack(raw_depths).astype(np.float32),
        "mask": np.stack(masks),
    }


def iter_patch_batches(
    dataset: ResidualDataset,
    indices: np.ndarray,
    *,
    patch_size: int,
    batch_size: int,
    training: bool,
    patches_per_sample: int = 1,
    rng: np.random.Generator | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield channels-last batches while preserving the supplied time split."""
    if batch_size < 1 or patches_per_sample < 1:
        raise ValueError("Batch size and patches per sample must be positive.")
    generator = rng or np.random.default_rng(0)
    records = (
        _training_records(
            indices,
            dataset.spatial_shape,
            patch_size,
            patches_per_sample,
            generator,
        )
        if training
        else _evaluation_records(indices, dataset.spatial_shape, patch_size)
    )
    for start in range(0, len(records), batch_size):
        yield _batch_from_records(dataset, records[start : start + batch_size], patch_size)


def create_train_state(
    model: ResidualUNet,
    rng: jax.Array,
    input_shape: tuple[int, int, int, int],
    optimizer: optax.GradientTransformation,
) -> train_state.TrainState:
    """Initialize model variables and an Optax-backed Flax training state."""
    variables = model.init(rng, jnp.zeros(input_shape, dtype=jnp.float32))
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )


@jax.jit
def training_step(
    state: train_state.TrainState,
    batch: dict[str, jax.Array],
    loss_weights: jax.Array,
    channel_scales: jax.Array,
) -> tuple[train_state.TrainState, LossComponents]:
    """Run one compiled gradient update."""

    def objective(params):
        prediction = state.apply_fn({"params": params}, batch["inputs"])
        components = residual_correction_loss(
            prediction,
            batch["target"],
            batch["raw_depth"],
            batch["mask"],
            mse_weight=loss_weights[0],
            mae_weight=loss_weights[1],
            negative_depth_weight=loss_weights[2],
            channel_scales=channel_scales,
        )
        return components.total, components

    (_, components), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
    return state.apply_gradients(grads=gradients), components


@jax.jit
def evaluation_step(
    state: train_state.TrainState,
    batch: dict[str, jax.Array],
    loss_weights: jax.Array,
    channel_scales: jax.Array,
) -> LossComponents:
    """Evaluate one batch without updating parameters."""
    prediction = state.apply_fn({"params": state.params}, batch["inputs"])
    return residual_correction_loss(
        prediction,
        batch["target"],
        batch["raw_depth"],
        batch["mask"],
        mse_weight=loss_weights[0],
        mae_weight=loss_weights[1],
        negative_depth_weight=loss_weights[2],
        channel_scales=channel_scales,
    )


def _mean_components(components: list[LossComponents]) -> dict[str, float]:
    if not components:
        raise ValueError("A training or validation epoch produced no batches.")
    names = LossComponents._fields
    return {
        name: float(np.mean([float(np.asarray(getattr(item, name))) for item in components]))
        for name in names
    }


def _learning_rate_schedule(config: Any, steps_per_epoch: int):
    schedule = str(config.schedule).lower()
    learning_rate = float(config.learning_rate)
    if schedule == "constant":
        return optax.constant_schedule(learning_rate)
    if schedule != "cosine":
        raise ValueError("Learning-rate schedule must be 'constant' or 'cosine'.")
    total_steps = max(1, int(config.epochs) * steps_per_epoch)
    warmup_steps = min(int(config.warmup_epochs) * steps_per_epoch, total_steps - 1)
    if warmup_steps == 0:
        return optax.cosine_decay_schedule(learning_rate, total_steps)
    return optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.1,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )


def save_checkpoint(
    state: train_state.TrainState,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> Path:
    """Save best Flax parameters and human-readable checkpoint metadata."""
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(flax.serialization.to_bytes(state.params))
    checkpoint.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return checkpoint


def load_checkpoint(template_params: Any, path: str | Path) -> Any:
    """Restore parameters using an initialized model tree as the schema."""
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Residual-network checkpoint does not exist: {checkpoint}")
    return flax.serialization.from_bytes(template_params, checkpoint.read_bytes())


def train_model(
    model: ResidualUNet,
    dataset: ResidualDataset,
    config: Any,
    *,
    seed: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
) -> TrainingResult:
    """Train with chronological validation and restore the best checkpoint."""
    patch_size = int(config.patch_size)
    batch_size = int(config.batch_size)
    patches_per_sample = int(config.patches_per_sample)
    steps_per_epoch = math.ceil(len(dataset.train_indices) * patches_per_sample / batch_size)
    schedule = _learning_rate_schedule(config, steps_per_epoch)
    optimizer_name = str(config.optimizer).lower()
    if optimizer_name == "adam":
        optimizer = optax.adam(schedule)
    elif optimizer_name == "adamw":
        optimizer = optax.adamw(schedule, weight_decay=float(config.weight_decay))
    else:
        raise ValueError("Optimizer must be 'adam' or 'adamw'.")
    if float(config.gradient_clip_norm) > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.gradient_clip_norm)),
            optimizer,
        )

    state = create_train_state(
        model,
        jax.random.PRNGKey(seed),
        (1, patch_size, patch_size, dataset.input_channels),
        optimizer,
    )
    weights = jnp.asarray(
        [
            config.loss.mse_weight,
            config.loss.mae_weight,
            config.loss.negative_depth_weight,
        ],
        dtype=jnp.float32,
    )
    channel_scales = jnp.asarray(
        dataset.metadata["normalization"]["target_residual_std"],
        dtype=jnp.float32,
    )
    rng = np.random.default_rng(seed)
    best_params = state.params
    best_loss = np.inf
    patience_reference_loss = np.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config.epochs) + 1):
        training_components: list[LossComponents] = []
        for batch in iter_patch_batches(
            dataset,
            dataset.train_indices,
            patch_size=patch_size,
            batch_size=batch_size,
            training=True,
            patches_per_sample=patches_per_sample,
            rng=rng,
        ):
            state, components = training_step(state, batch, weights, channel_scales)
            training_components.append(components)

        validation_components = [
            evaluation_step(state, batch, weights, channel_scales)
            for batch in iter_patch_batches(
                dataset,
                dataset.val_indices,
                patch_size=patch_size,
                batch_size=batch_size,
                training=False,
            )
        ]
        training_metrics = _mean_components(training_components)
        validation_metrics = _mean_components(validation_components)
        row: dict[str, float | int] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "learning_rate": float(np.asarray(schedule(state.step))),
        }
        history.append(row)

        validation_loss = validation_metrics["total"]
        absolute_improvement = validation_loss < best_loss
        significant_improvement = validation_loss < patience_reference_loss - float(
            config.early_stopping.min_delta
        )
        if absolute_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            best_params = jax.tree.map(lambda value: np.asarray(value), state.params)
            save_checkpoint(
                state,
                checkpoint_path,
                metadata={
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "input_channels": dataset.input_channels,
                    "output_channels": len(dataset.metadata["target_channels"]),
                    "target_residual_std": list(
                        dataset.metadata["normalization"]["target_residual_std"]
                    ),
                },
            )
        if significant_improvement:
            patience_reference_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config.early_stopping.patience):
            break

    fieldnames = list(history[0])
    with history_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    state = state.replace(params=jax.tree.map(jnp.asarray, best_params))
    return TrainingResult(
        state=state,
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        epochs_completed=len(history),
        history=history,
        checkpoint_path=Path(checkpoint_path),
        history_path=history_file,
    )
